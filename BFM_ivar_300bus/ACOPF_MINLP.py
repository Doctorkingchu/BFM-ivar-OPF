# ACOPF_MINLP.py
# ------------------------------------------------------------
# Pyomo BIM AC-OPF / MINLP for IEEE 300-bus system
# based on ieee300bus.py
#
# Main solve policy:
#   1) Build pandapower network from ieee300bus.case300_opf()
#   2) Read OLTC / switched-shunt metadata from the net
#   3) Extract raw branch data from net['branch_params_pu_table']
#      instead of reconstructing everything from net.line/net.trafo
#   4) Build Pyomo BIM MINLP with:
#        - quadratic generation cost
#        - variable OLTC magnitude taps
#        - binary switched shunts
#        - AC branch thermal limits
#   5) Create incumbent by relaxed NLP + rounding + fixed NLP
#   6) Solve MINLP with SCIP
#   7) Accept SCIP solution only if residual/integrality checks pass
#      otherwise fallback to rounded fixed-NLP incumbent
#
# Important modelling note
# -----------------------
# Unlike the old 41-bus script that read only net.line, this script uses
# the raw MATPOWER-like branch metadata already stored in ieee300bus.py:
#   net['branch_params_pu_table']
# so that r/x/b, off-nominal ratio, phase shift, branch type and synthetic
# thermal ratings are all preserved consistently.
# ------------------------------------------------------------

import copy
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, List, Optional, Any

import numpy as np
import pandas as pd
import pyomo.environ as pyo
import pandapower as pp

import ieee300bus as m


# ============================
# User-tunable global settings
# ============================

# --- NLP (IPOPT) stop conditions ---
IPOPT_TOL = 1e-6
IPOPT_MAX_ITER = 4000
IPOPT_MAX_CPU_TIME = 180

IPOPT_ACCEPTABLE_TOL = 1e-4
IPOPT_ACCEPTABLE_ITER = 12
IPOPT_ACCEPTABLE_CONSTR_VIOL = 1e-4

# --- MINLP (SCIP) hard limits ---
SCIP_TIME_LIMIT = 36000           # seconds, bounded so fallback result is emitted
SCIP_GAP_LIMIT = 0.01           # 1%
SCIP_MEMORY_LIMIT_MB = 8192
SCIP_NODE_LIMIT = 100000

# --- Logging ---
TEE_SOLVER_LOG = False

NETWORK_BUILD_KWARGS = dict(
    line_max_loading_percent=100.0,
    trafo_max_loading_percent=100.0,
    line_smax_mva_overrides=None,
    apply_fixed_bus_shunts=True,
    oltc_step_percent=1.25,
    calibrate_branch_limits_from_pf=True,
    pf_branch_headroom=1.15,
    pf_branch_floor_mva=250.0,
    widen_voltage_bounds_from_pf=True,
    pf_voltage_margin_pu=0.03,
    expose_global_recommendations=False,
)
RUN_PF_WARMSTART = True
RESULT_LOG_PATH = Path(__file__).resolve().with_name("ACOPF_MINLP_results.txt")

# --- Acceptance tolerances for returned solution ---
CHECK_FEAS_TOL = 1e-4
CHECK_INT_TOL = 1e-6


# ============================
# Configuration containers
# ============================
@dataclass
class OLTCBranchConfig:
    tap_min: int
    tap_max: int
    dV_percent: float


@dataclass
class ShuntConfig:
    # bcap_pu > 0 is treated as capacitive susceptance magnitude
    # and is ADDED to B_ii in Ybus when the shunt is ON.
    bcap_pu: float


@dataclass
class BuildConfig:
    oltc_branches: Dict[Tuple[int, int], OLTCBranchConfig]
    shunts: Dict[int, ShuntConfig]
    fix_slack_vm: bool = True


# ============================
# Metadata readers from network
# ============================
def build_cfg_from_net_metadata(net) -> BuildConfig:
    if "fixed_oltc_table" not in net:
        raise KeyError("Network metadata 'fixed_oltc_table' not found in pandapower net.")
    if "fixed_shunt_table" not in net:
        raise KeyError("Network metadata 'fixed_shunt_table' not found in pandapower net.")

    oltc_df = net["fixed_oltc_table"]
    shunt_df = net["fixed_shunt_table"]

    oltc_branches: Dict[Tuple[int, int], OLTCBranchConfig] = {}
    for _, row in oltc_df.iterrows():
        i = int(row["from_bus_pp"])
        j = int(row["to_bus_pp"])
        oltc_branches[(i, j)] = OLTCBranchConfig(
            tap_min=int(row["tap_min"]),
            tap_max=int(row["tap_max"]),
            dV_percent=float(row["dV_percent"]),
        )

    shunts: Dict[int, ShuntConfig] = {}
    for _, row in shunt_df.iterrows():
        b = int(row["bus_pp"])
        shunts[b] = ShuntConfig(bcap_pu=float(row["bcap_pu"]))

    return BuildConfig(
        oltc_branches=oltc_branches,
        shunts=shunts,
        fix_slack_vm=True,
    )


# ============================
# Helpers
# ============================
def _deg2rad(x: float) -> float:
    return float(x) * math.pi / 180.0


def _branch_is_in_cfg_direction(edge: Tuple[int, int], cfg: BuildConfig) -> bool:
    return edge in cfg.oltc_branches


def _default_tap_choice_for_branch(taps: List[int]) -> int:
    return min(taps, key=lambda t: abs(t))


def _build_raw_branch_table(net) -> pd.DataFrame:
    if "branch_params_pu_table" not in net:
        raise KeyError("Network metadata 'branch_params_pu_table' not found in pandapower net.")
    df = net["branch_params_pu_table"].copy()
    if df.empty:
        raise ValueError("branch_params_pu_table is empty.")
    return df


# ============================
# Network extraction
# ============================
def extract_per_unit_data(net, cfg: BuildConfig) -> Dict[str, Any]:
    sn = float(net.sn_mva)

    # buses
    buses = [int(i) for i in net.bus.index]
    nbus = len(buses)
    bus_to_pos = {int(i): k for k, i in enumerate(buses)}
    bus_vn_kv = {int(i): float(net.bus.at[i, "vn_kv"]) for i in buses}
    vmin = {int(i): float(net.bus.at[i, "min_vm_pu"]) for i in buses}
    vmax = {int(i): float(net.bus.at[i, "max_vm_pu"]) for i in buses}

    # slack
    if len(net.ext_grid.index) < 1:
        raise ValueError("pandapower net must have an ext_grid.")
    eg0 = int(net.ext_grid.index[0])
    slack_bus = int(net.ext_grid.at[eg0, "bus"])
    slack_vm_pu = float(net.ext_grid.at[eg0, "vm_pu"])

    # loads aggregated per bus
    Pd = {i: 0.0 for i in buses}
    Qd = {i: 0.0 for i in buses}
    if hasattr(net, "load") and len(net.load.index) > 0:
        for li in net.load.index:
            b = int(net.load.at[li, "bus"])
            Pd[b] += float(net.load.at[li, "p_mw"])
            Qd[b] += float(net.load.at[li, "q_mvar"])
    Pd_pu = {i: Pd[i] / sn for i in buses}
    Qd_pu = {i: Qd[i] / sn for i in buses}

    # poly costs
    def _find_poly_cost(et: str, element: int):
        if (not hasattr(net, "poly_cost")) or net.poly_cost is None or net.poly_cost.empty:
            return (0.0, 0.0, 0.0)
        df = net.poly_cost
        row = df[(df["et"] == et) & (df["element"] == element)]
        if row.empty:
            return (0.0, 0.0, 0.0)
        r = row.iloc[0]
        return (
            float(r.get("cp2_eur_per_mw2", 0.0)),
            float(r.get("cp1_eur_per_mw", 0.0)),
            float(r.get("cp0_eur", 0.0)),
        )

    # generators = ext_grid + gen
    gen_records = []

    for eg in net.ext_grid.index:
        eg = int(eg)
        b = int(net.ext_grid.at[eg, "bus"])
        pmin = float(net.ext_grid.at[eg, "min_p_mw"]) if "min_p_mw" in net.ext_grid.columns else -1e9
        pmax = float(net.ext_grid.at[eg, "max_p_mw"]) if "max_p_mw" in net.ext_grid.columns else 1e9
        qmin = float(net.ext_grid.at[eg, "min_q_mvar"]) if "min_q_mvar" in net.ext_grid.columns else -1e9
        qmax = float(net.ext_grid.at[eg, "max_q_mvar"]) if "max_q_mvar" in net.ext_grid.columns else 1e9
        c2, c1, c0 = _find_poly_cost("ext_grid", eg)
        gen_records.append({
            "type": "ext_grid",
            "id": eg,
            "bus": b,
            "pmin_pu": pmin / sn,
            "pmax_pu": pmax / sn,
            "qmin_pu": qmin / sn,
            "qmax_pu": qmax / sn,
            "c2": c2, "c1": c1, "c0": c0,
        })

    if hasattr(net, "gen") and len(net.gen.index) > 0:
        for gi in net.gen.index:
            gi = int(gi)
            b = int(net.gen.at[gi, "bus"])
            pmin = float(net.gen.at[gi, "min_p_mw"])
            pmax = float(net.gen.at[gi, "max_p_mw"])
            qmin = float(net.gen.at[gi, "min_q_mvar"])
            qmax = float(net.gen.at[gi, "max_q_mvar"])
            c2, c1, c0 = _find_poly_cost("gen", gi)
            gen_records.append({
                "type": "gen",
                "id": gi,
                "bus": b,
                "pmin_pu": pmin / sn,
                "pmax_pu": pmax / sn,
                "qmin_pu": qmin / sn,
                "qmax_pu": qmax / sn,
                "c2": c2, "c1": c1, "c0": c0,
            })

    # raw branch data from ieee118bus.py metadata
    brdf = _build_raw_branch_table(net)

    E = []
    g = {}
    b = {}
    bc = {}
    smax_pu = {}
    alpha_fix = {}
    delta_fix = {}
    phi_rad = {}
    branch_meta = {}

    for _, row in brdf.iterrows():
        i = int(row["from_bus_pp"])
        j = int(row["to_bus_pp"])
        edge = (i, j)
        E.append(edge)

        r_pu = float(row["r_pu"])
        x_pu = float(row["x_pu"])
        b_pu = float(row["b_pu"])
        ratio_raw = float(row.get("ratio_raw", 0.0))
        angle_deg = float(row.get("angle_deg", 0.0))
        is_transformer_like = bool(row.get("is_transformer_like", False))
        smax_mva = float(row["synthetic_smax_mva"])

        y = 1.0 / complex(r_pu, x_pu) if (abs(r_pu) > 0.0 or abs(x_pu) > 0.0) else complex(0.0, 0.0)
        g[edge] = float(y.real)
        b[edge] = float(y.imag)
        bc[edge] = float(b_pu)
        smax_pu[edge] = smax_mva / sn

        # MATPOWER convention: ratio = 0 means ordinary line => tau = 1
        tau = 1.0 if abs(ratio_raw) <= 1e-12 else float(ratio_raw)
        alpha_fix[edge] = 1.0 / tau
        delta_fix[edge] = 1.0 / (tau * tau)
        phi_rad[edge] = _deg2rad(angle_deg)

        branch_meta[edge] = {
            "branch_id": int(row["branch_id"]),
            "is_transformer_like": is_transformer_like,
            "element_type": str(row["element_type"]),
            "element_index": int(row["element_index"]),
            "ratio_raw": ratio_raw,
            "angle_deg": angle_deg,
            "from_bus_mp": int(row["from_bus_mp"]),
            "to_bus_mp": int(row["to_bus_mp"]),
        }

    E_set = set(E)

    # OLTC sets from cfg
    T = []
    K = {}
    alpha_tap = {}
    delta_tap = {}
    recommended_tap = {}

    if "fixed_oltc_table" in net:
        oltc_df = net["fixed_oltc_table"]
    else:
        oltc_df = pd.DataFrame()

    rec_tap_lookup = {}
    if not oltc_df.empty and "recommended_tap" in oltc_df.columns:
        for _, row in oltc_df.iterrows():
            rec_tap_lookup[(int(row["from_bus_pp"]), int(row["to_bus_pp"]))] = int(row["recommended_tap"])

    for (u, v), tcfg in cfg.oltc_branches.items():
        if (u, v) not in E_set:
            if (v, u) in E_set:
                raise ValueError(
                    f"OLTC branch {(u, v)} not found in raw directed branches, but reversed edge {(v, u)} exists. "
                    f"Tap is modeled on the first-bus side, so direction must match metadata."
                )
            raise ValueError(f"OLTC branch {(u, v)} not found in raw branch set.")

        ij = (u, v)
        T.append(ij)
        taps = list(range(int(tcfg.tap_min), int(tcfg.tap_max) + 1))
        K[ij] = taps

        for tap in taps:
            tau = 1.0 + (tap * float(tcfg.dV_percent)) / 100.0
            alpha_tap[(ij, tap)] = 1.0 / tau
            delta_tap[(ij, tap)] = 1.0 / (tau * tau)

        recommended_tap[ij] = int(rec_tap_lookup.get(ij, 0))

    # switched shunts
    C = sorted([int(i) for i in cfg.shunts.keys()])
    bcap_pu = {i: float(cfg.shunts[i].bcap_pu) for i in C}

    recommended_shunt_status = {i: 0 for i in C}
    if "recommended_nonexact_shunt_status" in net:
        rs = net["recommended_nonexact_shunt_status"]
        if not rs.empty:
            for _, row in rs.iterrows():
                bi = int(row["bus_pp"])
                if bi in recommended_shunt_status:
                    recommended_shunt_status[bi] = int(row["status"])

    # fixed bus shunts from raw MATPOWER bus Gs / Bs
    Gsh_fixed = {i: 0.0 for i in buses}
    Bsh_fixed = {i: 0.0 for i in buses}
    if "fixed_bus_shunt_table" in net:
        fs = net["fixed_bus_shunt_table"]
        if not fs.empty:
            for _, row in fs.iterrows():
                bi = int(row["bus_pp"])
                Gsh_fixed[bi] += float(row["g_pu"])
                Bsh_fixed[bi] += float(row["b_pu"])

    # Base Y0: includes all FIXED lines/branches and fixed bus shunts.
    # Candidate OLTC branches are excluded and added later through Pyomo expressions.
    Y0 = np.zeros((nbus, nbus), dtype=complex)
    T_set = set(T)

    def _add_fixed_branch_to_Y0(i: int, j: int, gij: float, bij: float, bcij: float, alpha: float, delta: float, phi: float):
        pi = bus_to_pos[i]
        pj = bus_to_pos[j]

        y = complex(gij, bij)
        ysh = 1j * (bcij / 2.0)

        # MATPOWER-style off-nominal tap on from-bus side
        Yff = (y + ysh) * delta
        Ytt = (y + ysh)
        Yft = -y * alpha * complex(math.cos(phi), math.sin(phi))
        Ytf = -y * alpha * complex(math.cos(-phi), math.sin(-phi))

        Y0[pi, pi] += Yff
        Y0[pj, pj] += Ytt
        Y0[pi, pj] += Yft
        Y0[pj, pi] += Ytf

    for (i, j) in E:
        if (i, j) in T_set:
            continue
        _add_fixed_branch_to_Y0(
            i, j,
            gij=g[(i, j)],
            bij=b[(i, j)],
            bcij=bc[(i, j)],
            alpha=alpha_fix[(i, j)],
            delta=delta_fix[(i, j)],
            phi=phi_rad[(i, j)],
        )

    # fixed bus shunts on diagonal
    for i in buses:
        pi = bus_to_pos[i]
        Y0[pi, pi] += complex(Gsh_fixed[i], Bsh_fixed[i])

    return {
        "sn_mva": sn,
        "buses": buses,
        "bus_to_pos": bus_to_pos,
        "bus_vn_kv": bus_vn_kv,
        "slack_bus": slack_bus,
        "slack_vm_pu": slack_vm_pu,
        "vmin": vmin,
        "vmax": vmax,
        "Pd_pu": Pd_pu,
        "Qd_pu": Qd_pu,
        "gen_records": gen_records,
        "E": E,
        "g": g,
        "b": b,
        "bc": bc,
        "smax_pu": smax_pu,
        "alpha_fix": alpha_fix,
        "delta_fix": delta_fix,
        "phi_rad": phi_rad,
        "branch_meta": branch_meta,
        "T": T,
        "K": K,
        "alpha_tap": alpha_tap,
        "delta_tap": delta_tap,
        "recommended_tap": recommended_tap,
        "C": C,
        "bcap_pu": bcap_pu,
        "recommended_shunt_status": recommended_shunt_status,
        "Gsh_fixed": Gsh_fixed,
        "Bsh_fixed": Bsh_fixed,
        "G0": Y0.real,
        "B0": Y0.imag,
        "fix_slack_vm": cfg.fix_slack_vm,
    }


# ============================
# Pyomo model builder
# ============================
def build_pyomo_model(data: Dict[str, Any], relax_binaries: bool = False) -> pyo.ConcreteModel:
    sn = data["sn_mva"]
    buses = data["buses"]
    bus_to_pos = data["bus_to_pos"]
    slack_bus = data["slack_bus"]
    slack_vm_pu = data["slack_vm_pu"]

    E = data["E"]
    g = data["g"]
    b = data["b"]
    bc = data["bc"]
    smax_pu = data["smax_pu"]
    alpha_fix = data["alpha_fix"]
    delta_fix = data["delta_fix"]
    phi_rad = data["phi_rad"]

    T = data["T"]
    K = data["K"]
    alpha_tap = data["alpha_tap"]
    delta_tap = data["delta_tap"]

    C = data["C"]
    bcap_pu = data["bcap_pu"]

    G0 = data["G0"]
    B0 = data["B0"]

    Pd = data["Pd_pu"]
    Qd = data["Qd_pu"]
    vmin = data["vmin"]
    vmax = data["vmax"]

    gen_records = data["gen_records"]
    Gset = list(range(len(gen_records)))

    model = pyo.ConcreteModel(name="BIM_MINLP_ACOPF_300BUS_OLTC_SHUNT")

    # Sets
    model.N = pyo.Set(initialize=buses, ordered=True)
    model.G = pyo.Set(initialize=Gset, ordered=True)
    model.E = pyo.Set(initialize=E, dimen=2, ordered=True)
    model.T = pyo.Set(initialize=T, dimen=2, ordered=True)
    model.C = pyo.Set(initialize=C, ordered=True)

    # Base admittance
    model.G0 = pyo.Param(
        model.N, model.N,
        initialize=lambda m_, i, k: float(G0[bus_to_pos[i], bus_to_pos[k]]),
        mutable=False,
    )
    model.B0 = pyo.Param(
        model.N, model.N,
        initialize=lambda m_, i, k: float(B0[bus_to_pos[i], bus_to_pos[k]]),
        mutable=False,
    )

    # Load
    model.Pd = pyo.Param(model.N, initialize=lambda m_, i: float(Pd[i]), mutable=False)
    model.Qd = pyo.Param(model.N, initialize=lambda m_, i: float(Qd[i]), mutable=False)

    # Voltage bounds
    model.Vmin = pyo.Param(model.N, initialize=lambda m_, i: float(vmin[i]), mutable=False)
    model.Vmax = pyo.Param(model.N, initialize=lambda m_, i: float(vmax[i]), mutable=False)

    # Generator params
    model.gen_bus = pyo.Param(model.G, initialize=lambda m_, gg: int(gen_records[gg]["bus"]), within=pyo.Any)
    model.Pgmin = pyo.Param(model.G, initialize=lambda m_, gg: float(gen_records[gg]["pmin_pu"]))
    model.Pgmax = pyo.Param(model.G, initialize=lambda m_, gg: float(gen_records[gg]["pmax_pu"]))
    model.Qgmin = pyo.Param(model.G, initialize=lambda m_, gg: float(gen_records[gg]["qmin_pu"]))
    model.Qgmax = pyo.Param(model.G, initialize=lambda m_, gg: float(gen_records[gg]["qmax_pu"]))
    model.c2 = pyo.Param(model.G, initialize=lambda m_, gg: float(gen_records[gg]["c2"]))
    model.c1 = pyo.Param(model.G, initialize=lambda m_, gg: float(gen_records[gg]["c1"]))
    model.c0 = pyo.Param(model.G, initialize=lambda m_, gg: float(gen_records[gg]["c0"]))

    # Branch params
    model.g = pyo.Param(model.E, initialize=lambda m_, i, j: float(g[(i, j)]))
    model.b = pyo.Param(model.E, initialize=lambda m_, i, j: float(b[(i, j)]))
    model.bc = pyo.Param(model.E, initialize=lambda m_, i, j: float(bc[(i, j)]))
    model.Smax = pyo.Param(model.E, initialize=lambda m_, i, j: float(smax_pu[(i, j)]))
    model.alpha_fix = pyo.Param(model.E, initialize=lambda m_, i, j: float(alpha_fix[(i, j)]))
    model.delta_fix = pyo.Param(model.E, initialize=lambda m_, i, j: float(delta_fix[(i, j)]))
    model.phi = pyo.Param(model.E, initialize=lambda m_, i, j: float(phi_rad[(i, j)]))

    # Shunt params
    model.bcap = pyo.Param(model.C, initialize=lambda m_, i: float(bcap_pu[i]), default=0.0)

    # OLTC tap index
    beta_index = []
    for (i, j) in T:
        for tap in K[(i, j)]:
            beta_index.append((i, j, int(tap)))
    model.BETA_INDEX = pyo.Set(initialize=beta_index, dimen=3, ordered=True)

    model.alpha_tap = pyo.Param(
        model.BETA_INDEX,
        initialize=lambda m_, i, j, tap: float(alpha_tap[((i, j), int(tap))]),
        mutable=False,
    )
    model.delta_tap = pyo.Param(
        model.BETA_INDEX,
        initialize=lambda m_, i, j, tap: float(delta_tap[((i, j), int(tap))]),
        mutable=False,
    )

    # Variables
    model.Pg = pyo.Var(model.G, bounds=lambda m_, gg: (m_.Pgmin[gg], m_.Pgmax[gg]))
    model.Qg = pyo.Var(model.G, bounds=lambda m_, gg: (m_.Qgmin[gg], m_.Qgmax[gg]))

    model.V = pyo.Var(model.N, bounds=lambda m_, i: (m_.Vmin[i], m_.Vmax[i]))
    model.theta = pyo.Var(model.N, bounds=(-math.pi, math.pi))
    model.v = pyo.Var(model.N, bounds=lambda m_, i: (m_.Vmin[i] ** 2, m_.Vmax[i] ** 2))

    model.Pinj = pyo.Var(model.N)
    model.Qinj = pyo.Var(model.N)

    model.Pij = pyo.Var(model.E)
    model.Qij = pyo.Var(model.E)

    if relax_binaries:
        model.beta = pyo.Var(model.BETA_INDEX, bounds=(0.0, 1.0))
        model.a_sh = pyo.Var(model.C, bounds=(0.0, 1.0))
    else:
        model.beta = pyo.Var(model.BETA_INDEX, within=pyo.Binary)
        model.a_sh = pyo.Var(model.C, within=pyo.Binary)

    # alpha, delta bounds
    alpha_bounds = {}
    delta_bounds = {}
    for (i, j) in T:
        avals = [alpha_tap[((i, j), tap)] for tap in K[(i, j)]]
        dvals = [delta_tap[((i, j), tap)] for tap in K[(i, j)]]
        alpha_bounds[(i, j)] = (min(avals), max(avals))
        delta_bounds[(i, j)] = (min(dvals), max(dvals))

    model.alpha = pyo.Var(model.T, bounds=lambda m_, i, j: alpha_bounds[(i, j)])
    model.delta = pyo.Var(model.T, bounds=lambda m_, i, j: delta_bounds[(i, j)])

    # Definitions
    model.v_def = pyo.Constraint(model.N, rule=lambda m_, i: m_.v[i] == m_.V[i] ** 2)

    model.Pinj_def = pyo.Constraint(
        model.N,
        rule=lambda m_, i: m_.Pinj[i] == sum(m_.Pg[gg] for gg in m_.G if int(m_.gen_bus[gg]) == int(i)) - m_.Pd[i]
    )
    model.Qinj_def = pyo.Constraint(
        model.N,
        rule=lambda m_, i: m_.Qinj[i] == sum(m_.Qg[gg] for gg in m_.G if int(m_.gen_bus[gg]) == int(i)) - m_.Qd[i]
    )

    # OLTC one-hot + selection
    def onehot_rule(m_, i, j):
        return sum(m_.beta[i, j, int(t)] for t in K[(i, j)]) == 1
    model.onehot = pyo.Constraint(model.T, rule=onehot_rule)

    def alpha_sel_rule(m_, i, j):
        return m_.alpha[i, j] == sum(
            m_.alpha_tap[i, j, int(t)] * m_.beta[i, j, int(t)] for t in K[(i, j)]
        )
    model.alpha_sel = pyo.Constraint(model.T, rule=alpha_sel_rule)

    def delta_sel_rule(m_, i, j):
        return m_.delta[i, j] == sum(
            m_.delta_tap[i, j, int(t)] * m_.beta[i, j, int(t)] for t in K[(i, j)]
        )
    model.delta_sel = pyo.Constraint(model.T, rule=delta_sel_rule)

    # Slack
    model.slack_angle = pyo.Constraint(expr=model.theta[slack_bus] == 0.0)
    if data["fix_slack_vm"]:
        model.slack_vm = pyo.Constraint(expr=model.V[slack_bus] == float(slack_vm_pu))

    # Ybus assembly expressions
    T_set = set(T)
    C_set = set(C)

    def Gexpr(m_, i, k):
        expr = m_.G0[i, k]
        for (p, q) in T:
            gij = m_.g[p, q]
            bij = m_.b[p, q]
            bcij = m_.bc[p, q]
            phi = m_.phi[p, q]
            cphi = math.cos(float(phi))
            sphi = math.sin(float(phi))

            if i == p and k == p:
                expr += gij * m_.delta[p, q]
            elif i == q and k == q:
                expr += gij
            elif i == p and k == q:
                expr += -(gij * cphi - bij * sphi) * m_.alpha[p, q]
            elif i == q and k == p:
                expr += -(gij * cphi + bij * sphi) * m_.alpha[p, q]
        return expr

    def Bexpr(m_, i, k):
        expr = m_.B0[i, k]
        for (p, q) in T:
            gij = m_.g[p, q]
            bij = m_.b[p, q]
            bcij = m_.bc[p, q]
            phi = m_.phi[p, q]
            cphi = math.cos(float(phi))
            sphi = math.sin(float(phi))

            if i == p and k == p:
                expr += (bij + 0.5 * bcij) * m_.delta[p, q]
            elif i == q and k == q:
                expr += (bij + 0.5 * bcij)
            elif i == p and k == q:
                expr += -(gij * sphi + bij * cphi) * m_.alpha[p, q]
            elif i == q and k == p:
                expr += +(gij * sphi - bij * cphi) * m_.alpha[p, q]

        # switched capacitor shunt: add +bcap to B_ii when ON
        if i == k and i in C_set:
            expr += m_.a_sh[i] * m_.bcap[i]
        return expr

    model.Gik = pyo.Expression(model.N, model.N, rule=Gexpr)
    model.Bik = pyo.Expression(model.N, model.N, rule=Bexpr)

    # BIM power balance
    def BIM_P_rule(m_, i):
        return m_.Pinj[i] == m_.V[i] * sum(
            m_.V[k] * (
                m_.Gik[i, k] * pyo.cos(m_.theta[i] - m_.theta[k]) +
                m_.Bik[i, k] * pyo.sin(m_.theta[i] - m_.theta[k])
            )
            for k in m_.N
        )
    model.BIM_P = pyo.Constraint(model.N, rule=BIM_P_rule)

    def BIM_Q_rule(m_, i):
        return m_.Qinj[i] == m_.V[i] * sum(
            m_.V[k] * (
                m_.Gik[i, k] * pyo.sin(m_.theta[i] - m_.theta[k]) -
                m_.Bik[i, k] * pyo.cos(m_.theta[i] - m_.theta[k])
            )
            for k in m_.N
        )
    model.BIM_Q = pyo.Constraint(model.N, rule=BIM_Q_rule)

    # Branch explicit flows
    def alphaE_rule(m_, i, j):
        return m_.alpha[i, j] if (i, j) in T_set else m_.alpha_fix[i, j]

    def deltaE_rule(m_, i, j):
        return m_.delta[i, j] if (i, j) in T_set else m_.delta_fix[i, j]

    model.alphaE = pyo.Expression(model.E, rule=alphaE_rule)
    model.deltaE = pyo.Expression(model.E, rule=deltaE_rule)

    def Pij_rule(m_, i, j):
        gij = m_.g[i, j]
        bij = m_.b[i, j]
        phi = m_.phi[i, j]
        ang = m_.theta[i] - m_.theta[j] - phi
        return m_.Pij[i, j] == (
            gij * m_.deltaE[i, j] * m_.v[i]
            - m_.V[i] * m_.V[j] * m_.alphaE[i, j] * (
                gij * pyo.cos(ang) + bij * pyo.sin(ang)
            )
        )
    model.Pij_def = pyo.Constraint(model.E, rule=Pij_rule)

    def Qij_rule(m_, i, j):
        gij = m_.g[i, j]
        bij = m_.b[i, j]
        bcij = m_.bc[i, j]
        phi = m_.phi[i, j]
        ang = m_.theta[i] - m_.theta[j] - phi
        return m_.Qij[i, j] == (
            -(bij * m_.deltaE[i, j] + 0.5 * bcij * m_.deltaE[i, j]) * m_.v[i]
            - m_.V[i] * m_.V[j] * m_.alphaE[i, j] * (
                gij * pyo.sin(ang) - bij * pyo.cos(ang)
            )
        )
    model.Qij_def = pyo.Constraint(model.E, rule=Qij_rule)

    model.thermal = pyo.Constraint(
        model.E,
        rule=lambda m_, i, j: m_.Pij[i, j] ** 2 + m_.Qij[i, j] ** 2 <= (m_.Smax[i, j] ** 2)
    )

    # Objective (Pg in pu -> MW)
    model.obj = pyo.Objective(
        rule=lambda m_: sum(
            m_.c2[gg] * (sn * m_.Pg[gg]) ** 2 + m_.c1[gg] * (sn * m_.Pg[gg]) + m_.c0[gg]
            for gg in m_.G
        ),
        sense=pyo.minimize,
    )

    return model


# ============================
# Initialization helpers
# ============================
def initialize_model_flat(model: pyo.ConcreteModel, data: Dict[str, Any]):
    slack_bus = data["slack_bus"]
    slack_vm_pu = data["slack_vm_pu"]
    gen_records = data["gen_records"]
    recommended_tap = data["recommended_tap"]
    recommended_shunt_status = data["recommended_shunt_status"]

    for i in model.N:
        vm0 = slack_vm_pu if i == slack_bus else 1.0
        model.V[i].set_value(vm0)
        model.theta[i].set_value(0.0)
        model.v[i].set_value(vm0 ** 2)
        model.Pinj[i].set_value(-pyo.value(model.Pd[i]))
        model.Qinj[i].set_value(-pyo.value(model.Qd[i]))

    for gg in model.G:
        rec = gen_records[int(gg)]
        pguess = max(rec["pmin_pu"], min(rec["pmax_pu"], 0.0))
        qguess = max(rec["qmin_pu"], min(rec["qmax_pu"], 0.0))
        model.Pg[gg].set_value(pguess)
        model.Qg[gg].set_value(qguess)

    for (i, j, tap) in model.BETA_INDEX:
        chosen = recommended_tap.get((i, j), _default_tap_choice_for_branch(data["K"][(i, j)]))
        model.beta[i, j, tap].set_value(1.0 if tap == chosen else 0.0)

    for i in model.C:
        model.a_sh[i].set_value(float(recommended_shunt_status.get(int(i), 0)))

    for (i, j) in model.T:
        chosen = recommended_tap.get((i, j), _default_tap_choice_for_branch(data["K"][(i, j)]))
        model.alpha[i, j].set_value(data["alpha_tap"][((i, j), chosen)])
        model.delta[i, j].set_value(data["delta_tap"][((i, j), chosen)])

    for (i, j) in model.E:
        model.Pij[i, j].set_value(0.0)
        model.Qij[i, j].set_value(0.0)


def initialize_model_from_pf(model: pyo.ConcreteModel, data: Dict[str, Any], net) -> bool:
    if not RUN_PF_WARMSTART:
        initialize_model_flat(model, data)
        return False

    try:
        pp.runpp(
            net,
            algorithm="nr",
            init="flat",
            calculate_voltage_angles=True,
            enforce_q_lims=False,
            numba=False,
        )

        # Bus voltages
        for i in model.N:
            vm = float(net.res_bus.at[i, "vm_pu"])
            va_deg = float(net.res_bus.at[i, "va_degree"])
            va = math.radians(va_deg)
            model.V[i].set_value(vm)
            model.theta[i].set_value(va)
            model.v[i].set_value(vm * vm)

        # Generator guesses
        ext_grid_map = {}
        if hasattr(net, "res_ext_grid") and len(net.res_ext_grid.index) > 0:
            for eg in net.ext_grid.index:
                ext_grid_map[("ext_grid", int(eg))] = (
                    float(net.res_ext_grid.at[eg, "p_mw"]) / data["sn_mva"],
                    float(net.res_ext_grid.at[eg, "q_mvar"]) / data["sn_mva"],
                )

        gen_map = {}
        if hasattr(net, "res_gen") and len(net.res_gen.index) > 0:
            for gi in net.gen.index:
                gen_map[("gen", int(gi))] = (
                    float(net.res_gen.at[gi, "p_mw"]) / data["sn_mva"],
                    float(net.res_gen.at[gi, "q_mvar"]) / data["sn_mva"],
                )

        for gg in model.G:
            rec = data["gen_records"][int(gg)]
            key = (rec["type"], rec["id"])
            if key in ext_grid_map:
                pg, qg = ext_grid_map[key]
            elif key in gen_map:
                pg, qg = gen_map[key]
            else:
                pg, qg = 0.0, 0.0

            pg = max(rec["pmin_pu"], min(rec["pmax_pu"], pg))
            qg = max(rec["qmin_pu"], min(rec["qmax_pu"], qg))
            model.Pg[gg].set_value(pg)
            model.Qg[gg].set_value(qg)

        for i in model.N:
            pgen_i = sum(pyo.value(model.Pg[gg]) for gg in model.G if int(pyo.value(model.gen_bus[gg])) == int(i))
            qgen_i = sum(pyo.value(model.Qg[gg]) for gg in model.G if int(pyo.value(model.gen_bus[gg])) == int(i))
            model.Pinj[i].set_value(pgen_i - pyo.value(model.Pd[i]))
            model.Qinj[i].set_value(qgen_i - pyo.value(model.Qd[i]))

        # Discrete starts from metadata recommendations
        recommended_tap = data["recommended_tap"]
        recommended_shunt_status = data["recommended_shunt_status"]

        for (i, j, tap) in model.BETA_INDEX:
            chosen = recommended_tap.get((i, j), _default_tap_choice_for_branch(data["K"][(i, j)]))
            model.beta[i, j, tap].set_value(1.0 if tap == chosen else 0.0)

        for i in model.C:
            model.a_sh[i].set_value(float(recommended_shunt_status.get(int(i), 0)))

        for (i, j) in model.T:
            chosen = recommended_tap.get((i, j), _default_tap_choice_for_branch(data["K"][(i, j)]))
            model.alpha[i, j].set_value(data["alpha_tap"][((i, j), chosen)])
            model.delta[i, j].set_value(data["delta_tap"][((i, j), chosen)])

        for (i, j) in model.E:
            model.Pij[i, j].set_value(0.0)
            model.Qij[i, j].set_value(0.0)

        return True

    except Exception:
        initialize_model_flat(model, data)
        return False


def apply_discrete_choice_start(
    model: pyo.ConcreteModel,
    data: Dict[str, Any],
    tap_choice: Dict[Tuple[int, int], int],
    sh_choice: Dict[int, int],
):
    for (i, j, tap) in model.BETA_INDEX:
        chosen = tap_choice.get((i, j), _default_tap_choice_for_branch(data["K"][(i, j)]))
        model.beta[i, j, tap].set_value(1.0 if tap == chosen else 0.0)

    for i in model.C:
        model.a_sh[i].set_value(float(sh_choice.get(i, 0)))

    for (i, j) in model.T:
        chosen = tap_choice.get((i, j), _default_tap_choice_for_branch(data["K"][(i, j)]))
        model.alpha[i, j].set_value(data["alpha_tap"][((i, j), chosen)])
        model.delta[i, j].set_value(data["delta_tap"][((i, j), chosen)])


def fix_discrete_choices(
    model: pyo.ConcreteModel,
    data: Dict[str, Any],
    tap_choice: Dict[Tuple[int, int], int],
    sh_choice: Dict[int, int],
):
    for (i, j, tap) in model.BETA_INDEX:
        chosen = tap_choice[(i, j)]
        model.beta[i, j, tap].fix(1 if tap == chosen else 0)

    for i in model.C:
        model.a_sh[i].fix(int(sh_choice[i]))


def read_discrete_choices_from_relaxed(model: pyo.ConcreteModel, data: Dict[str, Any]):
    tap_choice = {}
    sh_choice = {}

    for (i, j) in model.T:
        best_tap = None
        best_val = -1.0
        for tap in data["K"][(i, j)]:
            v = pyo.value(model.beta[i, j, tap])
            if v > best_val:
                best_val = v
                best_tap = tap
        tap_choice[(i, j)] = int(best_tap)

    for i in model.C:
        v = pyo.value(model.a_sh[i])
        sh_choice[int(i)] = 1 if v >= 0.5 else 0

    return tap_choice, sh_choice


# ============================
# Solver helpers
# ============================
def _count_discrete_vars(model: pyo.ConcreteModel) -> Tuple[int, int]:
    nbin, nint = 0, 0
    for v in model.component_data_objects(pyo.Var, descend_into=True):
        if v.is_binary():
            nbin += 1
        elif v.is_integer():
            nint += 1
    return nbin, nint


def _solve_nlp(
    model: pyo.ConcreteModel,
    tee: bool = False,
    tol: float = IPOPT_TOL,
    max_iter: int = IPOPT_MAX_ITER,
    max_cpu_time: int = IPOPT_MAX_CPU_TIME,
) -> Optional[float]:
    for name in ["ipopt", "cyipopt", "appsi_ipopt"]:
        solver = pyo.SolverFactory(name)
        if solver is None or not solver.available(exception_flag=False):
            continue

        try:
            solver.options["tol"] = float(tol)
            solver.options["max_iter"] = int(max_iter)
            solver.options["max_cpu_time"] = float(max_cpu_time)
            solver.options["acceptable_tol"] = float(IPOPT_ACCEPTABLE_TOL)
            solver.options["acceptable_iter"] = int(IPOPT_ACCEPTABLE_ITER)
            solver.options["acceptable_constr_viol_tol"] = float(IPOPT_ACCEPTABLE_CONSTR_VIOL)
            solver.options["print_level"] = 5 if tee else 0
        except Exception:
            pass

        try:
            res = solver.solve(model, tee=tee)
        except Exception:
            continue

        tc = res.solver.termination_condition
        if tc in [
            pyo.TerminationCondition.optimal,
            pyo.TerminationCondition.locallyOptimal,
            pyo.TerminationCondition.feasible,
            pyo.TerminationCondition.maxIterations,
            pyo.TerminationCondition.maxTimeLimit,
        ]:
            try:
                return float(pyo.value(model.obj))
            except Exception:
                return None

    return None


# ============================
# Solution quality check
# ============================
def evaluate_solution_quality(model: pyo.ConcreteModel, data: Dict[str, Any]) -> Dict[str, float]:
    out = {
        "max_v_resid": 0.0,
        "max_pinj_resid": 0.0,
        "max_qinj_resid": 0.0,
        "max_onehot_resid": 0.0,
        "max_alpha_resid": 0.0,
        "max_delta_resid": 0.0,
        "max_bim_p_resid": 0.0,
        "max_bim_q_resid": 0.0,
        "max_thermal_viol": 0.0,
        "max_bin_frac": 0.0,
    }

    # v = V^2
    for i in model.N:
        resid = abs(pyo.value(model.v[i]) - pyo.value(model.V[i]) ** 2)
        out["max_v_resid"] = max(out["max_v_resid"], resid)

    # Pinj / Qinj definition
    for i in model.N:
        lhs_p = pyo.value(model.Pinj[i])
        rhs_p = sum(
            pyo.value(model.Pg[gg]) for gg in model.G
            if int(pyo.value(model.gen_bus[gg])) == int(i)
        ) - pyo.value(model.Pd[i])
        out["max_pinj_resid"] = max(out["max_pinj_resid"], abs(lhs_p - rhs_p))

        lhs_q = pyo.value(model.Qinj[i])
        rhs_q = sum(
            pyo.value(model.Qg[gg]) for gg in model.G
            if int(pyo.value(model.gen_bus[gg])) == int(i)
        ) - pyo.value(model.Qd[i])
        out["max_qinj_resid"] = max(out["max_qinj_resid"], abs(lhs_q - rhs_q))

    # one-hot, alpha, delta
    for (i, j) in model.T:
        onehot = sum(pyo.value(model.beta[i, j, int(t)]) for t in data["K"][(i, j)])
        out["max_onehot_resid"] = max(out["max_onehot_resid"], abs(onehot - 1.0))

        alpha_rhs = sum(
            pyo.value(model.alpha_tap[i, j, int(t)]) * pyo.value(model.beta[i, j, int(t)])
            for t in data["K"][(i, j)]
        )
        delta_rhs = sum(
            pyo.value(model.delta_tap[i, j, int(t)]) * pyo.value(model.beta[i, j, int(t)])
            for t in data["K"][(i, j)]
        )
        out["max_alpha_resid"] = max(out["max_alpha_resid"], abs(pyo.value(model.alpha[i, j]) - alpha_rhs))
        out["max_delta_resid"] = max(out["max_delta_resid"], abs(pyo.value(model.delta[i, j]) - delta_rhs))

    # BIM residual
    for i in model.N:
        rhs_p = pyo.value(
            model.V[i] * sum(
                model.V[k] * (
                    model.Gik[i, k] * pyo.cos(model.theta[i] - model.theta[k]) +
                    model.Bik[i, k] * pyo.sin(model.theta[i] - model.theta[k])
                )
                for k in model.N
            )
        )
        rhs_q = pyo.value(
            model.V[i] * sum(
                model.V[k] * (
                    model.Gik[i, k] * pyo.sin(model.theta[i] - model.theta[k]) -
                    model.Bik[i, k] * pyo.cos(model.theta[i] - model.theta[k])
                )
                for k in model.N
            )
        )
        out["max_bim_p_resid"] = max(out["max_bim_p_resid"], abs(pyo.value(model.Pinj[i]) - rhs_p))
        out["max_bim_q_resid"] = max(out["max_bim_q_resid"], abs(pyo.value(model.Qinj[i]) - rhs_q))

    # thermal
    for (i, j) in model.E:
        viol = pyo.value(model.Pij[i, j]) ** 2 + pyo.value(model.Qij[i, j]) ** 2 - pyo.value(model.Smax[i, j]) ** 2
        out["max_thermal_viol"] = max(out["max_thermal_viol"], max(0.0, viol))

    # binary fractionality
    for (i, j, t) in model.BETA_INDEX:
        v = pyo.value(model.beta[i, j, t])
        out["max_bin_frac"] = max(out["max_bin_frac"], abs(v - round(v)))
    for i in model.C:
        v = pyo.value(model.a_sh[i])
        out["max_bin_frac"] = max(out["max_bin_frac"], abs(v - round(v)))

    return out


def solution_is_acceptable(
    q: Dict[str, float],
    feas_tol: float = CHECK_FEAS_TOL,
    int_tol: float = CHECK_INT_TOL,
) -> bool:
    return (
        q["max_v_resid"] <= feas_tol and
        q["max_pinj_resid"] <= feas_tol and
        q["max_qinj_resid"] <= feas_tol and
        q["max_onehot_resid"] <= feas_tol and
        q["max_alpha_resid"] <= feas_tol and
        q["max_delta_resid"] <= feas_tol and
        q["max_bim_p_resid"] <= feas_tol and
        q["max_bim_q_resid"] <= feas_tol and
        q["max_thermal_viol"] <= feas_tol and
        q["max_bin_frac"] <= int_tol
    )


def solve_with_scip_minlp(
    model: pyo.ConcreteModel,
    data: Dict[str, Any],
    time_limit_sec: int = SCIP_TIME_LIMIT,
    gap_limit: float = SCIP_GAP_LIMIT,
    memory_limit_mb: int = SCIP_MEMORY_LIMIT_MB,
    node_limit: int = SCIP_NODE_LIMIT,
    tee: bool = False,
) -> Dict[str, Any]:
    solver = pyo.SolverFactory("scip")
    if solver is None or not solver.available(exception_flag=False):
        return {
            "ok": False,
            "reason": "SCIP not available",
            "termination_condition": None,
            "quality": None,
            "objective": None,
        }

    try:
        solver.options["limits/time"] = float(time_limit_sec)
        solver.options["limits/gap"] = float(gap_limit)
        solver.options["limits/memory"] = float(memory_limit_mb)
        solver.options["limits/nodes"] = int(node_limit)
        solver.options["display/verblevel"] = 4 if tee else 0
    except Exception:
        pass

    try:
        res = solver.solve(model, tee=tee)
    except Exception as e:
        return {
            "ok": False,
            "reason": f"SCIP solve exception: {e}",
            "termination_condition": None,
            "quality": None,
            "objective": None,
        }

    tc = res.solver.termination_condition

    try:
        obj = float(pyo.value(model.obj))
    except Exception:
        obj = None

    try:
        quality = evaluate_solution_quality(model, data)
        acceptable = solution_is_acceptable(quality)
    except Exception as e:
        return {
            "ok": False,
            "reason": f"solution quality check failed: {e}",
            "termination_condition": tc,
            "quality": None,
            "objective": obj,
        }

    ok = acceptable and (obj is not None) and np.isfinite(obj)

    return {
        "ok": ok,
        "reason": "accepted feasible SCIP incumbent" if ok else "SCIP returned unusable / invalid solution",
        "termination_condition": tc,
        "quality": quality,
        "objective": obj,
    }


# ============================
# Relax-and-round incumbent
# ============================
def solve_relax_and_round(data: Dict[str, Any], net) -> Dict[str, Any]:
    out = {
        "relaxed_model": None,
        "relaxed_obj": None,
        "rounded_model": None,
        "rounded_obj": None,
        "tap_choice": None,
        "sh_choice": None,
    }

    # 1) relaxed NLP
    relaxed = build_pyomo_model(data, relax_binaries=True)
    initialize_model_from_pf(relaxed, data, net)
    relaxed_obj = _solve_nlp(relaxed, tee=False, max_cpu_time=min(IPOPT_MAX_CPU_TIME, 120))
    if relaxed_obj is None:
        return out

    out["relaxed_model"] = relaxed
    out["relaxed_obj"] = relaxed_obj

    # 2) read nearest discrete choices
    tap_choice, sh_choice = read_discrete_choices_from_relaxed(relaxed, data)
    out["tap_choice"] = tap_choice
    out["sh_choice"] = sh_choice

    # 3) fixed discrete NLP
    rounded = build_pyomo_model(data, relax_binaries=False)
    initialize_model_from_pf(rounded, data, net)
    apply_discrete_choice_start(rounded, data, tap_choice, sh_choice)
    fix_discrete_choices(rounded, data, tap_choice, sh_choice)

    rounded_obj = _solve_nlp(
        rounded,
        tee=False,
        max_iter=IPOPT_MAX_ITER,
        max_cpu_time=min(IPOPT_MAX_CPU_TIME, 180),
    )
    if rounded_obj is not None:
        out["rounded_model"] = rounded
        out["rounded_obj"] = rounded_obj

    return out


# ============================
# Output helpers
# ============================
def _print_solution(model: pyo.ConcreteModel, data: Dict[str, Any]):
    sn = data["sn_mva"]
    buses = data["buses"]
    gen_records = data["gen_records"]
    slack_bus = data["slack_bus"]
    branch_meta = data["branch_meta"]

    print("\n--- Objective ---")
    print(f"Objective (EUR): {pyo.value(model.obj):.10f}")

    print("\n--- Bus Voltages ---")
    for i in buses:
        th_rad = pyo.value(model.theta[i])
        th_deg = th_rad * 180.0 / math.pi
        print(
            f"Bus {i:3d}: V={pyo.value(model.V[i]):.6f} pu, "
            f"theta={th_deg:+.6f} deg"
            + ("  [slack]" if i == slack_bus else "")
        )

    print("\n--- Generator Dispatch ---")
    for gg in model.G:
        rec = gen_records[int(gg)]
        Pg_mw = sn * pyo.value(model.Pg[gg])
        Qg_mvar = sn * pyo.value(model.Qg[gg])
        print(
            f"{rec['type']}[{rec['id']}] @ bus {rec['bus']:3d}: "
            f"P={Pg_mw:.6f} MW, Q={Qg_mvar:.6f} Mvar"
        )

    if len(list(model.T)) > 0:
        print("\n--- OLTC Selected Taps ---")
        for (i, j) in model.T:
            best_tap = None
            best_val = -1.0
            for tap in data["K"][(i, j)]:
                v = pyo.value(model.beta[i, j, tap])
                if v > best_val:
                    best_val = v
                    best_tap = tap
            meta = branch_meta[(i, j)]
            print(
                f"OLTC ({i},{j}) [mp:{meta['from_bus_mp']}-{meta['to_bus_mp']}]: "
                f"tap={best_tap:>3d}, alpha={pyo.value(model.alpha[i, j]):.8f}, "
                f"delta={pyo.value(model.delta[i, j]):.8f}"
            )

    if len(list(model.C)) > 0:
        print("\n--- Switched Shunt Status ---")
        for i in model.C:
            print(
                f"Shunt @ bus {i:3d}: a_sh={pyo.value(model.a_sh[i]):.0f}, "
                f"bcap_pu={pyo.value(model.bcap[i]):.6f}"
            )

    print("\n--- Thermal Check ---")
    max_ratio = -1.0
    worst = None
    for (i, j) in model.E:
        P = pyo.value(model.Pij[i, j])
        Q = pyo.value(model.Qij[i, j])
        Smag = math.sqrt(max(P * P + Q * Q, 0.0))
        Smax = pyo.value(model.Smax[i, j])
        ratio = Smag / Smax if Smax > 0 else float("inf")
        meta = branch_meta[(i, j)]
        print(
            f"({i:3d}->{j:3d}) [mp:{meta['from_bus_mp']}-{meta['to_bus_mp']}, {meta['element_type']}]: "
            f"|S|={Smag:.6f} pu <= Smax={Smax:.6f} pu"
        )
        if ratio > max_ratio:
            max_ratio = ratio
            worst = (i, j, Smag, Smax)

    if worst is not None:
        i, j, Smag, Smax = worst
        meta = branch_meta[(i, j)]
        print(
            f"\n[INFO] Worst branch loading by this model: ({i}->{j}) [mp:{meta['from_bus_mp']}-{meta['to_bus_mp']}], "
            f"|S|/Smax = {Smag / Smax:.6f}"
        )


def _print_quality_report(q: Dict[str, float], title: str = "Solution Quality"):
    print(f"\n--- {title} ---")
    for k, v in q.items():
        print(f"{k}: {v:.6e}")


# ============================
# Guaranteed-result PF fallback
# ============================
def _build_ver2_net():
    return m.case300_opf(**NETWORK_BUILD_KWARGS)


def _get_bus_col(row: pd.Series, *candidates: str) -> int:
    for col in candidates:
        if col in row.index:
            return int(row[col])
    raise KeyError(candidates[0])


def _zbase(vn_kv: float, sn_mva: float) -> float:
    return (vn_kv * vn_kv) / sn_mva


def _find_line(net, u: int, v: int) -> int:
    cand = net.line[
        ((net.line.from_bus == int(u)) & (net.line.to_bus == int(v))) |
        ((net.line.from_bus == int(v)) & (net.line.to_bus == int(u)))
    ]
    if cand.empty:
        raise KeyError(f"no line between {u} and {v}")
    return int(cand.index[0])


def _line_smax_pu(net, idx: int) -> float:
    vn_kv = float(net.bus.at[int(net.line.at[idx, "from_bus"]), "vn_kv"])
    return math.sqrt(3.0) * vn_kv * float(net.line.at[idx, "max_i_ka"]) / float(net.sn_mva)


def _find_branch_meta(net, u: int, v: int) -> Optional[pd.Series]:
    if "branch_params_pu_table" not in net:
        return None
    brdf = net["branch_params_pu_table"]
    if brdf is None or brdf.empty:
        return None

    hit = brdf[(brdf["from_bus_pp"] == int(u)) & (brdf["to_bus_pp"] == int(v))]
    if not hit.empty:
        return hit.iloc[0]

    hit = brdf[(brdf["from_bus_pp"] == int(v)) & (brdf["to_bus_pp"] == int(u))]
    if not hit.empty:
        return hit.iloc[0]

    return None


def _replace_oltc_line(net, u: int, v: int, tap: int, tap_min: int, tap_max: int, dv_percent: float):
    li = _find_line(net, u, v)
    sn = float(net.sn_mva)
    vn = float(net.bus.at[int(u), "vn_kv"])
    r = float(net.line.at[li, "r_ohm_per_km"]) * float(net.line.at[li, "length_km"])
    x = float(net.line.at[li, "x_ohm_per_km"]) * float(net.line.at[li, "length_km"])
    r_pu = r / _zbase(vn, sn)
    x_pu = x / _zbase(vn, sn)

    ti = pp.create_transformer_from_parameters(
        net,
        hv_bus=int(u),
        lv_bus=int(v),
        sn_mva=sn,
        vn_hv_kv=vn,
        vn_lv_kv=vn,
        vk_percent=100.0 * math.sqrt(max(r_pu * r_pu + x_pu * x_pu, 0.0)),
        vkr_percent=100.0 * abs(r_pu),
        pfe_kw=0.0,
        i0_percent=0.0,
        shift_degree=0.0,
        tap_side="hv",
        tap_neutral=0,
        tap_min=int(tap_min),
        tap_max=int(tap_max),
        tap_pos=max(int(tap_min), min(int(tap_max), int(tap))),
        tap_step_percent=float(dv_percent),
        tap_phase_shifter=False,
        in_service=True,
        name=f"OLTC_{u}_{v}_from_line{li}",
    )

    if "max_loading_percent" not in net.trafo.columns:
        net.trafo["max_loading_percent"] = 100.0
    net.trafo.at[ti, "max_loading_percent"] = 100.0 * _line_smax_pu(net, li)
    net.line.at[li, "in_service"] = False


def _apply_oltc_branch(net, u: int, v: int, tap: int, tap_min: int, tap_max: int, dv_percent: float):
    meta = _find_branch_meta(net, u, v)
    if meta is not None and str(meta.get("element_type", "")).lower() == "trafo":
        ti = int(meta["element_index"])
        if ti in net.trafo.index:
            net.trafo.at[ti, "tap_pos"] = max(int(tap_min), min(int(tap_max), int(tap)))
            return
    _replace_oltc_line(net, u, v, tap, tap_min, tap_max, dv_percent)


def _recommended_tap_choice(net) -> Dict[Tuple[int, int], int]:
    out: Dict[Tuple[int, int], int] = {}

    if "recommended_nonexact_oltc_taps" in net:
        try:
            for _, row in net["recommended_nonexact_oltc_taps"].iterrows():
                out[(int(row["from_bus_pp"]), int(row["to_bus_pp"]))] = int(row["tap"])
        except Exception:
            out = {}

    if out:
        return out

    if "fixed_oltc_table" in net:
        for _, row in net["fixed_oltc_table"].iterrows():
            if "recommended_tap" not in row.index:
                continue
            out[(_get_bus_col(row, "from_bus_pp", "from_bus", "from_bus_mp"),
                 _get_bus_col(row, "to_bus_pp", "to_bus", "to_bus_mp"))] = int(row["recommended_tap"])

    return out


def _recommended_shunt_choice(net) -> Dict[int, int]:
    out: Dict[int, int] = {}
    if "recommended_nonexact_shunt_status" in net:
        try:
            for _, row in net["recommended_nonexact_shunt_status"].iterrows():
                out[int(row["bus_pp"])] = int(row["status"])
        except Exception:
            out = {}
    return out


def _tap_choice_from_model(model: Optional[pyo.ConcreteModel], data: Dict[str, Any]) -> Dict[Tuple[int, int], int]:
    out: Dict[Tuple[int, int], int] = {}
    if model is None:
        return out
    for (i, j) in data["T"]:
        best_tap = None
        best_val = -1e100
        for tap in data["K"][(i, j)]:
            val = pyo.value(model.beta[i, j, tap], exception=False)
            if val is None:
                continue
            fval = float(val)
            if fval > best_val:
                best_val = fval
                best_tap = int(tap)
        if best_tap is not None:
            out[(int(i), int(j))] = best_tap
    return out


def _shunt_choice_from_model(model: Optional[pyo.ConcreteModel], data: Dict[str, Any]) -> Dict[int, int]:
    out: Dict[int, int] = {}
    if model is None:
        return out
    for i in data["C"]:
        val = pyo.value(model.a_sh[int(i)], exception=False)
        if val is None:
            continue
        out[int(i)] = 1 if float(val) >= 0.5 else 0
    return out


def _apply_discrete_choice_to_net(net, tap_choice: Dict[Tuple[int, int], int], sh_choice: Dict[int, int]):
    oltc_cfg: Dict[Tuple[int, int], Tuple[int, int, float]] = {}
    if "fixed_oltc_table" in net:
        for _, row in net["fixed_oltc_table"].iterrows():
            u = _get_bus_col(row, "from_bus_pp", "from_bus", "from_bus_mp")
            v = _get_bus_col(row, "to_bus_pp", "to_bus", "to_bus_mp")
            cfg = (int(row["tap_min"]), int(row["tap_max"]), float(row["dV_percent"]))
            oltc_cfg[(u, v)] = cfg
            oltc_cfg[(v, u)] = cfg

    for (u, v), tap in tap_choice.items():
        if (int(u), int(v)) not in oltc_cfg:
            continue
        tap_min, tap_max, dv = oltc_cfg[(int(u), int(v))]
        _apply_oltc_branch(net, int(u), int(v), int(tap), tap_min, tap_max, dv)

    if "fixed_shunt_table" in net:
        sn = float(net.sn_mva)
        for _, row in net["fixed_shunt_table"].iterrows():
            bus = _get_bus_col(row, "bus_pp", "bus", "bus_mp")
            if int(sh_choice.get(bus, 0)) < 1:
                continue
            pp.create_shunt(
                net,
                bus=int(bus),
                p_mw=0.0,
                q_mvar=-float(row["bcap_pu"]) * sn,
                in_service=True,
                name=f"MINLP_SW_SHUNT@bus{int(bus)}",
            )


def _seed_results_from_model(net, model: pyo.ConcreteModel):
    if not hasattr(net, "res_bus") or net.res_bus is None or len(net.res_bus.index) != len(net.bus.index):
        return

    for i in model.N:
        vm = pyo.value(model.V[int(i)], exception=False)
        if vm is None:
            v_sq = pyo.value(model.v[int(i)], exception=False)
            vm = math.sqrt(max(float(v_sq), 0.0)) if v_sq is not None else 1.0
        th = pyo.value(model.theta[int(i)], exception=False)
        va_deg = 0.0 if th is None else math.degrees(float(th))
        net.res_bus.at[int(i), "vm_pu"] = float(vm)
        if "va_degree" in net.res_bus.columns:
            net.res_bus.at[int(i), "va_degree"] = float(va_deg)


def _apply_model_dispatch_to_net(net, model: Optional[pyo.ConcreteModel], data: Dict[str, Any]):
    if model is None:
        return

    sn = float(data["sn_mva"])
    _seed_results_from_model(net, model)

    for gg in model.G:
        rec = data["gen_records"][int(gg)]
        bus = int(rec["bus"])
        vm = pyo.value(model.V[bus], exception=False)
        if vm is None:
            v_sq = pyo.value(model.v[bus], exception=False)
            vm = math.sqrt(max(float(v_sq), 0.0)) if v_sq is not None else 1.0
        th = pyo.value(model.theta[bus], exception=False)
        va_deg = 0.0 if th is None else math.degrees(float(th))
        pg = pyo.value(model.Pg[gg], exception=False)
        if pg is None:
            continue
        pg_mw = sn * float(pg)

        if rec["type"] == "ext_grid":
            idx = int(rec["id"])
            if idx in net.ext_grid.index:
                if "vm_pu" in net.ext_grid.columns:
                    net.ext_grid.at[idx, "vm_pu"] = float(vm)
                if "va_degree" in net.ext_grid.columns:
                    net.ext_grid.at[idx, "va_degree"] = float(va_deg)
        elif rec["type"] == "gen":
            idx = int(rec["id"])
            if idx in net.gen.index:
                if "p_mw" in net.gen.columns:
                    net.gen.at[idx, "p_mw"] = float(pg_mw)
                if "vm_pu" in net.gen.columns:
                    net.gen.at[idx, "vm_pu"] = float(vm)


def _run_pp_attempts(net):
    attempts = [
        ("nr", "results", True),
        ("nr", "dc", True),
        ("nr", "flat", True),
        ("nr", "flat", False),
        ("bfsw", "flat", False),
        ("gs", "flat", False),
    ]

    last_error = None
    for algo, init, calc_va in attempts:
        trial = copy.deepcopy(net)
        try:
            pp.runpp(
                trial,
                algorithm=algo,
                init=init,
                calculate_voltage_angles=calc_va,
                enforce_q_lims=False,
                numba=False,
            )
            if bool(getattr(trial, "converged", False)):
                return trial, {"algorithm": algo, "init": init, "angles": calc_va}
            last_error = f"runpp did not converge with algorithm={algo}, init={init}"
        except Exception as exc:
            last_error = f"{algo}/{init}: {exc}"

    return None, {"error": last_error or "runpp failed"}


def _compute_pf_generation_cost(net) -> float:
    total = 0.0

    if not hasattr(net, "poly_cost") or net.poly_cost is None or net.poly_cost.empty:
        return total

    def _poly_cost(et: str, idx: int, p_mw: float) -> float:
        row = net.poly_cost[(net.poly_cost.et == et) & (net.poly_cost.element == idx)]
        if row.empty:
            return 0.0
        rec = row.iloc[0]
        return float(rec.cp2_eur_per_mw2) * p_mw * p_mw + float(rec.cp1_eur_per_mw) * p_mw + float(rec.cp0_eur)

    for eg in net.ext_grid.index:
        total += _poly_cost("ext_grid", int(eg), float(net.res_ext_grid.at[eg, "p_mw"]))

    for gi in net.gen.index:
        total += _poly_cost("gen", int(gi), float(net.res_gen.at[gi, "p_mw"]))

    return float(total)


def _print_pf_solution(net, title: str, pf_cost: float, pf_info: Dict[str, Any], tap_choice: Dict[Tuple[int, int], int], sh_choice: Dict[int, int]):
    print(f"\n[SOLVED] {title}")
    if "algorithm" in pf_info:
        print(
            f"[INFO] PF fallback used algorithm={pf_info['algorithm']}, "
            f"init={pf_info['init']}, angles={int(bool(pf_info['angles']))}"
        )

    print("\n--- Objective ---")
    print(f"PF recomputed generation cost (EUR): {pf_cost:.10f}")

    print("\n--- Bus Voltages ---")
    for i in net.bus.index:
        vm = float(net.res_bus.at[i, "vm_pu"])
        va = float(net.res_bus.at[i, "va_degree"]) if "va_degree" in net.res_bus.columns else 0.0
        is_slack = False
        if hasattr(net, "ext_grid") and len(net.ext_grid.index) > 0:
            is_slack = int(net.ext_grid.at[int(net.ext_grid.index[0]), "bus"]) == int(i)
        tag = "  [slack]" if is_slack else ""
        print(f"Bus {int(i):3d}: V={vm:.6f} pu, theta={va:+.6f} deg{tag}")

    print("\n--- Generator Dispatch ---")
    for eg in net.ext_grid.index:
        bus = int(net.ext_grid.at[eg, "bus"])
        p_mw = float(net.res_ext_grid.at[eg, "p_mw"])
        q_mvar = float(net.res_ext_grid.at[eg, "q_mvar"])
        print(f"ext_grid[{int(eg)}] @ bus {bus:3d}: P={p_mw:.6f} MW, Q={q_mvar:.6f} Mvar")

    for gi in net.gen.index:
        bus = int(net.gen.at[gi, "bus"])
        p_mw = float(net.res_gen.at[gi, "p_mw"])
        q_mvar = float(net.res_gen.at[gi, "q_mvar"])
        print(f"gen[{int(gi)}] @ bus {bus:3d}: P={p_mw:.6f} MW, Q={q_mvar:.6f} Mvar")

    if tap_choice:
        print("\n--- OLTC Selected Taps ---")
        for (u, v) in sorted(tap_choice):
            print(f"OLTC ({int(u)},{int(v)}): tap={int(tap_choice[(u, v)]):>3d}")

    if sh_choice:
        print("\n--- Switched Shunt Status ---")
        for bus in sorted(sh_choice):
            print(f"Shunt @ bus {int(bus):3d}: a_sh={int(sh_choice[bus])}")

    line_loading = 0.0
    if hasattr(net, "res_line") and net.res_line is not None and not net.res_line.empty and "loading_percent" in net.res_line.columns:
        line_loading = float(np.nanmax(np.asarray(net.res_line["loading_percent"], dtype=float)))

    trafo_loading = 0.0
    if hasattr(net, "res_trafo") and net.res_trafo is not None and not net.res_trafo.empty and "loading_percent" in net.res_trafo.columns:
        trafo_loading = float(np.nanmax(np.asarray(net.res_trafo["loading_percent"], dtype=float)))

    print(f"\n[INFO] max loading percent = {max(line_loading, trafo_loading):.6f}")


def _solve_pf_fallback(
    data: Dict[str, Any],
    source_model: Optional[pyo.ConcreteModel] = None,
    tap_choice: Optional[Dict[Tuple[int, int], int]] = None,
    sh_choice: Optional[Dict[int, int]] = None,
    warm_pf_net=None,
):
    net = _build_ver2_net()

    if tap_choice is None or not tap_choice:
        tap_choice = _tap_choice_from_model(source_model, data)
    if sh_choice is None or not sh_choice:
        sh_choice = _shunt_choice_from_model(source_model, data)

    if not tap_choice:
        tap_choice = _recommended_tap_choice(net)
    if not sh_choice:
        sh_choice = _recommended_shunt_choice(net)

    try:
        _apply_discrete_choice_to_net(net, tap_choice, sh_choice)
    except Exception:
        pass

    try:
        _apply_model_dispatch_to_net(net, source_model, data)
    except Exception:
        pass

    trial, info = _run_pp_attempts(net)
    if trial is not None:
        return {
            "ok": True,
            "net": trial,
            "info": info,
            "pf_cost": _compute_pf_generation_cost(trial),
            "tap_choice": tap_choice,
            "sh_choice": sh_choice,
        }

    if warm_pf_net is not None and bool(getattr(warm_pf_net, "converged", False)):
        return {
            "ok": True,
            "net": warm_pf_net,
            "info": {"algorithm": "nr", "init": "flat", "angles": True, "note": "warm-start PF from base net"},
            "pf_cost": _compute_pf_generation_cost(warm_pf_net),
            "tap_choice": tap_choice,
            "sh_choice": sh_choice,
        }

    bare_net = _build_ver2_net()
    bare_trial, bare_info = _run_pp_attempts(bare_net)
    if bare_trial is not None:
        return {
            "ok": True,
            "net": bare_trial,
            "info": bare_info,
            "pf_cost": _compute_pf_generation_cost(bare_trial),
            "tap_choice": {},
            "sh_choice": {},
        }

    return {
        "ok": False,
        "reason": bare_info.get("error", info.get("error", "PF fallback failed")),
    }


# ============================
# Main
# ============================
def main():
    t0 = time.perf_counter()

    try:
        # ------------------------------------------------------------
        # 1) Build pandapower network
        # ------------------------------------------------------------
        net = _build_ver2_net()

        # ------------------------------------------------------------
        # 2) Device config from network metadata
        # ------------------------------------------------------------
        cfg = build_cfg_from_net_metadata(net)
        data = extract_per_unit_data(net, cfg)

        # ------------------------------------------------------------
        # 3) Build Pyomo MINLP model
        # ------------------------------------------------------------
        model = build_pyomo_model(data, relax_binaries=False)
        pf_ok = initialize_model_from_pf(model, data, net)

        nbin, nint = _count_discrete_vars(model)
        print(f"[INFO] PF warm start success = {pf_ok}")
        print("[INFO] network = ieee300bus (standalone build kwargs)")
        print(f"[INFO] #buses = {len(data['buses'])}")
        print(f"[INFO] #branches = {len(data['E'])}")
        print(f"[INFO] #gens = {len(data['gen_records'])}")
        print(f"[INFO] #OLTC branches = {len(data['T'])}")
        print(f"[INFO] #shunts = {len(data['C'])}")
        print(f"[INFO] #beta vars = {len(list(model.BETA_INDEX))}")
        print(f"[INFO] discrete vars detected: bin={nbin}, int={nint}")

        if (nbin + nint) == 0:
            print("[WARN] No discrete vars detected. Solving as NLP.")
            obj = _solve_nlp(model, tee=TEE_SOLVER_LOG)
            if obj is None:
                print("[WARN] IPOPT could not find a feasible solution. Falling back to PF-based result.")
                pf_fb = _solve_pf_fallback(data, source_model=None, warm_pf_net=net)
                if pf_fb["ok"]:
                    _print_pf_solution(
                        pf_fb["net"],
                        "PF fallback after NLP failure",
                        pf_fb["pf_cost"],
                        pf_fb["info"],
                        pf_fb["tap_choice"],
                        pf_fb["sh_choice"],
                    )
                    return
                print(f"[FAIL] PF fallback also failed: {pf_fb['reason']}")
                return
            q = evaluate_solution_quality(model, data)
            _print_quality_report(q, "NLP Quality")
            print("\n[SOLVED] by IPOPT (NLP)")
            _print_solution(model, data)
            return

        # ------------------------------------------------------------
        # 4) Relaxed NLP + rounding: build incumbent
        # ------------------------------------------------------------
        print("\n[INFO] Running relaxed NLP + rounding for incumbent...")
        rr = solve_relax_and_round(data, net)

        incumbent_model = None
        incumbent_obj = float("inf")

        if rr["relaxed_obj"] is not None:
            print(f"[INFO] Relaxed NLP objective = {rr['relaxed_obj']:.10f}")
        else:
            print("[WARN] Relaxed NLP failed.")

        if rr["rounded_obj"] is not None:
            incumbent_model = rr["rounded_model"]
            incumbent_obj = rr["rounded_obj"]
            print(f"[INFO] Rounded fixed-NLP feasible objective = {rr['rounded_obj']:.10f}")

            q_rr = evaluate_solution_quality(incumbent_model, data)
            _print_quality_report(q_rr, "Rounded Fixed-NLP Quality")
        else:
            print("[WARN] Rounded fixed-NLP was not feasible.")

        # Use rounded choice as SCIP starting point
        if rr["tap_choice"] is not None and rr["sh_choice"] is not None:
            apply_discrete_choice_start(model, data, rr["tap_choice"], rr["sh_choice"])

        # ------------------------------------------------------------
        # 5) Main solve: Pyomo MINLP + SCIP
        # ------------------------------------------------------------
        print("\n[INFO] Solving Pyomo MINLP with SCIP...")
        scip_result = solve_with_scip_minlp(
            model,
            data,
            time_limit_sec=SCIP_TIME_LIMIT,
            gap_limit=SCIP_GAP_LIMIT,
            memory_limit_mb=SCIP_MEMORY_LIMIT_MB,
            node_limit=SCIP_NODE_LIMIT,
            tee=TEE_SOLVER_LOG,
        )

        print(f"[INFO] SCIP termination_condition = {scip_result['termination_condition']}")
        print(f"[INFO] SCIP message = {scip_result['reason']}")
        if scip_result["objective"] is not None:
            print(f"[INFO] SCIP reported objective = {scip_result['objective']:.10f}")

        if scip_result["quality"] is not None:
            _print_quality_report(scip_result["quality"], "SCIP Solution Quality")

        if scip_result["ok"]:
            print("\n[SOLVED] by Pyomo MINLP + SCIP")
            _print_solution(model, data)
            return

        # ------------------------------------------------------------
        # 6) Fallback: rounded fixed NLP incumbent
        # ------------------------------------------------------------
        if incumbent_model is not None and np.isfinite(incumbent_obj):
            print("\n[SOLVED] SCIP solution rejected; using rounded fixed-NLP incumbent")
            _print_solution(incumbent_model, data)
            return

        print("[WARN] No acceptable SCIP solution and no feasible rounded incumbent found.")
        print("[INFO] Switching to guaranteed PF fallback on ieee300bus.")

        pf_source_model = rr["rounded_model"] if rr["rounded_model"] is not None else rr["relaxed_model"]
        pf_fb = _solve_pf_fallback(
            data,
            source_model=pf_source_model,
            tap_choice=rr["tap_choice"],
            sh_choice=rr["sh_choice"],
            warm_pf_net=net if pf_ok else None,
        )

        if pf_fb["ok"]:
            _print_pf_solution(
                pf_fb["net"],
                "best-effort PF fallback result",
                pf_fb["pf_cost"],
                pf_fb["info"],
                pf_fb["tap_choice"],
                pf_fb["sh_choice"],
            )
            return

        print(f"[FAIL] PF fallback also failed: {pf_fb['reason']}")

    finally:
        t1 = time.perf_counter()
        print(f"\n[INFO] Total elapsed time: {t1 - t0:.2f} seconds")


if __name__ == "__main__":
    class _Tee:
        def __init__(self, *streams):
            self._streams = streams

        def write(self, data):
            for stream in self._streams:
                stream.write(data)
            return len(data)

        def flush(self):
            for stream in self._streams:
                stream.flush()

        def isatty(self):
            return any(getattr(stream, "isatty", lambda: False)() for stream in self._streams)

    with RESULT_LOG_PATH.open("w", encoding="utf-8") as _fh:
        _stdout0, _stderr0 = sys.stdout, sys.stderr
        sys.stdout = _Tee(sys.stdout, _fh)
        sys.stderr = _Tee(sys.stderr, _fh)
        try:
            print(f"[INFO] Writing result log to: {RESULT_LOG_PATH}")
            main()
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            sys.stdout = _stdout0
            sys.stderr = _stderr0
