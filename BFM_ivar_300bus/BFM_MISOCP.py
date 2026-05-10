# BFM_MISOCP.py
# ------------------------------------------------------------
# Pyomo Branch-Flow Model (BFM-cr) AC-OPF / MISOCP
# for the modified explicit 41-bus mesh network
#
# Adapted to:
#   ieee39busplus_modified_explicit.py
#
# Main changes
# ------------
# 1) Network import updated to the new 41-bus explicit mesh builder
# 2) OLTC candidate branches / tap ranges are read from network metadata
# 3) Shunt candidate buses / bcap values are read from network metadata
# 4) Default build settings aligned with the new non-exact mesh profile
#
# Notes
# -----
# - This is still a convex relaxation (BFM-cr), not the original exact BFM.
# - Exactness is NOT guaranteed. Check "max_soc_slack" after solve.
# - Solvers tried: cplex_direct -> cplex -> gurobi_direct -> gurobi -> scip
# ------------------------------------------------------------

import copy
import math
import time
from dataclasses import dataclass
from typing import Dict, Tuple, List, Any, Optional

import numpy as np
import pyomo.environ as pyo
from pyomo.opt import TerminationCondition
import pandapower as pp

import ieee300bus as mcase


# ============================
# Solver / numeric settings
# ============================
TEE_SOLVER_LOG = True

RUN_PF_WARMSTART = True

SOLVER_TIME_LIMIT = 36000.0
SOLVER_MIP_GAP = 1e-6
SOLVER_MIP_GAP_ABS = 0.0

RESID_TOL_ACCEPT = 1e-5
INT_TOL_ACCEPT = 1e-6

# fixed-discrete fallback
RUN_FIXED_DISCRETE_FALLBACK = True

# post-solve diagnostic / validation reporting
RUN_DW_AC_VALIDATION = True
DW_SOC_SLACK_SCREEN_TOL = 1e-4
DW_PP_INIT_MODES = ("results", "dc", "flat")

# reporting policy
REPORT_BEST_AVAILABLE_WHEN_REJECTED = True

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


# ----------------------------
# Configuration containers
# ----------------------------
@dataclass
class OLTCBranchConfig:
    tap_min: int
    tap_max: int
    dV_percent: float


@dataclass
class ShuntConfig:
    # qsh = a_sh * bcap_pu * (v / v_rated_sq)
    bcap_pu: float
    v_rated_sq: float = 1.0


@dataclass
class BuildConfig:
    oltc_branches: Dict[Tuple[int, int], OLTCBranchConfig]
    shunts: Dict[int, ShuntConfig]
    fix_slack_vm: bool = True


# ----------------------------
# Helpers
# ----------------------------
def _zbase_ohm(vn_kv: float, sn_mva: float) -> float:
    return (vn_kv ** 2) / sn_mva


def _find_poly_cost(net, et: str, element: int) -> Tuple[float, float, float]:
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


def _default_tap(taps: List[int]) -> int:
    return min(taps, key=lambda t: abs(t))


def build_cfg_from_net_metadata(net) -> BuildConfig:
    """
    Build OLTC / shunt config directly from the new network metadata.
    """
    oltc_branches: Dict[Tuple[int, int], OLTCBranchConfig] = {}
    shunts: Dict[int, ShuntConfig] = {}

    if "fixed_oltc_table" not in net:
        raise KeyError("Network is missing 'fixed_oltc_table' metadata.")
    if "fixed_shunt_table" not in net:
        raise KeyError("Network is missing 'fixed_shunt_table' metadata.")

    oltc_df = net["fixed_oltc_table"]
    shunt_df = net["fixed_shunt_table"]

    # ieee300bus_ver3 metadata uses pandapower-index columns:
    #   fixed_oltc_table  : from_bus_pp, to_bus_pp
    #   fixed_shunt_table : bus_pp
    for _, row in oltc_df.iterrows():
        i = int(row["from_bus_pp"])
        j = int(row["to_bus_pp"])
        oltc_branches[(i, j)] = OLTCBranchConfig(
            tap_min=int(row["tap_min"]),
            tap_max=int(row["tap_max"]),
            dV_percent=float(row["dV_percent"]),
        )

    for _, row in shunt_df.iterrows():
        b = int(row["bus_pp"])
        shunts[b] = ShuntConfig(
            bcap_pu=float(row["bcap_pu"]),
            v_rated_sq=1.0,
        )

    return BuildConfig(
        oltc_branches=oltc_branches,
        shunts=shunts,
        fix_slack_vm=True,
    )


# ----------------------------
# Data extraction
# ----------------------------
def extract_bfm300_fullmesh_data(net, cfg: BuildConfig) -> Dict[str, Any]:
    """
    Extract per-unit data for BFM on the FULL meshed graph for ieee300bus_ver3.

    Unlike the 41-bus builder, ieee300bus_ver3 stores transmission branches across
    both net.line and net.trafo. We therefore read the unified metadata table
    net["branch_params_pu_table"] and use its directed (from_bus_pp -> to_bus_pp)
    orientation for every branch.
    """
    sn = float(net.sn_mva)

    buses = [int(i) for i in net.bus.index]
    Vmin = {int(i): float(net.bus.at[i, "min_vm_pu"]) for i in buses}
    Vmax = {int(i): float(net.bus.at[i, "max_vm_pu"]) for i in buses}

    if len(net.ext_grid.index) < 1:
        raise ValueError("pandapower net must have an ext_grid (slack).")
    eg0 = int(net.ext_grid.index[0])
    slack_bus = int(net.ext_grid.at[eg0, "bus"])
    slack_vm_pu = float(net.ext_grid.at[eg0, "vm_pu"])

    Pd = {i: 0.0 for i in buses}
    Qd = {i: 0.0 for i in buses}
    if hasattr(net, "load") and len(net.load.index) > 0:
        for li in net.load.index:
            b = int(net.load.at[li, "bus"])
            Pd[b] += float(net.load.at[li, "p_mw"])
            Qd[b] += float(net.load.at[li, "q_mvar"])
    Pd_pu = {i: Pd[i] / sn for i in buses}
    Qd_pu = {i: Qd[i] / sn for i in buses}

    gen_records = []
    for eg in net.ext_grid.index:
        eg = int(eg)
        b = int(net.ext_grid.at[eg, "bus"])
        pmin = float(net.ext_grid.at[eg, "min_p_mw"]) if "min_p_mw" in net.ext_grid.columns else -1e9
        pmax = float(net.ext_grid.at[eg, "max_p_mw"]) if "max_p_mw" in net.ext_grid.columns else 1e9
        qmin = float(net.ext_grid.at[eg, "min_q_mvar"]) if "min_q_mvar" in net.ext_grid.columns else -1e9
        qmax = float(net.ext_grid.at[eg, "max_q_mvar"]) if "max_q_mvar" in net.ext_grid.columns else 1e9
        c2, c1, c0 = _find_poly_cost(net, "ext_grid", eg)
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
            c2, c1, c0 = _find_poly_cost(net, "gen", gi)
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

    if "branch_params_pu_table" not in net:
        raise KeyError("Network is missing 'branch_params_pu_table' metadata.")

    branch_df = net["branch_params_pu_table"].copy()
    req_cols = [
        "from_bus_pp", "to_bus_pp", "r_pu", "x_pu",
        "synthetic_smax_mva", "element_type", "element_index"
    ]
    for c in req_cols:
        if c not in branch_df.columns:
            raise KeyError(f"branch_params_pu_table is missing required column: {c}")

    # IEEE 300-bus contains a small number of parallel branches.
    # Pyomo arc sets cannot be indexed by duplicated (i,j), so we aggregate
    # same-direction parallel branches into one equivalent branch here.
    grouped_rows: Dict[Tuple[int, int], List[Dict[str, Any]]] = {}
    for _, row in branch_df.iterrows():
        fb = int(row["from_bus_pp"])
        tb = int(row["to_bus_pp"])
        grouped_rows.setdefault((fb, tb), []).append(row.to_dict())

    E = []
    r = {}
    x = {}
    ellmax = {}
    branch_meta_of_edge = {}

    for ij, rows in grouped_rows.items():
        fb, tb = ij
        E.append(ij)

        if len(rows) == 1:
            row = rows[0]
            r[ij] = float(row["r_pu"])
            x[ij] = float(row["x_pu"])

            smax_mva = float(row["synthetic_smax_mva"])
            smax_pu = smax_mva / sn
            ellmax[ij] = float(max(0.0, smax_pu * smax_pu))

            branch_meta_of_edge[ij] = {
                "aggregate": False,
                "element_type": str(row["element_type"]),
                "element_index": int(row["element_index"]),
                "from_bus_pp": fb,
                "to_bus_pp": tb,
                "parallel_count": 1,
            }
            continue

        # Guard against aggregating duplicate OLTC / transformer candidates.
        has_oltc_dup = any(tuple(sorted((fb, tb))) == tuple(sorted(e)) for e in cfg.oltc_branches.keys())
        if has_oltc_dup:
            raise ValueError(
                f"Parallel duplicate branch appears on OLTC candidate edge {ij}. "
                f"This case needs 3-index branch modeling, not 2-index aggregation."
            )

        # Equivalent impedance of parallel branches:
        #   y_eq = sum_k 1 / z_k,   z_eq = 1 / y_eq
        y_eq = 0.0 + 0.0j
        smax_total_mva = 0.0
        members = []
        for row in rows:
            rk = float(row["r_pu"])
            xk = float(row["x_pu"])
            zk = complex(rk, xk)
            if abs(zk) <= 1e-14:
                raise ValueError(f"Near-zero impedance parallel branch encountered at {ij}; cannot aggregate safely.")
            y_eq += 1.0 / zk
            smax_total_mva += float(row["synthetic_smax_mva"])
            members.append({
                "element_type": str(row["element_type"]),
                "element_index": int(row["element_index"]),
                "from_bus_pp": fb,
                "to_bus_pp": tb,
            })

        if abs(y_eq) <= 1e-14:
            raise ValueError(f"Failed to aggregate parallel branch admittance at {ij}.")

        z_eq = 1.0 / y_eq
        r[ij] = float(z_eq.real)
        x[ij] = float(z_eq.imag)

        smax_pu = smax_total_mva / sn
        ellmax[ij] = float(max(0.0, smax_pu * smax_pu))

        branch_meta_of_edge[ij] = {
            "aggregate": True,
            "members": members,
            "from_bus_pp": fb,
            "to_bus_pp": tb,
            "parallel_count": len(rows),
        }

    E_set = set(E)

    T = []
    K = {}
    delta_tap = {}
    alpha_tap = {}

    for (u, v), tcfg in cfg.oltc_branches.items():
        if (u, v) not in E_set:
            if (v, u) in E_set:
                raise ValueError(
                    f"OLTC edge {(u, v)} exists only as reversed directed edge {(v, u)} in branch table. "
                    f"Fix ordering in fixed_oltc_table or the builder metadata."
                )
            raise ValueError(f"OLTC edge {(u, v)} not found in branch_params_pu_table.")

        ij = (u, v)
        T.append(ij)
        taps = list(range(int(tcfg.tap_min), int(tcfg.tap_max) + 1))
        K[ij] = taps
        for tap in taps:
            tau = 1.0 + (tap * float(tcfg.dV_percent)) / 100.0
            alpha_tap[(ij, tap)] = 1.0 / tau
            delta_tap[(ij, tap)] = 1.0 / (tau * tau)

    C = sorted([int(i) for i in cfg.shunts.keys()])
    bcap_pu = {}
    v_rated_sq = {}
    Mq = {}
    for i in C:
        scfg = cfg.shunts[i]
        bcap_pu[i] = float(scfg.bcap_pu)
        v_rated_sq[i] = float(scfg.v_rated_sq) if float(scfg.v_rated_sq) > 0 else 1.0
        vU = float(Vmax[i] ** 2)
        Mq[i] = abs(bcap_pu[i]) * (vU / v_rated_sq[i]) + 1e-6

    return {
        "sn_mva": sn,
        "buses": buses,
        "slack_bus": slack_bus,
        "slack_vm_pu": slack_vm_pu,
        "Pd_pu": Pd_pu,
        "Qd_pu": Qd_pu,
        "Vmin": Vmin,
        "Vmax": Vmax,
        "gen_records": gen_records,
        "E": E,
        "r": r,
        "x": x,
        "ellmax": ellmax,
        "T": T,
        "K": K,
        "alpha_tap": alpha_tap,
        "delta_tap": delta_tap,
        "C": C,
        "bcap_pu": bcap_pu,
        "v_rated_sq": v_rated_sq,
        "Mq": Mq,
        "fix_slack_vm": cfg.fix_slack_vm,
        "branch_meta_of_edge": branch_meta_of_edge,
    }


# ----------------------------
# Model builder (MISOCP)
# ----------------------------
def build_pyomo_bfm_misocp_model(
    data: Dict[str, Any],
    relax_binaries: bool = False,
) -> pyo.ConcreteModel:
    sn = data["sn_mva"]
    buses = data["buses"]
    slack_bus = data["slack_bus"]
    slack_vm_pu = data["slack_vm_pu"]

    E = data["E"]
    T = data["T"]
    T_set = set(T)
    NT = [e for e in E if e not in T_set]

    r = data["r"]
    x = data["x"]
    ellmax = data["ellmax"]

    K = data["K"]
    delta_tap = data["delta_tap"]
    alpha_tap = data["alpha_tap"]

    C = data["C"]
    C_set = set(C)
    bcap_pu = data["bcap_pu"]
    v_rated_sq = data["v_rated_sq"]
    Mq = data["Mq"]

    Pd = data["Pd_pu"]
    Qd = data["Qd_pu"]
    Vmin = data["Vmin"]
    Vmax = data["Vmax"]

    gen_records = data["gen_records"]
    Gset = list(range(len(gen_records)))

    out_arcs = {i: [] for i in buses}
    in_arcs = {i: [] for i in buses}
    for (i, j) in E:
        out_arcs[i].append((i, j))
        in_arcs[j].append((i, j))

    OTAP_INDEX = []
    for (i, j) in T:
        for tap in K[(i, j)]:
            OTAP_INDEX.append((i, j, int(tap)))

    segPmax = {}
    segQmax = {}
    for (i, j, tap) in OTAP_INDEX:
        vmax_sq = Vmax[i] ** 2
        d = delta_tap[((i, j), tap)]
        smax = math.sqrt(max(0.0, ellmax[(i, j)] * vmax_sq * d))
        segPmax[(i, j, tap)] = smax
        segQmax[(i, j, tap)] = smax

    m = pyo.ConcreteModel(name="BFM300_BFMcr_MISOCP")

    # sets
    m.N = pyo.Set(initialize=buses, ordered=True)
    m.G = pyo.Set(initialize=Gset, ordered=True)
    m.E = pyo.Set(initialize=E, dimen=2, ordered=True)
    m.T = pyo.Set(initialize=T, dimen=2, ordered=True)
    m.NT = pyo.Set(initialize=NT, dimen=2, ordered=True)
    m.C = pyo.Set(initialize=C, ordered=True)
    m.OTAP = pyo.Set(initialize=OTAP_INDEX, dimen=3, ordered=True)

    # params
    m.Pd = pyo.Param(m.N, initialize=lambda mm, i: float(Pd[i]))
    m.Qd = pyo.Param(m.N, initialize=lambda mm, i: float(Qd[i]))

    m.Vmin_mag = pyo.Param(m.N, initialize=lambda mm, i: float(Vmin[i]))
    m.Vmax_mag = pyo.Param(m.N, initialize=lambda mm, i: float(Vmax[i]))
    m.Vmin_sq = pyo.Param(m.N, initialize=lambda mm, i: float(Vmin[i] ** 2))
    m.Vmax_sq = pyo.Param(m.N, initialize=lambda mm, i: float(Vmax[i] ** 2))

    m.r = pyo.Param(m.E, initialize=lambda mm, i, j: float(r[(i, j)]))
    m.x = pyo.Param(m.E, initialize=lambda mm, i, j: float(x[(i, j)]))
    m.ellmax = pyo.Param(m.E, initialize=lambda mm, i, j: float(ellmax[(i, j)]))

    m.gen_bus = pyo.Param(m.G, initialize=lambda mm, gg: int(gen_records[int(gg)]["bus"]), within=pyo.Any)
    m.Pgmin = pyo.Param(m.G, initialize=lambda mm, gg: float(gen_records[int(gg)]["pmin_pu"]))
    m.Pgmax = pyo.Param(m.G, initialize=lambda mm, gg: float(gen_records[int(gg)]["pmax_pu"]))
    m.Qgmin = pyo.Param(m.G, initialize=lambda mm, gg: float(gen_records[int(gg)]["qmin_pu"]))
    m.Qgmax = pyo.Param(m.G, initialize=lambda mm, gg: float(gen_records[int(gg)]["qmax_pu"]))
    m.c2 = pyo.Param(m.G, initialize=lambda mm, gg: float(gen_records[int(gg)]["c2"]))
    m.c1 = pyo.Param(m.G, initialize=lambda mm, gg: float(gen_records[int(gg)]["c1"]))
    m.c0 = pyo.Param(m.G, initialize=lambda mm, gg: float(gen_records[int(gg)]["c0"]))

    m.bcap = pyo.Param(m.C, initialize=lambda mm, i: float(bcap_pu[int(i)]))
    m.vrated = pyo.Param(m.C, initialize=lambda mm, i: float(v_rated_sq[int(i)]))
    m.Mq = pyo.Param(m.C, initialize=lambda mm, i: float(Mq[int(i)]))

    m.delta_tap = pyo.Param(
        m.OTAP,
        initialize=lambda mm, i, j, tap: float(delta_tap[((i, j), int(tap))])
    )
    m.alpha_tap = pyo.Param(
        m.OTAP,
        initialize=lambda mm, i, j, tap: float(alpha_tap[((i, j), int(tap))])
    )

    # variables
    m.Pg = pyo.Var(m.G, bounds=lambda mm, gg: (mm.Pgmin[gg], mm.Pgmax[gg]))
    m.Qg = pyo.Var(m.G, bounds=lambda mm, gg: (mm.Qgmin[gg], mm.Qgmax[gg]))

    m.v = pyo.Var(m.N, bounds=lambda mm, i: (mm.Vmin_sq[i], mm.Vmax_sq[i]))

    m.Pinj = pyo.Var(m.N)
    m.Qinj = pyo.Var(m.N)

    def _flow_bound(mm, i, j):
        vmax_sq = float(mm.Vmax_sq[i])
        smax = math.sqrt(max(0.0, float(mm.ellmax[i, j]) * vmax_sq))
        return (-smax, smax)

    m.Pij = pyo.Var(m.E, bounds=_flow_bound)
    m.Qij = pyo.Var(m.E, bounds=_flow_bound)
    m.ell = pyo.Var(m.E, bounds=lambda mm, i, j: (0.0, mm.ellmax[i, j]))

    m.qsh = pyo.Var(m.N)

    if relax_binaries:
        m.beta = pyo.Var(m.OTAP, bounds=(0.0, 1.0))
        m.a_sh = pyo.Var(m.C, bounds=(0.0, 1.0))
    else:
        m.beta = pyo.Var(m.OTAP, within=pyo.Binary)
        m.a_sh = pyo.Var(m.C, within=pyo.Binary)

    m.wtap = pyo.Var(m.OTAP, bounds=lambda mm, i, j, tap: (0.0, mm.Vmax_sq[i]))
    m.Pijt = pyo.Var(m.OTAP, bounds=lambda mm, i, j, tap: (-segPmax[(i, j, tap)], segPmax[(i, j, tap)]))
    m.Qijt = pyo.Var(m.OTAP, bounds=lambda mm, i, j, tap: (-segQmax[(i, j, tap)], segQmax[(i, j, tap)]))
    m.ellt = pyo.Var(m.OTAP, bounds=lambda mm, i, j, tap: (0.0, mm.ellmax[i, j]))

    # expressions
    def _delta_expr(mm, i, j):
        return sum(mm.delta_tap[i, j, tap] * mm.beta[i, j, tap] for tap in K[(i, j)])
    m.delta = pyo.Expression(m.T, rule=_delta_expr)

    def _alpha_expr(mm, i, j):
        return sum(mm.alpha_tap[i, j, tap] * mm.beta[i, j, tap] for tap in K[(i, j)])
    m.alpha = pyo.Expression(m.T, rule=_alpha_expr)

    def _deltaE_expr(mm, i, j):
        if (i, j) in T_set:
            return mm.delta[i, j]
        return 1.0
    m.deltaE = pyo.Expression(m.E, rule=_deltaE_expr)

    # constraints
    if data["fix_slack_vm"]:
        m.slack_v = pyo.Constraint(expr=m.v[slack_bus] == float(slack_vm_pu) ** 2)

    m.Pinj_def = pyo.Constraint(
        m.N,
        rule=lambda mm, i: mm.Pinj[i] == sum(mm.Pg[g] for g in mm.G if int(mm.gen_bus[g]) == int(i)) - mm.Pd[i]
    )
    m.Qinj_def = pyo.Constraint(
        m.N,
        rule=lambda mm, i: mm.Qinj[i] == sum(mm.Qg[g] for g in mm.G if int(mm.gen_bus[g]) == int(i)) - mm.Qd[i] + mm.qsh[i]
    )

    m.onehot = pyo.Constraint(
        m.T,
        rule=lambda mm, i, j: sum(mm.beta[i, j, tap] for tap in K[(i, j)]) == 1
    )

    m.aggP = pyo.Constraint(
        m.T,
        rule=lambda mm, i, j: mm.Pij[i, j] == sum(mm.Pijt[i, j, tap] for tap in K[(i, j)])
    )
    m.aggQ = pyo.Constraint(
        m.T,
        rule=lambda mm, i, j: mm.Qij[i, j] == sum(mm.Qijt[i, j, tap] for tap in K[(i, j)])
    )
    m.aggEll = pyo.Constraint(
        m.T,
        rule=lambda mm, i, j: mm.ell[i, j] == sum(mm.ellt[i, j, tap] for tap in K[(i, j)])
    )

    m.segP_ub = pyo.Constraint(
        m.OTAP,
        rule=lambda mm, i, j, tap: mm.Pijt[i, j, tap] <= segPmax[(i, j, tap)] * mm.beta[i, j, tap]
    )
    m.segP_lb = pyo.Constraint(
        m.OTAP,
        rule=lambda mm, i, j, tap: mm.Pijt[i, j, tap] >= -segPmax[(i, j, tap)] * mm.beta[i, j, tap]
    )
    m.segQ_ub = pyo.Constraint(
        m.OTAP,
        rule=lambda mm, i, j, tap: mm.Qijt[i, j, tap] <= segQmax[(i, j, tap)] * mm.beta[i, j, tap]
    )
    m.segQ_lb = pyo.Constraint(
        m.OTAP,
        rule=lambda mm, i, j, tap: mm.Qijt[i, j, tap] >= -segQmax[(i, j, tap)] * mm.beta[i, j, tap]
    )
    m.segEll_ub = pyo.Constraint(
        m.OTAP,
        rule=lambda mm, i, j, tap: mm.ellt[i, j, tap] <= mm.ellmax[i, j] * mm.beta[i, j, tap]
    )

    m.w1 = pyo.Constraint(
        m.OTAP,
        rule=lambda mm, i, j, tap: mm.wtap[i, j, tap] <= mm.Vmax_sq[i] * mm.beta[i, j, tap]
    )
    m.w2 = pyo.Constraint(
        m.OTAP,
        rule=lambda mm, i, j, tap: mm.wtap[i, j, tap] >= mm.Vmin_sq[i] * mm.beta[i, j, tap]
    )
    m.w3 = pyo.Constraint(
        m.OTAP,
        rule=lambda mm, i, j, tap: mm.wtap[i, j, tap] <= mm.v[i] - mm.Vmin_sq[i] * (1.0 - mm.beta[i, j, tap])
    )
    m.w4 = pyo.Constraint(
        m.OTAP,
        rule=lambda mm, i, j, tap: mm.wtap[i, j, tap] >= mm.v[i] - mm.Vmax_sq[i] * (1.0 - mm.beta[i, j, tap])
    )

    def qsh_zero_rule(mm, i):
        if int(i) in C_set:
            return pyo.Constraint.Skip
        return mm.qsh[i] == 0.0
    m.qsh_zero = pyo.Constraint(m.N, rule=qsh_zero_rule)

    m.qsh_ub = pyo.Constraint(m.C, rule=lambda mm, i: mm.qsh[int(i)] <= mm.Mq[i] * mm.a_sh[i])
    m.qsh_lb = pyo.Constraint(m.C, rule=lambda mm, i: mm.qsh[int(i)] >= -mm.Mq[i] * mm.a_sh[i])

    def qsh_match_pos(mm, i):
        q_target = mm.bcap[i] * (mm.v[int(i)] / mm.vrated[i])
        return mm.qsh[int(i)] - q_target <= mm.Mq[i] * (1.0 - mm.a_sh[i])

    def qsh_match_neg(mm, i):
        q_target = mm.bcap[i] * (mm.v[int(i)] / mm.vrated[i])
        return q_target - mm.qsh[int(i)] <= mm.Mq[i] * (1.0 - mm.a_sh[i])

    m.qsh_match_pos = pyo.Constraint(m.C, rule=qsh_match_pos)
    m.qsh_match_neg = pyo.Constraint(m.C, rule=qsh_match_neg)

    def bfm_P_balance(mm, i):
        out_sum = sum(mm.Pij[a, b] for (a, b) in out_arcs[int(i)])
        in_sum = sum((mm.Pij[a, b] - mm.r[a, b] * mm.ell[a, b]) for (a, b) in in_arcs[int(i)])
        return out_sum - in_sum == mm.Pinj[i]

    def bfm_Q_balance(mm, i):
        out_sum = sum(mm.Qij[a, b] for (a, b) in out_arcs[int(i)])
        in_sum = sum((mm.Qij[a, b] - mm.x[a, b] * mm.ell[a, b]) for (a, b) in in_arcs[int(i)])
        return out_sum - in_sum == mm.Qinj[i]

    m.BFM_P = pyo.Constraint(m.N, rule=bfm_P_balance)
    m.BFM_Q = pyo.Constraint(m.N, rule=bfm_Q_balance)

    def vdrop_nt(mm, i, j):
        rij = mm.r[i, j]
        xij = mm.x[i, j]
        return mm.v[j] == mm.v[i] - 2.0 * (rij * mm.Pij[i, j] + xij * mm.Qij[i, j]) + (rij * rij + xij * xij) * mm.ell[i, j]

    m.Vdrop_NT = pyo.Constraint(m.NT, rule=vdrop_nt)

    def vdrop_t(mm, i, j):
        rij = mm.r[i, j]
        xij = mm.x[i, j]
        return mm.v[j] == sum(
            mm.delta_tap[i, j, tap] * mm.wtap[i, j, tap]
            - 2.0 * (rij * mm.Pijt[i, j, tap] + xij * mm.Qijt[i, j, tap])
            + (rij * rij + xij * xij) * mm.ellt[i, j, tap]
            for tap in K[(i, j)]
        )

    m.Vdrop_T = pyo.Constraint(m.T, rule=vdrop_t)

    def soc_nt(mm, i, j):
        return mm.Pij[i, j] ** 2 + mm.Qij[i, j] ** 2 <= mm.v[i] * mm.ell[i, j]

    m.SOC_NT = pyo.Constraint(m.NT, rule=soc_nt)

    def soc_t(mm, i, j, tap):
        return mm.Pijt[i, j, tap] ** 2 + mm.Qijt[i, j, tap] ** 2 <= mm.delta_tap[i, j, tap] * mm.wtap[i, j, tap] * mm.ellt[i, j, tap]

    m.SOC_T = pyo.Constraint(m.OTAP, rule=soc_t)

    m.obj = pyo.Objective(
        expr=sum(
            m.c2[g] * (sn * m.Pg[g]) ** 2 + m.c1[g] * (sn * m.Pg[g]) + m.c0[g]
            for g in m.G
        ),
        sense=pyo.minimize
    )

    m._out_arcs = out_arcs
    m._in_arcs = in_arcs
    m._data_for_check = data
    return m


# ----------------------------
# Initialization
# ----------------------------
def _get_branch_pf_flow_pq(net, meta: Dict[str, Any]) -> Tuple[float, float]:
    # Aggregated parallel branch: sum PF flows of all member elements.
    if bool(meta.get("aggregate", False)):
        Ptot = 0.0
        Qtot = 0.0
        for sub in meta.get("members", []):
            Pk, Qk = _get_branch_pf_flow_pq(net, sub)
            Ptot += float(Pk)
            Qtot += float(Qk)
        return Ptot, Qtot

    et = meta["element_type"]
    idx = int(meta["element_index"])
    fb = int(meta["from_bus_pp"])
    tb = int(meta["to_bus_pp"])

    if et == "line":
        line_fb = int(net.line.at[idx, "from_bus"])
        line_tb = int(net.line.at[idx, "to_bus"])
        if fb == line_fb and tb == line_tb:
            return float(net.res_line.at[idx, "p_from_mw"]), float(net.res_line.at[idx, "q_from_mvar"])
        if fb == line_tb and tb == line_fb:
            return float(net.res_line.at[idx, "p_to_mw"]), float(net.res_line.at[idx, "q_to_mvar"])
        raise ValueError(f"Line orientation mismatch for edge {(fb, tb)} and line index {idx}")

    if et == "trafo":
        hv = int(net.trafo.at[idx, "hv_bus"])
        lv = int(net.trafo.at[idx, "lv_bus"])
        if fb == hv and tb == lv:
            return float(net.res_trafo.at[idx, "p_hv_mw"]), float(net.res_trafo.at[idx, "q_hv_mvar"])
        if fb == lv and tb == hv:
            return float(net.res_trafo.at[idx, "p_lv_mw"]), float(net.res_trafo.at[idx, "q_lv_mvar"])
        raise ValueError(f"Trafo orientation mismatch for edge {(fb, tb)} and trafo index {idx}")

    raise ValueError(f"Unsupported branch element_type: {et}")


# ----------------------------
# Initialization
# ----------------------------
def warmstart_from_pf(model: pyo.ConcreteModel, data: Dict[str, Any], net) -> bool:
    if not RUN_PF_WARMSTART:
        return False

    sn = data["sn_mva"]
    branch_meta_of_edge = data["branch_meta_of_edge"]
    T_set = set(data["T"])

    try:
        pp.runpp(
            net,
            algorithm="nr",
            init="flat",
            calculate_voltage_angles=True,
            enforce_q_lims=False,
            numba=False,
        )

        for i in model.N:
            vm = float(net.res_bus.at[int(i), "vm_pu"])
            v0 = vm * vm
            lb = float(pyo.value(model.Vmin_sq[int(i)]))
            ub = float(pyo.value(model.Vmax_sq[int(i)]))
            v0 = min(max(v0, lb), ub)
            model.v[int(i)].set_value(v0)

        ext_map = {}
        if hasattr(net, "res_ext_grid") and len(net.res_ext_grid.index) > 0:
            for eg in net.ext_grid.index:
                eg = int(eg)
                ext_map[("ext_grid", eg)] = (
                    float(net.res_ext_grid.at[eg, "p_mw"]) / sn,
                    float(net.res_ext_grid.at[eg, "q_mvar"]) / sn
                )

        gen_map = {}
        if hasattr(net, "res_gen") and len(net.res_gen.index) > 0:
            for gi in net.gen.index:
                gi = int(gi)
                gen_map[("gen", gi)] = (
                    float(net.res_gen.at[gi, "p_mw"]) / sn,
                    float(net.res_gen.at[gi, "q_mvar"]) / sn
                )

        for gg in model.G:
            rec = data["gen_records"][int(gg)]
            key = (rec["type"], rec["id"])
            if key in ext_map:
                pg, qg = ext_map[key]
            elif key in gen_map:
                pg, qg = gen_map[key]
            else:
                pg, qg = 0.0, 0.0

            pg = min(max(pg, float(model.Pgmin[gg])), float(model.Pgmax[gg]))
            qg = min(max(qg, float(model.Qgmin[gg])), float(model.Qgmax[gg]))
            model.Pg[gg].set_value(pg)
            model.Qg[gg].set_value(qg)

        for i in model.N:
            model.qsh[int(i)].set_value(0.0)
        for i in model.C:
            model.a_sh[i].set_value(0.0)

        for (i, j, tap) in model.OTAP:
            model.beta[i, j, tap].set_value(0.0)
            model.wtap[i, j, tap].set_value(0.0)
            model.Pijt[i, j, tap].set_value(0.0)
            model.Qijt[i, j, tap].set_value(0.0)
            model.ellt[i, j, tap].set_value(0.0)

        recommended_tap = {}
        if "recommended_nonexact_oltc_taps" in net:
            try:
                for _, row in net["recommended_nonexact_oltc_taps"].iterrows():
                    recommended_tap[(int(row["from_bus_pp"]), int(row["to_bus_pp"]))] = int(row["tap"])
            except Exception:
                recommended_tap = {}

        for (i, j) in model.T:
            taps = data["K"][(i, j)]
            pick = recommended_tap.get((int(i), int(j)), _default_tap(taps))
            if pick not in taps:
                pick = _default_tap(taps)
            for tap in taps:
                model.beta[i, j, tap].set_value(1.0 if tap == pick else 0.0)

        recommended_shunt = {}
        if "recommended_nonexact_shunt_status" in net:
            try:
                for _, row in net["recommended_nonexact_shunt_status"].iterrows():
                    recommended_shunt[int(row["bus_pp"])] = int(row["status"])
            except Exception:
                recommended_shunt = {}
        for i in model.C:
            model.a_sh[i].set_value(float(recommended_shunt.get(int(i), 0)))

        for (i, j) in model.E:
            meta = branch_meta_of_edge[(int(i), int(j))]
            Pmw, Qmvar = _get_branch_pf_flow_pq(net, meta)
            Ppu = Pmw / sn
            Qpu = Qmvar / sn

            smax = math.sqrt(max(0.0, float(pyo.value(model.ellmax[i, j])) * float(pyo.value(model.Vmax_sq[i]))))
            Ppu = min(max(Ppu, -smax), smax)
            Qpu = min(max(Qpu, -smax), smax)

            model.Pij[i, j].set_value(Ppu)
            model.Qij[i, j].set_value(Qpu)

            vi = float(pyo.value(model.v[i]))
            if (int(i), int(j)) in T_set:
                tap_vals = data["K"][(int(i), int(j))]
                pick = recommended_tap.get((int(i), int(j)), _default_tap(tap_vals))
                if pick not in tap_vals:
                    pick = _default_tap(tap_vals)
                delta0 = float(data["delta_tap"][((int(i), int(j)), pick)])
            else:
                delta0 = 1.0

            ell = (Ppu * Ppu + Qpu * Qpu) / max(delta0 * vi, 1e-8)
            ell = max(0.0, min(ell, float(pyo.value(model.ellmax[i, j]))))
            model.ell[i, j].set_value(ell)

            if (int(i), int(j)) in T_set:
                tap_vals = data["K"][(int(i), int(j))]
                pick = recommended_tap.get((int(i), int(j)), _default_tap(tap_vals))
                if pick not in tap_vals:
                    pick = _default_tap(tap_vals)
                for tap in tap_vals:
                    if tap == pick:
                        model.Pijt[i, j, tap].set_value(Ppu)
                        model.Qijt[i, j, tap].set_value(Qpu)
                        model.ellt[i, j, tap].set_value(ell)
                        model.wtap[i, j, tap].set_value(vi)
                    else:
                        model.Pijt[i, j, tap].set_value(0.0)
                        model.Qijt[i, j, tap].set_value(0.0)
                        model.ellt[i, j, tap].set_value(0.0)
                        model.wtap[i, j, tap].set_value(0.0)

        return True

    except Exception:
        return False


def set_default_discrete_and_fix(
    model: pyo.ConcreteModel,
    data: Dict[str, Any],
    net=None,
):
    recommended_tap = {}
    recommended_shunt = {}

    if net is not None and "recommended_nonexact_oltc_taps" in net:
        try:
            for _, row in net["recommended_nonexact_oltc_taps"].iterrows():
                recommended_tap[(int(row["from_bus_pp"]), int(row["to_bus_pp"]))] = int(row["tap"])
        except Exception:
            recommended_tap = {}

    if net is not None and "recommended_nonexact_shunt_status" in net:
        try:
            for _, row in net["recommended_nonexact_shunt_status"].iterrows():
                recommended_shunt[int(row["bus_pp"])] = int(row["status"])
        except Exception:
            recommended_shunt = {}

    for (i, j) in model.T:
        taps = data["K"][(i, j)]
        pick = recommended_tap.get((int(i), int(j)), _default_tap(taps))
        if pick not in taps:
            pick = _default_tap(taps)
        for tap in taps:
            model.beta[i, j, tap].fix(1 if tap == pick else 0)

    for i in model.C:
        model.a_sh[i].fix(int(recommended_shunt.get(int(i), 0)))


# ----------------------------
# Diagnostics
# ----------------------------
def evaluate_quality(model: pyo.ConcreteModel, data: Dict[str, Any]) -> Dict[str, float]:
    out_arcs = model._out_arcs
    in_arcs = model._in_arcs
    K = data["K"]

    max_bfp = 0.0
    max_bfq = 0.0
    max_vdrop = 0.0
    max_soc_violation = 0.0
    max_soc_slack = 0.0
    max_onehot = 0.0
    max_agg = 0.0
    max_wlink = 0.0
    max_sh = 0.0
    max_frac = 0.0

    for i in model.N:
        outP = sum(pyo.value(model.Pij[a, b]) for (a, b) in out_arcs[int(i)])
        inP = sum(pyo.value(model.Pij[a, b] - model.r[a, b] * model.ell[a, b]) for (a, b) in in_arcs[int(i)])
        max_bfp = max(max_bfp, abs((outP - inP) - pyo.value(model.Pinj[i])))

        outQ = sum(pyo.value(model.Qij[a, b]) for (a, b) in out_arcs[int(i)])
        inQ = sum(pyo.value(model.Qij[a, b] - model.x[a, b] * model.ell[a, b]) for (a, b) in in_arcs[int(i)])
        max_bfq = max(max_bfq, abs((outQ - inQ) - pyo.value(model.Qinj[i])))

    for (i, j) in model.NT:
        rij = pyo.value(model.r[i, j])
        xij = pyo.value(model.x[i, j])

        lhs_v = pyo.value(model.v[j])
        rhs_v = pyo.value(model.v[i] - 2.0 * (rij * model.Pij[i, j] + xij * model.Qij[i, j]) + (rij * rij + xij * xij) * model.ell[i, j])
        max_vdrop = max(max_vdrop, abs(lhs_v - rhs_v))

        lhs_soc = pyo.value(model.Pij[i, j] ** 2 + model.Qij[i, j] ** 2)
        rhs_soc = pyo.value(model.v[i] * model.ell[i, j])
        max_soc_violation = max(max_soc_violation, max(0.0, lhs_soc - rhs_soc))
        max_soc_slack = max(max_soc_slack, max(0.0, rhs_soc - lhs_soc))

    for (i, j) in model.T:
        rij = pyo.value(model.r[i, j])
        xij = pyo.value(model.x[i, j])

        s = sum(pyo.value(model.beta[i, j, tap]) for tap in K[(i, j)])
        max_onehot = max(max_onehot, abs(s - 1.0))

        aggP = sum(pyo.value(model.Pijt[i, j, tap]) for tap in K[(i, j)])
        aggQ = sum(pyo.value(model.Qijt[i, j, tap]) for tap in K[(i, j)])
        aggL = sum(pyo.value(model.ellt[i, j, tap]) for tap in K[(i, j)])
        max_agg = max(max_agg, abs(pyo.value(model.Pij[i, j]) - aggP))
        max_agg = max(max_agg, abs(pyo.value(model.Qij[i, j]) - aggQ))
        max_agg = max(max_agg, abs(pyo.value(model.ell[i, j]) - aggL))

        rhs_v = 0.0
        for tap in K[(i, j)]:
            rhs_v += pyo.value(
                model.delta_tap[i, j, tap] * model.wtap[i, j, tap]
                - 2.0 * (rij * model.Pijt[i, j, tap] + xij * model.Qijt[i, j, tap])
                + (rij * rij + xij * xij) * model.ellt[i, j, tap]
            )
        lhs_v = pyo.value(model.v[j])
        max_vdrop = max(max_vdrop, abs(lhs_v - rhs_v))

        for tap in K[(i, j)]:
            lhs_soc = pyo.value(model.Pijt[i, j, tap] ** 2 + model.Qijt[i, j, tap] ** 2)
            rhs_soc = pyo.value(model.delta_tap[i, j, tap] * model.wtap[i, j, tap] * model.ellt[i, j, tap])
            max_soc_violation = max(max_soc_violation, max(0.0, lhs_soc - rhs_soc))
            max_soc_slack = max(max_soc_slack, max(0.0, rhs_soc - lhs_soc))

            beta = pyo.value(model.beta[i, j, tap])
            w = pyo.value(model.wtap[i, j, tap])
            v = pyo.value(model.v[i])
            vmin_sq = pyo.value(model.Vmin_sq[i])
            vmax_sq = pyo.value(model.Vmax_sq[i])

            viols = [
                max(0.0, w - vmax_sq * beta),
                max(0.0, vmin_sq * beta - w),
                max(0.0, w - (v - vmin_sq * (1.0 - beta))),
                max(0.0, (v - vmax_sq * (1.0 - beta)) - w),
            ]
            max_wlink = max(max_wlink, max(viols))
            max_frac = max(max_frac, abs(beta - round(beta)))

    Cset = set(data["C"])
    for i in model.N:
        if int(i) not in Cset:
            max_sh = max(max_sh, abs(pyo.value(model.qsh[int(i)])))

    for i in model.C:
        a = pyo.value(model.a_sh[i])
        max_frac = max(max_frac, abs(a - round(a)))

    max_resid = max(
        max_bfp,
        max_bfq,
        max_vdrop,
        max_soc_violation,
        max_onehot,
        max_agg,
        max_wlink,
        max_sh,
    )

    return {
        "max_bfm_p": max_bfp,
        "max_bfm_q": max_bfq,
        "max_vdrop": max_vdrop,
        "max_soc_violation": max_soc_violation,
        "max_soc_slack": max_soc_slack,
        "max_onehot": max_onehot,
        "max_agg": max_agg,
        "max_wlink": max_wlink,
        "max_shunt": max_sh,
        "max_frac": max_frac,
        "max_resid": max_resid,
    }


def solution_is_acceptable(q: Dict[str, float]) -> bool:
    return (q["max_resid"] <= RESID_TOL_ACCEPT) and (q["max_frac"] <= INT_TOL_ACCEPT)


def _pick_selected_tap(model: pyo.ConcreteModel, data: Dict[str, Any], edge: Tuple[int, int]) -> Optional[int]:
    taps = data["K"].get(edge, [])
    best_tap = None
    best_val = -1e100
    for tap in taps:
        vv = pyo.value(model.beta[edge[0], edge[1], tap], exception=False)
        if vv is None:
            continue
        if float(vv) > best_val:
            best_val = float(vv)
            best_tap = int(tap)
    return best_tap


def _seed_validation_results(net, model: pyo.ConcreteModel):
    if not hasattr(net, "res_bus") or net.res_bus is None or net.res_bus.empty:
        return
    if "vm_pu" not in net.res_bus.columns:
        return

    for i in model.N:
        vm = math.sqrt(max(float(pyo.value(model.v[int(i)])), 0.0))
        if int(i) in net.res_bus.index:
            net.res_bus.at[int(i), "vm_pu"] = vm
            if "va_degree" in net.res_bus.columns:
                net.res_bus.at[int(i), "va_degree"] = 0.0


def _build_validation_net_from_solution(model: pyo.ConcreteModel, data: Dict[str, Any], net):
    net_ac = copy.deepcopy(net)
    sn = float(data["sn_mva"])
    branch_meta = data["branch_meta_of_edge"]

    # Apply discrete OLTC choices to transformer tap positions.
    for (i, j) in data["T"]:
        meta = branch_meta.get((int(i), int(j)))
        if meta is None or bool(meta.get("aggregate", False)):
            continue
        if str(meta.get("element_type", "")) != "trafo":
            continue
        idx = int(meta["element_index"])
        if idx not in net_ac.trafo.index or "tap_pos" not in net_ac.trafo.columns:
            continue
        best_tap = _pick_selected_tap(model, data, (int(i), int(j)))
        if best_tap is not None:
            net_ac.trafo.at[idx, "tap_pos"] = int(best_tap)

    # Add switched-shunt candidates selected by the MISOCP.
    for i in data["C"]:
        aval = pyo.value(model.a_sh[int(i)], exception=False)
        if aval is None or float(aval) < 0.5:
            continue
        pp.create_shunt(
            net_ac,
            bus=int(i),
            p_mw=0.0,
            q_mvar=float(-sn * data["bcap_pu"][int(i)]),
            name=f"DW_SwitchedShunt@bus{int(i)}",
        )

    # Apply dispatch setpoints while leaving the original OPF model untouched.
    for gg in model.G:
        rec = data["gen_records"][int(gg)]
        bus = int(rec["bus"])
        vm_pu = math.sqrt(max(float(pyo.value(model.v[bus])), 0.0))
        pg_mw = float(sn * pyo.value(model.Pg[gg]))

        if rec["type"] == "ext_grid":
            idx = int(rec["id"])
            if idx in net_ac.ext_grid.index:
                if "vm_pu" in net_ac.ext_grid.columns:
                    net_ac.ext_grid.at[idx, "vm_pu"] = vm_pu
        elif rec["type"] == "gen":
            idx = int(rec["id"])
            if idx in net_ac.gen.index:
                if "p_mw" in net_ac.gen.columns:
                    net_ac.gen.at[idx, "p_mw"] = pg_mw
                if "vm_pu" in net_ac.gen.columns:
                    net_ac.gen.at[idx, "vm_pu"] = vm_pu

    _seed_validation_results(net_ac, model)
    return net_ac


def _compute_ac_generation_cost(net_ac, data: Dict[str, Any]) -> Optional[float]:
    total = 0.0
    for rec in data["gen_records"]:
        idx = int(rec["id"])
        if rec["type"] == "ext_grid":
            if not hasattr(net_ac, "res_ext_grid") or idx not in net_ac.res_ext_grid.index:
                return None
            p_mw = float(net_ac.res_ext_grid.at[idx, "p_mw"])
        elif rec["type"] == "gen":
            if not hasattr(net_ac, "res_gen") or idx not in net_ac.res_gen.index:
                return None
            p_mw = float(net_ac.res_gen.at[idx, "p_mw"])
        else:
            continue
        total += float(rec["c2"]) * p_mw * p_mw + float(rec["c1"]) * p_mw + float(rec["c0"])
    return float(total)


def evaluate_dw_postsolve(
    model: pyo.ConcreteModel,
    data: Dict[str, Any],
    net,
    raw_obj: Optional[float],
    quality: Optional[Dict[str, float]],
) -> Dict[str, Any]:
    report: Dict[str, Any] = {
        "raw_objective": raw_obj,
        "soc_screen_tol": float(DW_SOC_SLACK_SCREEN_TOL),
        "soc_screen_pass": None,
        "max_soc_slack": None,
        "ac_validation_ok": False,
        "ac_validation_init": None,
        "ac_validated_cost": None,
        "ac_cost_gap": None,
        "ac_max_vm_violation": None,
        "ac_max_line_loading_pct": None,
        "ac_max_trafo_loading_pct": None,
        "ac_max_loading_pct": None,
        "error": None,
    }

    if quality is not None:
        report["max_soc_slack"] = float(quality["max_soc_slack"])
        report["soc_screen_pass"] = bool(float(quality["max_soc_slack"]) <= DW_SOC_SLACK_SCREEN_TOL)

    if (not RUN_DW_AC_VALIDATION) or (raw_obj is None):
        return report

    try:
        base_net = _build_validation_net_from_solution(model, data, net)
    except Exception as e:
        report["error"] = f"build-validation-net failed: {e}"
        return report

    last_error = None
    for init_mode in DW_PP_INIT_MODES:
        net_try = copy.deepcopy(base_net)
        try:
            pp.runpp(
                net_try,
                algorithm="nr",
                init=init_mode,
                calculate_voltage_angles=True,
                enforce_q_lims=True,
                numba=False,
            )
            if not bool(getattr(net_try, "converged", False)):
                last_error = f"runpp did not converge with init={init_mode}"
                continue

            ac_cost = _compute_ac_generation_cost(net_try, data)
            vm = np.asarray(net_try.res_bus["vm_pu"], dtype=float)
            vmin = np.asarray(net_try.bus["min_vm_pu"], dtype=float)
            vmax = np.asarray(net_try.bus["max_vm_pu"], dtype=float)
            vm_violation = float(max(np.max(np.maximum(vmin - vm, 0.0)), np.max(np.maximum(vm - vmax, 0.0))))

            line_loading = 0.0
            if hasattr(net_try, "res_line") and net_try.res_line is not None and (not net_try.res_line.empty):
                if "loading_percent" in net_try.res_line.columns:
                    line_loading = float(np.nanmax(np.asarray(net_try.res_line["loading_percent"], dtype=float)))

            trafo_loading = 0.0
            if hasattr(net_try, "res_trafo") and net_try.res_trafo is not None and (not net_try.res_trafo.empty):
                if "loading_percent" in net_try.res_trafo.columns:
                    trafo_loading = float(np.nanmax(np.asarray(net_try.res_trafo["loading_percent"], dtype=float)))

            report["ac_validation_ok"] = True
            report["ac_validation_init"] = init_mode
            report["ac_validated_cost"] = ac_cost
            report["ac_cost_gap"] = (None if ac_cost is None else float(ac_cost - raw_obj))
            report["ac_max_vm_violation"] = vm_violation
            report["ac_max_line_loading_pct"] = line_loading
            report["ac_max_trafo_loading_pct"] = trafo_loading
            report["ac_max_loading_pct"] = float(max(line_loading, trafo_loading))
            return report
        except Exception as e:
            last_error = f"init={init_mode}: {e}"

    report["error"] = last_error
    return report


def print_dw_postsolve_report(report: Dict[str, Any], label: str = "main"):
    print(f"\n--- DW Post-Solve Report ({label}) ---")

    if report.get("raw_objective") is not None:
        print(f"[DW] raw objective         = {report['raw_objective']:.10f}")

    if report.get("max_soc_slack") is not None:
        status = "PASS" if report.get("soc_screen_pass") else "FAIL"
        print(
            f"[DW] max_soc_slack        = {report['max_soc_slack']:.3e} "
            f"({status}, tol={report['soc_screen_tol']:.1e})"
        )

    if report.get("ac_validation_ok"):
        print(f"[DW] AC validation        = converged (init={report['ac_validation_init']})")
        if report.get("ac_validated_cost") is not None:
            print(f"[DW] AC-validated cost    = {report['ac_validated_cost']:.10f}")
        if report.get("ac_cost_gap") is not None:
            print(f"[DW] AC cost gap          = {report['ac_cost_gap']:+.10f}")
        if report.get("ac_max_vm_violation") is not None:
            print(f"[DW] max |V| violation    = {report['ac_max_vm_violation']:.3e}")
        if report.get("ac_max_loading_pct") is not None:
            print(f"[DW] max loading percent  = {report['ac_max_loading_pct']:.3f}")
    else:
        err = report.get("error", "not run")
        print(f"[DW] AC validation        = failed ({err})")


# ----------------------------
# Solver
# ----------------------------
def _configure_solver(opt, solver_name: str, tee: bool):
    try:
        if solver_name.startswith("cplex"):
            opt.options["timelimit"] = float(SOLVER_TIME_LIMIT)
            try:
                opt.options["mipgap"] = float(SOLVER_MIP_GAP)
            except Exception:
                pass

        elif solver_name.startswith("gurobi"):
            opt.options["TimeLimit"] = float(SOLVER_TIME_LIMIT)
            opt.options["MIPGap"] = float(SOLVER_MIP_GAP)
            opt.options["MIPGapAbs"] = float(SOLVER_MIP_GAP_ABS)
            opt.options["NumericFocus"] = 2

        elif solver_name == "scip":
            opt.options["limits/time"] = float(SOLVER_TIME_LIMIT)
            opt.options["limits/gap"] = float(SOLVER_MIP_GAP)
            opt.options["limits/absgap"] = float(SOLVER_MIP_GAP_ABS)
            opt.options["display/verblevel"] = 5 if tee else 0
    except Exception:
        pass


def solve_misocp(model: pyo.ConcreteModel, tee: bool = True) -> Dict[str, Any]:
    candidates = [
        "cplex_direct",
        "cplex",
        "gurobi_direct",
        "gurobi",
        "scip",
    ]

    last_error = None

    for sname in candidates:
        opt = pyo.SolverFactory(sname)
        if opt is None or not opt.available(exception_flag=False):
            continue

        _configure_solver(opt, sname, tee)

        t0 = time.perf_counter()
        try:
            res = opt.solve(model, tee=tee)
            elapsed = time.perf_counter() - t0
        except Exception as e:
            last_error = f"{sname}: {e}"
            continue

        term = res.solver.termination_condition
        try:
            obj_raw = pyo.value(model.obj, exception=False)
            obj = float(obj_raw) if obj_raw is not None else None
        except Exception:
            obj = None

        infeas_terms = {
            TerminationCondition.infeasible,
            TerminationCondition.infeasibleOrUnbounded,
            TerminationCondition.invalidProblem,
        }
        if (term in infeas_terms) or (obj is None) or (not np.isfinite(obj)):
            return {
                "solver": sname,
                "ok": False,
                "reason": f"no loaded solution ({term})",
                "obj": None,
                "quality": None,
                "termination": term,
                "elapsed": elapsed,
            }

        q = evaluate_quality(model, model._data_for_check)
        ok = solution_is_acceptable(q)

        return {
            "solver": sname,
            "ok": ok,
            "reason": "accepted" if ok else "rejected by quality",
            "obj": obj,
            "quality": q,
            "termination": term,
            "elapsed": elapsed,
        }

    return {
        "solver": None,
        "ok": False,
        "reason": f"No solver succeeded. Last error: {last_error}",
        "obj": None,
        "quality": None,
        "termination": None,
        "elapsed": None,
    }


# ----------------------------
# Reporting
# ----------------------------
def print_solution(model: pyo.ConcreteModel, data: Dict[str, Any], title: str):
    sn = data["sn_mva"]
    slack = data["slack_bus"]
    gen_records = data["gen_records"]

    print(f"\n==================== {title} ====================")
    print(f"Objective (EUR): {pyo.value(model.obj):.10f}")

    q = evaluate_quality(model, data)
    print("\n--- Quality ---")
    for k, v in q.items():
        print(f"{k}: {v:.3e}")

    if q["max_soc_slack"] > 1e-5:
        print("\n[WARN] max_soc_slack is not near zero.")
        print("       This means the SOCP relaxation may be inexact.")

    print("\n--- Bus voltages ---")
    for i in data["buses"]:
        v = pyo.value(model.v[i])
        V = math.sqrt(max(v, 0.0))
        tag = " [slack]" if i == slack else ""
        print(f"Bus {i:2d}: v={v:.6f}, |V|={V:.6f}{tag}")

    print("\n--- Generators (MW/MVAr) ---")
    for gg in model.G:
        rec = gen_records[int(gg)]
        Pg = sn * pyo.value(model.Pg[gg])
        Qg = sn * pyo.value(model.Qg[gg])
        print(f"{rec['type']}[{rec['id']}] @ bus {rec['bus']:2d}: P={Pg:.4f} MW, Q={Qg:.4f} MVAr")

    if len(data["T"]) > 0:
        print("\n--- OLTC taps ---")
        for (i, j) in model.T:
            taps = data["K"][(i, j)]
            best_t, best_v = None, -1.0
            for t in taps:
                vv = pyo.value(model.beta[i, j, t])
                if vv > best_v:
                    best_v = vv
                    best_t = t
            print(
                f"OLTC ({i}->{j}): tap={best_t:>3d}, "
                f"delta={pyo.value(model.delta[i, j]):.6f}, "
                f"alpha={pyo.value(model.alpha[i, j]):.6f}"
            )

    if len(data["C"]) > 0:
        print("\n--- Shunts ---")
        for i in model.C:
            a = int(round(pyo.value(model.a_sh[i])))
            qpu = pyo.value(model.qsh[int(i)])
            print(f"Shunt @ bus {int(i):2d}: a_sh={a}, qsh={qpu:.6f} pu  => {sn*qpu:.3f} MVAr")

    print("=========================================================\n")


def maybe_print_best_available_solution(
    solve_res: Dict[str, Any],
    model: Optional[pyo.ConcreteModel],
    data: Dict[str, Any],
    title: str,
) -> bool:
    if (not REPORT_BEST_AVAILABLE_WHEN_REJECTED) or (model is None):
        return False
    if solve_res.get("obj") is None:
        return False

    print(
        "\n[WARN] Reporting best-available solution despite non-acceptance."
        f" Reason: {solve_res.get('reason')}"
    )
    print_solution(model, data, title=title)
    return True


# ----------------------------
# Main
# ----------------------------
def main():
    t_all0 = time.perf_counter()

    net = mcase.case300_opf(**NETWORK_BUILD_KWARGS)

    cfg = build_cfg_from_net_metadata(net)
    data = extract_bfm300_fullmesh_data(net, cfg)

    if "pf_calibrated_branch_limits" in net:
        brcal = net["pf_calibrated_branch_limits"]
        try:
            n_tight = int((brcal["smax_new_mva"] > brcal["smax_old_mva"] + 1e-9).sum())
        except Exception:
            n_tight = len(brcal)
        print(f"[INFO] PF-calibrated branch limits updated on {n_tight} branches")
    if "pf_widened_voltage_bounds" in net:
        vw = net["pf_widened_voltage_bounds"]
        try:
            n_wide = int(((vw["min_new"] < vw["min_old"] - 1e-12) | (vw["max_new"] > vw["max_old"] + 1e-12)).sum())
        except Exception:
            n_wide = len(vw)
        print(f"[INFO] PF-widened voltage bounds updated on {n_wide} buses")

    print(f"[INFO] #buses = {len(data['buses'])}")
    print("[INFO] network = ieee300bus (standalone build kwargs)")
    print(f"[INFO] #branches (directed, full) = {len(data['E'])}")
    print(f"[INFO] #gens = {len(data['gen_records'])}")
    print(f"[INFO] #OLTC branches = {len(data['T'])}")
    print(f"[INFO] #shunts = {len(data['C'])}")
    print(f"[INFO] total Pd (MW)  = {data['sn_mva'] * sum(data['Pd_pu'].values()):.4f}")
    print(f"[INFO] total Qd (MVAr)= {data['sn_mva'] * sum(data['Qd_pu'].values()):.4f}")

    triples = [(r["type"], r["id"], r["c2"], r["c1"], r["c0"]) for r in data["gen_records"]]
    all_zero_cost = all(
        (abs(c2) < 1e-15 and abs(c1) < 1e-15 and abs(c0) < 1e-15)
        for (_, _, c2, c1, c0) in triples
    )
    print("[INFO] cost triples (type,id,c2,c1,c0) sample:", triples[:5], " ...")
    if all_zero_cost:
        print("[WARN] All generator cost coefficients are zero -> objective is constant.")

    model = build_pyomo_bfm_misocp_model(data, relax_binaries=False)

    pf_ok = warmstart_from_pf(model, data, net)
    print(f"[INFO] PF warm start success = {pf_ok}")

    print("\n[INFO] Solving BFM-cr MISOCP on IEEE 300-bus ...")
    res = solve_misocp(model, tee=TEE_SOLVER_LOG)

    print(f"[INFO] solver      = {res['solver']}")
    print(f"[INFO] elapsed     = {res['elapsed']}")
    print(f"[INFO] termination = {res['termination']}")
    print(f"[INFO] accepted    = {res['ok']} ({res['reason']})")

    if res["quality"] is not None:
        q = res["quality"]
        print(f"[INFO] max_resid     = {q['max_resid']:.3e}")
        print(f"[INFO] max_frac      = {q['max_frac']:.3e}")
        print(f"[INFO] max_soc_slack = {q['max_soc_slack']:.3e}")

    if res["obj"] is not None:
        print(f"[INFO] objective = {res['obj']:.10f}")

    if res["obj"] is not None:
        dw_report = evaluate_dw_postsolve(
            model,
            data,
            net,
            raw_obj=res["obj"],
            quality=res["quality"],
        )
        print_dw_postsolve_report(dw_report, label="main")

    if res["ok"]:
        print_solution(model, data, title="IEEE 300-bus BFM-cr MISOCP Solution")
    else:
        print("[WARN] Main MISOCP solution rejected or no solver succeeded.")

        reported_any = False

        print("\n[INFO] Solving continuous relaxation for diagnosis ...")
        relax_model = build_pyomo_bfm_misocp_model(data, relax_binaries=True)
        _ = warmstart_from_pf(relax_model, data, net)
        relax_res = solve_misocp(relax_model, tee=False)
        print(f"[INFO] relaxed solver       = {relax_res['solver']}")
        print(f"[INFO] relaxed termination = {relax_res['termination']}")
        print(f"[INFO] relaxed accepted    = {relax_res['ok']} ({relax_res['reason']})")
        if relax_res["quality"] is not None:
            qr = relax_res["quality"]
            print(f"[INFO] relaxed max_resid     = {qr['max_resid']:.3e}")
            print(f"[INFO] relaxed max_soc_slack = {qr['max_soc_slack']:.3e}")
        if relax_res["obj"] is not None:
            print(f"[INFO] relaxed objective     = {relax_res['obj']:.10f}")
            dw_relax = evaluate_dw_postsolve(
                relax_model,
                data,
                net,
                raw_obj=relax_res["obj"],
                quality=relax_res["quality"],
            )
            print_dw_postsolve_report(dw_relax, label="relaxed")

        if RUN_FIXED_DISCRETE_FALLBACK:
            print("\n[INFO] Solving fixed-discrete fallback (recommended tap/shunt when available) ...")
            fixed_model = build_pyomo_bfm_misocp_model(data, relax_binaries=False)
            _ = warmstart_from_pf(fixed_model, data, net)
            set_default_discrete_and_fix(fixed_model, data, net=net)

            fixed_res = solve_misocp(fixed_model, tee=False)
            print(f"[INFO] fallback solver       = {fixed_res['solver']}")
            print(f"[INFO] fallback termination  = {fixed_res['termination']}")
            print(f"[INFO] fallback accepted     = {fixed_res['ok']} ({fixed_res['reason']})")

            if fixed_res["quality"] is not None:
                qf = fixed_res["quality"]
                print(f"[INFO] fallback max_resid     = {qf['max_resid']:.3e}")
                print(f"[INFO] fallback max_soc_slack = {qf['max_soc_slack']:.3e}")

            if fixed_res["obj"] is not None:
                dw_fallback = evaluate_dw_postsolve(
                    fixed_model,
                    data,
                    net,
                    raw_obj=fixed_res["obj"],
                    quality=fixed_res["quality"],
                )
                print_dw_postsolve_report(dw_fallback, label="fallback")

            if fixed_res["ok"]:
                print_solution(fixed_model, data, title="IEEE 300-bus Fallback Fixed-Discrete BFM-cr Solution")
                reported_any = True
            else:
                print("[WARN] Fallback solve also failed.")
                reported_any = maybe_print_best_available_solution(
                    fixed_res,
                    fixed_model,
                    data,
                    title="IEEE 300-bus Best-Available Fixed-Discrete BFM-cr Solution (Unaccepted)",
                )

        if not reported_any:
            reported_any = maybe_print_best_available_solution(
                res,
                model,
                data,
                title="IEEE 300-bus Best-Available MISOCP Solution (Unaccepted)",
            )

        if not reported_any:
            reported_any = maybe_print_best_available_solution(
                relax_res,
                relax_model,
                data,
                title="IEEE 300-bus Best-Available Continuous-Relaxation Solution (Unaccepted)",
            )

        if not reported_any:
            print("[WARN] No reportable solution vector was available.")

    t_all1 = time.perf_counter()
    print(f"\n[INFO] Total elapsed time: {t_all1 - t_all0:.2f} seconds")


if __name__ == "__main__":
    main()
