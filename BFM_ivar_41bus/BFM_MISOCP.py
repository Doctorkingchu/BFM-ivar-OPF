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

import math
import time
from dataclasses import dataclass
from typing import Dict, Tuple, List, Any, Optional

import numpy as np
import pyomo.environ as pyo
import pandapower as pp

import ieee41bus as mcase


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

    for _, row in oltc_df.iterrows():
        i = int(row["from_bus"])
        j = int(row["to_bus"])
        oltc_branches[(i, j)] = OLTCBranchConfig(
            tap_min=int(row["tap_min"]),
            tap_max=int(row["tap_max"]),
            dV_percent=float(row["dV_percent"]),
        )

    for _, row in shunt_df.iterrows():
        b = int(row["bus"])
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
def extract_bfm41_fullmesh_data(net, cfg: BuildConfig) -> Dict[str, Any]:
    """
    Extract per-unit data for BFM on the FULL meshed graph:
      - E: directed edges exactly as stored in net.line (from_bus -> to_bus), ALL kept
      - r,x in pu
      - ellmax = (Smax_pu)^2
      - OLTC only if (i,j) exists exactly in E
      - Shunt uses bcap in pu: qsh = a * bcap * (v / v_rated_sq)
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

    # loads (pu)
    Pd = {i: 0.0 for i in buses}
    Qd = {i: 0.0 for i in buses}
    if hasattr(net, "load") and len(net.load.index) > 0:
        for li in net.load.index:
            b = int(net.load.at[li, "bus"])
            Pd[b] += float(net.load.at[li, "p_mw"])
            Qd[b] += float(net.load.at[li, "q_mvar"])
    Pd_pu = {i: Pd[i] / sn for i in buses}
    Qd_pu = {i: Qd[i] / sn for i in buses}

    # generators (ext_grid + gen)
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

    # directed edges E from net.line (ALL)
    E = []
    r = {}
    x = {}
    ellmax = {}
    line_id_of_edge = {}

    for lid in net.line.index:
        lid = int(lid)
        fb = int(net.line.at[lid, "from_bus"])
        tb = int(net.line.at[lid, "to_bus"])
        ij = (fb, tb)

        if ij in r:
            raise ValueError(f"Duplicate directed edge in net.line: {ij}")

        E.append(ij)
        line_id_of_edge[ij] = lid

        vn_kv = float(net.bus.at[fb, "vn_kv"])
        zb = _zbase_ohm(vn_kv, sn)

        r_ohm = float(net.line.at[lid, "r_ohm_per_km"]) * float(net.line.at[lid, "length_km"])
        x_ohm = float(net.line.at[lid, "x_ohm_per_km"]) * float(net.line.at[lid, "length_km"])

        r_pu = r_ohm / zb
        x_pu = x_ohm / zb

        r[ij] = float(r_pu)
        x[ij] = float(x_pu)

        Imax = float(net.line.at[lid, "max_i_ka"]) if "max_i_ka" in net.line.columns else 1e9
        Smax_mva = math.sqrt(3.0) * vn_kv * Imax
        Smax_pu = Smax_mva / sn
        ellmax[ij] = float(Smax_pu * Smax_pu)

    E_set = set(E)

    # OLTC data
    T = []
    K = {}
    delta_tap = {}
    alpha_tap = {}

    for (u, v), tcfg in cfg.oltc_branches.items():
        if (u, v) not in E_set:
            if (v, u) in E_set:
                raise ValueError(
                    f"OLTC edge {(u, v)} exists only as reversed directed edge {(v, u)} in net.line. "
                    f"Fix ordering in network line table or change OLTC direction."
                )
            raise ValueError(f"OLTC edge {(u, v)} not found in net.line directed edges.")

        ij = (u, v)
        T.append(ij)
        taps = list(range(int(tcfg.tap_min), int(tcfg.tap_max) + 1))
        K[ij] = taps
        for tap in taps:
            tau = 1.0 + (tap * float(tcfg.dV_percent)) / 100.0
            alpha_tap[(ij, tap)] = 1.0 / tau
            delta_tap[(ij, tap)] = 1.0 / (tau * tau)

    # shunts
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
        "line_id_of_edge": line_id_of_edge,
    }


# ----------------------------
# Model builder (MISOCP)
# ----------------------------
def build_pyomo_bfm_misocp_model(data: Dict[str, Any], relax_binaries: bool = False) -> pyo.ConcreteModel:
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

    m = pyo.ConcreteModel(name="BFM41_BFMcr_MISOCP")

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
def warmstart_from_pf(model: pyo.ConcreteModel, data: Dict[str, Any], net) -> bool:
    if not RUN_PF_WARMSTART:
        return False

    sn = data["sn_mva"]
    line_id_of_edge = data["line_id_of_edge"]
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
            model.v[int(i)].set_value(vm * vm)

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

        for (i, j) in model.T:
            pick = _default_tap(data["K"][(i, j)])
            for tap in data["K"][(i, j)]:
                model.beta[i, j, tap].set_value(1.0 if tap == pick else 0.0)

        for (i, j) in model.E:
            lid = line_id_of_edge[(int(i), int(j))]
            Ppu = float(net.res_line.at[lid, "p_from_mw"]) / sn
            Qpu = float(net.res_line.at[lid, "q_from_mvar"]) / sn

            model.Pij[i, j].set_value(Ppu)
            model.Qij[i, j].set_value(Qpu)

            vi = float(pyo.value(model.v[i]))
            if (int(i), int(j)) in T_set:
                pick = _default_tap(data["K"][(int(i), int(j))])
                delta0 = float(data["delta_tap"][((int(i), int(j)), pick)])
            else:
                delta0 = 1.0

            ell = (Ppu * Ppu + Qpu * Qpu) / max(delta0 * vi, 1e-8)
            ell = max(0.0, min(ell, float(pyo.value(model.ellmax[i, j]))))
            model.ell[i, j].set_value(ell)

            if (int(i), int(j)) in T_set:
                pick = _default_tap(data["K"][(int(i), int(j))])
                for tap in data["K"][(int(i), int(j))]:
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


def set_default_discrete_and_fix(model: pyo.ConcreteModel, data: Dict[str, Any]):
    for (i, j) in model.T:
        pick = _default_tap(data["K"][(i, j)])
        for tap in data["K"][(i, j)]:
            model.beta[i, j, tap].fix(1 if tap == pick else 0)
    for i in model.C:
        model.a_sh[i].fix(0)


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
            obj = float(pyo.value(model.obj))
        except Exception:
            obj = None

        q = evaluate_quality(model, model._data_for_check)
        ok = (obj is not None) and np.isfinite(obj) and solution_is_acceptable(q)

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


# ----------------------------
# Main
# ----------------------------
def main():
    t_all0 = time.perf_counter()

    # 1) Build modified explicit 41-bus mesh network
    net = mcase.busmeshed39_opf(
        slack_vm_pu=1.0,
        line_max_loading_percent=1e6,
        stress_q_over_p=0.95,
        stress_load_mw_each=300.0,
        r_loss_scale=3.0,
        max_i_ka_base=2.0,
        max_i_ka_stress=1.0,
        line_smax_pu_overrides=None,   # use builder defaults
    )

    # 2) Build config directly from network metadata
    cfg = build_cfg_from_net_metadata(net)

    # 3) Extract data
    data = extract_bfm41_fullmesh_data(net, cfg)

    print(f"[INFO] #buses = {len(data['buses'])}")
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

    # 4) Build MISOCP model
    model = build_pyomo_bfm_misocp_model(data, relax_binaries=False)

    # warmstart
    pf_ok = warmstart_from_pf(model, data, net)
    print(f"[INFO] PF warm start success = {pf_ok}")

    # 5) Solve MISOCP
    print("\n[INFO] Solving BFM-cr MISOCP ...")
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

    if res["ok"]:
        print_solution(model, data, title="BFM-cr MISOCP Solution")
    else:
        print("[WARN] Main MISOCP solution rejected or no solver succeeded.")

        if RUN_FIXED_DISCRETE_FALLBACK:
            print("\n[INFO] Solving fixed-discrete fallback (default tap, shunt off) ...")
            fixed_model = build_pyomo_bfm_misocp_model(data, relax_binaries=False)
            _ = warmstart_from_pf(fixed_model, data, net)
            set_default_discrete_and_fix(fixed_model, data)

            fixed_res = solve_misocp(fixed_model, tee=False)
            print(f"[INFO] fallback solver       = {fixed_res['solver']}")
            print(f"[INFO] fallback termination  = {fixed_res['termination']}")
            print(f"[INFO] fallback accepted     = {fixed_res['ok']} ({fixed_res['reason']})")

            if fixed_res["quality"] is not None:
                qf = fixed_res["quality"]
                print(f"[INFO] fallback max_resid     = {qf['max_resid']:.3e}")
                print(f"[INFO] fallback max_soc_slack = {qf['max_soc_slack']:.3e}")

            if fixed_res["ok"]:
                print_solution(fixed_model, data, title="Fallback Fixed-Discrete BFM-cr Solution")
            else:
                print("[WARN] Fallback solve also failed.")

    t_all1 = time.perf_counter()
    print(f"\n[INFO] Total elapsed time: {t_all1 - t_all0:.2f} seconds")


if __name__ == "__main__":
    main()