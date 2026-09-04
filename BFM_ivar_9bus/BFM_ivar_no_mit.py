# BFM_ivar_no_mit.py
# ------------------------------------------------------------
# Pyomo Branch-Flow OPF with outer iteration on ell (BFM-it)
# + NEW: theta update step by substitution (radial BFS propagation)
#
# Key points (matches your CVXPY bfmit_opf_with_theta.py logic):
#   For t = 1..T:
#     1) Solve MIQCP OPF with ell fixed at ell^{t-1}
#     2) Read solution (Pij^t, Qij^t, v^t, OLTC taps, shunt)
#     3) Build dir_flag from previous P (P_prev) (OLTC forced forward)
#     4) Build physical sending-end flows (Pphys, Qphys) consistent with ell_fix
#     5) Update theta^t using substitution:
#          (theta_s^t - theta_r^t)
#            = (xPphys - rQphys)/sqrt(v_s v_r) - sin(dprev) + dprev
#        where v_s is v(send) for non-OLTC, and u_send = delta*v_hv for OLTC
#     6) Update ell^t with damping:
#          ell_calc = (Pphys^2 + Qphys^2)/v_send_eff
#          ell_new  = (1-damp)*ell_prev + damp*ell_calc
#     7) Stop if sum|ell_new - ell_prev| <= eps and ell bounds satisfied
#
# Notes:
# - Model orientation is a rooted tree away from slack (i->j). Pij,Qij are signed.
# - Direction for theta/ell update is handled by dir_flag + Pphys/Qphys.
# - If solver fails at t=1, check device bounds / big-M / voltage limits / costs / Smax.
#
# Solve:
#   - Try GUROBI first, then SCIP (both via Pyomo interface).
# ------------------------------------------------------------

import math
import time
from dataclasses import dataclass
from typing import Dict, Tuple, List, Any, Optional

import numpy as np
import pyomo.environ as pyo

import ieee9bus as m  # radial: must provide busradial9_opf()


# ============================
# Global settings
# ============================
TEE_SOLVER_LOG = False

# Outer iteration
OUTER_MAX_ITERS = 50
OUTER_EPS = 1e-6
DENOM_EPS = 1e-9

# Damping / stabilization (NEW)
DEFAULT_DAMPING_ELL = 0.6
DEFAULT_THETA_DAMPING = 1.0
DEFAULT_CLIP_DTHETA_PREV = math.pi

# MIP/QCQP solver settings
SCIP_TIME_LIMIT = 600
SCIP_GAP_LIMIT = 1e-4
SCIP_NODE_LIMIT = 200000
SCIP_MEMORY_LIMIT_MB = 8192

GUROBI_TIME_LIMIT = 600
GUROBI_MIPGAP = 1e-4


# ----------------------------
# Configuration containers
# ----------------------------
@dataclass
class OLTCBranchConfig:
    tap_min: int
    tap_max: int
    dV_percent: float  # e.g., 1.25 means 1.25% per tap step


@dataclass
class ShuntConfig:
    q_rated_mvar: float
    v_rated_pu: float  # typically 1.0 (so v_rated_sq = 1.0)


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
    return (float(r.get("cp2_eur_per_mw2", 0.0)),
            float(r.get("cp1_eur_per_mw", 0.0)),
            float(r.get("cp0_eur", 0.0)))


def _orient_radial_edges(buses: List[int], undirected_edges: List[Tuple[int, int]], root: int):
    """
    Orient an undirected tree away from root.
    Returns:
      E_dir : list[(parent, child)]
      parent: dict[node]->parent or None for root
      children: dict[node]->list[child]
    """
    adj = {i: [] for i in buses}
    for u, v in undirected_edges:
        adj[u].append(v)
        adj[v].append(u)

    parent = {root: None}
    children = {i: [] for i in buses}
    q = [root]
    seen = {root}

    while q:
        u = q.pop(0)
        for v in adj[u]:
            if v in seen:
                continue
            seen.add(v)
            parent[v] = u
            children[u].append(v)
            q.append(v)

    if len(seen) != len(buses):
        raise ValueError("Network is not connected (cannot orient as a tree).")

    E_dir = []
    for v in buses:
        if v == root:
            continue
        pu = parent[v]
        if pu is None:
            raise ValueError("Unexpected: non-root node has no parent.")
        E_dir.append((pu, v))

    return E_dir, parent, children


def extract_bfm_per_unit_data(net, cfg: BuildConfig) -> Dict[str, Any]:
    """
    Extract per-unit data for BFM on a radial network.
    - Builds a rooted orientation away from the slack bus.
    """
    sn = float(net.sn_mva)

    # buses
    buses = [int(i) for i in net.bus.index]
    bus_vn_kv = {int(i): float(net.bus.at[i, "vn_kv"]) for i in buses}
    Vmin = {int(i): float(net.bus.at[i, "min_vm_pu"]) for i in buses}
    Vmax = {int(i): float(net.bus.at[i, "max_vm_pu"]) for i in buses}

    # slack (assume first ext_grid)
    if len(net.ext_grid.index) < 1:
        raise ValueError("pandapower net must have an ext_grid (slack).")
    eg0 = int(net.ext_grid.index[0])
    slack_bus = int(net.ext_grid.at[eg0, "bus"])
    slack_vm_pu = float(net.ext_grid.at[eg0, "vm_pu"])

    # loads aggregated per bus (MW/Mvar -> pu)
    Pd = {i: 0.0 for i in buses}
    Qd = {i: 0.0 for i in buses}
    if hasattr(net, "load") and len(net.load.index) > 0:
        for li in net.load.index:
            b = int(net.load.at[li, "bus"])
            Pd[b] += float(net.load.at[li, "p_mw"])
            Qd[b] += float(net.load.at[li, "q_mvar"])
    Pd_pu = {i: Pd[i] / sn for i in buses}
    Qd_pu = {i: Qd[i] / sn for i in buses}

    # generators = ext_grid + gen
    gen_records = []

    # ext_grid as generator
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

    # gen table
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

    # sanity: convex cost
    for rec in gen_records:
        if float(rec["c2"]) < -1e-12:
            raise ValueError("Nonconvex quadratic cost detected (cp2 < 0).")

    # undirected lines + per-unit params
    undirected_edges = []
    r_pu_undir = {}
    x_pu_undir = {}
    Smax_pu_undir = {}
    ellmax_undir = {}

    for e_id in net.line.index:
        e_id = int(e_id)
        fb = int(net.line.at[e_id, "from_bus"])
        tb = int(net.line.at[e_id, "to_bus"])
        key = (min(fb, tb), max(fb, tb))
        undirected_edges.append(key)

        r_ohm = float(net.line.at[e_id, "r_ohm_per_km"]) * float(net.line.at[e_id, "length_km"])
        x_ohm = float(net.line.at[e_id, "x_ohm_per_km"]) * float(net.line.at[e_id, "length_km"])

        zbase = _zbase_ohm(bus_vn_kv[fb], sn)
        r_pu = r_ohm / zbase
        x_pu = x_ohm / zbase

        r_pu_undir[key] = float(r_pu)
        x_pu_undir[key] = float(x_pu)

        Imax = float(net.line.at[e_id, "max_i_ka"]) if "max_i_ka" in net.line.columns else 1e9
        Vkv = bus_vn_kv[fb]
        Smax_mva = math.sqrt(3.0) * Vkv * Imax
        Smax_pu = Smax_mva / sn
        Smax_pu_undir[key] = float(Smax_pu)

        # ellmax in pu: I_pu^2. With Vnom=1, I_pu ~= Smax_pu
        ellmax_undir[key] = float(Smax_pu * Smax_pu)

    # orient edges as rooted tree
    E_dir, parent, children = _orient_radial_edges(buses, undirected_edges, root=slack_bus)

    # directed params
    r = {}
    x = {}
    Smax = {}
    ellmax = {}
    for (i, j) in E_dir:
        key = (min(i, j), max(i, j))
        r[(i, j)] = r_pu_undir[key]
        x[(i, j)] = x_pu_undir[key]
        Smax[(i, j)] = Smax_pu_undir[key]
        ellmax[(i, j)] = ellmax_undir[key]

    # OLTC sets from cfg (map to oriented arc)
    T = []
    K = {}
    delta_tap = {}  # 1/tau^2
    alpha_tap = {}  # 1/tau

    E_set = set(E_dir)
    for (u, v), tcfg in cfg.oltc_branches.items():
        if (u, v) in E_set:
            ij = (u, v)
        elif (v, u) in E_set:
            ij = (v, u)
        else:
            raise ValueError(f"OLTC branch {(u, v)} not found in oriented radial edges.")
        T.append(ij)

        taps = list(range(int(tcfg.tap_min), int(tcfg.tap_max) + 1))
        K[ij] = taps
        for tap in taps:
            tau = 1.0 + (tap * float(tcfg.dV_percent)) / 100.0
            alpha_tap[(ij, tap)] = 1.0 / tau
            delta_tap[(ij, tap)] = 1.0 / (tau * tau)

    # shunt set C and params
    C = sorted([int(i) for i in cfg.shunts.keys()])
    q_rated_pu = {}
    v_rated_sq = {}
    Mq = {}

    for i in C:
        scfg = cfg.shunts[i]
        qpu = float(scfg.q_rated_mvar) / sn
        vr = float(scfg.v_rated_pu)
        vrsq = vr * vr
        q_rated_pu[i] = qpu
        v_rated_sq[i] = vrsq

        vmax_sq = (Vmax[i] ** 2)
        Mq[i] = abs(qpu) * (vmax_sq / vrsq) + 1e-6

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
        "E": E_dir,
        "parent": parent,
        "children": children,
        "r": r,
        "x": x,
        "Smax": Smax,
        "ellmax": ellmax,
        "T": T,
        "K": K,
        "alpha_tap": alpha_tap,
        "delta_tap": delta_tap,
        "C": C,
        "q_rated_pu": q_rated_pu,
        "v_rated_sq": v_rated_sq,
        "Mq": Mq,
        "fix_slack_vm": cfg.fix_slack_vm,
    }


# ----------------------------
# Pyomo model (ell fixed, MIQCP)
# ----------------------------
def build_pyomo_bfmit_model(
    data: Dict[str, Any],
    ell_fix: Dict[Tuple[int, int], float],
    relax_binaries: bool = False,
    warm_start: Optional[Dict[str, Any]] = None,
) -> pyo.ConcreteModel:

    sn = data["sn_mva"]
    buses = data["buses"]
    slack_bus = data["slack_bus"]
    slack_vm_pu = data["slack_vm_pu"]

    E = data["E"]
    r = data["r"]
    x = data["x"]
    Smax = data["Smax"]

    T = data["T"]
    K = data["K"]
    delta_tap = data["delta_tap"]
    alpha_tap = data["alpha_tap"]

    C = data["C"]
    q_rated_pu = data["q_rated_pu"]
    v_rated_sq = data["v_rated_sq"]
    Mq = data["Mq"]

    Pd = data["Pd_pu"]
    Qd = data["Qd_pu"]
    Vmin = data["Vmin"]
    Vmax = data["Vmax"]

    gen_records = data["gen_records"]
    Gset = list(range(len(gen_records)))

    # adjacency for nodal balance
    out_arcs = {i: [] for i in buses}
    in_arcs = {i: [] for i in buses}
    for (i, j) in E:
        out_arcs[i].append((i, j))
        in_arcs[j].append((i, j))

    T_set = set(T)
    C_set = set(C)

    model = pyo.ConcreteModel(name="BFMit_MIQCP_OLTC_SHUNT_THETA")

    # Sets
    model.N = pyo.Set(initialize=buses, ordered=True)
    model.G = pyo.Set(initialize=Gset, ordered=True)
    model.E = pyo.Set(initialize=E, dimen=2, ordered=True)
    model.T = pyo.Set(initialize=T, dimen=2, ordered=True)
    model.C = pyo.Set(initialize=C, ordered=True)

    # Params
    model.Pd = pyo.Param(model.N, initialize=lambda m, i: float(Pd[i]), mutable=False)
    model.Qd = pyo.Param(model.N, initialize=lambda m, i: float(Qd[i]), mutable=False)

    model.Vmin = pyo.Param(model.N, initialize=lambda m, i: float(Vmin[i]), mutable=False)
    model.Vmax = pyo.Param(model.N, initialize=lambda m, i: float(Vmax[i]), mutable=False)

    model.r = pyo.Param(model.E, initialize=lambda m, i, j: float(r[(i, j)]), mutable=False)
    model.x = pyo.Param(model.E, initialize=lambda m, i, j: float(x[(i, j)]), mutable=False)
    model.Smax = pyo.Param(model.E, initialize=lambda m, i, j: float(Smax[(i, j)]), mutable=False)

    # Fixed ell (Param)
    model.ell_fix = pyo.Param(
        model.E,
        initialize=lambda m, i, j: float(ell_fix[(i, j)]),
        mutable=False
    )

    # Generator params
    model.gen_bus = pyo.Param(model.G, initialize=lambda m, gg: int(gen_records[int(gg)]["bus"]), within=pyo.Any)
    model.Pgmin = pyo.Param(model.G, initialize=lambda m, gg: float(gen_records[int(gg)]["pmin_pu"]))
    model.Pgmax = pyo.Param(model.G, initialize=lambda m, gg: float(gen_records[int(gg)]["pmax_pu"]))
    model.Qgmin = pyo.Param(model.G, initialize=lambda m, gg: float(gen_records[int(gg)]["qmin_pu"]))
    model.Qgmax = pyo.Param(model.G, initialize=lambda m, gg: float(gen_records[int(gg)]["qmax_pu"]))
    model.c2 = pyo.Param(model.G, initialize=lambda m, gg: float(gen_records[int(gg)]["c2"]))
    model.c1 = pyo.Param(model.G, initialize=lambda m, gg: float(gen_records[int(gg)]["c1"]))
    model.c0 = pyo.Param(model.G, initialize=lambda m, gg: float(gen_records[int(gg)]["c0"]))

    # Shunt params
    model.qrated = pyo.Param(model.C, initialize=lambda m, i: float(q_rated_pu[int(i)]), mutable=False)
    model.vrated = pyo.Param(model.C, initialize=lambda m, i: float(v_rated_sq[int(i)]), mutable=False)
    model.Mq = pyo.Param(model.C, initialize=lambda m, i: float(Mq[int(i)]), mutable=False)

    # OLTC index set (i,j,tap)
    beta_index = []
    for (i, j) in T:
        for tap in K[(i, j)]:
            beta_index.append((i, j, int(tap)))
    model.BETA_INDEX = pyo.Set(initialize=beta_index, dimen=3, ordered=True)

    model.delta_tap = pyo.Param(
        model.BETA_INDEX,
        initialize=lambda m, i, j, tap: float(delta_tap[((i, j), int(tap))]),
        mutable=False
    )
    model.alpha_tap = pyo.Param(
        model.BETA_INDEX,
        initialize=lambda m, i, j, tap: float(alpha_tap[((i, j), int(tap))]),
        mutable=False
    )

    # Variables
    model.Pg = pyo.Var(model.G, bounds=lambda m, gg: (m.Pgmin[gg], m.Pgmax[gg]))
    model.Qg = pyo.Var(model.G, bounds=lambda m, gg: (m.Qgmin[gg], m.Qgmax[gg]))

    # squared voltage v in [Vmin^2, Vmax^2]
    model.v = pyo.Var(model.N, bounds=lambda m, i: (m.Vmin[i] ** 2, m.Vmax[i] ** 2))

    # net injections
    model.Pinj = pyo.Var(model.N)
    model.Qinj = pyo.Var(model.N)

    # branch variables (signed allowed)
    model.Pij = pyo.Var(model.E)
    model.Qij = pyo.Var(model.E)

    # shunt vars (defined for all buses; forced to 0 if not in C)
    model.qsh = pyo.Var(model.N)

    # binaries
    if relax_binaries:
        model.beta = pyo.Var(model.BETA_INDEX, bounds=(0.0, 1.0))
        model.a_sh = pyo.Var(model.C, bounds=(0.0, 1.0))
    else:
        model.beta = pyo.Var(model.BETA_INDEX, within=pyo.Binary)
        model.a_sh = pyo.Var(model.C, within=pyo.Binary)

    # McCormick helper vars: tv = beta * v[i]
    model.tv = pyo.Var(model.BETA_INDEX)

    # ----------------------------
    # Constraints
    # ----------------------------
    # Slack voltage fix (squared)
    if data["fix_slack_vm"]:
        model.slack_v = pyo.Constraint(expr=model.v[slack_bus] == float(slack_vm_pu) ** 2)

    # Net injections
    def Pinj_rule(m, i):
        return m.Pinj[i] == sum(m.Pg[gg] for gg in m.G if int(m.gen_bus[gg]) == int(i)) - m.Pd[i]
    model.Pinj_def = pyo.Constraint(model.N, rule=Pinj_rule)

    def Qinj_rule(m, i):
        return m.Qinj[i] == sum(m.Qg[gg] for gg in m.G if int(m.gen_bus[gg]) == int(i)) - m.Qd[i] + m.qsh[i]
    model.Qinj_def = pyo.Constraint(model.N, rule=Qinj_rule)

    # OLTC one-hot
    def onehot_rule(m, i, j):
        taps = K[(i, j)]
        return sum(m.beta[i, j, int(t)] for t in taps) == 1
    model.onehot = pyo.Constraint(model.T, rule=onehot_rule)

    # McCormick for tv = beta * v[i]
    def tv_ub1(m, i, j, tap):
        vU = (m.Vmax[int(i)] ** 2)
        return m.tv[i, j, tap] <= vU * m.beta[i, j, tap]
    model.tv_ub1 = pyo.Constraint(model.BETA_INDEX, rule=tv_ub1)

    def tv_lb1(m, i, j, tap):
        vL = (m.Vmin[int(i)] ** 2)
        return m.tv[i, j, tap] >= vL * m.beta[i, j, tap]
    model.tv_lb1 = pyo.Constraint(model.BETA_INDEX, rule=tv_lb1)

    def tv_ub2(m, i, j, tap):
        vL = (m.Vmin[int(i)] ** 2)
        return m.tv[i, j, tap] <= m.v[int(i)] - vL * (1.0 - m.beta[i, j, tap])
    model.tv_ub2 = pyo.Constraint(model.BETA_INDEX, rule=tv_ub2)

    def tv_lb2(m, i, j, tap):
        vU = (m.Vmax[int(i)] ** 2)
        return m.tv[i, j, tap] >= m.v[int(i)] - vU * (1.0 - m.beta[i, j, tap])
    model.tv_lb2 = pyo.Constraint(model.BETA_INDEX, rule=tv_lb2)

    # Expression: v_send(i,j) = v_i for non-OLTC, or sum(delta_tap * tv) for OLTC
    def vsend_rule(m, i, j):
        if (i, j) not in T_set:
            return m.v[int(i)]
        taps = K[(i, j)]
        return sum(m.delta_tap[i, j, int(t)] * m.tv[i, j, int(t)] for t in taps)
    model.vsend = pyo.Expression(model.E, rule=vsend_rule)

    # Switched shunt big-M
    def qsh_zero_rule(m, i):
        if int(i) in C_set:
            return pyo.Constraint.Skip
        return m.qsh[i] == 0.0
    model.qsh_zero = pyo.Constraint(model.N, rule=qsh_zero_rule)

    def qsh_bound_pos(m, i):
        return m.qsh[int(i)] <= m.Mq[i] * m.a_sh[i]
    model.qsh_bound_pos = pyo.Constraint(model.C, rule=qsh_bound_pos)

    def qsh_bound_neg(m, i):
        return m.qsh[int(i)] >= -m.Mq[i] * m.a_sh[i]
    model.qsh_bound_neg = pyo.Constraint(model.C, rule=qsh_bound_neg)

    def qsh_match_pos_rule(m, i):
        q_target = m.qrated[i] * (m.v[int(i)] / m.vrated[i])
        return m.qsh[int(i)] - q_target <= m.Mq[i] * (1.0 - m.a_sh[i])
    model.qsh_match_pos = pyo.Constraint(model.C, rule=qsh_match_pos_rule)

    def qsh_match_neg_rule(m, i):
        q_target = m.qrated[i] * (m.v[int(i)] / m.vrated[i])
        return q_target - m.qsh[int(i)] <= m.Mq[i] * (1.0 - m.a_sh[i])
    model.qsh_match_neg = pyo.Constraint(model.C, rule=qsh_match_neg_rule)

    # BFM nodal balance with fixed ell
    def bfm_P_balance_rule(m, i):
        out_sum = sum(m.Pij[a, b] for (a, b) in out_arcs[int(i)])
        in_sum = sum((m.Pij[a, b] - m.r[a, b] * m.ell_fix[a, b]) for (a, b) in in_arcs[int(i)])
        return out_sum - in_sum == m.Pinj[i]
    model.BFM_P = pyo.Constraint(model.N, rule=bfm_P_balance_rule)

    def bfm_Q_balance_rule(m, i):
        out_sum = sum(m.Qij[a, b] for (a, b) in out_arcs[int(i)])
        in_sum = sum((m.Qij[a, b] - m.x[a, b] * m.ell_fix[a, b]) for (a, b) in in_arcs[int(i)])
        return out_sum - in_sum == m.Qinj[i]
    model.BFM_Q = pyo.Constraint(model.N, rule=bfm_Q_balance_rule)

    # Voltage drop
    def vdrop_rule(m, i, j):
        rij = m.r[i, j]
        xij = m.x[i, j]
        z2 = rij * rij + xij * xij
        return m.v[int(j)] == m.vsend[i, j] - 2.0 * (rij * m.Pij[i, j] + xij * m.Qij[i, j]) + z2 * m.ell_fix[i, j]
    model.Vdrop = pyo.Constraint(model.E, rule=vdrop_rule)

    # Branch thermal limit (convex QC)
    def thermal_rule(m, i, j):
        return m.Pij[i, j] ** 2 + m.Qij[i, j] ** 2 <= (m.Smax[i, j] ** 2)
    model.Thermal = pyo.Constraint(model.E, rule=thermal_rule)

    # Objective: Pg in pu -> MW
    model.obj = pyo.Objective(
        rule=lambda m: sum(
            m.c2[gg] * (sn * m.Pg[gg]) ** 2 + m.c1[gg] * (sn * m.Pg[gg]) + m.c0[gg]
            for gg in m.G
        ),
        sense=pyo.minimize
    )

    # Warm start
    if warm_start is not None:
        if "v" in warm_start:
            for i in buses:
                if i in warm_start["v"]:
                    model.v[i].value = float(warm_start["v"][i])
        if "Pij" in warm_start:
            for (i, j) in E:
                if (i, j) in warm_start["Pij"]:
                    model.Pij[i, j].value = float(warm_start["Pij"][(i, j)])
        if "Qij" in warm_start:
            for (i, j) in E:
                if (i, j) in warm_start["Qij"]:
                    model.Qij[i, j].value = float(warm_start["Qij"][(i, j)])
        if "Pg" in warm_start:
            for gg in Gset:
                if gg in warm_start["Pg"]:
                    model.Pg[gg].value = float(warm_start["Pg"][gg])
        if "Qg" in warm_start:
            for gg in Gset:
                if gg in warm_start["Qg"]:
                    model.Qg[gg].value = float(warm_start["Qg"][gg])
        if "beta" in warm_start:
            for (i, j, tap) in model.BETA_INDEX:
                if (i, j, tap) in warm_start["beta"]:
                    model.beta[i, j, tap].value = float(warm_start["beta"][(i, j, tap)])
        if "a_sh" in warm_start:
            for i in C:
                if i in warm_start["a_sh"]:
                    model.a_sh[i].value = float(warm_start["a_sh"][i])

    return model


# ----------------------------
# Solver
# ----------------------------
def solve_miqcp(model: pyo.ConcreteModel, tee: bool = False) -> Dict[str, Any]:
    """
    Try GUROBI first, then SCIP.
    Returns dict with ok/status/termination_condition.
    """
    # 1) GUROBI
    solver = pyo.SolverFactory("gurobi")
    if solver is not None and solver.available(exception_flag=False):
        try:
            solver.options["TimeLimit"] = float(GUROBI_TIME_LIMIT)
            solver.options["MIPGap"] = float(GUROBI_MIPGAP)
            # keep NonConvex safe; shouldn't hurt if convex
            solver.options["NonConvex"] = 2
        except Exception:
            pass

        res = solver.solve(model, tee=tee)
        tc = res.solver.termination_condition
        st = res.solver.status
        ok = tc in [
            pyo.TerminationCondition.optimal,
            pyo.TerminationCondition.locallyOptimal,
            pyo.TerminationCondition.feasible,
            pyo.TerminationCondition.maxTimeLimit
        ]
        return {"ok": ok, "solver": "gurobi", "termination": str(tc), "status": str(st)}

    # 2) SCIP (AMPL .nl interface — required for MIQCP on Pyomo+SCIP)
    solver = pyo.SolverFactory("scip", solver_io="nl")
    if solver is not None and solver.available(exception_flag=False):
        try:
            solver.options["limits/time"] = float(SCIP_TIME_LIMIT)
            solver.options["limits/gap"] = float(SCIP_GAP_LIMIT)
            solver.options["limits/nodes"] = int(SCIP_NODE_LIMIT)
            solver.options["limits/memory"] = float(SCIP_MEMORY_LIMIT_MB)
            solver.options["display/verblevel"] = 4 if tee else 0
        except Exception:
            pass

        res = solver.solve(model, tee=tee)
        tc = res.solver.termination_condition
        st = res.solver.status
        ok = tc in [
            pyo.TerminationCondition.optimal,
            pyo.TerminationCondition.locallyOptimal,
            pyo.TerminationCondition.feasible,
            pyo.TerminationCondition.maxTimeLimit
        ]
        return {"ok": ok, "solver": "scip", "termination": str(tc), "status": str(st)}

    raise RuntimeError("No MIQCP-capable solver found. Install/enable GUROBI or SCIP (Pyomo interface).")


# ----------------------------
# NEW: theta update by substitution (radial BFS)
# ----------------------------
def update_theta_by_substitution_pyomo(
    data: Dict[str, Any],
    v_sol: Dict[int, float],                 # bus -> v(pu^2)
    Pphys: Dict[Tuple[int, int], float],     # edge -> Pphys (pu)
    Qphys: Dict[Tuple[int, int], float],     # edge -> Qphys (pu)
    theta_prev: Dict[int, float],            # bus -> theta(rad)
    send_bus: Dict[Tuple[int, int], int],    # edge -> sending bus id
    recv_bus: Dict[Tuple[int, int], int],    # edge -> receiving bus id
    u_send_edge: Dict[Tuple[int, int], float],  # OLTC: u_send=delta*v_hv, else nan
    eps_v: float = 1e-9,
    clip_dtheta_prev: float = math.pi,
    theta_damping: float = 1.0,
) -> Dict[int, float]:
    """
    Edge equation:
      theta_s^t - theta_r^t
        = (xPphys - rQphys)/sqrt(v_s v_r) - sin(dprev) + dprev
    Radial => recover theta by BFS from slack with theta[slack]=0.
    """
    buses = [int(b) for b in data["buses"]]
    E = data["E"]
    T_set = set(data["T"])
    slack = int(data["slack_bus"])

    # 1) compute b_e per edge
    b_e = {}
    for (i, j) in E:
        s = int(send_bus[(i, j)])
        rcv = int(recv_bus[(i, j)])

        if (i, j) in T_set:
            vs = float(u_send_edge.get((i, j), float("nan")))
            if not np.isfinite(vs):
                vs = float(v_sol[s])
        else:
            vs = float(v_sol[s])
        vr = float(v_sol[rcv])

        denom = math.sqrt(max(vs, eps_v) * max(vr, eps_v))

        dprev = float(theta_prev.get(s, 0.0) - theta_prev.get(rcv, 0.0))
        if clip_dtheta_prev is not None and clip_dtheta_prev > 0:
            dprev = float(np.clip(dprev, -clip_dtheta_prev, clip_dtheta_prev))

        rij = float(data["r"][(i, j)])
        xij = float(data["x"][(i, j)])
        rhs = (xij * float(Pphys[(i, j)]) - rij * float(Qphys[(i, j)])) / denom
        b_e[(i, j)] = rhs - math.sin(dprev) + dprev

    # 2) incidence adjacency (bus -> edges)
    adj = {b: [] for b in buses}
    for (i, j) in E:
        s = int(send_bus[(i, j)])
        rcv = int(recv_bus[(i, j)])
        adj[s].append((i, j))
        adj[rcv].append((i, j))

    # 3) BFS
    theta_new = {b: float("nan") for b in buses}
    theta_new[slack] = 0.0
    q = [slack]

    while q:
        u = q.pop(0)
        for ekey in adj[u]:
            s = int(send_bus[ekey])
            rcv = int(recv_bus[ekey])

            if u == s:
                v = rcv
                if not np.isfinite(theta_new[v]):
                    theta_new[v] = theta_new[u] - b_e[ekey]
                    q.append(v)
            elif u == rcv:
                v = s
                if not np.isfinite(theta_new[v]):
                    theta_new[v] = theta_new[u] + b_e[ekey]
                    q.append(v)

    # fallback
    for b in buses:
        if not np.isfinite(theta_new[b]):
            theta_new[b] = 0.0

    # damping
    theta_out = {}
    for b in buses:
        thp = float(theta_prev.get(b, 0.0))
        thn = float(theta_new[b])
        theta_out[b] = (1.0 - float(theta_damping)) * thp + float(theta_damping) * thn
    theta_out[slack] = 0.0
    return theta_out


# ----------------------------
# Outer iteration (BFM-it) + theta update
# ----------------------------
def run_bfmit_outer_iteration(
    data: Dict[str, Any],
    max_iters: int = OUTER_MAX_ITERS,
    eps: float = OUTER_EPS,
    tee: bool = False,
    damping_ell: float = DEFAULT_DAMPING_ELL,
    theta_damping: float = DEFAULT_THETA_DAMPING,
    clip_dtheta_prev: float = DEFAULT_CLIP_DTHETA_PREV,
) -> Dict[str, Any]:

    buses = [int(b) for b in data["buses"]]
    E = data["E"]
    T_set = set(data["T"])
    ellmax = data["ellmax"]

    # 1) init
    ell_prev = {(i, j): 0.0 for (i, j) in E}
    P_prev = {(i, j): 0.0 for (i, j) in E}  # NEW: direction decision uses previous P
    theta_prev = {b: 0.0 for b in buses}     # NEW
    theta_prev[int(data["slack_bus"])] = 0.0

    warm = {
        "v": {i: 1.0 for i in buses},
        "Pij": {(i, j): 0.0 for (i, j) in E},
        "Qij": {(i, j): 0.0 for (i, j) in E},
        "Pg": {},
        "Qg": {},
        "beta": {},
        "a_sh": {},
    }

    best = {"iter": 0, "obj": float("inf"), "model": None, "ell": None, "theta": None}

    for t in range(1, max_iters + 1):
        t_iter0 = time.perf_counter()

        # 2.1) solve OPF with ell fixed at ell^{t-1}
        model = build_pyomo_bfmit_model(data, ell_fix=ell_prev, relax_binaries=False, warm_start=warm)

        solinfo = solve_miqcp(model, tee=tee)
        if not solinfo["ok"]:
            print(f"[t={t:02d}] MIQCP not solved. solver={solinfo['solver']} "
                  f"termination={solinfo['termination']} status={solinfo['status']} STOP.")
            break

        obj = pyo.value(model.obj)
        if obj < best["obj"]:
            best["obj"] = obj
            best["iter"] = t
            best["model"] = model

        # ----------------------------
        # 2.2) read solution
        # ----------------------------
        v_sol = {int(i): float(pyo.value(model.v[i])) for i in buses}
        P_sol = {(i, j): float(pyo.value(model.Pij[i, j])) for (i, j) in E}
        Q_sol = {(i, j): float(pyo.value(model.Qij[i, j])) for (i, j) in E}

        # OLTC u_send (= delta*v_hv)
        u_send_edge = {}
        for (i, j) in E:
            if (i, j) in T_set:
                u_send_edge[(i, j)] = float(pyo.value(model.vsend[i, j]))
            else:
                u_send_edge[(i, j)] = float("nan")

        # update warm-start dict from this solution
        warm["v"] = {i: float(pyo.value(model.v[i])) for i in buses}
        warm["Pij"] = {(i, j): float(pyo.value(model.Pij[i, j])) for (i, j) in E}
        warm["Qij"] = {(i, j): float(pyo.value(model.Qij[i, j])) for (i, j) in E}
        warm["Pg"] = {int(gg): float(pyo.value(model.Pg[gg])) for gg in model.G}
        warm["Qg"] = {int(gg): float(pyo.value(model.Qg[gg])) for gg in model.G}
        warm["a_sh"] = {int(i): float(pyo.value(model.a_sh[i])) for i in model.C}
        warm["beta"] = {(i, j, tap): float(pyo.value(model.beta[i, j, tap])) for (i, j, tap) in model.BETA_INDEX}

        # ----------------------------
        # NEW: dir_flag + send/recv (OLTC forced forward)
        # ----------------------------
        dir_flag = {}
        send_bus = {}
        recv_bus = {}
        for (i, j) in E:
            if (i, j) in T_set:
                dir_flag[(i, j)] = 1
            else:
                dir_flag[(i, j)] = 1 if P_prev[(i, j)] >= 0.0 else -1

            if dir_flag[(i, j)] == 1:
                send_bus[(i, j)] = int(i)
                recv_bus[(i, j)] = int(j)
            else:
                send_bus[(i, j)] = int(j)
                recv_bus[(i, j)] = int(i)

        # ----------------------------
        # NEW: physical sending-end flows (consistent with ell_fix=ell_prev)
        # ----------------------------
        Pphys = {}
        Qphys = {}
        for (i, j) in E:
            rij = float(data["r"][(i, j)])
            xij = float(data["x"][(i, j)])
            if dir_flag[(i, j)] == 1:
                Pphys[(i, j)] = P_sol[(i, j)]
                Qphys[(i, j)] = Q_sol[(i, j)]
            else:
                Pphys[(i, j)] = -P_sol[(i, j)] + rij * float(ell_prev[(i, j)])
                Qphys[(i, j)] = -Q_sol[(i, j)] + xij * float(ell_prev[(i, j)])

        # ----------------------------
        # NEW: theta update (BFS propagation)
        # ----------------------------
        theta_new = update_theta_by_substitution_pyomo(
            data=data,
            v_sol=v_sol,
            Pphys=Pphys,
            Qphys=Qphys,
            theta_prev=theta_prev,
            send_bus=send_bus,
            recv_bus=recv_bus,
            u_send_edge=u_send_edge,
            eps_v=1e-9,
            clip_dtheta_prev=clip_dtheta_prev,
            theta_damping=theta_damping,
        )

        # ----------------------------
        # NEW: ell update (damped) using effective sending v
        # ----------------------------
        ell_new = {}
        sum_diff = 0.0
        max_viol = 0.0

        for (i, j) in E:
            s = int(send_bus[(i, j)])

            if (i, j) in T_set:
                v_send_eff = float(u_send_edge[(i, j)])
                if not np.isfinite(v_send_eff):
                    v_send_eff = float(v_sol[s])
            else:
                v_send_eff = float(v_sol[s])

            denom = max(v_send_eff, DENOM_EPS)
            ell_calc = (Pphys[(i, j)] ** 2 + Qphys[(i, j)] ** 2) / denom

            ell_damped = (1.0 - float(damping_ell)) * float(ell_prev[(i, j)]) + float(damping_ell) * ell_calc
            ell_new[(i, j)] = ell_damped

            sum_diff += abs(ell_damped - ell_prev[(i, j)])

            viol = 0.0
            if ell_damped < -1e-8:
                viol = -ell_damped
            if ell_damped > ellmax[(i, j)] + 1e-8:
                viol = ell_damped - ellmax[(i, j)]
            max_viol = max(max_viol, viol)

        t_iter1 = time.perf_counter()
        print(f"[t={t:02d}] obj={obj:.6f}  sum|dell|={sum_diff:.3e}  "
              f"max_ell_viol={max_viol:.3e}  iter_time={t_iter1-t_iter0:.2f}s")

        # 2.4) termination
        if (sum_diff <= eps) and (max_viol <= 1e-8):
            best["ell"] = ell_new
            best["theta"] = theta_new
            best["model"] = model
            best["iter"] = t
            best["obj"] = obj
            print(f"[CONVERGED] at t={t} with eps={eps}")
            return best

        # update states
        ell_prev = ell_new
        P_prev = P_sol
        theta_prev = theta_new

    # not converged
    best["ell"] = ell_prev
    best["theta"] = theta_prev
    return best


# ----------------------------
# Output
# ----------------------------
def _print_solution(model: pyo.ConcreteModel, data: Dict[str, Any],
                    ell: Dict[Tuple[int, int], float],
                    theta: Optional[Dict[int, float]] = None):
    sn = data["sn_mva"]
    buses = data["buses"]
    gen_records = data["gen_records"]
    slack_bus = data["slack_bus"]
    T_set = set(data["T"])
    K = data["K"]

    print("\n--- Bus Voltages ---")
    for i in buses:
        v = pyo.value(model.v[i])
        V = math.sqrt(max(v, 0.0))
        tag = "  [slack]" if i == slack_bus else ""
        print(f"Bus {i}: v={v:.6f}, V={V:.6f}{tag}")

    if theta is not None:
        print("\n--- Bus Angles (theta, rad / deg) ---")
        for i in buses:
            th = float(theta.get(int(i), 0.0))
            print(f"Bus {i}: theta={th:+.6f} rad  ({math.degrees(th):+.6f} deg)")

    print("\n--- Generator Dispatch (MW / Mvar) ---")
    for gg in model.G:
        rec = gen_records[int(gg)]
        Pg_mw = sn * pyo.value(model.Pg[gg])
        Qg_mvar = sn * pyo.value(model.Qg[gg])
        print(f"{rec['type']}[{rec['id']}] @ bus {rec['bus']}: P={Pg_mw:.4f} MW, Q={Qg_mvar:.4f} Mvar")

    if len(list(model.T)) > 0:
        print("\n--- OLTC taps (beta one-hot) ---")
        for (i, j) in model.T:
            best_tap = None
            best_val = -1.0
            for tap in K[(i, j)]:
                vbeta = float(pyo.value(model.beta[i, j, int(tap)]))
                if vbeta > best_val:
                    best_val = vbeta
                    best_tap = int(tap)
            alpha = float(pyo.value(model.alpha_tap[i, j, best_tap]))
            delta = float(pyo.value(model.delta_tap[i, j, best_tap]))
            tau = 1.0 / alpha
            print(f"OLTC ({i}->{j}): tap={best_tap} (beta={best_val:.6f})  tau={tau:.6f}  alpha={alpha:.6f}  delta={delta:.6f}")

    if len(list(model.C)) > 0:
        print("\n--- Switched shunt status ---")
        for i in model.C:
            a = float(pyo.value(model.a_sh[i]))
            q = float(pyo.value(model.qsh[int(i)]))
            print(f"Shunt @ bus {int(i)}: a_sh={a:.6f}, qsh(pu)={q:.6f}  => {sn*q:.4f} Mvar")

    print("\n--- Branch flows & ell (next) ---")
    for (i, j) in model.E:
        P = float(pyo.value(model.Pij[i, j]))
        Q = float(pyo.value(model.Qij[i, j]))
        Smag = math.sqrt(max(P * P + Q * Q, 0.0))
        denom = float(pyo.value(model.vsend[i, j])) if (i, j) in T_set else float(pyo.value(model.v[i]))
        denom = max(denom, DENOM_EPS)
        ell_calc = (P * P + Q * Q) / denom
        print(f"({i}->{j}): P={P:+.6f} pu, Q={Q:+.6f} pu, |S|={Smag:.6f} pu, "
              f"ell_calc(naive)={ell_calc:.6f}, ell_used_next={ell[(i,j)]:.6f}")


# ----------------------------
# Main
# ----------------------------
def main():
    t0 = time.perf_counter()

    # Build pandapower net
    net = m.busradial9_opf(slack_vm_pu=1.0, line_max_loading_percent=1e6)

    # OLTC configs
    oltc_branches: Dict[Tuple[int, int], OLTCBranchConfig] = {
        (3, 4): OLTCBranchConfig(tap_min=-8, tap_max=8, dV_percent=1.25),
        (5, 6): OLTCBranchConfig(tap_min=-6, tap_max=6, dV_percent=1.25),
        (7, 1): OLTCBranchConfig(tap_min=-4, tap_max=4, dV_percent=1.25),
    }

    # Shunt configs
    shunts: Dict[int, ShuntConfig] = {
        1: ShuntConfig(q_rated_mvar=10.0, v_rated_pu=1.0),
        4: ShuntConfig(q_rated_mvar=15.0, v_rated_pu=1.0),
        5: ShuntConfig(q_rated_mvar=20.0, v_rated_pu=1.0),
        6: ShuntConfig(q_rated_mvar=10.0, v_rated_pu=1.0),
        8: ShuntConfig(q_rated_mvar=8.0,  v_rated_pu=1.0),
    }

    cfg = BuildConfig(oltc_branches=oltc_branches, shunts=shunts, fix_slack_vm=True)
    data = extract_bfm_per_unit_data(net, cfg)

    print(f"[INFO] #buses={len(data['buses'])}, #edges={len(data['E'])}, #OLTC={len(data['T'])}, #shunt={len(data['C'])}")
    print(f"[INFO] Outer iteration: max_iters={OUTER_MAX_ITERS}, eps={OUTER_EPS}")
    print(f"[INFO] Damping: ell={DEFAULT_DAMPING_ELL}, theta={DEFAULT_THETA_DAMPING}, clip_dtheta_prev={DEFAULT_CLIP_DTHETA_PREV}")

    sol = run_bfmit_outer_iteration(
        data=data,
        max_iters=OUTER_MAX_ITERS,
        eps=OUTER_EPS,
        tee=TEE_SOLVER_LOG,
        damping_ell=DEFAULT_DAMPING_ELL,
        theta_damping=DEFAULT_THETA_DAMPING,
        clip_dtheta_prev=DEFAULT_CLIP_DTHETA_PREV,
    )

    best_model = sol["model"]
    if best_model is None:
        print("[FAIL] No solution produced.")
        return

    print("\n[SOLVED] Best/Last iterate summary")
    print(f"  iter  = {sol['iter']}")
    print(f"  obj   = {sol['obj']:.6f}")
    _print_solution(best_model, data, sol["ell"], theta=sol.get("theta", None))

    t1 = time.perf_counter()
    print(f"\n[INFO] Total elapsed time: {t1 - t0:.2f} seconds.")


if __name__ == "__main__":
    main()
