# BFM_MISOCP.py
# ------------------------------------------------------------
# Pyomo Branch-Flow AC-OPF / MISOCP (radial) with OLTC + switched shunt
#
# Changes vs your MINLP:
#  1) Replace nonconvex equality:
#        P^2 + Q^2 = (deltaE * v_i) * ell
#     with SOC relaxation:
#        P^2 + Q^2 <= w_ij * ell
#     where w_ij = deltaE * v_i is handled exactly via one-hot (big-M).
#
#  2) Replace voltage drop term deltaE*v_i with w_ij to avoid bilinearities:
#        v_j = w_ij - 2(rP + xQ) + (r^2 + x^2) ell
#
# Solvers:
#   - Prefer: gurobi (MIQCP/MISOCP), cplex, mosek (in that order)
# ------------------------------------------------------------

import math
import time
from dataclasses import dataclass
from typing import Dict, Tuple, List, Any, Optional

import numpy as np
import pyomo.environ as pyo

import ieee9bus as m  # radial: must provide busradial9_opf()


# ============================
# Solver settings (MISOCP/MIQCP)
# ============================
SOLVER_TIME_LIMIT = 3600
SOLVER_MIP_GAP = 0.01
TEE_SOLVER_LOG = False


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
    - Line charging bc kept at 0.0 (consistent with your net setup).
    - Precomputes bounds and big-M for w_ij = (deltaE)*v_i.
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

    # ext_grid as a generator
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

    # undirected lines + per-unit params
    undirected_edges = []
    r_pu_undir = {}
    x_pu_undir = {}
    bc_pu_undir = {}
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

        zbase = _zbase_ohm(bus_vn_kv[fb], sn)  # your net: same kv
        r_pu = r_ohm / zbase
        x_pu = x_ohm / zbase

        r_pu_undir[key] = float(r_pu)
        x_pu_undir[key] = float(x_pu)
        bc_pu_undir[key] = 0.0

        Imax = float(net.line.at[e_id, "max_i_ka"]) if "max_i_ka" in net.line.columns else 1e9
        Vkv = bus_vn_kv[fb]
        Smax_mva = math.sqrt(3.0) * Vkv * Imax
        Smax_pu = Smax_mva / sn
        Smax_pu_undir[key] = float(Smax_pu)

        ellmax_undir[key] = float(Smax_pu * Smax_pu)  # Ipu^2 ~ S^2 at 1pu

    # orient edges as a rooted tree
    E_dir, parent, children = _orient_radial_edges(buses, undirected_edges, root=slack_bus)

    # directed params
    r, x, bc, Smax, ellmax = {}, {}, {}, {}, {}
    for (i, j) in E_dir:
        key = (min(i, j), max(i, j))
        r[(i, j)] = r_pu_undir[key]
        x[(i, j)] = x_pu_undir[key]
        bc[(i, j)] = bc_pu_undir[key]
        Smax[(i, j)] = Smax_pu_undir[key]
        ellmax[(i, j)] = ellmax_undir[key]

    # OLTC sets from cfg (map to oriented arc)
    T = []
    K = {}
    alpha_tap = {}
    delta_tap = {}

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
            alpha_tap[((ij[0], ij[1]), tap)] = 1.0 / tau
            delta_tap[((ij[0], ij[1]), tap)] = 1.0 / (tau * tau)

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
        Mq[i] = qpu * (Vmax[i] * Vmax[i]) / vrsq + 1e-6

    # Precompute bounds for w_ij = (deltaE)*v_i and big-M for OLTC arcs
    T_set = set(T)

    w_lb = {}
    w_ub = {}
    Mw_w = {}  # only for OLTC arcs in T

    for (i, j) in E_dir:
        if (i, j) in T_set:
            taps = K[(i, j)]
            deltas = [delta_tap[((i, j), int(t))] for t in taps]
            dmin = min(deltas)
            dmax = max(deltas)
            wL = dmin * (Vmin[i] ** 2)
            wU = dmax * (Vmax[i] ** 2)
            w_lb[(i, j)] = float(wL)
            w_ub[(i, j)] = float(wU)
            Mw_w[(i, j)] = float((wU - wL) + 1e-6)
        else:
            w_lb[(i, j)] = float(Vmin[i] ** 2)
            w_ub[(i, j)] = float(Vmax[i] ** 2)

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
        "bc": bc,
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
        "w_lb": w_lb,
        "w_ub": w_ub,
        "Mw_w": Mw_w,
        "fix_slack_vm": cfg.fix_slack_vm,
    }


# ----------------------------
# Pyomo model builder (BFM MISOCP)
# ----------------------------
def build_pyomo_bfm_misocp_model(data: Dict[str, Any]) -> pyo.ConcreteModel:
    sn = data["sn_mva"]
    buses = data["buses"]
    slack_bus = data["slack_bus"]
    slack_vm_pu = data["slack_vm_pu"]

    E = data["E"]
    r = data["r"]
    x = data["x"]
    Smax = data["Smax"]
    ellmax = data["ellmax"]

    T = data["T"]
    K = data["K"]
    delta_tap_map = data["delta_tap"]

    C = data["C"]
    q_rated_pu = data["q_rated_pu"]
    v_rated_sq = data["v_rated_sq"]
    Mq = data["Mq"]

    Pd = data["Pd_pu"]
    Qd = data["Qd_pu"]
    Vmin = data["Vmin"]
    Vmax = data["Vmax"]

    w_lb = data["w_lb"]
    w_ub = data["w_ub"]
    Mw_w = data["Mw_w"]

    gen_records = data["gen_records"]
    Gset = list(range(len(gen_records)))

    # adjacency lists for nodal balance
    out_arcs = {i: [] for i in buses}
    in_arcs = {i: [] for i in buses}
    for (i, j) in E:
        out_arcs[i].append((i, j))
        in_arcs[j].append((i, j))

    model = pyo.ConcreteModel(name="BFM_MISOCP_OLTC_SHUNT")

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
    model.ellmax = pyo.Param(model.E, initialize=lambda m, i, j: float(ellmax[(i, j)]), mutable=False)

    # w bounds on each directed edge
    model.w_lb = pyo.Param(model.E, initialize=lambda m, i, j: float(w_lb[(i, j)]), mutable=False)
    model.w_ub = pyo.Param(model.E, initialize=lambda m, i, j: float(w_ub[(i, j)]), mutable=False)

    # Generator params
    model.gen_bus = pyo.Param(model.G, initialize=lambda m, gg: int(gen_records[int(gg)]["bus"]), within=pyo.Any)
    model.Pgmin = pyo.Param(model.G, initialize=lambda m, gg: float(gen_records[int(gg)]["pmin_pu"]))
    model.Pgmax = pyo.Param(model.G, initialize=lambda m, gg: float(gen_records[int(gg)]["pmax_pu"]))
    model.Qgmin = pyo.Param(model.G, initialize=lambda m, gg: float(gen_records[int(gg)]["qmin_pu"]))
    model.Qgmax = pyo.Param(model.G, initialize=lambda m, gg: float(gen_records[int(gg)]["qmax_pu"]))
    model.c2 = pyo.Param(model.G, initialize=lambda m, gg: float(gen_records[int(gg)]["c2"]))
    model.c1 = pyo.Param(model.G, initialize=lambda m, gg: float(gen_records[int(gg)]["c1"]))
    model.c0 = pyo.Param(model.G, initialize=lambda m, gg: float(gen_records[int(gg)]["c0"]))

    # Shunt params (only for i in C)
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
        initialize=lambda m, i, j, tap: float(delta_tap_map[((i, j), int(tap))]),
        mutable=False
    )

    # Big-M for w selection (only for OLTC arcs)
    if len(T) > 0:
        model.Mw = pyo.Param(model.T, initialize=lambda m, i, j: float(Mw_w[(i, j)]), mutable=False)

    # Variables
    model.Pg = pyo.Var(model.G, bounds=lambda m, gg: (m.Pgmin[gg], m.Pgmax[gg]))
    model.Qg = pyo.Var(model.G, bounds=lambda m, gg: (m.Qgmin[gg], m.Qgmax[gg]))

    # squared voltage v in [Vmin^2, Vmax^2]
    model.v = pyo.Var(model.N, bounds=lambda m, i: (m.Vmin[i] ** 2, m.Vmax[i] ** 2))

    # net injections
    model.Pinj = pyo.Var(model.N)
    model.Qinj = pyo.Var(model.N)

    # branch variables
    model.Pij = pyo.Var(model.E)
    model.Qij = pyo.Var(model.E)
    model.ell = pyo.Var(model.E, bounds=lambda m, i, j: (0.0, m.ellmax[i, j]))

    # w_ij = (deltaE)*v_i  (deltaE=1 for non-OLTC, delta(tap) for OLTC)
    model.w = pyo.Var(model.E, bounds=lambda m, i, j: (m.w_lb[i, j], m.w_ub[i, j]))

    # shunt vars: capacitor -> nonnegative
    model.qsh = pyo.Var(model.N)

    # binaries
    model.beta = pyo.Var(model.BETA_INDEX, within=pyo.Binary) if len(beta_index) > 0 else pyo.Var()
    model.a_sh = pyo.Var(model.C, within=pyo.Binary) if len(C) > 0 else pyo.Var()

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

    # Switched shunt big-M (capacitor: qsh >= 0, and qsh=0 if OFF)
    C_set = set(C)

    def qsh_zero_rule(m, i):
        if int(i) in C_set:
            return pyo.Constraint.Skip
        return m.qsh[i] == 0.0
    model.qsh_zero = pyo.Constraint(model.N, rule=qsh_zero_rule)

    def qsh_upper_rule(m, i):
        return m.qsh[i] <= m.Mq[i] * m.a_sh[i]
    model.qsh_upper = pyo.Constraint(model.C, rule=qsh_upper_rule)

    def qsh_lower_rule(m, i):
        return m.qsh[i] >= 0.0
    model.qsh_lower = pyo.Constraint(model.C, rule=qsh_lower_rule)

    def qsh_match_pos_rule(m, i):
        q_target = m.qrated[i] * (m.v[int(i)] / m.vrated[i])
        return m.qsh[int(i)] - q_target <= m.Mq[i] * (1.0 - m.a_sh[i])
    model.qsh_match_pos = pyo.Constraint(model.C, rule=qsh_match_pos_rule)

    def qsh_match_neg_rule(m, i):
        q_target = m.qrated[i] * (m.v[int(i)] / m.vrated[i])
        return q_target - m.qsh[int(i)] <= m.Mq[i] * (1.0 - m.a_sh[i])
    model.qsh_match_neg = pyo.Constraint(model.C, rule=qsh_match_neg_rule)

    # OLTC: one-hot selection
    def onehot_rule(m, i, j):
        taps = K[(i, j)]
        return sum(m.beta[i, j, int(t)] for t in taps) == 1
    model.onehot = pyo.Constraint(model.T, rule=onehot_rule) if len(T) > 0 else pyo.ConstraintList()

    # w constraints:
    #  - non-OLTC arcs: w_ij = v_i (linear)
    #  - OLTC arcs: w_ij = delta_tap(i,j,t)*v_i for the chosen tap t (exact via big-M)
    T_set = set(T)

    def w_nonoltc_rule(m, i, j):
        if (i, j) in T_set:
            return pyo.Constraint.Skip
        return m.w[i, j] == m.v[i]
    model.w_nonoltc = pyo.Constraint(model.E, rule=w_nonoltc_rule)

    def w_oltc_pos_rule(m, i, j, tap):
        # w - delta(tap)*v <= Mw*(1-beta)
        return m.w[i, j] - m.delta_tap[i, j, tap] * m.v[i] <= m.Mw[i, j] * (1.0 - m.beta[i, j, tap])
    def w_oltc_neg_rule(m, i, j, tap):
        # w - delta(tap)*v >= -Mw*(1-beta)
        return m.w[i, j] - m.delta_tap[i, j, tap] * m.v[i] >= -m.Mw[i, j] * (1.0 - m.beta[i, j, tap])

    if len(beta_index) > 0:
        model.w_oltc_pos = pyo.Constraint(model.BETA_INDEX, rule=w_oltc_pos_rule)
        model.w_oltc_neg = pyo.Constraint(model.BETA_INDEX, rule=w_oltc_neg_rule)
    else:
        model.w_oltc_pos = pyo.ConstraintList()
        model.w_oltc_neg = pyo.ConstraintList()

    # Branch-flow nodal balance (same sign convention)
    def bfm_P_balance_rule(m, i):
        out_sum = sum(m.Pij[a, b] for (a, b) in out_arcs[int(i)])
        in_sum = sum((m.Pij[a, b] - m.r[a, b] * m.ell[a, b]) for (a, b) in in_arcs[int(i)])
        return out_sum - in_sum == m.Pinj[i]
    model.BFM_P = pyo.Constraint(model.N, rule=bfm_P_balance_rule)

    def bfm_Q_balance_rule(m, i):
        out_sum = sum(m.Qij[a, b] for (a, b) in out_arcs[int(i)])
        in_sum = sum((m.Qij[a, b] - m.x[a, b] * m.ell[a, b]) for (a, b) in in_arcs[int(i)])
        return out_sum - in_sum == m.Qinj[i]
    model.BFM_Q = pyo.Constraint(model.N, rule=bfm_Q_balance_rule)

    # Voltage drop (now linear in variables, because delta*v replaced by w)
    def vdrop_rule(m, i, j):
        rij = m.r[i, j]
        xij = m.x[i, j]
        return m.v[j] == m.w[i, j] - 2.0 * (rij * m.Pij[i, j] + xij * m.Qij[i, j]) + (rij * rij + xij * xij) * m.ell[i, j]
    model.Vdrop = pyo.Constraint(model.E, rule=vdrop_rule)

    # SOC relaxation of current-power relation:
    # Rotated SOC: P^2+Q^2 <= w*ell, with w>=0, ell>=0
    # Use standard SOC form:
    #   || [ 2P, 2Q, w-ell ] ||_2 <= w + ell
    def soc_rule(m, i, j):
        P = m.Pij[i, j]
        Q = m.Qij[i, j]
        wv = m.w[i, j]
        ellv = m.ell[i, j]
        # squared SOC (no sqrt) : sum_squares <= (affine)^2
        return (2.0 * P) ** 2 + (2.0 * Q) ** 2 + (wv - ellv) ** 2 <= (wv + ellv) ** 2
    model.SOC_Irel = pyo.Constraint(model.E, rule=soc_rule)

    # Thermal limit
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

    return model


# ----------------------------
# Initialization (helps MIP a lot)
# ----------------------------
def initialize_bfm_misocp(model: pyo.ConcreteModel, data: Dict[str, Any]):
    """
    Warm start:
    - flat v=1
    - tap closest to 0, shunt OFF
    - approximate Pg/Qg
    - flows from subtree aggregation (ignoring losses)
    - set w = v or delta*v according to chosen tap
    - ell from SOC equality approx: ell ≈ (P^2+Q^2)/max(w,eps)
    """
    buses = data["buses"]
    slack = data["slack_bus"]
    children = data["children"]
    Pd = data["Pd_pu"]
    Qd = data["Qd_pu"]
    T_set = set(data["T"])

    # v flat
    for i in buses:
        model.v[i].value = 1.0
    if data["fix_slack_vm"]:
        model.v[slack].value = float(data["slack_vm_pu"]) ** 2

    # binaries init: tap closest to 0, shunt OFF
    if hasattr(model, "BETA_INDEX") and len(list(model.BETA_INDEX)) > 0:
        for (i, j, tap) in model.BETA_INDEX:
            model.beta[i, j, tap].value = 0.0
        for (i, j) in model.T:
            taps = sorted([tap for (ii, jj, tap) in model.BETA_INDEX if (ii == i and jj == j)])
            pick = min(taps, key=lambda t: abs(t))
            model.beta[i, j, pick].value = 1.0

    if hasattr(model, "C") and len(list(model.C)) > 0:
        for i in model.C:
            model.a_sh[i].value = 0.0
    for i in model.N:
        model.qsh[i].value = 0.0

    # Pg/Qg init: non-slack gens to mid, slack balances
    total_loadP = sum(Pd[i] for i in buses)
    total_loadQ = sum(Qd[i] for i in buses)

    slack_gen_indices = []
    for gg in model.G:
        bus = int(model.gen_bus[gg])
        pmin, pmax = float(model.Pgmin[gg]), float(model.Pgmax[gg])
        qmin, qmax = float(model.Qgmin[gg]), float(model.Qgmax[gg])
        pmid = 0.5 * (pmin + pmax)
        qmid = 0.0 if (qmin <= 0.0 <= qmax) else 0.5 * (qmin + qmax)

        # treat ext_grid at slack as balancing gen
        if bus == slack:
            slack_gen_indices.append(int(gg))
        else:
            model.Pg[gg].value = pmid
            model.Qg[gg].value = qmid

    if slack_gen_indices:
        otherP = sum(model.Pg[gg].value for gg in model.G if int(gg) not in slack_gen_indices)
        otherQ = sum(model.Qg[gg].value for gg in model.G if int(gg) not in slack_gen_indices)
        slackP = total_loadP - otherP
        slackQ = total_loadQ - otherQ
        for gg in slack_gen_indices:
            model.Pg[gg].value = min(max(slackP, float(model.Pgmin[gg])), float(model.Pgmax[gg]))
            model.Qg[gg].value = min(max(slackQ, float(model.Qgmin[gg])), float(model.Qgmax[gg]))

    # compute net injections initial (qsh=0)
    Pinj0 = {i: -Pd[i] for i in buses}
    Qinj0 = {i: -Qd[i] for i in buses}
    for gg in model.G:
        b = int(model.gen_bus[gg])
        Pinj0[b] += float(model.Pg[gg].value)
        Qinj0[b] += float(model.Qg[gg].value)

    # demand = -Pinj (positive if net consumption)
    demandP = {i: -Pinj0[i] for i in buses}
    demandQ = {i: -Qinj0[i] for i in buses}

    # postorder traversal
    order = []
    stack = [slack]
    while stack:
        u = stack.pop()
        order.append(u)
        for v in children[u]:
            stack.append(v)
    order.reverse()

    subP = {i: demandP[i] for i in buses}
    subQ = {i: demandQ[i] for i in buses}
    for u in order:
        for v in children[u]:
            subP[u] += subP[v]
            subQ[u] += subQ[v]

    # set branch flows and w, ell
    for (i, j) in model.E:
        model.Pij[i, j].value = subP[j]
        model.Qij[i, j].value = subQ[j]

        # set w = v_i (non-OLTC) or delta(tap)*v_i (OLTC)
        vi = float(model.v[i].value)
        if (i, j) in T_set:
            # find chosen tap
            chosen = None
            for (ii, jj, tap) in model.BETA_INDEX:
                if ii == i and jj == j and abs(float(model.beta[ii, jj, tap].value) - 1.0) < 1e-9:
                    chosen = tap
                    break
            if chosen is None:
                # fallback: pick first
                chosen = sorted([tap for (ii, jj, tap) in model.BETA_INDEX if (ii == i and jj == j)])[0]
            delta = float(pyo.value(model.delta_tap[i, j, chosen]))
            model.w[i, j].value = delta * vi
        else:
            model.w[i, j].value = vi

        wv = float(model.w[i, j].value)
        P = float(model.Pij[i, j].value)
        Q = float(model.Qij[i, j].value)
        model.ell[i, j].value = max((P * P + Q * Q) / max(wv, 1e-6), 0.0)


# ----------------------------
# Solver utilities (MISOCP/MIQCP)
# ----------------------------
def solve_misocp(model: pyo.ConcreteModel,
                time_limit_sec: int = SOLVER_TIME_LIMIT,
                mip_gap: float = SOLVER_MIP_GAP,
                tee: bool = False) -> bool:
    """
    Try a MISOCP-capable solver through Pyomo.
    Prefer gurobi, then cplex, then mosek.
    """
    candidates = ["gurobi", "cplex", "mosek"]

    last_err = None
    for name in candidates:
        solver = pyo.SolverFactory(name)
        if solver is None or not solver.available(exception_flag=False):
            continue
        try:
            # generic common options (solver-specific keys may differ; we try safely)
            if name == "gurobi":
                solver.options["TimeLimit"] = float(time_limit_sec)
                solver.options["MIPGap"] = float(mip_gap)
                solver.options["OutputFlag"] = 1 if tee else 0
                # If your environment mistakenly flags SOC-QC as nonconvex, you can uncomment:
                # solver.options["NonConvex"] = 2
            elif name == "cplex":
                solver.options["timelimit"] = float(time_limit_sec)
                solver.options["mip_tolerances_mipgap"] = float(mip_gap)
            elif name == "mosek":
                # MOSEK option names can vary by interface; keep minimal
                solver.options["MSK_DPAR_MIO_TOL_REL_GAP"] = float(mip_gap)
                solver.options["MSK_DPAR_OPTIMIZER_MAX_TIME"] = float(time_limit_sec)

            res = solver.solve(model, tee=tee)
            tc = res.solver.termination_condition
            if tc in [
                pyo.TerminationCondition.optimal,
                pyo.TerminationCondition.locallyOptimal,
                pyo.TerminationCondition.feasible,
                pyo.TerminationCondition.maxTimeLimit,
            ]:
                return True
            last_err = f"{name}: termination_condition={tc}"
        except Exception as e:
            last_err = f"{name}: {repr(e)}"
            continue

    if last_err is not None:
        print("[FAIL] No solver succeeded. Last error:", last_err)
    else:
        print("[FAIL] No MISOCP solver available. Install/enable gurobi/cplex/mosek.")
    return False


# ----------------------------
# Output
# ----------------------------
def _print_solution(model: pyo.ConcreteModel, data: Dict[str, Any]):
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

    print("\n--- Generator Dispatch (MW / Mvar) ---")
    for gg in model.G:
        rec = gen_records[int(gg)]
        Pg_mw = sn * pyo.value(model.Pg[gg])
        Qg_mvar = sn * pyo.value(model.Qg[gg])
        print(f"{rec['type']}[{rec['id']}] @ bus {rec['bus']}: P={Pg_mw:.4f} MW, Q={Qg_mvar:.4f} Mvar")

    if len(list(model.T)) > 0:
        print("\n--- OLTC taps (beta one-hot) ---")
        for (i, j) in model.T:
            best = None
            for (ii, jj, tap) in model.BETA_INDEX:
                if ii == i and jj == j:
                    bv = pyo.value(model.beta[ii, jj, tap])
                    if best is None or bv > best[1]:
                        best = (tap, bv)
            if best is not None:
                tap = best[0]
                delta = pyo.value(model.delta_tap[i, j, tap])
                print(f"OLTC ({i}->{j}): tap={tap} (beta={best[1]:.6f}), delta={delta:.6f}, "
                      f"w={pyo.value(model.w[i,j]):.6f}, v_send={pyo.value(model.v[i]):.6f}")

    if len(list(model.C)) > 0:
        print("\n--- Switched shunt status ---")
        for i in model.C:
            print(f"Shunt @ bus {int(i)}: a_sh={pyo.value(model.a_sh[i]):.6f}, "
                  f"qsh(pu)={pyo.value(model.qsh[int(i)]):.6f}  => {sn*pyo.value(model.qsh[int(i)]):.4f} Mvar")

    print("\n--- Branch flows check ---")
    for (i, j) in model.E:
        P = pyo.value(model.Pij[i, j])
        Q = pyo.value(model.Qij[i, j])
        ell = pyo.value(model.ell[i, j])
        wv = pyo.value(model.w[i, j])
        Smag = math.sqrt(max(P * P + Q * Q, 0.0))
        print(f"({i}->{j}): P={P:+.6f} pu, Q={Q:+.6f} pu, |S|={Smag:.6f} pu <= {pyo.value(model.Smax[i,j]):.6f}, "
              f"ell={ell:.6f}, w={wv:.6f}, "
              f"(P^2+Q^2)={P*P+Q*Q:.6f} <= w*ell={wv*ell:.6f}")


# ----------------------------
# Main
# ----------------------------
def main():
    t0 = time.perf_counter()
    try:
        # Build pandapower net (same as your BIM code)
        net = m.busradial9_opf(slack_vm_pu=1.0, line_max_loading_percent=1e6)

        # SAME discrete devices as your BIM MINLP code
        oltc_branches: Dict[Tuple[int, int], OLTCBranchConfig] = {
            (3, 4): OLTCBranchConfig(tap_min=-8, tap_max=8, dV_percent=1.25),
            (5, 6): OLTCBranchConfig(tap_min=-6, tap_max=6, dV_percent=1.25),
            (7, 1): OLTCBranchConfig(tap_min=-4, tap_max=4, dV_percent=1.25),
        }

        shunts: Dict[int, ShuntConfig] = {
            1: ShuntConfig(q_rated_mvar=10.0, v_rated_pu=1.0),
            4: ShuntConfig(q_rated_mvar=15.0, v_rated_pu=1.0),
            5: ShuntConfig(q_rated_mvar=20.0, v_rated_pu=1.0),
            6: ShuntConfig(q_rated_mvar=10.0, v_rated_pu=1.0),
            8: ShuntConfig(q_rated_mvar=8.0,  v_rated_pu=1.0),
        }

        cfg = BuildConfig(oltc_branches=oltc_branches, shunts=shunts, fix_slack_vm=True)
        data = extract_bfm_per_unit_data(net, cfg)

        # Build BFM-MISOCP model
        model = build_pyomo_bfm_misocp_model(data)
        initialize_bfm_misocp(model, data)

        print(f"[INFO] #OLTC branches={len(data['T'])}, #SwitchedShunts={len(data['C'])}, #BetaIndex={len(list(model.BETA_INDEX))}")

        print("[INFO] Solving as MISOCP/MIQCP (SOC-relaxed BFM).")
        ok = solve_misocp(model, time_limit_sec=SOLVER_TIME_LIMIT, mip_gap=SOLVER_MIP_GAP, tee=TEE_SOLVER_LOG)
        if not ok:
            print("[FAIL] Could not obtain a feasible solution with available MISOCP solvers.")
            return

        print("\n[SOLVED] by MISOCP-capable solver (SOC-relaxed BFM).")
        print("Objective (EUR):", pyo.value(model.obj))
        _print_solution(model, data)

    finally:
        t1 = time.perf_counter()
        print(f"\n[INFO] Total elapsed time: {t1 - t0:.2f} seconds.")


if __name__ == "__main__":
    main()
