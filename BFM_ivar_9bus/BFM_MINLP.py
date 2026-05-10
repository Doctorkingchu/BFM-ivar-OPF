# BFM_MINLP.py
# ------------------------------------------------------------
# Pyomo Branch-Flow AC-OPF / MINLP (radial) with OLTC + switched shunt
# - Nonconvex equality kept: P^2 + Q^2 = v * ell (or delta*v*ell for OLTC)
# - Same discrete device locations/specs as your BIM MINLP code
#
# Solve order:
#   1) Try SCIP (nonconvex MIQCQP/MINLP)
#   2) Fallback: NLP-based Branch-and-Bound heuristic (IPOPT per node)
# ------------------------------------------------------------

import math
import time
from dataclasses import dataclass
from typing import Dict, Tuple, List, Any, Optional

import numpy as np
import pyomo.environ as pyo

import ieee9bus as m  # radial: must provide busradial9_opf()


# ============================
# Solver settings
# ============================
IPOPT_TOL = 1e-6
IPOPT_MAX_ITER = 2000
IPOPT_MAX_CPU_TIME = 60  # sec per NLP

IPOPT_ACCEPTABLE_TOL = 1e-4
IPOPT_ACCEPTABLE_ITER = 10
IPOPT_ACCEPTABLE_CONSTR_VIOL = 1e-4

SCIP_TIME_LIMIT = 3600
SCIP_GAP_LIMIT = 0.01
SCIP_MEMORY_LIMIT_MB = 4096
SCIP_NODE_LIMIT = 30000

HEURISTIC_BB_MAX_NODES = 40
HEURISTIC_NODE_IPOPT_TIME = 20
HEURISTIC_FRAC_TOL = 1e-6

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
    - Line charging bc is read but your net sets it to 0.0 (consistent with your BFM equations).
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
    line_key_by_eid = {}
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
        line_key_by_eid[e_id] = key

        r_ohm = float(net.line.at[e_id, "r_ohm_per_km"]) * float(net.line.at[e_id, "length_km"])
        x_ohm = float(net.line.at[e_id, "x_ohm_per_km"]) * float(net.line.at[e_id, "length_km"])

        zbase = _zbase_ohm(bus_vn_kv[fb], sn)  # all buses same kv in your net
        r_pu = r_ohm / zbase
        x_pu = x_ohm / zbase

        r_pu_undir[key] = float(r_pu)
        x_pu_undir[key] = float(x_pu)
        bc_pu_undir[key] = 0.0  # your net sets c_nf_per_km=0; keep explicit

        Imax = float(net.line.at[e_id, "max_i_ka"]) if "max_i_ka" in net.line.columns else 1e9
        Vkv = bus_vn_kv[fb]
        Smax_mva = math.sqrt(3.0) * Vkv * Imax
        Smax_pu = Smax_mva / sn
        Smax_pu_undir[key] = float(Smax_pu)

        # current pu = Smax_pu at V=1pu; so ellmax = Imax_pu^2 = Smax_pu^2
        ellmax_undir[key] = float(Smax_pu * Smax_pu)

    # orient edges as a rooted tree
    E_dir, parent, children = _orient_radial_edges(buses, undirected_edges, root=slack_bus)

    # directed params
    r = {}
    x = {}
    bc = {}
    Smax = {}
    ellmax = {}
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
        # allow user to specify undirected pair; map to oriented if exists either way
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
        # tight big-M: max q when ON occurs at v = Vmax^2
        Mq[i] = qpu * (Vmax[i] * Vmax[i]) / vrsq + 1e-6

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
        "fix_slack_vm": cfg.fix_slack_vm,
    }


# ----------------------------
# Pyomo model builder (BFM)
# ----------------------------
def build_pyomo_bfm_model(data: Dict[str, Any], relax_binaries: bool = False) -> pyo.ConcreteModel:
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
    alpha_tap = data["alpha_tap"]
    delta_tap = data["delta_tap"]

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

    # adjacency lists for nodal balance
    out_arcs = {i: [] for i in buses}
    in_arcs = {i: [] for i in buses}
    for (i, j) in E:
        out_arcs[i].append((i, j))
        in_arcs[j].append((i, j))

    model = pyo.ConcreteModel(name="BFM_MINLP_OLTC_SHUNT")

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

    model.alpha_tap = pyo.Param(
        model.BETA_INDEX,
        initialize=lambda m, i, j, tap: float(alpha_tap[((i, j), int(tap))]),
        mutable=False
    )
    model.delta_tap = pyo.Param(
        model.BETA_INDEX,
        initialize=lambda m, i, j, tap: float(delta_tap[((i, j), int(tap))]),
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

    # branch variables
    model.Pij = pyo.Var(model.E)
    model.Qij = pyo.Var(model.E)
    model.ell = pyo.Var(model.E, bounds=lambda m, i, j: (0.0, m.ellmax[i, j]))

    # shunt vars: capacitor -> nonnegative (tighten)
    # for i not in C we will fix to 0 by constraint
    model.qsh = pyo.Var(model.N)

    if relax_binaries:
        model.beta = pyo.Var(model.BETA_INDEX, bounds=(0.0, 1.0))
        model.a_sh = pyo.Var(model.C, bounds=(0.0, 1.0))
    else:
        model.beta = pyo.Var(model.BETA_INDEX, within=pyo.Binary)
        model.a_sh = pyo.Var(model.C, within=pyo.Binary)

    model.alpha = pyo.Var(model.T)
    model.delta = pyo.Var(model.T)

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

    # OLTC: one-hot + alpha/delta selection
    def onehot_rule(m, i, j):
        taps = K[(i, j)]
        return sum(m.beta[i, j, int(t)] for t in taps) == 1
    model.onehot = pyo.Constraint(model.T, rule=onehot_rule)

    def alpha_sel_rule(m, i, j):
        taps = K[(i, j)]
        return m.alpha[i, j] == sum(m.alpha_tap[i, j, int(t)] * m.beta[i, j, int(t)] for t in taps)
    model.alpha_sel = pyo.Constraint(model.T, rule=alpha_sel_rule)

    def delta_sel_rule(m, i, j):
        taps = K[(i, j)]
        return m.delta[i, j] == sum(m.delta_tap[i, j, int(t)] * m.beta[i, j, int(t)] for t in taps)
    model.delta_sel = pyo.Constraint(model.T, rule=delta_sel_rule)

    T_set = set(T)

    # deltaE(i,j) = delta if OLTC else 1
    model.deltaE = pyo.Expression(model.E, rule=lambda m, i, j: m.delta[i, j] if (i, j) in T_set else 1.0)

    # Switched shunt big-M (capacitor: qsh >= 0, and qsh=0 if OFF)
    # for buses not in C: force qsh=0
    C_set = set(C)

    def qsh_zero_rule(m, i):
        if int(i) in C_set:
            return pyo.Constraint.Skip
        return m.qsh[i] == 0.0
    model.qsh_zero = pyo.Constraint(model.N, rule=qsh_zero_rule)

    # for shunt buses: 0 <= qsh <= Mq * a_sh, and |qsh - q_target| <= Mq*(1-a_sh)
    def qsh_upper_rule(m, i):
        return m.qsh[i] <= m.Mq[i] * m.a_sh[i]
    model.qsh_upper = pyo.Constraint(model.C, rule=qsh_upper_rule)

    def qsh_lower_rule(m, i):
        return m.qsh[i] >= 0.0
    model.qsh_lower = pyo.Constraint(model.C, rule=qsh_lower_rule)

    def qsh_match_pos_rule(m, i):
        # qsh - q_target <= M(1-a)
        q_target = m.qrated[i] * (m.v[int(i)] / m.vrated[i])
        return m.qsh[int(i)] - q_target <= m.Mq[i] * (1.0 - m.a_sh[i])
    model.qsh_match_pos = pyo.Constraint(model.C, rule=qsh_match_pos_rule)

    def qsh_match_neg_rule(m, i):
        # q_target - qsh <= M(1-a)
        q_target = m.qrated[i] * (m.v[int(i)] / m.vrated[i])
        return q_target - m.qsh[int(i)] <= m.Mq[i] * (1.0 - m.a_sh[i])
    model.qsh_match_neg = pyo.Constraint(model.C, rule=qsh_match_neg_rule)

    # Branch-flow nodal balance (your exact sign convention)
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

    # Voltage drop (non-OLTC: deltaE=1; OLTC: deltaE=delta)
    def vdrop_rule(m, i, j):
        rij = m.r[i, j]
        xij = m.x[i, j]
        return m.v[j] == m.deltaE[i, j] * m.v[i] - 2.0 * (rij * m.Pij[i, j] + xij * m.Qij[i, j]) + (rij * rij + xij * xij) * m.ell[i, j]
    model.Vdrop = pyo.Constraint(model.E, rule=vdrop_rule)

    # Current-power relation (nonconvex equality)
    def current_power_rule(m, i, j):
        return m.Pij[i, j] ** 2 + m.Qij[i, j] ** 2 == m.deltaE[i, j] * m.v[i] * m.ell[i, j]
    model.Irel = pyo.Constraint(model.E, rule=current_power_rule)

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
# Initialization (helps IPOPT a lot)
# ----------------------------
def initialize_bfm(model: pyo.ConcreteModel, data: Dict[str, Any]):
    """
    Provide a reasonable warm start:
    - flat v=1
    - Pg: use pandapower initial p_mw for gens, slack set to balance approx
    - flows: subtree aggregation ignoring losses; ell from P^2+Q^2 ≈ v*ell
    - taps: 0 (closest), shunt OFF
    """
    sn = data["sn_mva"]
    buses = data["buses"]
    slack = data["slack_bus"]
    children = data["children"]
    Pd = data["Pd_pu"]
    Qd = data["Qd_pu"]
    gen_records = data["gen_records"]
    E = data["E"]

    # v flat
    for i in buses:
        model.v[i].value = 1.0
    if data["fix_slack_vm"]:
        model.v[slack].value = float(data["slack_vm_pu"]) ** 2

    # binaries initial
    # beta: pick tap closest to 0 (if exists)
    for (i, j) in model.T:
        taps = sorted([t for (_, _, t) in model.BETA_INDEX if _ == i and _ == j])  # not used
    # safer explicit loop:
    for (i, j, tap) in model.BETA_INDEX:
        model.beta[i, j, tap].value = 0.0
    for (i, j) in model.T:
        taps = sorted([tap for (ii, jj, tap) in model.BETA_INDEX if (ii == i and jj == j)])
        pick = min(taps, key=lambda t: abs(t))
        model.beta[i, j, pick].value = 1.0

    for i in model.C:
        model.a_sh[i].value = 0.0
    for i in model.N:
        model.qsh[i].value = 0.0

    # Pg/Qg init: set non-slack gens to mid of bounds, slack to satisfy approximate P balance
    # (we don't have p0 in gen_records; they contain bounds + costs only. So do mid.)
    total_loadP = sum(Pd[i] for i in buses)
    total_loadQ = sum(Qd[i] for i in buses)

    Pg_init = []
    Qg_init = []
    slack_gen_indices = []
    for gg in model.G:
        bus = int(model.gen_bus[gg])
        pmin, pmax = float(model.Pgmin[gg]), float(model.Pgmax[gg])
        qmin, qmax = float(model.Qgmin[gg]), float(model.Qgmax[gg])
        pmid = 0.5 * (pmin + pmax)
        qmid = 0.0 if (qmin <= 0.0 <= qmax) else 0.5 * (qmin + qmax)

        if bus == slack and gen_records[int(gg)]["type"] == "ext_grid":
            slack_gen_indices.append(int(gg))
            Pg_init.append(None)
            Qg_init.append(None)
        else:
            model.Pg[gg].value = pmid
            model.Qg[gg].value = qmid
            Pg_init.append(pmid)
            Qg_init.append(qmid)

    # balance slack P approximately
    if slack_gen_indices:
        otherP = sum(model.Pg[gg].value for gg in model.G if int(gg) not in slack_gen_indices)
        otherQ = sum(model.Qg[gg].value for gg in model.G if int(gg) not in slack_gen_indices)
        slackP = total_loadP - otherP
        slackQ = total_loadQ - otherQ
        for gg in slack_gen_indices:
            # clip to bounds
            model.Pg[gg].value = min(max(slackP, float(model.Pgmin[gg])), float(model.Pgmax[gg]))
            model.Qg[gg].value = min(max(slackQ, float(model.Qgmin[gg])), float(model.Qgmax[gg]))

    # compute net injections initial (using qsh=0)
    Pinj0 = {i: -Pd[i] for i in buses}
    Qinj0 = {i: -Qd[i] for i in buses}
    for gg in model.G:
        b = int(model.gen_bus[gg])
        Pinj0[b] += float(model.Pg[gg].value)
        Qinj0[b] += float(model.Qg[gg].value)

    # subtree aggregation for flows (ignoring losses)
    # compute subtree sum of (-Pinj)?? Use your nodal balance sign convention:
    # For a leaf load: Pinj negative => upstream P should be positive.
    # We'll compute required upstream flow as demand = -Pinj (positive if net consumption).
    demandP = {i: -Pinj0[i] for i in buses}
    demandQ = {i: -Qinj0[i] for i in buses}

    # postorder traversal
    order = []
    stack = [slack]
    parent = {slack: None}
    while stack:
        u = stack.pop()
        order.append(u)
        for v in children[u]:
            parent[v] = u
            stack.append(v)
    order.reverse()  # leaves first

    subP = {i: demandP[i] for i in buses}
    subQ = {i: demandQ[i] for i in buses}
    for u in order:
        for v in children[u]:
            subP[u] += subP[v]
            subQ[u] += subQ[v]

    # set branch flows Pij,Qij (parent->child) = subtree demand at child
    for (i, j) in E:
        model.Pij[i, j].value = subP[j]
        model.Qij[i, j].value = subQ[j]
        # ell from equality approx: ell = (P^2+Q^2)/(delta*v)
        delta = 1.0
        if (i, j) in set(data["T"]):
            # use nominal delta at tap=0 if exists; else 1
            delta = 1.0
        vij = float(model.v[i].value)
        model.ell[i, j].value = max((subP[j] ** 2 + subQ[j] ** 2) / max(vij * delta, 1e-6), 0.0)


# ----------------------------
# Solver utilities
# ----------------------------
def _count_discrete_vars(model: pyo.ConcreteModel) -> Tuple[int, int]:
    from pyomo.environ import Var
    nbin, nint = 0, 0
    for v in model.component_data_objects(Var, descend_into=True):
        if v.is_binary():
            nbin += 1
        elif v.is_integer():
            nint += 1
    return nbin, nint


def _solve_nlp(model: pyo.ConcreteModel,
               tee: bool = False,
               tol: float = IPOPT_TOL,
               max_iter: int = IPOPT_MAX_ITER,
               max_cpu_time: int = IPOPT_MAX_CPU_TIME) -> Optional[float]:
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

        res = solver.solve(model, tee=tee)
        tc = res.solver.termination_condition
        if tc in [
            pyo.TerminationCondition.optimal,
            pyo.TerminationCondition.locallyOptimal,
            pyo.TerminationCondition.feasible,
            pyo.TerminationCondition.maxIterations,
            pyo.TerminationCondition.maxTimeLimit,
        ]:
            return pyo.value(model.obj)

        return None

    raise RuntimeError("No NLP solver available (ipopt/cyipopt/appsi_ipopt).")


def _try_solve_with_scip_minlp(model: pyo.ConcreteModel,
                              time_limit_sec: int = SCIP_TIME_LIMIT,
                              gap_limit: float = SCIP_GAP_LIMIT,
                              memory_limit_mb: int = SCIP_MEMORY_LIMIT_MB,
                              node_limit: int = SCIP_NODE_LIMIT,
                              tee: bool = False) -> bool:
    solver = pyo.SolverFactory("scip")
    if solver is None or not solver.available(exception_flag=False):
        return False

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
        tc = res.solver.termination_condition
        if tc in [
            pyo.TerminationCondition.optimal,
            pyo.TerminationCondition.locallyOptimal,
            pyo.TerminationCondition.feasible,
            pyo.TerminationCondition.maxTimeLimit,
        ]:
            return True
    except Exception:
        return False

    return False


def solve_with_nlp_branch_and_bound(data: Dict[str, Any],
                                   max_nodes: int = HEURISTIC_BB_MAX_NODES,
                                   frac_tol: float = HEURISTIC_FRAC_TOL,
                                   tee: bool = False) -> Dict[str, Any]:
    best = {"obj": float("inf"), "sol": None}
    node_count = 0

    # branch variables: beta (i,j,tap) + a_sh(i)
    beta_keys = list(data["alpha_tap"].keys())  # ((i,j),tap)
    beta_triplets = [(ij[0], ij[1], int(tap)) for (ij, tap) in beta_keys]
    shunt_keys = list(data["C"])

    branch_vars = [("beta", key) for key in beta_triplets] + [("a_sh", i) for i in shunt_keys]

    def _get_val(model, kind, key):
        if kind == "beta":
            i, j, tap = key
            return pyo.value(model.beta[i, j, tap])
        else:
            i = key
            return pyo.value(model.a_sh[i])

    def _fix(model, kind, key, val):
        if kind == "beta":
            i, j, tap = key
            model.beta[i, j, tap].fix(val)
        else:
            i = key
            model.a_sh[i].fix(val)

    def _is_integral(x):
        return (x is not None) and ((x <= frac_tol) or (x >= 1.0 - frac_tol))

    def _pick_branch(model):
        best_pick = None
        best_frac = 0.0
        for kind, key in branch_vars:
            v = _get_val(model, kind, key)
            if _is_integral(v):
                continue
            frac = abs(v - round(v))
            if frac > best_frac:
                best_frac = frac
                best_pick = (kind, key, v)
        return best_pick

    def dfs(fixings: List[Tuple[str, Any, int]]):
        nonlocal node_count, best
        if node_count >= max_nodes:
            return
        node_count += 1

        model = build_pyomo_bfm_model(data, relax_binaries=True)
        initialize_bfm(model, data)
        for kind, key, val in fixings:
            _fix(model, kind, key, val)

        obj = _solve_nlp(
            model,
            tee=tee,
            tol=IPOPT_TOL,
            max_iter=min(IPOPT_MAX_ITER, 2000),
            max_cpu_time=HEURISTIC_NODE_IPOPT_TIME
        )
        if obj is None:
            return
        if obj >= best["obj"] - 1e-9:
            return

        # check integrality
        all_int = True
        for kind, key in branch_vars:
            v = _get_val(model, kind, key)
            if not _is_integral(v):
                all_int = False
                break

        if all_int:
            best["obj"] = obj
            best["sol"] = model
            return

        pick = _pick_branch(model)
        if pick is None:
            best["obj"] = obj
            best["sol"] = model
            return

        kind, key, v = pick
        order = [1, 0] if v >= 0.5 else [0, 1]
        for val in order:
            dfs(fixings + [(kind, key, val)])

    dfs([])
    return {"best_obj": best["obj"], "best_model": best["sol"], "nodes": node_count}


# ----------------------------
# Output
# ----------------------------
def _print_solution(model: pyo.ConcreteModel, data: Dict[str, Any]):
    sn = data["sn_mva"]
    buses = data["buses"]
    gen_records = data["gen_records"]
    slack_bus = data["slack_bus"]

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
                    v = pyo.value(model.beta[ii, jj, tap])
                    if best is None or v > best[1]:
                        best = (tap, v)
            if best is not None:
                print(f"OLTC ({i}->{j}): tap={best[0]} (beta={best[1]:.6f}), "
                      f"alpha={pyo.value(model.alpha[i, j]):.6f}, delta={pyo.value(model.delta[i, j]):.6f}")

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
        Smag = math.sqrt(max(P * P + Q * Q, 0.0))
        print(f"({i}->{j}): P={P:+.6f} pu, Q={Q:+.6f} pu, |S|={Smag:.6f} pu <= {pyo.value(model.Smax[i,j]):.6f}, ell={ell:.6f}")


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

        # Build BFM model
        model = build_pyomo_bfm_model(data, relax_binaries=False)
        initialize_bfm(model, data)

        # Detect discrete vars
        nbin, nint = _count_discrete_vars(model)
        print(f"[INFO] discrete vars detected: bin={nbin}, int={nint}")
        print(f"[INFO] #OLTC branches={len(data['T'])}, #SwitchedShunts={len(data['C'])}, #BetaIndex={len(list(model.BETA_INDEX))}")

        # Try SCIP MINLP first
        if (nbin + nint) > 0:
            print("[INFO] Discrete vars present -> trying SCIP (nonconvex MINLP).")
            ok = _try_solve_with_scip_minlp(
                model,
                time_limit_sec=SCIP_TIME_LIMIT,
                gap_limit=SCIP_GAP_LIMIT,
                memory_limit_mb=SCIP_MEMORY_LIMIT_MB,
                node_limit=SCIP_NODE_LIMIT,
                tee=TEE_SOLVER_LOG
            )
            if ok:
                print("\n[SOLVED] by SCIP (BFM MINLP).")
                print("Objective (EUR):", pyo.value(model.obj))
                _print_solution(model, data)
                return

            print("\n[WARN] SCIP failed/limited. Falling back to NLP-B&B heuristic (IPOPT per node).")

            result = solve_with_nlp_branch_and_bound(data, max_nodes=HEURISTIC_BB_MAX_NODES, tee=False)
            best_model = result["best_model"]
            if best_model is None:
                print("[FAIL] No feasible solution found by NLP-B&B heuristic.")
                return

            print("\n[SOLVED] by NLP-B&B heuristic (BFM).")
            print("Nodes explored:", result["nodes"])
            print("Best objective (EUR):", result["best_obj"])
            _print_solution(best_model, data)
            return

        # No discrete vars case (shouldn't happen here)
        print("[INFO] No discrete vars -> solving NLP (IPOPT).")
        obj = _solve_nlp(model, tee=TEE_SOLVER_LOG)
        if obj is None:
            print("[FAIL] IPOPT could not find a feasible solution.")
            return
        print("\n[SOLVED] by IPOPT (BFM NLP).")
        print("Objective (EUR):", obj)
        _print_solution(model, data)

    finally:
        t1 = time.perf_counter()
        print(f"\n[INFO] Total elapsed time: {t1 - t0:.2f} seconds.")


if __name__ == "__main__":
    main()
