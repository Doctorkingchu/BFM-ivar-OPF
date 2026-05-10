# BFM_MINLP.py
# ------------------------------------------------------------
# Pyomo Branch-Flow Model (BFM) AC-OPF / MINLP
# for modified explicit 41-bus mesh network
#
#   - Uses ALL directed branches in net.line (from_bus -> to_bus) as E
#   - OLTC: one-hot tap selection -> delta_ij (used in vdrop and Irel)
#   - Switched shunt capacitor: binary on/off,
#         qsh = a * bcap * (v / v_rated_sq)
#   - Nonconvex equality kept:
#         P_ij^2 + Q_ij^2 = deltaE_ij * v_i * ell_ij
#
# Exploration 강화:
#   - SCIP: gap ~ 0, huge node/time, optimality emphasis,
#           tighter feasibility tol, verbose log
#   - If SCIP ends too fast / suspicious, run heuristic NLP-B&B
#
# Network module:
#   ieee39busplus_modified_explicit.py
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
# Solver settings (more exploration)
# ============================
# IPOPT (feasibility NLP / per-node NLP)
IPOPT_TOL = 1e-7
IPOPT_MAX_ITER = 4000
IPOPT_MAX_CPU_TIME = 180

IPOPT_ACCEPTABLE_TOL = 1e-5
IPOPT_ACCEPTABLE_ITER = 20
IPOPT_ACCEPTABLE_CONSTR_VIOL = 1e-5

# SCIP (nonconvex MINLP) - exploration 강화
SCIP_TIME_LIMIT = 36000
SCIP_GAP_LIMIT = 1e-9
SCIP_ABSGAP_LIMIT = 0.0
SCIP_MEMORY_LIMIT_MB = 8192
SCIP_NODE_LIMIT = 5_000_000

TEE_SOLVER_LOG = True
RUN_PF_WARMSTART = True

RESID_TOL_ACCEPT = 1e-4
INT_TOL_ACCEPT = 1e-6

RUN_HEURISTIC_BB_IF_SCIP_SUSPICIOUS = True
HEURISTIC_BB_MAX_NODES = 300
HEURISTIC_NODE_IPOPT_TIME = 60
HEURISTIC_FRAC_TOL = 1e-6

SCIP_TOO_FAST_SEC = 10.0


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
# Metadata readers
# ----------------------------
def read_network_device_metadata(net) -> Dict[str, Any]:
    if "fixed_oltc_table" not in net:
        raise KeyError("Network metadata 'fixed_oltc_table' not found.")
    if "fixed_shunt_table" not in net:
        raise KeyError("Network metadata 'fixed_shunt_table' not found.")

    oltc_df = net["fixed_oltc_table"]
    shunt_df = net["fixed_shunt_table"]

    oltc_branches: Dict[Tuple[int, int], OLTCBranchConfig] = {}
    oltc_edges_ordered: List[Tuple[int, int]] = []

    for _, row in oltc_df.iterrows():
        i = int(row["from_bus"])
        j = int(row["to_bus"])
        oltc_edges_ordered.append((i, j))
        oltc_branches[(i, j)] = OLTCBranchConfig(
            tap_min=int(row["tap_min"]),
            tap_max=int(row["tap_max"]),
            dV_percent=float(row["dV_percent"]),
        )

    shunts: Dict[int, ShuntConfig] = {}
    for _, row in shunt_df.iterrows():
        b = int(row["bus"])
        shunts[b] = ShuntConfig(
            bcap_pu=float(row["bcap_pu"]),
            v_rated_sq=1.0,
        )

    return {
        "oltc_edges_ordered": oltc_edges_ordered,
        "oltc_branches": oltc_branches,
        "shunts": shunts,
    }


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


# ----------------------------
# Data extraction
# ----------------------------
def extract_bfm_fullmesh_data(net, cfg: BuildConfig) -> Dict[str, Any]:
    """
    Extract per-unit data for BFM on the FULL meshed graph:
      - E: directed edges exactly as stored in net.line (from_bus -> to_bus), ALL kept
      - r,x in pu
      - ellmax = (Smax_pu)^2
      - OLTC only if (i,j) exists exactly in E (tap on i-side)
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

    # OLTC on directed edges only
    T = []
    K = {}
    delta_tap = {}
    alpha_tap = {}

    for (u, v), tcfg in cfg.oltc_branches.items():
        if (u, v) not in E_set:
            if (v, u) in E_set:
                raise ValueError(
                    f"OLTC edge {(u, v)} exists only as reversed directed edge {(v, u)} in net.line. "
                    f"Fix metadata direction or line ordering."
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
# Model builder
# ----------------------------
def build_pyomo_bfm_model(data: Dict[str, Any], relax_binaries: bool = False) -> pyo.ConcreteModel:
    sn = data["sn_mva"]
    buses = data["buses"]
    slack_bus = data["slack_bus"]
    slack_vm_pu = data["slack_vm_pu"]

    E = data["E"]
    r = data["r"]
    x = data["x"]
    ellmax = data["ellmax"]

    T = data["T"]
    K = data["K"]
    delta_tap = data["delta_tap"]
    alpha_tap = data["alpha_tap"]

    C = data["C"]
    bcap_pu = data["bcap_pu"]
    v_rated_sq = data["v_rated_sq"]
    Mq = data["Mq"]

    Pd = data["Pd_pu"]
    Qd = data["Qd_pu"]
    Vmin = data["Vmin"]
    Vmax = data["Vmax"]

    gen_records = data["gen_records"]
    Gset = list(range(len(gen_records)))

    T_set = set(T)
    C_set = set(C)

    out_arcs = {i: [] for i in buses}
    in_arcs = {i: [] for i in buses}
    for (i, j) in E:
        out_arcs[i].append((i, j))
        in_arcs[j].append((i, j))

    m = pyo.ConcreteModel(name="BFM_MINLP_modified_explicit_41bus")

    m.N = pyo.Set(initialize=buses, ordered=True)
    m.G = pyo.Set(initialize=Gset, ordered=True)
    m.E = pyo.Set(initialize=E, dimen=2, ordered=True)
    m.T = pyo.Set(initialize=T, dimen=2, ordered=True)
    m.C = pyo.Set(initialize=C, ordered=True)

    # params
    m.Pd = pyo.Param(m.N, initialize=lambda mm, i: float(Pd[i]), mutable=False)
    m.Qd = pyo.Param(m.N, initialize=lambda mm, i: float(Qd[i]), mutable=False)

    m.Vmin = pyo.Param(m.N, initialize=lambda mm, i: float(Vmin[i]), mutable=False)
    m.Vmax = pyo.Param(m.N, initialize=lambda mm, i: float(Vmax[i]), mutable=False)

    m.r = pyo.Param(m.E, initialize=lambda mm, i, j: float(r[(i, j)]), mutable=False)
    m.x = pyo.Param(m.E, initialize=lambda mm, i, j: float(x[(i, j)]), mutable=False)
    m.ellmax = pyo.Param(m.E, initialize=lambda mm, i, j: float(ellmax[(i, j)]), mutable=False)

    # gen params
    m.gen_bus = pyo.Param(m.G, initialize=lambda mm, gg: int(gen_records[int(gg)]["bus"]), within=pyo.Any)
    m.Pgmin = pyo.Param(m.G, initialize=lambda mm, gg: float(gen_records[int(gg)]["pmin_pu"]))
    m.Pgmax = pyo.Param(m.G, initialize=lambda mm, gg: float(gen_records[int(gg)]["pmax_pu"]))
    m.Qgmin = pyo.Param(m.G, initialize=lambda mm, gg: float(gen_records[int(gg)]["qmin_pu"]))
    m.Qgmax = pyo.Param(m.G, initialize=lambda mm, gg: float(gen_records[int(gg)]["qmax_pu"]))
    m.c2 = pyo.Param(m.G, initialize=lambda mm, gg: float(gen_records[int(gg)]["c2"]))
    m.c1 = pyo.Param(m.G, initialize=lambda mm, gg: float(gen_records[int(gg)]["c1"]))
    m.c0 = pyo.Param(m.G, initialize=lambda mm, gg: float(gen_records[int(gg)]["c0"]))

    # shunt params
    m.bcap = pyo.Param(m.C, initialize=lambda mm, i: float(bcap_pu[int(i)]), mutable=False)
    m.vrated = pyo.Param(m.C, initialize=lambda mm, i: float(v_rated_sq[int(i)]), mutable=False)
    m.Mq = pyo.Param(m.C, initialize=lambda mm, i: float(Mq[int(i)]), mutable=False)

    # OLTC tap index
    beta_index = []
    for (i, j) in T:
        for tap in K[(i, j)]:
            beta_index.append((i, j, int(tap)))
    m.BETA_INDEX = pyo.Set(initialize=beta_index, dimen=3, ordered=True)

    m.delta_tap = pyo.Param(
        m.BETA_INDEX,
        initialize=lambda mm, i, j, tap: float(delta_tap[((i, j), int(tap))]),
        mutable=False
    )
    m.alpha_tap = pyo.Param(
        m.BETA_INDEX,
        initialize=lambda mm, i, j, tap: float(alpha_tap[((i, j), int(tap))]),
        mutable=False
    )

    # variables
    m.Pg = pyo.Var(m.G, bounds=lambda mm, gg: (mm.Pgmin[gg], mm.Pgmax[gg]))
    m.Qg = pyo.Var(m.G, bounds=lambda mm, gg: (mm.Qgmin[gg], mm.Qgmax[gg]))

    m.v = pyo.Var(m.N, bounds=lambda mm, i: (mm.Vmin[i] ** 2, mm.Vmax[i] ** 2))

    m.Pinj = pyo.Var(m.N)
    m.Qinj = pyo.Var(m.N)

    def _p_bounds(mm, i, j):
        vmax_sq = float(mm.Vmax[i] ** 2)
        smax = math.sqrt(float(mm.ellmax[i, j]) * vmax_sq)
        return (-smax, smax)

    def _q_bounds(mm, i, j):
        vmax_sq = float(mm.Vmax[i] ** 2)
        smax = math.sqrt(float(mm.ellmax[i, j]) * vmax_sq)
        return (-smax, smax)

    m.Pij = pyo.Var(m.E, bounds=_p_bounds)
    m.Qij = pyo.Var(m.E, bounds=_q_bounds)
    m.ell = pyo.Var(m.E, bounds=lambda mm, i, j: (0.0, mm.ellmax[i, j]))

    m.qsh = pyo.Var(m.N)

    if relax_binaries:
        m.beta = pyo.Var(m.BETA_INDEX, bounds=(0.0, 1.0))
        m.a_sh = pyo.Var(m.C, bounds=(0.0, 1.0))
    else:
        m.beta = pyo.Var(m.BETA_INDEX, within=pyo.Binary)
        m.a_sh = pyo.Var(m.C, within=pyo.Binary)

    delta_bounds = {}
    for (i, j) in T:
        vals = [delta_tap[((i, j), tap)] for tap in K[(i, j)]]
        delta_bounds[(i, j)] = (min(vals), max(vals))
    m.delta = pyo.Var(m.T, bounds=lambda mm, i, j: delta_bounds[(i, j)])

    # constraints
    if data["fix_slack_vm"]:
        m.slack_v = pyo.Constraint(expr=m.v[slack_bus] == float(slack_vm_pu) ** 2)

    m.Pinj_def = pyo.Constraint(
        m.N,
        rule=lambda mm, i: mm.Pinj[i] == sum(mm.Pg[gg] for gg in mm.G if int(mm.gen_bus[gg]) == int(i)) - mm.Pd[i]
    )
    m.Qinj_def = pyo.Constraint(
        m.N,
        rule=lambda mm, i: mm.Qinj[i] == sum(mm.Qg[gg] for gg in mm.G if int(mm.gen_bus[gg]) == int(i)) - mm.Qd[i] + mm.qsh[i]
    )

    # OLTC one-hot + delta selection
    m.onehot = pyo.Constraint(
        m.T,
        rule=lambda mm, i, j: sum(mm.beta[i, j, int(t)] for t in K[(i, j)]) == 1
    )
    m.delta_sel = pyo.Constraint(
        m.T,
        rule=lambda mm, i, j: mm.delta[i, j] == sum(mm.delta_tap[i, j, int(t)] * mm.beta[i, j, int(t)] for t in K[(i, j)])
    )
    m.deltaE = pyo.Expression(m.E, rule=lambda mm, i, j: mm.delta[i, j] if (i, j) in T_set else 1.0)

    # shunt: qsh=0 for non-C
    def qsh_zero_rule(mm, i):
        if int(i) in C_set:
            return pyo.Constraint.Skip
        return mm.qsh[i] == 0.0
    m.qsh_zero = pyo.Constraint(m.N, rule=qsh_zero_rule)

    m.qsh_nonneg = pyo.Constraint(m.C, rule=lambda mm, i: mm.qsh[int(i)] >= 0.0)
    m.qsh_upper = pyo.Constraint(m.C, rule=lambda mm, i: mm.qsh[int(i)] <= mm.Mq[i] * mm.a_sh[i])

    def qsh_match_pos(mm, i):
        q_target = mm.bcap[i] * (mm.v[int(i)] / mm.vrated[i])
        return mm.qsh[int(i)] - q_target <= mm.Mq[i] * (1.0 - mm.a_sh[i])

    def qsh_match_neg(mm, i):
        q_target = mm.bcap[i] * (mm.v[int(i)] / mm.vrated[i])
        return q_target - mm.qsh[int(i)] <= mm.Mq[i] * (1.0 - mm.a_sh[i])

    m.qsh_match_pos = pyo.Constraint(m.C, rule=qsh_match_pos)
    m.qsh_match_neg = pyo.Constraint(m.C, rule=qsh_match_neg)

    # BFM balance
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

    def vdrop(mm, i, j):
        rij = mm.r[i, j]
        xij = mm.x[i, j]
        return mm.v[j] == mm.deltaE[i, j] * mm.v[i] - 2.0 * (rij * mm.Pij[i, j] + xij * mm.Qij[i, j]) + (rij * rij + xij * xij) * mm.ell[i, j]
    m.Vdrop = pyo.Constraint(m.E, rule=vdrop)

    def irel(mm, i, j):
        return mm.Pij[i, j] ** 2 + mm.Qij[i, j] ** 2 == mm.deltaE[i, j] * mm.v[i] * mm.ell[i, j]
    m.Irel = pyo.Constraint(m.E, rule=irel)

    m.obj = pyo.Objective(
        expr=sum(m.c2[g] * (sn * m.Pg[g]) ** 2 + m.c1[g] * (sn * m.Pg[g]) + m.c0[g] for g in m.G),
        sense=pyo.minimize
    )

    m._out_arcs = out_arcs
    m._in_arcs = in_arcs
    return m


# ----------------------------
# Initialization
# ----------------------------
def warmstart_from_pf(model: pyo.ConcreteModel, data: Dict[str, Any], net) -> bool:
    if not RUN_PF_WARMSTART:
        return False

    sn = data["sn_mva"]
    line_id_of_edge = data["line_id_of_edge"]

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

        for (i, j, tap) in model.BETA_INDEX:
            model.beta[i, j, tap].set_value(0.0)
        for (i, j) in model.T:
            pick = _default_tap(data["K"][(i, j)])
            for tap in data["K"][(i, j)]:
                model.beta[i, j, tap].set_value(1.0 if tap == pick else 0.0)
            model.delta[i, j].set_value(data["delta_tap"][((i, j), pick)])

        for i in model.C:
            model.a_sh[i].set_value(0.0)
        for i in model.N:
            model.qsh[int(i)].set_value(0.0)

        for (i, j) in model.E:
            lid = line_id_of_edge[(int(i), int(j))]
            Ppu = float(net.res_line.at[lid, "p_from_mw"]) / sn
            Qpu = float(net.res_line.at[lid, "q_from_mvar"]) / sn
            model.Pij[i, j].set_value(Ppu)
            model.Qij[i, j].set_value(Qpu)

            vi = float(pyo.value(model.v[i]))
            deltaE = float(pyo.value(model.deltaE[i, j]))
            ell = (Ppu * Ppu + Qpu * Qpu) / max(deltaE * vi, 1e-8)
            ell = max(0.0, min(ell, float(pyo.value(model.ellmax[i, j]))))
            model.ell[i, j].set_value(ell)

        return True

    except Exception:
        return False


def set_default_discrete_and_fix(model: pyo.ConcreteModel, data: Dict[str, Any]):
    for (i, j) in model.T:
        taps = data["K"][(i, j)]
        pick = _default_tap(taps)
        for t in taps:
            model.beta[i, j, t].fix(1 if t == pick else 0)
        model.delta[i, j].fix(data["delta_tap"][((i, j), pick)])
    for i in model.C:
        model.a_sh[i].fix(0)


# ----------------------------
# Diagnostics
# ----------------------------
def evaluate_quality(model: pyo.ConcreteModel, data: Dict[str, Any]) -> Dict[str, float]:
    out_arcs = model._out_arcs
    in_arcs = model._in_arcs

    max_bfp = 0.0
    max_bfq = 0.0
    max_vdrop = 0.0
    max_irel = 0.0
    max_onehot = 0.0
    max_delta = 0.0
    max_sh = 0.0
    max_frac = 0.0

    for i in model.N:
        outP = sum(pyo.value(model.Pij[a, b]) for (a, b) in out_arcs[int(i)])
        inP = sum(pyo.value(model.Pij[a, b] - model.r[a, b] * model.ell[a, b]) for (a, b) in in_arcs[int(i)])
        max_bfp = max(max_bfp, abs((outP - inP) - pyo.value(model.Pinj[i])))

        outQ = sum(pyo.value(model.Qij[a, b]) for (a, b) in out_arcs[int(i)])
        inQ = sum(pyo.value(model.Qij[a, b] - model.x[a, b] * model.ell[a, b]) for (a, b) in in_arcs[int(i)])
        max_bfq = max(max_bfq, abs((outQ - inQ) - pyo.value(model.Qinj[i])))

    for (i, j) in model.E:
        rij = pyo.value(model.r[i, j])
        xij = pyo.value(model.x[i, j])

        lhs = pyo.value(model.v[j])
        rhs = pyo.value(
            model.deltaE[i, j] * model.v[i]
            - 2.0 * (rij * model.Pij[i, j] + xij * model.Qij[i, j])
            + (rij * rij + xij * xij) * model.ell[i, j]
        )
        max_vdrop = max(max_vdrop, abs(lhs - rhs))

        lhs2 = pyo.value(model.Pij[i, j] ** 2 + model.Qij[i, j] ** 2)
        rhs2 = pyo.value(model.deltaE[i, j] * model.v[i] * model.ell[i, j])
        max_irel = max(max_irel, abs(lhs2 - rhs2))

    for (i, j) in model.T:
        taps = data["K"][(i, j)]
        s = sum(pyo.value(model.beta[i, j, t]) for t in taps)
        max_onehot = max(max_onehot, abs(s - 1.0))
        rhs = sum(pyo.value(model.delta_tap[i, j, t]) * pyo.value(model.beta[i, j, t]) for t in taps)
        max_delta = max(max_delta, abs(pyo.value(model.delta[i, j]) - rhs))
        for t in taps:
            v = pyo.value(model.beta[i, j, t])
            max_frac = max(max_frac, abs(v - round(v)))

    for i in model.C:
        v = pyo.value(model.a_sh[i])
        max_frac = max(max_frac, abs(v - round(v)))

    Cset = set(data["C"])
    for i in model.N:
        if int(i) not in Cset:
            max_sh = max(max_sh, abs(pyo.value(model.qsh[int(i)])))

    max_resid = max(max_bfp, max_bfq, max_vdrop, max_irel, max_onehot, max_delta, max_sh)
    return {
        "max_bfm_p": max_bfp,
        "max_bfm_q": max_bfq,
        "max_vdrop": max_vdrop,
        "max_irel": max_irel,
        "max_onehot": max_onehot,
        "max_delta": max_delta,
        "max_shunt": max_sh,
        "max_frac": max_frac,
        "max_resid": max_resid,
    }


def solution_is_acceptable(q: Dict[str, float]) -> bool:
    return (q["max_resid"] <= RESID_TOL_ACCEPT) and (q["max_frac"] <= INT_TOL_ACCEPT)


# ----------------------------
# Solvers
# ----------------------------
def solve_ipopt(
    model: pyo.ConcreteModel,
    tee: bool = False,
    tol: float = IPOPT_TOL,
    max_iter: int = IPOPT_MAX_ITER,
    max_cpu_time: int = IPOPT_MAX_CPU_TIME
) -> Optional[float]:
    for name in ["ipopt", "cyipopt", "appsi_ipopt"]:
        opt = pyo.SolverFactory(name)
        if opt is None or not opt.available(exception_flag=False):
            continue
        try:
            opt.options["tol"] = float(tol)
            opt.options["max_iter"] = int(max_iter)
            opt.options["max_cpu_time"] = float(max_cpu_time)
            opt.options["acceptable_tol"] = float(IPOPT_ACCEPTABLE_TOL)
            opt.options["acceptable_iter"] = int(IPOPT_ACCEPTABLE_ITER)
            opt.options["acceptable_constr_viol_tol"] = float(IPOPT_ACCEPTABLE_CONSTR_VIOL)
            opt.options["print_level"] = 5 if tee else 0
        except Exception:
            pass
        try:
            res = opt.solve(model, tee=tee)
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


def solve_scip_minlp(model: pyo.ConcreteModel, tee: bool = True) -> Dict[str, Any]:
    opt = pyo.SolverFactory("scip", solver_io="nl")
    if opt is None or not opt.available(exception_flag=False):
        opt = pyo.SolverFactory("scip")
    if opt is None or not opt.available(exception_flag=False):
        return {
            "ok": False,
            "reason": "SCIP not available",
            "obj": None,
            "quality": None,
            "termination": None,
            "elapsed": None
        }

    try:
        opt.options["limits/time"] = float(SCIP_TIME_LIMIT)
        opt.options["limits/nodes"] = int(SCIP_NODE_LIMIT)
        opt.options["limits/gap"] = float(SCIP_GAP_LIMIT)
        opt.options["limits/absgap"] = float(SCIP_ABSGAP_LIMIT)
        opt.options["emphasis/optimality"] = 1
        opt.options["numerics/feastol"] = 1e-8
        opt.options["numerics/dualfeastol"] = 1e-8
        opt.options["display/verblevel"] = 5 if tee else 0
    except Exception:
        pass

    t0 = time.perf_counter()
    try:
        res = opt.solve(model, tee=tee)
    except Exception as e:
        return {
            "ok": False,
            "reason": f"SCIP exception: {e}",
            "obj": None,
            "quality": None,
            "termination": None,
            "elapsed": time.perf_counter() - t0
        }

    elapsed = time.perf_counter() - t0
    term = res.solver.termination_condition

    try:
        obj = float(pyo.value(model.obj))
    except Exception:
        obj = None

    q = evaluate_quality(model, model._data_for_check)
    ok = (obj is not None) and np.isfinite(obj) and solution_is_acceptable(q)

    return {
        "ok": ok,
        "reason": "accepted" if ok else "rejected by quality",
        "obj": obj,
        "quality": q,
        "termination": term,
        "elapsed": elapsed
    }


# ----------------------------
# Heuristic NLP-based B&B
# ----------------------------
def solve_with_nlp_branch_and_bound(
    data: Dict[str, Any],
    net,
    max_nodes: int = HEURISTIC_BB_MAX_NODES,
    frac_tol: float = HEURISTIC_FRAC_TOL,
    tee: bool = False
) -> Dict[str, Any]:
    best = {"obj": float("inf"), "sol": None}
    node_count = 0

    beta_triplets = []
    for (i, j) in data["T"]:
        for tap in data["K"][(i, j)]:
            beta_triplets.append((i, j, int(tap)))
    shunt_keys = list(data["C"])
    branch_vars = [("beta", key) for key in beta_triplets] + [("a_sh", i) for i in shunt_keys]

    def _get_val(model, kind, key):
        if kind == "beta":
            i, j, tap = key
            return pyo.value(model.beta[i, j, tap])
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
        best_frac = -1.0
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

        m = build_pyomo_bfm_model(data, relax_binaries=True)
        m._data_for_check = data

        _ = warmstart_from_pf(m, data, net)

        for kind, key, val in fixings:
            _fix(m, kind, key, val)

        obj = solve_ipopt(
            m,
            tee=tee,
            tol=IPOPT_TOL,
            max_iter=min(IPOPT_MAX_ITER, 2500),
            max_cpu_time=HEURISTIC_NODE_IPOPT_TIME,
        )
        if obj is None:
            return
        if obj >= best["obj"] - 1e-9:
            return

        all_int = True
        for kind, key in branch_vars:
            v = _get_val(m, kind, key)
            if not _is_integral(v):
                all_int = False
                break

        if all_int:
            best["obj"] = obj
            best["sol"] = m
            return

        pick = _pick_branch(m)
        if pick is None:
            best["obj"] = obj
            best["sol"] = m
            return

        kind, key, v = pick
        order = [1, 0] if v >= 0.5 else [0, 1]
        for val in order:
            dfs(fixings + [(kind, key, val)])

    dfs([])
    return {"best_obj": best["obj"], "best_model": best["sol"], "nodes": node_count}


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
    print("\n--- Quality (max residuals) ---")
    for k, v in q.items():
        print(f"{k}: {v:.3e}")

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
            alpha = sum(
                pyo.value(model.alpha_tap[i, j, t]) * pyo.value(model.beta[i, j, t])
                for t in taps
            )
            print(f"OLTC ({i}->{j}): tap={best_t:>3d}, delta={pyo.value(model.delta[i,j]):.6f}, alpha={alpha:.6f}")

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

    # 1) Build modified explicit 41-bus mesh
    net = mcase.busmeshed39_opf(
        slack_vm_pu=1.0,
        line_max_loading_percent=1e6,
        stress_q_over_p=0.95,
        stress_load_mw_each=300.0,
        r_loss_scale=3.0,
        max_i_ka_base=2.0,
        max_i_ka_stress=1.0,
        line_smax_pu_overrides=None,
    )

    # 2) Read metadata and build config
    meta = read_network_device_metadata(net)
    cfg = BuildConfig(
        oltc_branches=meta["oltc_branches"],
        shunts=meta["shunts"],
        fix_slack_vm=True,
    )

    # 3) Extract data
    data = extract_bfm_fullmesh_data(net, cfg)

    print(f"[INFO] #buses = {len(data['buses'])}")
    print(f"[INFO] #branches (directed, full) = {len(data['E'])}")
    print(f"[INFO] #gens = {len(data['gen_records'])}")
    print(f"[INFO] #OLTC branches = {len(data['T'])}")
    print(f"[INFO] #shunts = {len(data['C'])}")
    print(f"[INFO] total Pd (MW) = {data['sn_mva'] * sum(data['Pd_pu'].values()):.4f}")
    print(f"[INFO] total Qd (MVAr)= {data['sn_mva'] * sum(data['Qd_pu'].values()):.4f}")

    triples = [(r["type"], r["id"], r["c2"], r["c1"], r["c0"]) for r in data["gen_records"]]
    all_zero_cost = all(
        (abs(c2) < 1e-15 and abs(c1) < 1e-15 and abs(c0) < 1e-15)
        for (_, _, c2, c1, c0) in triples
    )
    print("[INFO] cost triples (type,id,c2,c1,c0) sample:", triples[:5], " ...")
    if all_zero_cost:
        print("[WARN] All generator cost coefficients are zero -> objective is constant.")

    # 4) Build MINLP model
    model = build_pyomo_bfm_model(data, relax_binaries=False)
    model._data_for_check = data

    pf_ok = warmstart_from_pf(model, data, net)
    print(f"[INFO] PF warm start success = {pf_ok}")

    # 5) Fixed-discrete backup NLP
    print("\n[INFO] Solving fixed-discrete NLP (tap≈0, shunt=0) with IPOPT ...")
    fixed_nlp = build_pyomo_bfm_model(data, relax_binaries=False)
    fixed_nlp._data_for_check = data
    _ = warmstart_from_pf(fixed_nlp, data, net)
    set_default_discrete_and_fix(fixed_nlp, data)

    fixed_obj = solve_ipopt(fixed_nlp, tee=False, max_cpu_time=min(IPOPT_MAX_CPU_TIME, 180))
    fixed_ok = False
    if fixed_obj is not None:
        q_fixed = evaluate_quality(fixed_nlp, data)
        fixed_ok = q_fixed["max_resid"] <= 1e-3
        print(f"[INFO] Fixed NLP objective = {fixed_obj:.10f}, max_resid={q_fixed['max_resid']:.3e}")
    else:
        print("[WARN] Fixed-discrete NLP failed.")

    # 6) SCIP MINLP
    print("\n[INFO] Solving MINLP with SCIP (exploration-forced settings)...")
    scip_res = solve_scip_minlp(model, tee=TEE_SOLVER_LOG)

    print(f"[INFO] SCIP elapsed = {scip_res['elapsed']:.2f} sec")
    print(f"[INFO] SCIP termination = {scip_res['termination']}")
    if scip_res["quality"] is not None:
        q = scip_res["quality"]
        print(
            f"[INFO] SCIP max_resid={q['max_resid']:.3e}, "
            f"max_frac={q['max_frac']:.3e}, max_onehot={q['max_onehot']:.3e}"
        )
    if scip_res["obj"] is not None:
        print(f"[INFO] SCIP objective = {scip_res['obj']:.10f}")
    print(f"[INFO] SCIP accepted = {scip_res['ok']} ({scip_res['reason']})")

    scip_suspicious = (
        (not scip_res["ok"]) or
        (scip_res["elapsed"] is not None and scip_res["elapsed"] <= SCIP_TOO_FAST_SEC)
    )

    if scip_res["ok"]:
        print_solution(model, data, title="SCIP MINLP Solution (ACCEPTED)")
    else:
        print("[WARN] SCIP solution rejected by residual/integrality check.")
        if fixed_ok:
            print_solution(fixed_nlp, data, title="Fallback: Fixed-discrete NLP (IPOPT)")
        else:
            print("[WARN] No feasible fixed-discrete NLP fallback available.")

    # 7) Heuristic NLP-B&B
    if RUN_HEURISTIC_BB_IF_SCIP_SUSPICIOUS and scip_suspicious:
        print("\n[INFO] Running heuristic NLP Branch-and-Bound ...")
        result = solve_with_nlp_branch_and_bound(
            data,
            net,
            max_nodes=HEURISTIC_BB_MAX_NODES,
            frac_tol=HEURISTIC_FRAC_TOL,
            tee=False
        )
        best_model = result["best_model"]
        if best_model is None:
            print("[WARN] Heuristic NLP-B&B did not find an integer-feasible improvement.")
        else:
            print(f"[INFO] Heuristic NLP-B&B nodes explored = {result['nodes']}")
            print(f"[INFO] Heuristic NLP-B&B best objective = {result['best_obj']:.10f}")
            print_solution(best_model, data, title="Heuristic NLP-B&B Best")

    t_all1 = time.perf_counter()
    print(f"\n[INFO] Total elapsed time: {t_all1 - t_all0:.2f} seconds")


if __name__ == "__main__":
    main()