# BFM_ivar_no_mit.py
# ------------------------------------------------------------
# BFM-ag on modified explicit 41-bus (meshed, keep all lines)
# with OLTC + shunt
# - Subproblem: MIQCP with ell fixed
# - Soft vdrop + Soft KCL (always feasible)
# - Output: robust
#
# Network module:
#   ieee41bus.py
# ------------------------------------------------------------

import math
import time
from dataclasses import dataclass
from typing import Dict, Tuple, Any, Optional, List

import numpy as np
import pyomo.environ as pyo
from pyomo.opt import TerminationCondition

import ieee41bus as mcase


# ============================================================
# User options
# ============================================================
FIX_SLACK_VM = True

# If None -> use all metadata-defined OLTC/shunts from the network
ACTIVE_OLTC_EDGES: Optional[List[Tuple[int, int]]] = None
ACTIVE_SHUNT_BUSES: Optional[List[int]] = None

# Build kwargs for the modified explicit 41-bus network
NETWORK_BUILD_KWARGS = dict(
    slack_vm_pu=1.0,
    line_max_loading_percent=1e6,
    stress_q_over_p=0.95,
    stress_load_mw_each=300.0,
    r_loss_scale=3.0,
    max_i_ka_base=2.0,
    max_i_ka_stress=1.0,
    line_smax_pu_overrides=None,
)

# ============================================================
# Global settings
# ============================================================
TEE_SOLVER_LOG = True

OUTER_MAX_ITERS = 40
OUTER_EPS = 1e-5
DENOM_EPS = 1e-10

ELL_GAMMA = 0.7
THETA_GAMMA = 1.0
CLIP_DTHETA_PREV = math.pi
THETA_RIDGE = 1e-8

# Soft penalties
RHO_VDROP = 2e5
RHO_KCL   = 2e7

SCIP_TIME_LIMIT = 120.0
SCIP_GAP_LIMIT = 1e-4
SCIP_NODE_LIMIT = 300000
SCIP_MEMORY_LIMIT_MB = 8192
SCIP_FEASTOL = 1e-5
SCIP_DUALFEASTOL = 1e-7


# ============================================================
# Utilities
# ============================================================
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


def _clip(v: float, lo: Optional[float], hi: Optional[float]) -> float:
    if lo is not None and v < lo:
        return float(lo)
    if hi is not None and v > hi:
        return float(hi)
    return float(v)


def _val(vardata, default: float = 0.0) -> float:
    x = getattr(vardata, "value", None)
    if x is None:
        return float(default)
    return float(x)


def _pval(x, default: float = 0.0) -> float:
    try:
        v = pyo.value(x, exception=False)
        if v is None:
            return float(default)
        return float(v)
    except Exception:
        return float(default)


def _solution_complete(m: pyo.ConcreteModel, data: Dict[str, Any]) -> bool:
    for i in data["buses"]:
        if m.v[i].value is None:
            return False
    for g in m.G:
        if m.Pg[g].value is None:
            return False
    for (i, j) in data["E"]:
        if m.Pij[i, j].value is None or m.Qij[i, j].value is None:
            return False
        if m.s_vdrop[i, j].value is None:
            return False
    return True


def _default_tap_choice(taps: List[int]) -> int:
    return min(taps, key=lambda t: abs(int(t)))


def _round_onehot_from_warm(
    data: Dict[str, Any],
    warm_beta: Dict[Tuple[int, int, int], float]
) -> Dict[Tuple[int, int, int], int]:
    out = {}
    for (i, j) in data["T"]:
        taps = data["K"][(i, j)]
        best_t, best_v = None, -1e100
        for t in taps:
            v = float(warm_beta.get((i, j, int(t)), 0.0))
            if v > best_v:
                best_v, best_t = v, int(t)
        if best_t is None:
            best_t = _default_tap_choice([int(t) for t in taps])
        for t in taps:
            out[(i, j, int(t))] = 1 if int(t) == int(best_t) else 0
    return out


def _pick_tap_from_beta(model: pyo.ConcreteModel, data: Dict[str, Any], i: int, j: int) -> int:
    taps = data["K"][(i, j)]
    best_t, best_v = None, -1e100
    for t in taps:
        vb = _val(model.beta[i, j, int(t)], 0.0)
        if vb > best_v:
            best_v, best_t = vb, int(t)
    if best_t is None:
        best_t = _default_tap_choice([int(t) for t in taps])
    return int(best_t)


def _delta_of_selected_tap(data: Dict[str, Any], i: int, j: int, tap: int) -> float:
    return float(data["delta_tap"][((i, j), int(tap))])


def _alpha_of_selected_tap(data: Dict[str, Any], i: int, j: int, tap: int) -> float:
    return float(data["alpha_tap"][((i, j), int(tap))])


# ============================================================
# Metadata readers
# ============================================================
@dataclass
class OLTCBranchConfig:
    tap_min: int
    tap_max: int
    dV_percent: float


@dataclass
class BuildConfig:
    oltc_branches: Dict[Tuple[int, int], OLTCBranchConfig]
    shunt_bcap_pu: Dict[int, float]
    fix_slack_vm: bool = True


def read_network_oltc_metadata(
    net,
    active_oltc_edges: Optional[List[Tuple[int, int]]] = None
) -> Dict[Tuple[int, int], OLTCBranchConfig]:
    if "fixed_oltc_table" not in net:
        raise KeyError("Network metadata 'fixed_oltc_table' not found.")

    oltc_df = net["fixed_oltc_table"].copy()
    available_arcs = {
        (int(net.line.at[lid, "from_bus"]), int(net.line.at[lid, "to_bus"]))
        for lid in net.line.index
    }

    requested = None
    if active_oltc_edges is not None:
        requested = {tuple(map(int, e)) for e in active_oltc_edges}

    oltc_branches: Dict[Tuple[int, int], OLTCBranchConfig] = {}

    for _, row in oltc_df.iterrows():
        i0 = int(row["from_bus"])
        j0 = int(row["to_bus"])

        if requested is not None and (i0, j0) not in requested and (j0, i0) not in requested:
            continue

        if (i0, j0) in available_arcs:
            i, j = i0, j0
        elif (j0, i0) in available_arcs:
            i, j = j0, i0
        else:
            raise ValueError(f"OLTC metadata edge ({i0},{j0}) not found in net.line.")

        oltc_branches[(i, j)] = OLTCBranchConfig(
            tap_min=int(row["tap_min"]),
            tap_max=int(row["tap_max"]),
            dV_percent=float(row["dV_percent"]),
        )

    if len(oltc_branches) == 0:
        raise ValueError("No active OLTC branches found.")

    return oltc_branches


def read_network_shunt_metadata(
    net,
    active_shunt_buses: Optional[List[int]] = None
) -> Dict[int, float]:
    if "fixed_shunt_table" not in net:
        raise KeyError("Network metadata 'fixed_shunt_table' not found.")

    sh_df = net["fixed_shunt_table"].copy()
    requested = None
    if active_shunt_buses is not None:
        requested = {int(b) for b in active_shunt_buses}

    shunt_bcap_pu: Dict[int, float] = {}
    for _, row in sh_df.iterrows():
        b = int(row["bus"])
        if requested is not None and b not in requested:
            continue
        shunt_bcap_pu[b] = float(row["bcap_pu"])

    if len(shunt_bcap_pu) == 0:
        raise ValueError("No active shunt buses found.")

    return shunt_bcap_pu


# ============================================================
# Data extraction (meshed, keep ALL net.line rows)
# ============================================================
def extract_data_meshed_keep_lines(net, cfg: BuildConfig) -> Dict[str, Any]:
    sn = float(net.sn_mva)

    buses = [int(i) for i in net.bus.index]
    bus_vn_kv = {int(i): float(net.bus.at[i, "vn_kv"]) for i in buses}
    Vmin_pu = {int(i): float(net.bus.at[i, "min_vm_pu"]) for i in buses}
    Vmax_pu = {int(i): float(net.bus.at[i, "max_vm_pu"]) for i in buses}

    if len(net.ext_grid.index) < 1:
        raise ValueError("pandapower net must have ext_grid.")
    eg0 = int(net.ext_grid.index[0])
    slack_bus = int(net.ext_grid.at[eg0, "bus"])
    slack_vm_pu = float(net.ext_grid.at[eg0, "vm_pu"])

    # loads
    Pd = {i: 0.0 for i in buses}
    Qd = {i: 0.0 for i in buses}
    if hasattr(net, "load") and len(net.load.index) > 0:
        for li in net.load.index:
            b = int(net.load.at[li, "bus"])
            Pd[b] += float(net.load.at[li, "p_mw"])
            Qd[b] += float(net.load.at[li, "q_mvar"])
    Pd_pu = {i: Pd[i] / sn for i in buses}
    Qd_pu = {i: Qd[i] / sn for i in buses}

    # generators ext_grid + gen
    gen_records = []
    for eg in net.ext_grid.index:
        eg = int(eg)
        b = int(net.ext_grid.at[eg, "bus"])
        pmin = float(net.ext_grid.at[eg, "min_p_mw"]) if "min_p_mw" in net.ext_grid.columns else -1e9
        pmax = float(net.ext_grid.at[eg, "max_p_mw"]) if "max_p_mw" in net.ext_grid.columns else 1e9
        qmin = float(net.ext_grid.at[eg, "min_q_mvar"]) if "min_q_mvar" in net.ext_grid.columns else -1e9
        qmax = float(net.ext_grid.at[eg, "max_q_mvar"]) if "max_q_mvar" in net.ext_grid.columns else 1e9
        c2, c1, c0 = _find_poly_cost(net, "ext_grid", eg)
        gen_records.append(dict(
            type="ext_grid", id=eg, bus=b,
            pmin_pu=pmin / sn, pmax_pu=pmax / sn,
            qmin_pu=qmin / sn, qmax_pu=qmax / sn,
            c2=c2, c1=c1, c0=c0
        ))

    if hasattr(net, "gen") and len(net.gen.index) > 0:
        for gi in net.gen.index:
            gi = int(gi)
            b = int(net.gen.at[gi, "bus"])
            pmin = float(net.gen.at[gi, "min_p_mw"])
            pmax = float(net.gen.at[gi, "max_p_mw"])
            qmin = float(net.gen.at[gi, "min_q_mvar"])
            qmax = float(net.gen.at[gi, "max_q_mvar"])
            c2, c1, c0 = _find_poly_cost(net, "gen", gi)
            gen_records.append(dict(
                type="gen", id=gi, bus=b,
                pmin_pu=pmin / sn, pmax_pu=pmax / sn,
                qmin_pu=qmin / sn, qmax_pu=qmax / sn,
                c2=c2, c1=c1, c0=c0
            ))

    oltc_dir_set = set(cfg.oltc_branches.keys())

    E: List[Tuple[int, int]] = []
    r = {}
    x = {}
    Smax = {}
    ellmax = {}
    Eset = set()

    for lid in net.line.index:
        lid = int(lid)
        fb0 = int(net.line.at[lid, "from_bus"])
        tb0 = int(net.line.at[lid, "to_bus"])

        # enforce OLTC direction if this line is OLTC metadata edge
        if (fb0, tb0) in oltc_dir_set:
            fb, tb = fb0, tb0
        elif (tb0, fb0) in oltc_dir_set:
            fb, tb = tb0, fb0
        else:
            fb, tb = fb0, tb0

        if (fb, tb) in Eset:
            raise ValueError(f"Duplicate directed edge {(fb, tb)} found in net.line.")
        Eset.add((fb, tb))
        E.append((fb, tb))

        vn_kv = bus_vn_kv[fb]
        zb = _zbase_ohm(vn_kv, sn)

        r_ohm = float(net.line.at[lid, "r_ohm_per_km"]) * float(net.line.at[lid, "length_km"])
        x_ohm = float(net.line.at[lid, "x_ohm_per_km"]) * float(net.line.at[lid, "length_km"])
        r_pu = r_ohm / zb
        x_pu = x_ohm / zb

        r[(fb, tb)] = float(r_pu)
        x[(fb, tb)] = float(x_pu)

        Imax = float(net.line.at[lid, "max_i_ka"]) if "max_i_ka" in net.line.columns else 1e9
        Smax_mva = math.sqrt(3.0) * vn_kv * Imax
        Smax_pu = Smax_mva / sn
        Smax[(fb, tb)] = float(Smax_pu)

        vmin_sq = max(Vmin_pu[fb] ** 2, 1e-6)
        ellmax[(fb, tb)] = float((Smax_pu ** 2) / vmin_sq)

    out_arcs = {i: [] for i in buses}
    in_arcs = {i: [] for i in buses}
    for (i, j) in E:
        out_arcs[i].append((i, j))
        in_arcs[j].append((i, j))

    # OLTC constants
    T = []
    K = {}
    alpha_tap = {}
    delta_tap = {}

    E_dir_set = set(E)
    for (i, j), tcfg in cfg.oltc_branches.items():
        if (i, j) not in E_dir_set:
            raise ValueError(f"OLTC edge {(i, j)} not found in directed E.")
        T.append((i, j))
        taps = list(range(int(tcfg.tap_min), int(tcfg.tap_max) + 1))
        K[(i, j)] = taps
        for tap in taps:
            tau = 1.0 + (tap * float(tcfg.dV_percent)) / 100.0
            alpha_tap[((i, j), int(tap))] = 1.0 / tau
            delta_tap[((i, j), int(tap))] = 1.0 / (tau * tau)

    C = sorted([int(i) for i in cfg.shunt_bcap_pu.keys()])
    bcap = {int(i): float(cfg.shunt_bcap_pu[int(i)]) for i in C}

    return dict(
        sn_mva=sn,
        buses=buses,
        slack_bus=slack_bus,
        slack_vm_pu=slack_vm_pu,
        Vmin_pu=Vmin_pu,
        Vmax_pu=Vmax_pu,
        Pd_pu=Pd_pu,
        Qd_pu=Qd_pu,
        gen_records=gen_records,
        E=E,
        out_arcs=out_arcs,
        in_arcs=in_arcs,
        r=r,
        x=x,
        Smax=Smax,
        ellmax=ellmax,
        T=T,
        K=K,
        alpha_tap=alpha_tap,
        delta_tap=delta_tap,
        C=C,
        bcap=bcap,
        fix_slack_vm=cfg.fix_slack_vm,
    )


# ============================================================
# Subproblem (ell fixed): SOFT vdrop + SOFT KCL
# ============================================================
def build_subproblem(
    data: Dict[str, Any],
    ell_fix: Dict[Tuple[int, int], float],
    relax_binaries: bool = False,
    warm: Optional[Dict[str, Any]] = None
) -> pyo.ConcreteModel:

    sn = data["sn_mva"]
    buses = data["buses"]
    E = data["E"]
    out_arcs = data["out_arcs"]
    in_arcs = data["in_arcs"]

    slack_bus = data["slack_bus"]
    slack_vm_pu = data["slack_vm_pu"]

    Pd = data["Pd_pu"]
    Qd = data["Qd_pu"]
    Vmin = data["Vmin_pu"]
    Vmax = data["Vmax_pu"]

    r = data["r"]
    x = data["x"]
    Smax = data["Smax"]

    T = data["T"]
    K = data["K"]
    delta_tap = data["delta_tap"]
    alpha_tap = data["alpha_tap"]

    C = data["C"]
    bcap = data["bcap"]

    gen_records = data["gen_records"]
    Gset = list(range(len(gen_records)))

    T_set = set(T)
    C_set = set(C)

    m = pyo.ConcreteModel("BFM_ag_subproblem_softvdrop_softkcl")

    m.N = pyo.Set(initialize=buses, ordered=True)
    m.E = pyo.Set(initialize=E, dimen=2, ordered=True)
    m.T = pyo.Set(initialize=T, dimen=2, ordered=True)
    m.C = pyo.Set(initialize=C, ordered=True)
    m.G = pyo.Set(initialize=Gset, ordered=True)

    m.Pd = pyo.Param(m.N, initialize=lambda mm, i: float(Pd[int(i)]))
    m.Qd = pyo.Param(m.N, initialize=lambda mm, i: float(Qd[int(i)]))
    m.Vmin = pyo.Param(m.N, initialize=lambda mm, i: float(Vmin[int(i)]))
    m.Vmax = pyo.Param(m.N, initialize=lambda mm, i: float(Vmax[int(i)]))

    m.r = pyo.Param(m.E, initialize=lambda mm, i, j: float(r[(int(i), int(j))]))
    m.x = pyo.Param(m.E, initialize=lambda mm, i, j: float(x[(int(i), int(j))]))
    m.Smax = pyo.Param(m.E, initialize=lambda mm, i, j: float(Smax[(int(i), int(j))]))
    m.ell_fix = pyo.Param(m.E, initialize=lambda mm, i, j: float(ell_fix[(int(i), int(j))]))

    m.gen_bus = pyo.Param(m.G, initialize=lambda mm, g: int(gen_records[int(g)]["bus"]), within=pyo.Any)
    m.Pgmin = pyo.Param(m.G, initialize=lambda mm, g: float(gen_records[int(g)]["pmin_pu"]))
    m.Pgmax = pyo.Param(m.G, initialize=lambda mm, g: float(gen_records[int(g)]["pmax_pu"]))
    m.Qgmin = pyo.Param(m.G, initialize=lambda mm, g: float(gen_records[int(g)]["qmin_pu"]))
    m.Qgmax = pyo.Param(m.G, initialize=lambda mm, g: float(gen_records[int(g)]["qmax_pu"]))
    m.c2 = pyo.Param(m.G, initialize=lambda mm, g: float(gen_records[int(g)]["c2"]))
    m.c1 = pyo.Param(m.G, initialize=lambda mm, g: float(gen_records[int(g)]["c1"]))
    m.c0 = pyo.Param(m.G, initialize=lambda mm, g: float(gen_records[int(g)]["c0"]))

    m.bcap = pyo.Param(m.C, initialize=lambda mm, i: float(bcap[int(i)]))

    beta_index = []
    for (i, j) in T:
        for tap in K[(i, j)]:
            beta_index.append((int(i), int(j), int(tap)))
    m.BETA_INDEX = pyo.Set(initialize=beta_index, dimen=3, ordered=True)

    m.delta_tap = pyo.Param(
        m.BETA_INDEX,
        initialize=lambda mm, i, j, tap: float(delta_tap[((int(i), int(j)), int(tap))])
    )
    m.alpha_tap = pyo.Param(
        m.BETA_INDEX,
        initialize=lambda mm, i, j, tap: float(alpha_tap[((int(i), int(j)), int(tap))])
    )

    # vars
    m.Pg = pyo.Var(m.G, bounds=lambda mm, g: (mm.Pgmin[g], mm.Pgmax[g]))
    m.Qg = pyo.Var(m.G, bounds=lambda mm, g: (mm.Qgmin[g], mm.Qgmax[g]))
    m.v = pyo.Var(m.N, bounds=lambda mm, i: (mm.Vmin[i] ** 2, mm.Vmax[i] ** 2))

    m.Pij = pyo.Var(m.E, bounds=lambda mm, i, j: (-mm.Smax[i, j], mm.Smax[i, j]))
    m.Qij = pyo.Var(m.E, bounds=lambda mm, i, j: (-mm.Smax[i, j], mm.Smax[i, j]))
    m.Pinj = pyo.Var(m.N)
    m.Qinj = pyo.Var(m.N)

    # shunt
    if relax_binaries:
        m.a_sh = pyo.Var(m.C, bounds=(0.0, 1.0))
    else:
        m.a_sh = pyo.Var(m.C, within=pyo.Binary)
    m.z = pyo.Var(m.C, bounds=lambda mm, i: (0.0, mm.Vmax[int(i)] ** 2))

    # oltc
    if relax_binaries:
        m.beta = pyo.Var(m.BETA_INDEX, bounds=(0.0, 1.0))
    else:
        m.beta = pyo.Var(m.BETA_INDEX, within=pyo.Binary)

    def _tv_bounds(mm, i, j, tap):
        return (0.0, mm.Vmax[int(i)] ** 2)
    m.tv = pyo.Var(m.BETA_INDEX, bounds=_tv_bounds)

    # SOFT vdrop slack
    m.s_vdrop = pyo.Var(m.E, within=pyo.NonNegativeReals)

    # SOFT KCL slacks
    m.sP_pos = pyo.Var(m.N, within=pyo.NonNegativeReals)
    m.sP_neg = pyo.Var(m.N, within=pyo.NonNegativeReals)
    m.sQ_pos = pyo.Var(m.N, within=pyo.NonNegativeReals)
    m.sQ_neg = pyo.Var(m.N, within=pyo.NonNegativeReals)

    # slack fix
    if FIX_SLACK_VM and data["fix_slack_vm"]:
        m.slack_v = pyo.Constraint(expr=m.v[slack_bus] == float(slack_vm_pu) ** 2)

    # Pinj/Qinj
    def _Pinj(mm, i):
        gen_sum = sum(mm.Pg[g] for g in mm.G if int(mm.gen_bus[g]) == int(i))
        return mm.Pinj[i] == gen_sum - mm.Pd[i]
    m.Pinj_def = pyo.Constraint(m.N, rule=_Pinj)

    def _qsh_expr(mm, i):
        i = int(i)
        if i not in C_set:
            return 0.0
        return mm.bcap[i] * mm.z[i]
    m.qsh = pyo.Expression(m.N, rule=_qsh_expr)

    def _Qinj(mm, i):
        gen_sum = sum(mm.Qg[g] for g in mm.G if int(mm.gen_bus[g]) == int(i))
        return mm.Qinj[i] == gen_sum - mm.Qd[i] + mm.qsh[i]
    m.Qinj_def = pyo.Constraint(m.N, rule=_Qinj)

    # McCormick for z = a*v
    m.con_sh = pyo.ConstraintList()
    for i in C:
        vL = float(Vmin[i] ** 2)
        vU = float(Vmax[i] ** 2)
        m.con_sh.add(m.z[i] <= vU * m.a_sh[i])
        m.con_sh.add(m.z[i] >= vL * m.a_sh[i])
        m.con_sh.add(m.z[i] <= m.v[i] - vL * (1 - m.a_sh[i]))
        m.con_sh.add(m.z[i] >= m.v[i] - vU * (1 - m.a_sh[i]))

    # OLTC one-hot
    def _onehot(mm, i, j):
        taps = K[(int(i), int(j))]
        return sum(mm.beta[int(i), int(j), int(t)] for t in taps) == 1
    m.onehot = pyo.Constraint(m.T, rule=_onehot)

    # McCormick for tv = beta*v_i
    m.con_tv = pyo.ConstraintList()
    for (i, j, tap) in m.BETA_INDEX:
        vL = float(Vmin[i] ** 2)
        vU = float(Vmax[i] ** 2)
        m.con_tv.add(m.tv[i, j, tap] <= vU * m.beta[i, j, tap])
        m.con_tv.add(m.tv[i, j, tap] >= vL * m.beta[i, j, tap])
        m.con_tv.add(m.tv[i, j, tap] <= m.v[i] - vL * (1 - m.beta[i, j, tap]))
        m.con_tv.add(m.tv[i, j, tap] >= m.v[i] - vU * (1 - m.beta[i, j, tap]))

    def _vsend(mm, i, j):
        if (int(i), int(j)) not in T_set:
            return mm.v[int(i)]
        taps = K[(int(i), int(j))]
        return sum(mm.delta_tap[int(i), int(j), int(t)] * mm.tv[int(i), int(j), int(t)] for t in taps)

    # SOFT KCL
    def _bfmP(mm, i):
        i = int(i)
        out_sum = sum(mm.Pij[a, b] for (a, b) in out_arcs[i])
        in_sum  = sum((mm.Pij[a, b] - mm.r[a, b] * mm.ell_fix[a, b]) for (a, b) in in_arcs[i])
        return out_sum - in_sum + mm.sP_pos[i] - mm.sP_neg[i] == mm.Pinj[i]
    m.BFM_P = pyo.Constraint(m.N, rule=_bfmP)

    def _bfmQ(mm, i):
        i = int(i)
        out_sum = sum(mm.Qij[a, b] for (a, b) in out_arcs[i])
        in_sum  = sum((mm.Qij[a, b] - mm.x[a, b] * mm.ell_fix[a, b]) for (a, b) in in_arcs[i])
        return out_sum - in_sum + mm.sQ_pos[i] - mm.sQ_neg[i] == mm.Qinj[i]
    m.BFM_Q = pyo.Constraint(m.N, rule=_bfmQ)

    # SOFT voltage drop
    m.con_vdrop = pyo.ConstraintList()
    for (i, j) in E:
        i = int(i)
        j = int(j)
        rij = m.r[i, j]
        xij = m.x[i, j]
        z2 = rij * rij + xij * xij
        rhs = _vsend(m, i, j) - 2.0 * (rij * m.Pij[i, j] + xij * m.Qij[i, j]) + z2 * m.ell_fix[i, j]
        res = m.v[j] - rhs
        m.con_vdrop.add(res <= m.s_vdrop[i, j])
        m.con_vdrop.add(-res <= m.s_vdrop[i, j])

    # thermal (convex QC)
    def _thermal(mm, i, j):
        i, j = int(i), int(j)
        return mm.Pij[i, j] ** 2 + mm.Qij[i, j] ** 2 <= (mm.Smax[i, j] ** 2)
    m.Thermal = pyo.Constraint(m.E, rule=_thermal)

    # objective + penalties
    m.obj = pyo.Objective(
        expr=sum(
            m.c2[g] * (sn * m.Pg[g]) ** 2 + m.c1[g] * (sn * m.Pg[g]) + m.c0[g]
            for g in m.G
        )
        + float(RHO_VDROP) * sum(m.s_vdrop[i, j] for (i, j) in m.E)
        + float(RHO_KCL) * sum(m.sP_pos[i] + m.sP_neg[i] + m.sQ_pos[i] + m.sQ_neg[i] for i in m.N),
        sense=pyo.minimize
    )

    # Warm start
    if warm is not None:
        if "v" in warm:
            for i in buses:
                if i in warm["v"]:
                    m.v[i].value = _clip(float(warm["v"][i]), m.v[i].lb, m.v[i].ub)

        if "Pg" in warm:
            for g in Gset:
                if g in warm["Pg"]:
                    m.Pg[g].value = _clip(float(warm["Pg"][g]), m.Pg[g].lb, m.Pg[g].ub)

        if "Qg" in warm:
            for g in Gset:
                if g in warm["Qg"]:
                    m.Qg[g].value = _clip(float(warm["Qg"][g]), m.Qg[g].lb, m.Qg[g].ub)

        if "Pij" in warm:
            for (i, j) in E:
                if (i, j) in warm["Pij"]:
                    m.Pij[i, j].value = _clip(float(warm["Pij"][(i, j)]), m.Pij[i, j].lb, m.Pij[i, j].ub)

        if "Qij" in warm:
            for (i, j) in E:
                if (i, j) in warm["Qij"]:
                    m.Qij[i, j].value = _clip(float(warm["Qij"][(i, j)]), m.Qij[i, j].lb, m.Qij[i, j].ub)

        if "a_sh" in warm:
            for i in C:
                if i in warm["a_sh"]:
                    val = float(warm["a_sh"][i])
                    if relax_binaries:
                        m.a_sh[i].value = _clip(val, 0.0, 1.0)
                    else:
                        m.a_sh[i].value = 1 if val >= 0.5 else 0

        if "beta" in warm:
            if relax_binaries:
                for (i, j, tap) in m.BETA_INDEX:
                    if (i, j, tap) in warm["beta"]:
                        m.beta[i, j, tap].value = _clip(float(warm["beta"][(i, j, tap)]), 0.0, 1.0)
            else:
                rounded_beta = _round_onehot_from_warm(data, warm["beta"])
                for (i, j, tap) in m.BETA_INDEX:
                    m.beta[i, j, tap].value = int(rounded_beta.get((i, j, tap), 0))

    m._data = data
    return m


# ============================================================
# Solve with SCIP
# ============================================================
def solve_with_scip(model: pyo.ConcreteModel, timelimit: float, mipgap: float, tee: bool) -> bool:
    opt = pyo.SolverFactory("scip", solver_io="nl")
    if opt is None or not opt.available(exception_flag=False):
        opt = pyo.SolverFactory("scip")
    if opt is None or not opt.available(exception_flag=False):
        return False

    try:
        opt.options["limits/time"] = float(timelimit)
        opt.options["limits/gap"] = float(mipgap)
        opt.options["limits/nodes"] = int(SCIP_NODE_LIMIT)
        opt.options["limits/memory"] = float(SCIP_MEMORY_LIMIT_MB)
        opt.options["numerics/feastol"] = float(SCIP_FEASTOL)
        opt.options["numerics/dualfeastol"] = float(SCIP_DUALFEASTOL)
        opt.options["display/verblevel"] = 4 if tee else 0
    except Exception:
        pass

    res = opt.solve(model, tee=tee, load_solutions=True)
    tc = res.solver.termination_condition

    if tc == TerminationCondition.infeasible:
        return False
    return _solution_complete(model, model._data)


def solve_subproblem_robust(
    data: Dict[str, Any],
    ell_fix: Dict[Tuple[int, int], float],
    warm: Optional[Dict[str, Any]],
    tee: bool
) -> Tuple[pyo.ConcreteModel, str]:

    m_bin = build_subproblem(data, ell_fix, relax_binaries=False, warm=warm)
    if solve_with_scip(m_bin, SCIP_TIME_LIMIT, SCIP_GAP_LIMIT, tee):
        return m_bin, "binary"

    if tee:
        print("[WARN] binary infeasible/no incumbent -> RELAXED ...")

    m_relax = build_subproblem(data, ell_fix, relax_binaries=True, warm=warm)
    if solve_with_scip(m_relax, SCIP_TIME_LIMIT, SCIP_GAP_LIMIT, False):
        return m_relax, "relaxed_only"

    raise RuntimeError("RELAXED also infeasible/no-solution (even with soft vdrop + soft KCL).")


# ============================================================
# Theta recovery (LS on meshed graph)
# ============================================================
def recover_theta_ls(
    buses: List[int],
    edges: List[Tuple[int, int]],
    d_edge: Dict[Tuple[int, int], float],
    slack: int,
    ridge: float = 1e-8
) -> Dict[int, float]:

    n = len(buses)
    idx = {b: k for k, b in enumerate(buses)}
    L = np.zeros((n, n), dtype=float)
    bvec = np.zeros(n, dtype=float)

    for (i, j) in edges:
        ii = idx[int(i)]
        jj = idx[int(j)]
        d = float(d_edge[(i, j)])
        w = 1.0
        L[ii, ii] += w
        L[jj, jj] += w
        L[ii, jj] -= w
        L[jj, ii] -= w
        bvec[ii] += w * d
        bvec[jj] -= w * d

    s = idx[int(slack)]
    keep = [k for k in range(n) if k != s]
    Lr = L[np.ix_(keep, keep)] + ridge * np.eye(len(keep))
    br = bvec[keep]

    try:
        xr = np.linalg.solve(Lr, br)
    except np.linalg.LinAlgError:
        xr = np.linalg.lstsq(Lr, br, rcond=None)[0]

    theta = {int(b): 0.0 for b in buses}
    for kk, k in enumerate(keep):
        theta[int(buses[k])] = float(xr[kk])
    theta[int(slack)] = 0.0
    return theta


# ============================================================
# Objective breakdown
# ============================================================
def objective_breakdown(model: pyo.ConcreteModel, data: Dict[str, Any]) -> Dict[str, float]:
    sn = float(data["sn_mva"])
    gen_cost = 0.0
    for g in model.G:
        Pg = _val(model.Pg[g], 0.0)
        gen_cost += (
            float(_pval(model.c2[g])) * (sn * Pg) ** 2
            + float(_pval(model.c1[g])) * (sn * Pg)
            + float(_pval(model.c0[g]))
        )

    pen_vdrop = float(RHO_VDROP) * sum(_val(model.s_vdrop[i, j], 0.0) for (i, j) in data["E"])
    pen_kcl = float(RHO_KCL) * sum(
        _val(model.sP_pos[i], 0.0) + _val(model.sP_neg[i], 0.0)
        + _val(model.sQ_pos[i], 0.0) + _val(model.sQ_neg[i], 0.0)
        for i in data["buses"]
    )
    total = gen_cost + pen_vdrop + pen_kcl
    return dict(gen_cost=gen_cost, pen_vdrop=pen_vdrop, pen_kcl=pen_kcl, total=total)


# ============================================================
# Outer loop: BFM-ag
# ============================================================
def run_bfm_ag(data: Dict[str, Any], max_iters: int, eps: float, tee: bool) -> Dict[str, Any]:
    buses = data["buses"]
    E = data["E"]
    slack = int(data["slack_bus"])
    T_set = set(data["T"])
    ellmax = data["ellmax"]

    ell_prev = {(i, j): 0.0 for (i, j) in E}
    theta_prev = {int(b): 0.0 for b in buses}
    theta_prev[slack] = 0.0

    warm = {
        "v": {i: 1.0 for i in buses},
        "Pij": {(i, j): 0.0 for (i, j) in E},
        "Qij": {(i, j): 0.0 for (i, j) in E},
        "Pg": {},
        "Qg": {},
        "beta": {},
        "a_sh": {},
    }

    best = {
        "iter": 0,
        "gen_cost": float("inf"),
        "total": float("inf"),
        "model": None,
        "ell": None,
        "theta": None,
        "tag": ""
    }
    last = {"iter": 0, "model": None, "ell": None, "theta": None, "tag": ""}

    for t in range(1, max_iters + 1):
        t0 = time.perf_counter()

        model, tag = solve_subproblem_robust(data, ell_prev, warm=warm, tee=tee)

        # read solution
        v_sol = {int(i): _val(model.v[i], 1.0) for i in buses}
        P_sol = {(i, j): _val(model.Pij[i, j], 0.0) for (i, j) in E}
        Q_sol = {(i, j): _val(model.Qij[i, j], 0.0) for (i, j) in E}

        # update warm
        warm["v"] = dict(v_sol)
        warm["Pij"] = dict(P_sol)
        warm["Qij"] = dict(Q_sol)
        warm["Pg"] = {int(g): _val(model.Pg[g], 0.0) for g in model.G}
        warm["Qg"] = {int(g): _val(model.Qg[g], 0.0) for g in model.G}
        warm["a_sh"] = {int(i): _val(model.a_sh[i], 0.0) for i in data["C"]} if len(data["C"]) > 0 else {}
        warm["beta"] = {(i, j, tap): _val(model.beta[i, j, tap], 0.0) for (i, j, tap) in model.BETA_INDEX} if len(data["T"]) > 0 else {}

        # edge target dtheta
        d_edge = {}
        for (i, j) in E:
            i = int(i)
            j = int(j)
            rij = float(data["r"][(i, j)])
            xij = float(data["x"][(i, j)])

            if (i, j) in T_set:
                tap = _pick_tap_from_beta(model, data, i, j)
                delt = _delta_of_selected_tap(data, i, j, tap)
                vsend = max(delt * v_sol[i], DENOM_EPS)
            else:
                vsend = max(v_sol[i], DENOM_EPS)

            vj = max(v_sol[j], DENOM_EPS)
            denom = math.sqrt(vsend * vj)

            dprev = float(theta_prev[i] - theta_prev[j])
            dprev = float(np.clip(dprev, -CLIP_DTHETA_PREV, CLIP_DTHETA_PREV))

            rhs = (xij * P_sol[(i, j)] - rij * Q_sol[(i, j)]) / denom
            d_edge[(i, j)] = float(rhs - math.sin(dprev) + dprev)

        # theta LS recovery + damping
        theta_ls = recover_theta_ls(buses=buses, edges=E, d_edge=d_edge, slack=slack, ridge=THETA_RIDGE)
        theta_new = {}
        for b in buses:
            thp = float(theta_prev[int(b)])
            thn = float(theta_ls[int(b)])
            theta_new[int(b)] = (1.0 - THETA_GAMMA) * thp + THETA_GAMMA * thn
        theta_new[slack] = 0.0

        # ell update
        ell_new = {}
        sum_diff = 0.0
        max_viol = 0.0
        for (i, j) in E:
            i = int(i)
            j = int(j)
            P = P_sol[(i, j)]
            Q = Q_sol[(i, j)]
            S2 = P * P + Q * Q

            if (i, j) in T_set:
                tap = _pick_tap_from_beta(model, data, i, j)
                delt = _delta_of_selected_tap(data, i, j, tap)
                vsend = max(delt * v_sol[i], DENOM_EPS)
            else:
                vsend = max(v_sol[i], DENOM_EPS)

            ell_calc = S2 / vsend
            ell_damped = (1.0 - ELL_GAMMA) * ell_prev[(i, j)] + ELL_GAMMA * ell_calc
            ell_clamped = min(max(ell_damped, 0.0), float(ellmax[(i, j)]))

            ell_new[(i, j)] = ell_clamped
            sum_diff += abs(ell_clamped - ell_prev[(i, j)])

            viol = 0.0
            if ell_clamped < -1e-8:
                viol = -ell_clamped
            if ell_clamped > float(ellmax[(i, j)]) + 1e-8:
                viol = ell_clamped - float(ellmax[(i, j)])
            max_viol = max(max_viol, viol)

        max_vdrop_slack = max(_val(model.s_vdrop[i, j], 0.0) for (i, j) in E)
        max_kcl_slack = max(
            _val(model.sP_pos[i], 0.0) + _val(model.sP_neg[i], 0.0)
            + _val(model.sQ_pos[i], 0.0) + _val(model.sQ_neg[i], 0.0)
            for i in buses
        )

        obj = objective_breakdown(model, data)
        t1 = time.perf_counter()
        print(
            f"[t={t:02d}] tag={tag:>14s}  "
            f"gen_cost={obj['gen_cost']:,.6f}  total={obj['total']:,.6f}  "
            f"sum|dell|={sum_diff:.3e}  max_ell_viol={max_viol:.3e}  "
            f"max_vdrop_slack={max_vdrop_slack:.3e}  max_kcl_slack={max_kcl_slack:.3e}  "
            f"time={t1-t0:.2f}s"
        )

        last.update({"iter": t, "model": model, "ell": ell_new, "theta": theta_new, "tag": tag})

        if obj["gen_cost"] < best["gen_cost"]:
            best.update({
                "iter": t,
                "gen_cost": obj["gen_cost"],
                "total": obj["total"],
                "model": model,
                "ell": ell_new,
                "theta": theta_new,
                "tag": tag
            })

        if (sum_diff <= eps) and (max_viol <= 1e-8):
            print(f"[CONVERGED] t={t}, eps={eps}")
            break

        ell_prev = ell_new
        theta_prev = theta_new

    return {"best": best, "last": last}


# ============================================================
# Reporting
# ============================================================
def print_report(
    title: str,
    model: pyo.ConcreteModel,
    data: Dict[str, Any],
    ell: Dict[Tuple[int, int], float],
    theta: Optional[Dict[int, float]] = None
):
    sn = float(data["sn_mva"])
    buses = data["buses"]
    E = data["E"]
    T_set = set(data["T"])
    slack = int(data["slack_bus"])

    obj = objective_breakdown(model, data)

    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)
    print("Objective breakdown (EUR):")
    print(f"  gen_cost  = {obj['gen_cost']:,.6f}")
    print(f"  pen_vdrop = {obj['pen_vdrop']:,.6f}")
    print(f"  pen_kcl   = {obj['pen_kcl']:,.6f}")
    print(f"  total     = {obj['total']:,.6f}")

    max_vdrop_slack = max(_val(model.s_vdrop[i, j], 0.0) for (i, j) in E)
    max_kcl_slack = max(
        _val(model.sP_pos[i], 0.0) + _val(model.sP_neg[i], 0.0)
        + _val(model.sQ_pos[i], 0.0) + _val(model.sQ_neg[i], 0.0)
        for i in buses
    )
    print(f"  max_vdrop_slack = {max_vdrop_slack:.6e}")
    print(f"  max_kcl_slack   = {max_kcl_slack:.6e}")

    print("\n--- Bus Voltages ---")
    for i in buses:
        v = _val(model.v[i], 1.0)
        V = math.sqrt(max(v, 0.0))
        tag = " [slack]" if i == slack else ""
        print(f"  bus {i:2d}: v={v:.6f}, |V|={V:.6f}{tag}")

    if theta is not None:
        print("\n--- Bus Angles (theta) ---")
        for i in buses:
            th = float(theta.get(int(i), 0.0))
            print(f"  bus {i:2d}: theta={th:+.6f} rad  ({math.degrees(th):+.6f} deg)")

    print("\n--- Generators (MW / MVAr) ---")
    for g in model.G:
        rec = data["gen_records"][int(g)]
        Pg_mw = sn * _val(model.Pg[g], 0.0)
        Qg_mvar = sn * _val(model.Qg[g], 0.0)
        print(f"  {rec['type']}[{rec['id']}] @ bus {rec['bus']:2d}: P={Pg_mw:+.4f} MW, Q={Qg_mvar:+.4f} MVAr")

    if len(data["T"]) > 0:
        print("\n--- OLTC taps (chosen) ---")
        for (i, j) in data["T"]:
            tap = _pick_tap_from_beta(model, data, i, j)
            alpha = _alpha_of_selected_tap(data, i, j, tap)
            delta = _delta_of_selected_tap(data, i, j, tap)
            tau = 1.0 / alpha
            print(f"  ({i}->{j}): tap={tap:>3d}, tau={tau:.6f}, alpha={alpha:.6f}, delta={delta:.6f}")

    if len(data["C"]) > 0:
        print("\n--- Switched Shunts ---")
        for i in data["C"]:
            a = _val(model.a_sh[i], 0.0)
            z = _val(model.z[i], a * _val(model.v[i], 1.0))
            qpu = float(data["bcap"][i]) * z
            print(f"  bus {i:2d}: a_sh={a:.6f}, qsh={sn*qpu:+.4f} MVAr (pu={qpu:+.6f})")

    kcl_list = []
    for i in buses:
        sp = _val(model.sP_pos[i], 0.0) + _val(model.sP_neg[i], 0.0)
        sq = _val(model.sQ_pos[i], 0.0) + _val(model.sQ_neg[i], 0.0)
        kcl_list.append((max(sp, sq), sp, sq, i))
    kcl_list.sort(reverse=True)
    print("\n--- KCL slack (top 10 buses) ---")
    for k, sp, sq, i in kcl_list[:10]:
        print(f"  bus {i:2d}: slackP={sp:.3e}, slackQ={sq:.3e}")

    print("\n--- Branch flows (P,Q,|S|) and ell_next ---")
    for (i, j) in E:
        P = _val(model.Pij[i, j], 0.0)
        Q = _val(model.Qij[i, j], 0.0)
        Smag = math.sqrt(max(P * P + Q * Q, 0.0))

        vi = _val(model.v[i], 1.0)
        if (i, j) in T_set:
            tap = _pick_tap_from_beta(model, data, i, j)
            delt = _delta_of_selected_tap(data, i, j, tap)
            denom = max(delt * vi, DENOM_EPS)
        else:
            denom = max(vi, DENOM_EPS)

        ell_calc = (P * P + Q * Q) / denom
        s_vdrop = _val(model.s_vdrop[i, j], 0.0)

        print(
            f"  ({i:2d}->{j:2d})  P={P:+.6f} pu  Q={Q:+.6f} pu  |S|={Smag:.6f} pu  "
            f"ell_calc={ell_calc:.6f}  ell_next={ell[(i, j)]:.6f}  vdrop_slack={s_vdrop:.3e}"
        )


# ============================================================
# Main
# ============================================================
def main():
    t0 = time.perf_counter()

    net = mcase.busmeshed39_opf(**NETWORK_BUILD_KWARGS)

    oltc_cfg = read_network_oltc_metadata(net, active_oltc_edges=ACTIVE_OLTC_EDGES)
    shunt_cfg = read_network_shunt_metadata(net, active_shunt_buses=ACTIVE_SHUNT_BUSES)

    cfg = BuildConfig(
        oltc_branches=oltc_cfg,
        shunt_bcap_pu=shunt_cfg,
        fix_slack_vm=True
    )

    data = extract_data_meshed_keep_lines(net, cfg)

    print("[INFO] Data summary (modified explicit 41-bus, keep all lines)")
    print(f"  #buses  = {len(data['buses'])}")
    print(f"  #lines  = {len(data['E'])}  (directed, one per net.line row)")
    print(f"  #gens   = {len(data['gen_records'])}")
    print(f"  #OLTC   = {len(data['T'])}  (beta vars = {sum(len(data['K'][ij]) for ij in data['T'])})")
    print(f"  #shunts = {len(data['C'])}")
    print(f"  outer: max_iters={OUTER_MAX_ITERS}, eps={OUTER_EPS}")
    print(f"  ell_gamma={ELL_GAMMA}, theta_gamma={THETA_GAMMA}, ridge={THETA_RIDGE}")
    print(f"  penalties: rho_vdrop={RHO_VDROP:g}, rho_kcl={RHO_KCL:g}")

    sol = run_bfm_ag(data, max_iters=OUTER_MAX_ITERS, eps=OUTER_EPS, tee=TEE_SOLVER_LOG)

    best = sol["best"]
    last = sol["last"]

    print("\n[SOLVED] Summary")
    print(f"  best(iter={best['iter']})  gen_cost={best['gen_cost']:.6f}  total={best['total']:.6f}  tag={best['tag']}")
    print(f"  last(iter={last['iter']})  tag={last['tag']}")

    if best["model"] is not None:
        print_report(
            title=f"BEST (by generation cost) @ iter={best['iter']}  tag={best['tag']}",
            model=best["model"],
            data=data,
            ell=best["ell"],
            theta=best["theta"],
        )

    if last["model"] is not None and last["iter"] != best["iter"]:
        print_report(
            title=f"LAST ITERATE @ iter={last['iter']}  tag={last['tag']}",
            model=last["model"],
            data=data,
            ell=last["ell"],
            theta=last["theta"],
        )

    t1 = time.perf_counter()
    print(f"\n[INFO] Total elapsed time: {t1 - t0:.2f} seconds.")


if __name__ == "__main__":
    main()