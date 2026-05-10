# BFM_it.py
# ------------------------------------------------------------
# Pyomo MIQCP BFM-it for modified explicit 41-bus system
# (meshed allowed, NO line dropping)
#
# FIXES APPLIED:
#   (1) Robust infeasible handling:
#       - If SCIP returns infeasible / no incumbent, we DO NOT read vsend/tv.
#       - We fall back to: relax -> round -> fix (QCQP) to get an incumbent.
#       - Outer loop has backtracking on ell_fix (blend toward last-feasible ell)
#         when a subproblem becomes infeasible.
#
#   (2) Warm-start clipping:
#       - All warm-start values (v, Pg, Qg, beta, a_sh, Pij, Qij) are clipped to bounds
#         to avoid Pyomo W1002 bound warnings and to improve solver robustness.
#
#   (3) SCIP feasibility tolerance:
#       - Set numerics/feastol a bit looser to avoid rejecting incumbents
#         due to ~1e-6 level residuals.
#
# Network module:
#   ieee39busplus_modified_explicit.py
# ------------------------------------------------------------

import math
import time
from dataclasses import dataclass
from typing import Dict, Tuple, Any, Optional, List

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

# Build kwargs for modified explicit 41-bus network
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

# Outer iteration
OUTER_MAX_ITERS = 30
OUTER_EPS = 1e-5
DENOM_EPS = 1e-10

# damping for ell update (1.0 = no damping)
ELL_DAMPING_GAMMA = 0.7

# Per-iteration solver limits (SCIP)
TIME_LIMIT_BINARY = 120.0
TIME_LIMIT_RELAX = 60.0
TIME_LIMIT_FIXED = 60.0
MIP_GAP = 1e-4
NODE_LIMIT = 300000
MEM_LIMIT_MB = 8192

# Termination feasibility tolerance for ell
ELL_VIOL_TOL = 1e-8

# Backtracking on ell_fix when a subproblem is infeasible
BACKTRACK_MAX_TRIES = 6
BACKTRACK_BLEND = 0.5

# SCIP numerical tolerance
SCIP_FEASTOL = 1e-5


# ============================================================
# Helpers
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


def _default_tap_choice(taps: List[int]) -> int:
    return min(taps, key=lambda t: abs(int(t)))


def _has_loaded_solution(model: pyo.ConcreteModel) -> bool:
    for v in model.component_data_objects(pyo.Var, active=True, descend_into=True):
        if v.value is not None:
            return True
    return False


def _clip(val: float, lo: Optional[float], hi: Optional[float]) -> float:
    if (lo is not None) and (val < lo):
        return float(lo)
    if (hi is not None) and (val > hi):
        return float(hi)
    return float(val)


def _set_init_clipped(var: pyo.Var, idx, val: float):
    v = var[idx]
    lb = v.lb
    ub = v.ub
    v.value = _clip(float(val), lb, ub)


def _safe_value(x, default: float = 0.0) -> float:
    try:
        v = pyo.value(x)
        if v is None:
            return float(default)
        return float(v)
    except Exception:
        return float(default)


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
        raise ValueError("No active OLTC branches found from network metadata.")

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
        raise ValueError("No active shunt buses found from network metadata.")

    return shunt_bcap_pu


# ============================================================
# Data extraction (meshed, keep ALL lines, directed as net.line rows)
# ============================================================
def extract_bfmit_per_unit_data_meshed(net, cfg: BuildConfig) -> Dict[str, Any]:
    sn = float(net.sn_mva)

    buses = [int(i) for i in net.bus.index]
    bus_vn_kv = {int(i): float(net.bus.at[i, "vn_kv"]) for i in buses}
    Vmin_pu = {int(i): float(net.bus.at[i, "min_vm_pu"]) for i in buses}
    Vmax_pu = {int(i): float(net.bus.at[i, "max_vm_pu"]) for i in buses}

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
            "type": "ext_grid", "id": eg, "bus": b,
            "pmin_pu": pmin / sn, "pmax_pu": pmax / sn,
            "qmin_pu": qmin / sn, "qmax_pu": qmax / sn,
            "c2": c2, "c1": c1, "c0": c0
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
                "type": "gen", "id": gi, "bus": b,
                "pmin_pu": pmin / sn, "pmax_pu": pmax / sn,
                "qmin_pu": qmin / sn, "qmax_pu": qmax / sn,
                "c2": c2, "c1": c1, "c0": c0
            })

    oltc_dir_set = set(cfg.oltc_branches.keys())

    E: List[Tuple[int, int]] = []
    r: Dict[Tuple[int, int], float] = {}
    x: Dict[Tuple[int, int], float] = {}
    Smax: Dict[Tuple[int, int], float] = {}
    ellmax: Dict[Tuple[int, int], float] = {}

    E_set = set()

    for lid in net.line.index:
        lid = int(lid)
        fb0 = int(net.line.at[lid, "from_bus"])
        tb0 = int(net.line.at[lid, "to_bus"])

        if (fb0, tb0) in oltc_dir_set:
            fb, tb = fb0, tb0
        elif (tb0, fb0) in oltc_dir_set:
            fb, tb = tb0, fb0
        else:
            fb, tb = fb0, tb0

        if (fb, tb) in E_set:
            raise ValueError(
                f"Duplicate directed line row detected: {(fb, tb)}. "
                f"Your model dictionaries assume unique (from,to) per net.line row."
            )
        E_set.add((fb, tb))
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

    T = []
    K: Dict[Tuple[int, int], List[int]] = {}
    delta_tap: Dict[Tuple[Tuple[int, int], int], float] = {}
    alpha_tap: Dict[Tuple[Tuple[int, int], int], float] = {}

    E_dir_set = set(E)
    for (i, j), tcfg in cfg.oltc_branches.items():
        if (i, j) not in E_dir_set:
            raise ValueError(
                f"OLTC edge {(i, j)} not found in directed line list E. "
                f"Check net.line orientation and metadata."
            )
        T.append((i, j))
        taps = list(range(int(tcfg.tap_min), int(tcfg.tap_max) + 1))
        K[(i, j)] = taps
        for tap in taps:
            tau = 1.0 + (tap * float(tcfg.dV_percent)) / 100.0
            alpha_tap[((i, j), tap)] = 1.0 / tau
            delta_tap[((i, j), tap)] = 1.0 / (tau * tau)

    C = sorted([int(i) for i in cfg.shunt_bcap_pu.keys()])
    bcap = {int(i): float(cfg.shunt_bcap_pu[int(i)]) for i in C}

    return {
        "sn_mva": sn,
        "buses": buses,
        "slack_bus": slack_bus,
        "slack_vm_pu": slack_vm_pu,
        "Vmin_pu": Vmin_pu,
        "Vmax_pu": Vmax_pu,
        "Pd_pu": Pd_pu,
        "Qd_pu": Qd_pu,
        "gen_records": gen_records,
        "E": E,
        "out_arcs": out_arcs,
        "in_arcs": in_arcs,
        "r": r,
        "x": x,
        "Smax": Smax,
        "ellmax": ellmax,
        "T": T,
        "K": K,
        "alpha_tap": alpha_tap,
        "delta_tap": delta_tap,
        "C": C,
        "bcap": bcap,
        "fix_slack_vm": cfg.fix_slack_vm,
    }


# ============================================================
# Build MIQCP subproblem (ell fixed)
# ============================================================
def build_pyomo_bfmit_model(
    data: Dict[str, Any],
    ell_fix: Dict[Tuple[int, int], float],
    relax_binaries: bool = False,
    warm_start: Optional[Dict[str, Any]] = None,
) -> pyo.ConcreteModel:

    sn = data["sn_mva"]
    buses = data["buses"]
    E = data["E"]
    out_arcs = data["out_arcs"]
    in_arcs = data["in_arcs"]

    r = data["r"]
    x = data["x"]
    Smax = data["Smax"]

    T = data["T"]
    K = data["K"]
    delta_tap = data["delta_tap"]

    C = data["C"]
    bcap = data["bcap"]

    Pd = data["Pd_pu"]
    Qd = data["Qd_pu"]
    Vmin = data["Vmin_pu"]
    Vmax = data["Vmax_pu"]

    slack_bus = data["slack_bus"]
    slack_vm_pu = data["slack_vm_pu"]

    gen_records = data["gen_records"]
    Gset = list(range(len(gen_records)))

    T_set = set(T)
    C_set = set(C)

    m = pyo.ConcreteModel(name="BFMit_MIQCP_modified41")

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

    m.ell_fix = pyo.Param(
        m.E,
        initialize=lambda mm, i, j: float(ell_fix[(int(i), int(j))]),
        mutable=False
    )

    m.gen_bus = pyo.Param(m.G, initialize=lambda mm, g: int(gen_records[int(g)]["bus"]), within=pyo.Any)
    m.Pgmin = pyo.Param(m.G, initialize=lambda mm, g: float(gen_records[int(g)]["pmin_pu"]))
    m.Pgmax = pyo.Param(m.G, initialize=lambda mm, g: float(gen_records[int(g)]["pmax_pu"]))
    m.Qgmin = pyo.Param(m.G, initialize=lambda mm, g: float(gen_records[int(g)]["qmin_pu"]))
    m.Qgmax = pyo.Param(m.G, initialize=lambda mm, g: float(gen_records[int(g)]["qmax_pu"]))
    m.c2 = pyo.Param(m.G, initialize=lambda mm, g: float(gen_records[int(g)]["c2"]))
    m.c1 = pyo.Param(m.G, initialize=lambda mm, g: float(gen_records[int(g)]["c1"]))
    m.c0 = pyo.Param(m.G, initialize=lambda mm, g: float(gen_records[int(g)]["c0"]))

    m.bcap = pyo.Param(m.C, initialize=lambda mm, i: float(bcap[int(i)]), mutable=False)

    beta_index = []
    for (i, j) in T:
        for tap in K[(i, j)]:
            beta_index.append((int(i), int(j), int(tap)))
    m.BETA_INDEX = pyo.Set(initialize=beta_index, dimen=3, ordered=True)

    m.delta_tap = pyo.Param(
        m.BETA_INDEX,
        initialize=lambda mm, i, j, tap: float(delta_tap[((int(i), int(j)), int(tap))]),
        mutable=False
    )

    m.Pg = pyo.Var(m.G, bounds=lambda mm, g: (mm.Pgmin[g], mm.Pgmax[g]))
    m.Qg = pyo.Var(m.G, bounds=lambda mm, g: (mm.Qgmin[g], mm.Qgmax[g]))

    m.v = pyo.Var(m.N, bounds=lambda mm, i: (mm.Vmin[i] ** 2, mm.Vmax[i] ** 2))

    m.Pij = pyo.Var(m.E)
    m.Qij = pyo.Var(m.E)

    if relax_binaries:
        m.a_sh = pyo.Var(m.C, bounds=(0.0, 1.0))
    else:
        m.a_sh = pyo.Var(m.C, within=pyo.Binary)

    m.z = pyo.Var(m.C)
    m.qsh = pyo.Var(m.N)

    if relax_binaries:
        m.beta = pyo.Var(m.BETA_INDEX, bounds=(0.0, 1.0))
    else:
        m.beta = pyo.Var(m.BETA_INDEX, within=pyo.Binary)

    m.tv = pyo.Var(m.BETA_INDEX)

    m.Pinj = pyo.Var(m.N)
    m.Qinj = pyo.Var(m.N)

    if FIX_SLACK_VM and data["fix_slack_vm"]:
        m.slack_v = pyo.Constraint(expr=m.v[slack_bus] == float(slack_vm_pu) ** 2)

    def _Pinj_rule(mm, i):
        inj_gen = sum(mm.Pg[g] for g in mm.G if int(mm.gen_bus[g]) == int(i))
        return mm.Pinj[i] == inj_gen - mm.Pd[i]
    m.Pinj_def = pyo.Constraint(m.N, rule=_Pinj_rule)

    def _Qinj_rule(mm, i):
        inj_gen = sum(mm.Qg[g] for g in mm.G if int(mm.gen_bus[g]) == int(i))
        return mm.Qinj[i] == inj_gen - mm.Qd[i] + mm.qsh[i]
    m.Qinj_def = pyo.Constraint(m.N, rule=_Qinj_rule)

    m.con_sh = pyo.ConstraintList()
    for i in buses:
        if i in C_set:
            m.con_sh.add(m.qsh[i] == m.bcap[i] * m.z[i])
        else:
            m.con_sh.add(m.qsh[i] == 0.0)

    for i in C:
        vL = float(Vmin[i] ** 2)
        vU = float(Vmax[i] ** 2)
        m.con_sh.add(m.z[i] <= vU * m.a_sh[i])
        m.con_sh.add(m.z[i] >= vL * m.a_sh[i])
        m.con_sh.add(m.z[i] <= m.v[i] - vL * (1.0 - m.a_sh[i]))
        m.con_sh.add(m.z[i] >= m.v[i] - vU * (1.0 - m.a_sh[i]))

    def _onehot_rule(mm, i, j):
        taps = K[(int(i), int(j))]
        return sum(mm.beta[int(i), int(j), int(t)] for t in taps) == 1
    m.onehot = pyo.Constraint(m.T, rule=_onehot_rule)

    m.con_tv = pyo.ConstraintList()
    for (i, j, tap) in m.BETA_INDEX:
        vL = float(Vmin[i] ** 2)
        vU = float(Vmax[i] ** 2)
        m.con_tv.add(m.tv[i, j, tap] <= vU * m.beta[i, j, tap])
        m.con_tv.add(m.tv[i, j, tap] >= vL * m.beta[i, j, tap])
        m.con_tv.add(m.tv[i, j, tap] <= m.v[i] - vL * (1.0 - m.beta[i, j, tap]))
        m.con_tv.add(m.tv[i, j, tap] >= m.v[i] - vU * (1.0 - m.beta[i, j, tap]))

    def _vsend_rule(mm, i, j):
        if (int(i), int(j)) not in T_set:
            return mm.v[int(i)]
        taps = K[(int(i), int(j))]
        return sum(mm.delta_tap[int(i), int(j), int(t)] * mm.tv[int(i), int(j), int(t)] for t in taps)
    m.vsend = pyo.Expression(m.E, rule=_vsend_rule)

    def _bfmP(mm, i):
        i = int(i)
        out_sum = sum(mm.Pij[a, b] for (a, b) in out_arcs[i])
        in_sum = sum((mm.Pij[a, b] - mm.r[a, b] * mm.ell_fix[a, b]) for (a, b) in in_arcs[i])
        return out_sum - in_sum == mm.Pinj[i]
    m.BFM_P = pyo.Constraint(m.N, rule=_bfmP)

    def _bfmQ(mm, i):
        i = int(i)
        out_sum = sum(mm.Qij[a, b] for (a, b) in out_arcs[i])
        in_sum = sum((mm.Qij[a, b] - mm.x[a, b] * mm.ell_fix[a, b]) for (a, b) in in_arcs[i])
        return out_sum - in_sum == mm.Qinj[i]
    m.BFM_Q = pyo.Constraint(m.N, rule=_bfmQ)

    def _vdrop(mm, i, j):
        i, j = int(i), int(j)
        rij = mm.r[i, j]
        xij = mm.x[i, j]
        z2 = rij * rij + xij * xij
        return mm.v[j] == mm.vsend[i, j] - 2.0 * (rij * mm.Pij[i, j] + xij * mm.Qij[i, j]) + z2 * mm.ell_fix[i, j]
    m.Vdrop = pyo.Constraint(m.E, rule=_vdrop)

    def _thermal(mm, i, j):
        i, j = int(i), int(j)
        return mm.Pij[i, j] ** 2 + mm.Qij[i, j] ** 2 <= (mm.Smax[i, j] ** 2)
    m.Thermal = pyo.Constraint(m.E, rule=_thermal)

    m.obj = pyo.Objective(
        expr=sum(m.c2[g] * (sn * m.Pg[g]) ** 2 + m.c1[g] * (sn * m.Pg[g]) + m.c0[g] for g in m.G),
        sense=pyo.minimize
    )

    if warm_start is not None:
        if "v" in warm_start:
            for i in buses:
                if i in warm_start["v"]:
                    _set_init_clipped(m.v, i, warm_start["v"][i])

        if "Pg" in warm_start:
            for g in Gset:
                if g in warm_start["Pg"]:
                    _set_init_clipped(m.Pg, g, warm_start["Pg"][g])

        if "Qg" in warm_start:
            for g in Gset:
                if g in warm_start["Qg"]:
                    _set_init_clipped(m.Qg, g, warm_start["Qg"][g])

        if "Pij" in warm_start:
            for (i, j) in E:
                if (i, j) in warm_start["Pij"]:
                    m.Pij[i, j].value = float(warm_start["Pij"][(i, j)])

        if "Qij" in warm_start:
            for (i, j) in E:
                if (i, j) in warm_start["Qij"]:
                    m.Qij[i, j].value = float(warm_start["Qij"][(i, j)])

        if "a_sh" in warm_start:
            for i in C:
                if i in warm_start["a_sh"]:
                    val = float(warm_start["a_sh"][i])
                    val = _clip(val, 0.0, 1.0)
                    m.a_sh[i].value = val

        if "beta" in warm_start:
            for (i, j, tap) in m.BETA_INDEX:
                key = (i, j, tap)
                if key in warm_start["beta"]:
                    val = float(warm_start["beta"][key])
                    val = _clip(val, 0.0, 1.0)
                    m.beta[i, j, tap].value = val

    m._data = data
    return m


# ============================================================
# Solver helpers
# ============================================================
def _solve_with_scip(model: pyo.ConcreteModel, timelimit: float, mipgap: float, tee: bool) -> bool:
    opt = pyo.SolverFactory("scip", solver_io="nl")
    if opt is None or (not opt.available(exception_flag=False)):
        opt = pyo.SolverFactory("scip")
    if opt is None or (not opt.available(exception_flag=False)):
        return False

    try:
        opt.options["limits/time"] = float(timelimit)
        opt.options["limits/gap"] = float(mipgap)
        opt.options["limits/nodes"] = int(NODE_LIMIT)
        opt.options["limits/memory"] = float(MEM_LIMIT_MB)
        opt.options["numerics/feastol"] = float(SCIP_FEASTOL)
        opt.options["display/verblevel"] = 4 if tee else 0
    except Exception:
        pass

    res = opt.solve(model, tee=tee, load_solutions=True)
    tc = res.solver.termination_condition
    st = res.solver.status
    msg = (getattr(res.solver, "message", "") or "").strip()

    if tee:
        print(f"[INFO] SCIP status={st}, termination={tc}")
        if msg:
            print(f"[INFO] SCIP message: {msg}")

    if tc == TerminationCondition.infeasible:
        return False

    return _has_loaded_solution(model)


def solve_miqcp_robust(
    data: Dict[str, Any],
    ell_fix: Dict[Tuple[int, int], float],
    warm_start: Optional[Dict[str, Any]],
    tee: bool,
) -> Tuple[pyo.ConcreteModel, str]:
    m_bin = build_pyomo_bfmit_model(data, ell_fix, relax_binaries=False, warm_start=warm_start)
    ok_bin = _solve_with_scip(m_bin, TIME_LIMIT_BINARY, MIP_GAP, tee)
    if ok_bin:
        return m_bin, "binary"

    if tee:
        print("[WARN] No incumbent in binary solve -> solving RELAXED (continuous) ...")

    m_relax = build_pyomo_bfmit_model(data, ell_fix, relax_binaries=True, warm_start=warm_start)
    ok_relax = _solve_with_scip(m_relax, TIME_LIMIT_RELAX, MIP_GAP, False)
    if (not ok_relax) or (not _has_loaded_solution(m_relax)):
        return m_relax, "relaxed_only"

    tap_choice: Dict[Tuple[int, int], int] = {}
    for (i, j) in data["T"]:
        taps = data["K"][(i, j)]
        best_t, best_v = None, -1e100
        for t in taps:
            v = float(_safe_value(m_relax.beta[i, j, int(t)], 0.0))
            if v > best_v:
                best_v, best_t = v, int(t)
        if best_t is None:
            best_t = _default_tap_choice([int(t) for t in taps])
        tap_choice[(i, j)] = int(best_t)

    sh_choice: Dict[int, int] = {}
    for i in data["C"]:
        v = float(_safe_value(m_relax.a_sh[i], 0.0))
        sh_choice[int(i)] = 1 if v >= 0.5 else 0

    if tee:
        print("[INFO] Solving FIXED-discrete QCQP (from relax->round) ...")

    m_fix = build_pyomo_bfmit_model(data, ell_fix, relax_binaries=False, warm_start=warm_start)

    for (i, j) in data["T"]:
        pick = tap_choice[(i, j)]
        for t in data["K"][(i, j)]:
            val = 1.0 if int(t) == int(pick) else 0.0
            m_fix.beta[i, j, int(t)].fix(val)

    for i in data["C"]:
        m_fix.a_sh[i].fix(float(sh_choice[int(i)]))

    ok_fix = _solve_with_scip(m_fix, TIME_LIMIT_FIXED, MIP_GAP, tee)
    if ok_fix:
        return m_fix, "fixed_from_relax"

    if tee:
        print("[WARN] Fixed-discrete QCQP failed; returning RELAXED solution (continuous) to proceed safely.")
    return m_relax, "relaxed_only"


# ============================================================
# Outer iteration (BFM-it) with ell backtracking
# ============================================================
def run_bfmit(data: Dict[str, Any], max_iters: int, eps: float, tee: bool) -> Dict[str, Any]:
    buses = data["buses"]
    E = data["E"]
    T_set = set(data["T"])
    ellmax = data["ellmax"]

    ell_prev = {(i, j): 0.0 for (i, j) in E}
    ell_last_ok = {(i, j): 0.0 for (i, j) in E}

    warm = {
        "v": {i: 1.0 for i in buses},
        "Pij": {(i, j): 0.0 for (i, j) in E},
        "Qij": {(i, j): 0.0 for (i, j) in E},
        "Pg": {},
        "Qg": {},
        "beta": {},
        "a_sh": {},
    }

    best = {"iter": 0, "obj": float("inf"), "model": None, "ell": None, "tag": ""}

    for t in range(1, max_iters + 1):
        t0 = time.perf_counter()

        ell_try = dict(ell_prev)
        model = None
        tag = ""
        solved = False

        for bt in range(1, BACKTRACK_MAX_TRIES + 1):
            try:
                model_bt, tag_bt = solve_miqcp_robust(data, ell_try, warm_start=warm, tee=tee)
                if _has_loaded_solution(model_bt):
                    model, tag = model_bt, tag_bt
                    solved = True
                    break
            except Exception as e:
                if tee:
                    print(f"[WARN] Solve attempt failed (bt={bt}): {e}")

            ell_try = {
                e: (1.0 - BACKTRACK_BLEND) * float(ell_last_ok[e]) + BACKTRACK_BLEND * float(ell_try[e])
                for e in E
            }
            if tee:
                print(f"[WARN] Backtracking ell_fix (bt={bt}/{BACKTRACK_MAX_TRIES}) -> blending toward last feasible ell.")

        if not solved or model is None or (not _has_loaded_solution(model)):
            print(f"[FAIL] t={t}: could not obtain an incumbent after backtracking. Stopping.")
            break

        obj = float(pyo.value(model.obj))
        if obj < best["obj"]:
            best.update({"iter": t, "obj": obj, "model": model, "tag": tag})

        warm["v"] = {i: float(_safe_value(model.v[i], 1.0)) for i in buses}
        warm["Pij"] = {(i, j): float(_safe_value(model.Pij[i, j], 0.0)) for (i, j) in E}
        warm["Qij"] = {(i, j): float(_safe_value(model.Qij[i, j], 0.0)) for (i, j) in E}
        warm["Pg"] = {int(g): float(_safe_value(model.Pg[g], 0.0)) for g in model.G}
        warm["Qg"] = {int(g): float(_safe_value(model.Qg[g], 0.0)) for g in model.G}
        warm["a_sh"] = {int(i): float(_safe_value(model.a_sh[i], 0.0)) for i in model.C} if len(list(model.C)) > 0 else {}
        warm["beta"] = {(i, j, tap): float(_safe_value(model.beta[i, j, tap], 0.0)) for (i, j, tap) in model.BETA_INDEX} if len(list(model.T)) > 0 else {}

        ell_new = {}
        sum_diff = 0.0
        max_viol = 0.0

        for (i, j) in E:
            P = float(_safe_value(model.Pij[i, j], 0.0))
            Q = float(_safe_value(model.Qij[i, j], 0.0))
            S2 = P * P + Q * Q

            if (i, j) in T_set:
                denom = float(_safe_value(model.vsend[i, j], 1.0))
            else:
                denom = float(_safe_value(model.v[i], 1.0))

            denom = max(denom, DENOM_EPS)
            ell_raw = S2 / denom
            ell_raw = min(max(ell_raw, 0.0), float(ellmax[(i, j)]))

            ell_damped = (1.0 - ELL_DAMPING_GAMMA) * float(ell_try[(i, j)]) + ELL_DAMPING_GAMMA * ell_raw
            ell_clamped = min(max(ell_damped, 0.0), float(ellmax[(i, j)]))

            ell_new[(i, j)] = ell_clamped
            sum_diff += abs(ell_clamped - float(ell_try[(i, j)]))

            viol = 0.0
            if ell_clamped < -ELL_VIOL_TOL:
                viol = -ell_clamped
            if ell_clamped > float(ellmax[(i, j)]) + ELL_VIOL_TOL:
                viol = ell_clamped - float(ellmax[(i, j)])
            max_viol = max(max_viol, viol)

        t1 = time.perf_counter()
        print(
            f"[t={t:02d}] tag={tag:>14s}  obj={obj:,.6f}  "
            f"sum|dell|={sum_diff:.3e}  max_ell_viol={max_viol:.3e}  time={t1-t0:.2f}s"
        )

        ell_last_ok = dict(ell_try)

        if (sum_diff <= eps) and (max_viol <= ELL_VIOL_TOL):
            best["ell"] = ell_new
            best["model"] = model
            best["iter"] = t
            best["obj"] = obj
            best["tag"] = tag
            print(f"[CONVERGED] t={t}, eps={eps}")
            return best

        ell_prev = ell_new

    best["ell"] = ell_prev
    return best


# ============================================================
# Reporting
# ============================================================
def print_solution(model: pyo.ConcreteModel, data: Dict[str, Any], ell: Dict[Tuple[int, int], float]):
    sn = data["sn_mva"]
    buses = data["buses"]
    slack = data["slack_bus"]
    gen_records = data["gen_records"]
    T_set = set(data["T"])

    print("\n====================")
    print("BFM-it MIQCP RESULT (modified explicit 41-bus)")
    print("====================")
    print(f"Objective (EUR): {float(pyo.value(model.obj)):.6f}")

    print("\n--- Bus Voltages (pu) ---")
    for i in buses:
        v = float(_safe_value(model.v[i], 1.0))
        V = math.sqrt(max(v, 0.0))
        tag = " [slack]" if int(i) == int(slack) else ""
        print(f"Bus {i:2d}: V={V:.6f}{tag}")

    print("\n--- Generator Dispatch (MW / MVAr) ---")
    for g in model.G:
        rec = gen_records[int(g)]
        Pg_mw = sn * float(_safe_value(model.Pg[g], 0.0))
        Qg_mvar = sn * float(_safe_value(model.Qg[g], 0.0))
        print(f"{rec['type']}[{rec['id']}] @ bus {rec['bus']:2d}: P={Pg_mw:.4f} MW, Q={Qg_mvar:.4f} MVAr")

    if len(data["T"]) > 0:
        print("\n--- OLTC taps (chosen) ---")
        for (i, j) in data["T"]:
            taps = data["K"][(i, j)]
            best_t, best_v = None, -1.0
            for t in taps:
                vb = float(_safe_value(model.beta[i, j, int(t)], 0.0))
                if vb > best_v:
                    best_v, best_t = vb, int(t)
            delta = float(data["delta_tap"][((i, j), int(best_t))])
            alpha = float(data["alpha_tap"][((i, j), int(best_t))])
            print(f"OLTC ({i}->{j}): tap={best_t:>3d}  beta={best_v:.6f}  alpha={alpha:.6f}  delta={delta:.6f}")

    if len(data["C"]) > 0:
        print("\n--- Switched Shunts ---")
        for i in data["C"]:
            a = float(_safe_value(model.a_sh[i], 0.0))
            qpu = float(_safe_value(model.qsh[i], 0.0))
            print(f"Shunt @ bus {i:2d}: a_sh={a:.0f}  qsh={sn*qpu:.4f} MVAr")

    print("\n--- Branch flows & ell (next) ---")
    for (i, j) in data["E"]:
        P = float(_safe_value(model.Pij[i, j], 0.0))
        Q = float(_safe_value(model.Qij[i, j], 0.0))
        Smag = math.sqrt(max(P * P + Q * Q, 0.0))
        denom = float(_safe_value(model.vsend[i, j], 1.0)) if (i, j) in T_set else float(_safe_value(model.v[i], 1.0))
        denom = max(denom, DENOM_EPS)
        ell_calc = (P * P + Q * Q) / denom
        print(f"({i:2d}->{j:2d}) |S|={Smag:.6f} pu  ell_calc={ell_calc:.6f}  ell_next={ell[(i,j)]:.6f}")


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

    data = extract_bfmit_per_unit_data_meshed(net, cfg)

    print("[INFO] Data summary (modified explicit 41-bus, keep all lines)")
    print(f"  #buses   = {len(data['buses'])}")
    print(f"  #lines   = {len(data['E'])}   (directed, one per net.line row)")
    print(f"  #gens    = {len(data['gen_records'])}")
    print(f"  #OLTC    = {len(data['T'])}   (beta vars = {sum(len(data['K'][ij]) for ij in data['T'])})")
    print(f"  #shunts  = {len(data['C'])}")
    print(f"  outer: max_iters={OUTER_MAX_ITERS}, eps={OUTER_EPS}, ell_gamma={ELL_DAMPING_GAMMA}")
    print(f"  backtrack: max_tries={BACKTRACK_MAX_TRIES}, blend={BACKTRACK_BLEND}")
    print(f"  SCIP: time(bin/relax/fix)={TIME_LIMIT_BINARY}/{TIME_LIMIT_RELAX}/{TIME_LIMIT_FIXED}s, gap={MIP_GAP}, feastol={SCIP_FEASTOL}")

    sol = run_bfmit(data, max_iters=OUTER_MAX_ITERS, eps=OUTER_EPS, tee=TEE_SOLVER_LOG)

    best_model = sol["model"]
    if best_model is None:
        print("[FAIL] No solution produced.")
        return

    print("\n[SOLVED] Best/Last iterate summary")
    print(f"  iter = {sol['iter']}")
    print(f"  obj  = {sol['obj']:.6f}")
    print(f"  tag  = {sol.get('tag','')}")
    print_solution(best_model, data, sol["ell"])

    t1 = time.perf_counter()
    print(f"\n[INFO] Total elapsed time: {t1 - t0:.2f} seconds.")


if __name__ == "__main__":
    main()