# BFM_ivar_mit.py
# ------------------------------------------------------------
# Pyomo Branch-Flow OPF with outer iteration on ell (BFM-it)
# + theta update step by substitution (radial BFS propagation)
# + Objective enhancement (aligned with MIQCP_BFMit_obj.py):
#     (1) Loss proxy linearization: l_lin = b0 + aP*P + aQ*Q + aV*v_send  (>=0)
#         Objective uses affine loss cost via (cP,cQ,cV); b0 used for l_lin>=0 only.
#     (2) Proximal stabilization in EXPANDED form:
#           rho*(x^2 - 2*x_prev*x)  (constant term removed)
#     (3) w_loss smoothing (beta_wloss) and optional nonnegativity
#     (4) Warm-start CLIPPED into var bounds (avoid Pyomo W1002 spam)
#
# Added anti-oscillation features:
#   (A) ell is updated in MODEL ORIENTATION only (no dir_flag-based v_send flip)
#   (B) ell_fix (fed to MIQCP) is EMA-smoothed ell* from previous iterate (ell EMA fix)
#   (C) theta damping is wrap-aware (prevents jumps near +/-pi)
#   (D) optional state damping on (P,Q,v) used for next iterate references
#
# NEW (requested): plateau early stop for sum|dell|
#   - If sum|dell| stops improving and stays within a small band for a window,
#     terminate outer iterations early (return best solution so far).
#
# Also FIXED: best model / best ell / best theta consistency
#   - when best objective is updated, we also store the corresponding ell_state/theta.
#
# Notes:
# - Model orientation is a rooted tree away from slack (i->j). Pij,Qij are signed.
# - Direction for theta update is handled by dir_flag + Pphys/Qphys.
# - Loss proxy uses previous-iterate references (Pbar,Qbar,v_send_bar) and
#   uses OLTC effective sending voltage v_send = vsend (delta*v_hv).
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
OUTER_MAX_ITERS = 100
OUTER_EPS = 1e-6
DENOM_EPS = 1e-9

# Damping / stabilization
DEFAULT_DAMPING_ELL = 0.6
DEFAULT_THETA_DAMPING = 1.0
DEFAULT_CLIP_DTHETA_PREV = math.pi

# MIP/QCQP solver settings
SCIP_TIME_LIMIT = 600
SCIP_GAP_LIMIT = 1e-6
SCIP_NODE_LIMIT = 200000
SCIP_MEMORY_LIMIT_MB = 8192

GUROBI_TIME_LIMIT = 600
GUROBI_MIPGAP = 1e-6

# ============================
# Objective enhancement knobs
# ============================
USE_LOSS_PROXY = True
LOSS_LIN_NONNEG_CONSTRAINT = True  # l_lin >= 0 (recommended)
USE_PROXIMAL = True
USE_THETA_PROJ_PROX = True

# Prox weights (match the scale used in MIQCP_BFMit_obj.py more closely)
RHO_P = 10.0
RHO_Q = 10.0
RHO_V = 10.0
RHO_THETA = 1.0

# w_loss smoothing (like beta_wloss in your IPOPT version)
BETA_WLOSS = 0.8
LOSS_WEIGHT_SCALE = 1.0
WLOSS_NONNEG = True

# numeric floors for Taylor coeffs (avoid blow-up if v_send small)
EPS_V_LIN = 1e-6
V_SEND_FLOOR = 0.8 ** 2

# Warm-start clipping
CLIP_WARMSTART = True

# ============================
# anti-oscillation knobs
# ============================
# (1) ell EMA fix: ell_fix (fed to MIQCP) is EMA-smoothed ell*
USE_ELL_EMA_FIX = True
BETA_ELL = 0.85          # 0.8~0.95 (bigger = more stable, slower)
ELL_CLIP_NONNEG = True
ELL_CLIP_MAX = True

# (2) optional state damping for iterate references
USE_STATE_DAMPING = True
DAMPING_X = 0.6          # 0.4~0.8 (smaller = more damping)

# (3) theta damping: wrap-aware
USE_WRAP_THETA_DAMP = True

# ============================
# NEW: Plateau early-stop knobs (sum|dell|)
# ============================
USE_PLATEAU_STOP = True

# start checking after enough iterations (avoid early false-stop)
PLATEAU_MIN_ITERS = 20

# how many recent iterations to judge "plateau"
PLATEAU_WINDOW = 12

# "band" thresholds for recent sum|dell|
# range_tol = max(abs_tol, rel_tol * mean_recent)
PLATEAU_ABS_RANGE_TOL = 1e-5
PLATEAU_REL_RANGE_TOL = 1e-3     # 0.1% of mean_recent

# step thresholds inside the window (average |Δ sum|dell||)
# step_tol = max(abs_tol, rel_tol * mean_recent)
PLATEAU_ABS_STEP_TOL = 1e-6
PLATEAU_REL_STEP_TOL = 1e-3      # 0.1% of mean_recent

# (optional) require that we are NOT already truly converged by OUTER_EPS
# i.e., plateau triggers only when sum_diff is still "large"
PLATEAU_REQUIRE_NONCONVERGED = True


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


def _clip_to_bounds(val: float, lb: Optional[float], ub: Optional[float]) -> float:
    if val is None or not np.isfinite(val):
        return val
    x = float(val)
    if lb is not None and np.isfinite(lb):
        x = max(x, float(lb))
    if ub is not None and np.isfinite(ub):
        x = min(x, float(ub))
    return x


def _clip_to_vardata_bounds(vardata, val):
    """
    Clip numeric val into Pyomo VarData bounds.
    Returns float or None.
    """
    if val is None:
        return None
    try:
        x = float(val)
    except Exception:
        return None

    lb, ub = vardata.bounds  # numeric, expression, or None
    if lb is not None:
        try:
            lbv = float(pyo.value(lb))
            if x < lbv:
                x = lbv
        except Exception:
            pass
    if ub is not None:
        try:
            ubv = float(pyo.value(ub))
            if x > ubv:
                x = ubv
        except Exception:
            pass
    return x


def _wrap_pi(x: float) -> float:
    """Map angle to (-pi, pi]."""
    return (x + math.pi) % (2.0 * math.pi) - math.pi


def _damped_angle(prev: float, new: float, eta: float) -> float:
    """prev + eta * wrap(new - prev)"""
    return prev + float(eta) * _wrap_pi(new - prev)


def _plateau_check_sumdiff(sum_hist: List[float], eps: float) -> Tuple[bool, Dict[str, float]]:
    """
    Decide whether sum|dell| has plateaued over the last PLATEAU_WINDOW iterations.

    Condition (robust to small oscillations):
      - recent_range = max(recent) - min(recent) <= range_tol
      - avg_step = mean(|recent[k]-recent[k-1]|) <= step_tol
      - optionally require not already converged (mean_recent > eps)

    Returns:
      (is_plateau, stats_dict)
    """
    if (not USE_PLATEAU_STOP) or (len(sum_hist) < PLATEAU_WINDOW):
        return False, {}

    recent = np.array(sum_hist[-PLATEAU_WINDOW:], dtype=float)
    mean_recent = float(np.mean(recent))
    recent_range = float(np.max(recent) - np.min(recent))

    if PLATEAU_WINDOW >= 2:
        diffs = np.abs(np.diff(recent))
        avg_step = float(np.mean(diffs))
        max_step = float(np.max(diffs)) if diffs.size > 0 else 0.0
    else:
        avg_step = 0.0
        max_step = 0.0

    range_tol = max(float(PLATEAU_ABS_RANGE_TOL), float(PLATEAU_REL_RANGE_TOL) * max(mean_recent, 1e-12))
    step_tol = max(float(PLATEAU_ABS_STEP_TOL), float(PLATEAU_REL_STEP_TOL) * max(mean_recent, 1e-12))

    if PLATEAU_REQUIRE_NONCONVERGED and (mean_recent <= 10.0 * eps):
        return False, {
            "mean_recent": mean_recent, "recent_range": recent_range, "avg_step": avg_step, "max_step": max_step,
            "range_tol": range_tol, "step_tol": step_tol,
        }

    is_plateau = (recent_range <= range_tol) and (avg_step <= step_tol)

    return is_plateau, {
        "mean_recent": mean_recent,
        "recent_range": recent_range,
        "avg_step": avg_step,
        "max_step": max_step,
        "range_tol": range_tol,
        "step_tol": step_tol,
    }


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

    # identify slack generator record (ext_grid) for w_loss
    slack_gen_idx = None
    slack_c2 = 0.0
    slack_c1 = 0.0
    for idx, rec in enumerate(gen_records):
        if rec["type"] == "ext_grid":
            slack_gen_idx = idx
            slack_c2 = float(rec["c2"])
            slack_c1 = float(rec["c1"])
            break

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
        "slack_gen_idx": slack_gen_idx,
        "slack_c2": slack_c2,
        "slack_c1": slack_c1,
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
# Loss proxy coefficient builder (Taylor at previous iterate)
# ----------------------------
def compute_loss_proxy_coeffs(
    *,
    data: Dict[str, Any],
    Pbar: Dict[Tuple[int, int], float],
    Qbar: Dict[Tuple[int, int], float],
    vbar: Dict[int, float],
    ubar: Optional[Dict[Tuple[int, int], float]],
    w_loss: float,
) -> Dict[str, Dict[Any, float]]:
    """
    Build Taylor coefficients for l(P,Q,v_send) = (P^2+Q^2)/v_send around (Pbar,Qbar,vsend_bar).

    Returns dict with:
      aP[(i,j)], aQ[(i,j)], aV[(i,j)], b0[(i,j)]
      cP[(i,j)], cQ[(i,j)], cV[(i,j)]   where objective uses:
         sum( cP*P + cQ*Q + cV*v_send )
      (b0 is used for l_lin >= 0 constraint; constant term is not needed in objective)
    """
    sn = float(data["sn_mva"])
    E = data["E"]
    T_set = set(data["T"])
    r = data["r"]

    aP: Dict[Tuple[int, int], float] = {}
    aQ: Dict[Tuple[int, int], float] = {}
    aV: Dict[Tuple[int, int], float] = {}
    b0: Dict[Tuple[int, int], float] = {}

    cP: Dict[Tuple[int, int], float] = {}
    cQ: Dict[Tuple[int, int], float] = {}
    cV: Dict[Tuple[int, int], float] = {}

    for (i, j) in E:
        # reference v_send
        if (i, j) in T_set and (ubar is not None) and ((i, j) in ubar) and np.isfinite(ubar[(i, j)]):
            vS = float(ubar[(i, j)])
        else:
            vS = float(vbar[int(i)])

        vS = max(vS, V_SEND_FLOOR, EPS_V_LIN)

        P0 = float(Pbar[(i, j)])
        Q0 = float(Qbar[(i, j)])

        # lbar
        l0 = (P0 * P0 + Q0 * Q0) / vS

        # gradients
        aP_ = 2.0 * P0 / vS
        aQ_ = 2.0 * Q0 / vS
        aV_ = -(P0 * P0 + Q0 * Q0) / (vS * vS)

        # intercept
        b0_ = l0 - aP_ * P0 - aQ_ * Q0 - aV_ * vS

        aP[(i, j)] = aP_
        aQ[(i, j)] = aQ_
        aV[(i, j)] = aV_
        b0[(i, j)] = b0_

        # loss cost weight gamma = (w_loss * S_base) * r_ij
        gamma = float(w_loss) * sn * float(r[(i, j)])
        cP[(i, j)] = gamma * aP_
        cQ[(i, j)] = gamma * aQ_
        cV[(i, j)] = gamma * aV_

    return {"aP": aP, "aQ": aQ, "aV": aV, "b0": b0, "cP": cP, "cQ": cQ, "cV": cV}


# ----------------------------
# Pyomo model (ell fixed, MIQCP)
# ----------------------------
def build_pyomo_bfmit_model(
    data: Dict[str, Any],
    ell_fix: Dict[Tuple[int, int], float],
    relax_binaries: bool = False,
    warm_start: Optional[Dict[str, Any]] = None,
    loss_proxy: Optional[Dict[str, Dict[Any, float]]] = None,
    prox_prev: Optional[Dict[str, Dict[Any, float]]] = None,
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

    model = pyo.ConcreteModel(name="BFMit_MIQCP_OLTC_SHUNT_THETA_ENHANCED")

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

    # ----------------------------
    # Loss proxy coefficients + proximal references (mutable Params)
    # ----------------------------
    def _init_zero_edge(m, i, j):
        return 0.0

    model.aP_edge = pyo.Param(model.E, initialize=_init_zero_edge, mutable=True)
    model.aQ_edge = pyo.Param(model.E, initialize=_init_zero_edge, mutable=True)
    model.aV_edge = pyo.Param(model.E, initialize=_init_zero_edge, mutable=True)
    model.b0_edge = pyo.Param(model.E, initialize=_init_zero_edge, mutable=True)

    model.cP_edge = pyo.Param(model.E, initialize=_init_zero_edge, mutable=True)
    model.cQ_edge = pyo.Param(model.E, initialize=_init_zero_edge, mutable=True)
    model.cV_edge = pyo.Param(model.E, initialize=_init_zero_edge, mutable=True)

    model.Pprev = pyo.Param(model.E, initialize=_init_zero_edge, mutable=True)
    model.Qprev = pyo.Param(model.E, initialize=_init_zero_edge, mutable=True)
    model.vprev = pyo.Param(model.N, initialize=lambda m, i: 1.0, mutable=True)

    # ----------------------------
    # Variables
    # ----------------------------
    model.Pg = pyo.Var(model.G, bounds=lambda m, gg: (m.Pgmin[gg], m.Pgmax[gg]))
    model.Qg = pyo.Var(model.G, bounds=lambda m, gg: (m.Qgmin[gg], m.Qgmax[gg]))

    # squared voltage v in [Vmin^2, Vmax^2]
    model.v = pyo.Var(model.N, bounds=lambda m, i: (m.Vmin[i] ** 2, m.Vmax[i] ** 2))

    # net injections
    model.Pinj = pyo.Var(model.N)
    model.Qinj = pyo.Var(model.N)

    # branch variables (signed allowed), add bounds to help MIQCP
    model.Pij = pyo.Var(model.E, bounds=lambda m, i, j: (-m.Smax[i, j], m.Smax[i, j]))
    model.Qij = pyo.Var(model.E, bounds=lambda m, i, j: (-m.Smax[i, j], m.Smax[i, j]))

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
    def _tv_bounds(m, i, j, tap):
        vU = (m.Vmax[int(i)] ** 2)
        return (0.0, float(vU))
    model.tv = pyo.Var(model.BETA_INDEX, bounds=_tv_bounds)

    # ----------------------------
    # Constraints
    # ----------------------------
    # Slack voltage fix (squared)
    if data["fix_slack_vm"]:
        model.slack_v = pyo.Constraint(expr=model.v[slack_bus] == float(slack_vm_pu) ** 2)

    # Net injections
    def Pinj_rule(m, i):
        z = 0.0 * m.Pg[next(iter(m.G))] if len(Gset) > 0 else 0.0
        return m.Pinj[i] == (z + pyo.quicksum(m.Pg[gg] for gg in m.G if int(m.gen_bus[gg]) == int(i))) - m.Pd[i]
    model.Pinj_def = pyo.Constraint(model.N, rule=Pinj_rule)

    def Qinj_rule(m, i):
        z = 0.0 * m.Qg[next(iter(m.G))] if len(Gset) > 0 else 0.0
        return m.Qinj[i] == (z + pyo.quicksum(m.Qg[gg] for gg in m.G if int(m.gen_bus[gg]) == int(i))) - m.Qd[i] + m.qsh[i]
    model.Qinj_def = pyo.Constraint(model.N, rule=Qinj_rule)

    # OLTC one-hot
    def onehot_rule(m, i, j):
        taps = K[(i, j)]
        return pyo.quicksum(m.beta[i, j, int(t)] for t in taps) == 1
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
    T_set = set(T)

    def vsend_rule(m, i, j):
        if (i, j) not in T_set:
            return m.v[int(i)]
        taps = K[(i, j)]
        return pyo.quicksum(m.delta_tap[i, j, int(t)] * m.tv[i, j, int(t)] for t in taps)
    model.vsend = pyo.Expression(model.E, rule=vsend_rule)

    # Switched shunt big-M
    C_set = set(C)

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
    out_arcs = {i: [] for i in buses}
    in_arcs = {i: [] for i in buses}
    for (i, j) in E:
        out_arcs[i].append((i, j))
        in_arcs[j].append((i, j))

    def bfm_P_balance_rule(m, i):
        z = 0.0 * m.Pij[next(iter(m.E))] if len(E) > 0 else 0.0
        out_sum = z + pyo.quicksum(m.Pij[a, b] for (a, b) in out_arcs[int(i)])
        in_sum = z + pyo.quicksum((m.Pij[a, b] - m.r[a, b] * m.ell_fix[a, b]) for (a, b) in in_arcs[int(i)])
        return out_sum - in_sum == m.Pinj[i]
    model.BFM_P = pyo.Constraint(model.N, rule=bfm_P_balance_rule)

    def bfm_Q_balance_rule(m, i):
        z = 0.0 * m.Qij[next(iter(m.E))] if len(E) > 0 else 0.0
        out_sum = z + pyo.quicksum(m.Qij[a, b] for (a, b) in out_arcs[int(i)])
        in_sum = z + pyo.quicksum((m.Qij[a, b] - m.x[a, b] * m.ell_fix[a, b]) for (a, b) in in_arcs[int(i)])
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

    # Loss linearization expression + nonnegativity (optional)
    def l_lin_rule(m, i, j):
        return m.b0_edge[i, j] + m.aP_edge[i, j] * m.Pij[i, j] + m.aQ_edge[i, j] * m.Qij[i, j] + m.aV_edge[i, j] * m.vsend[i, j]
    model.l_lin = pyo.Expression(model.E, rule=l_lin_rule)

    if LOSS_LIN_NONNEG_CONSTRAINT and USE_LOSS_PROXY:
        def l_lin_nonneg_rule(m, i, j):
            return m.l_lin[i, j] >= 0.0
        model.l_lin_nonneg = pyo.Constraint(model.E, rule=l_lin_nonneg_rule)

    # ----------------------------
    # Fill Params from inputs (loss_proxy / prox_prev)
    # ----------------------------
    if loss_proxy is not None:
        for (i, j) in E:
            model.aP_edge[i, j] = float(loss_proxy.get("aP", {}).get((i, j), 0.0))
            model.aQ_edge[i, j] = float(loss_proxy.get("aQ", {}).get((i, j), 0.0))
            model.aV_edge[i, j] = float(loss_proxy.get("aV", {}).get((i, j), 0.0))
            model.b0_edge[i, j] = float(loss_proxy.get("b0", {}).get((i, j), 0.0))

            model.cP_edge[i, j] = float(loss_proxy.get("cP", {}).get((i, j), 0.0))
            model.cQ_edge[i, j] = float(loss_proxy.get("cQ", {}).get((i, j), 0.0))
            model.cV_edge[i, j] = float(loss_proxy.get("cV", {}).get((i, j), 0.0))

    if prox_prev is not None:
        for (i, j) in E:
            model.Pprev[i, j] = float(prox_prev.get("Pij", {}).get((i, j), 0.0))
            model.Qprev[i, j] = float(prox_prev.get("Qij", {}).get((i, j), 0.0))
        for i in buses:
            model.vprev[i] = float(prox_prev.get("v", {}).get(i, 1.0))

    # ----------------------------
    # Objective (enhanced)
    # ----------------------------
    def obj_rule(m):
        # base gen cost (MW scale)
        gen_cost = pyo.quicksum(
            m.c2[gg] * (sn * m.Pg[gg]) ** 2 + m.c1[gg] * (sn * m.Pg[gg]) + m.c0[gg]
            for gg in m.G
        )

        # loss proxy cost term (affine in vars)
        loss_term = pyo.quicksum(
            m.cP_edge[i, j] * m.Pij[i, j] +
            m.cQ_edge[i, j] * m.Qij[i, j] +
            m.cV_edge[i, j] * m.vsend[i, j]
            for (i, j) in m.E
        )

        # proximal terms (EXPANDED form): rho*(x^2 - 2*x_prev*x)
        prox = 0.0
        if USE_PROXIMAL:
            prox += float(RHO_P) * pyo.quicksum(m.Pij[i, j] ** 2 - 2.0 * m.Pprev[i, j] * m.Pij[i, j] for (i, j) in m.E)
            prox += float(RHO_Q) * pyo.quicksum(m.Qij[i, j] ** 2 - 2.0 * m.Qprev[i, j] * m.Qij[i, j] for (i, j) in m.E)
            prox += float(RHO_V) * pyo.quicksum(m.v[i] ** 2 - 2.0 * m.vprev[i] * m.v[i] for i in m.N)

        theta_proj = 0.0
        if USE_THETA_PROJ_PROX:
            theta_proj = float(RHO_THETA) * pyo.quicksum(
                (m.x[i, j] * (m.Pij[i, j] - m.Pprev[i, j]) - m.r[i, j] * (m.Qij[i, j] - m.Qprev[i, j])) ** 2
                for (i, j) in m.E
            )

        return gen_cost + (loss_term if USE_LOSS_PROXY else 0.0) + prox + theta_proj

    model.obj = pyo.Objective(rule=obj_rule, sense=pyo.minimize)

    # ----------------------------
    # Warm start (CLIP to bounds)
    # ----------------------------
    if warm_start is not None:
        if "v" in warm_start:
            for i in buses:
                if i in warm_start["v"]:
                    val = float(warm_start["v"][i])
                    model.v[i].value = _clip_to_vardata_bounds(model.v[i], val) if CLIP_WARMSTART else val

        if "Pij" in warm_start:
            for (i, j) in E:
                if (i, j) in warm_start["Pij"]:
                    val = float(warm_start["Pij"][(i, j)])
                    model.Pij[i, j].value = _clip_to_vardata_bounds(model.Pij[i, j], val) if CLIP_WARMSTART else val

        if "Qij" in warm_start:
            for (i, j) in E:
                if (i, j) in warm_start["Qij"]:
                    val = float(warm_start["Qij"][(i, j)])
                    model.Qij[i, j].value = _clip_to_vardata_bounds(model.Qij[i, j], val) if CLIP_WARMSTART else val

        if "Pg" in warm_start:
            for gg in Gset:
                if gg in warm_start["Pg"]:
                    val = float(warm_start["Pg"][gg])
                    model.Pg[gg].value = _clip_to_vardata_bounds(model.Pg[gg], val) if CLIP_WARMSTART else val

        if "Qg" in warm_start:
            for gg in Gset:
                if gg in warm_start["Qg"]:
                    val = float(warm_start["Qg"][gg])
                    model.Qg[gg].value = _clip_to_vardata_bounds(model.Qg[gg], val) if CLIP_WARMSTART else val

        if "beta" in warm_start:
            for (i, j, tap) in model.BETA_INDEX:
                if (i, j, tap) in warm_start["beta"]:
                    val = float(warm_start["beta"][(i, j, tap)])
                    if relax_binaries:
                        model.beta[i, j, tap].value = _clip_to_vardata_bounds(model.beta[i, j, tap], val)
                    else:
                        model.beta[i, j, tap].value = 1.0 if val >= 0.5 else 0.0

        if "a_sh" in warm_start:
            for i in C:
                if i in warm_start["a_sh"]:
                    val = float(warm_start["a_sh"][i])
                    if relax_binaries:
                        model.a_sh[i].value = _clip_to_vardata_bounds(model.a_sh[i], val)
                    else:
                        model.a_sh[i].value = 1.0 if val >= 0.5 else 0.0

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
            solver.options["NonConvex"] = 2
            solver.options["MIPFocus"] = 1
        except Exception:
            pass

        res = solver.solve(model, tee=tee, warmstart=True)
        tc = res.solver.termination_condition
        st = res.solver.status
        ok = tc in [
            pyo.TerminationCondition.optimal,
            pyo.TerminationCondition.locallyOptimal,
            pyo.TerminationCondition.feasible,
            pyo.TerminationCondition.maxTimeLimit
        ]
        return {"ok": ok, "solver": "gurobi", "termination": str(tc), "status": str(st)}

    # 2) SCIP (AMPL .nl interface — required for MIQCP on Pyomo+SCIP;
    # NL interface does not accept the `warmstart` kwarg, so we omit it.)
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
# theta update by substitution (radial BFS)
# ----------------------------
def update_theta_by_substitution_pyomo(
    data: Dict[str, Any],
    v_sol: Dict[int, float],                    # bus -> v(pu^2)
    Pphys: Dict[Tuple[int, int], float],        # edge -> Pphys (pu)
    Qphys: Dict[Tuple[int, int], float],        # edge -> Qphys (pu)
    theta_prev: Dict[int, float],               # bus -> theta(rad)
    send_bus: Dict[Tuple[int, int], int],       # edge -> sending bus id
    recv_bus: Dict[Tuple[int, int], int],       # edge -> receiving bus id
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

        # wrap-aware previous angle difference
        dprev = _wrap_pi(float(theta_prev.get(s, 0.0) - theta_prev.get(rcv, 0.0)))
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

    # damping (wrap-aware recommended)
    theta_out = {}
    for b in buses:
        thp = float(theta_prev.get(b, 0.0))
        thn = float(theta_new[b])
        if USE_WRAP_THETA_DAMP:
            theta_out[b] = _damped_angle(thp, thn, float(theta_damping))
        else:
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
    K = data["K"]
    delta_tap = data["delta_tap"]

    # --- iterate states (for references / next iterate) ---
    ell_fix = {(i, j): 0.0 for (i, j) in E}     # ell* fed to MIQCP (EMA-smoothed)
    ell_state = {(i, j): 0.0 for (i, j) in E}   # monitoring / damped ell state
    P_prev = {(i, j): 0.0 for (i, j) in E}
    Q_prev = {(i, j): 0.0 for (i, j) in E}
    v_prev = {int(b): 1.0 for b in buses}
    u_send_prev = {(i, j): float("nan") for (i, j) in E}  # OLTC effective vsend(prev)

    theta_prev = {b: 0.0 for b in buses}
    theta_prev[int(data["slack_bus"])] = 0.0

    warm = {
        "v": dict(v_prev),
        "Pij": dict(P_prev),
        "Qij": dict(Q_prev),
        "Pg": {},
        "Qg": {},
        "beta": {},
        "a_sh": {},
        "u_send": dict(u_send_prev),
    }

    # FIXED: best stores consistent (model, ell, theta) of that iterate
    best = {
        "iter": 0,
        "obj": float("inf"),
        "model": None,
        "ell": None,
        "theta": None,
    }

    # w_loss smoothing state
    w_loss_sm = 0.0
    slack_gen_idx = data.get("slack_gen_idx", None)
    slack_c2 = float(data.get("slack_c2", 0.0))
    slack_c1 = float(data.get("slack_c1", 0.0))
    sn = float(data["sn_mva"])

    # NEW: history for plateau stop
    sumdiff_hist: List[float] = []
    obj_hist: List[float] = []

    for t in range(1, max_iters + 1):
        t_iter0 = time.perf_counter()

        # Save previous states for direction decisions and convergence metrics
        P_prev_sign = dict(P_prev)
        ell_state_prev = dict(ell_state)

        # ------------------------------------------------------------
        # build ell_raw from PREVIOUS iterate (MODEL ORIENTATION)
        # ------------------------------------------------------------
        ell_raw = {}
        for (i, j) in E:
            if (i, j) in T_set:
                vs = float(u_send_prev.get((i, j), float("nan")))
                if not np.isfinite(vs):
                    vs = float(v_prev[int(i)])
            else:
                vs = float(v_prev[int(i)])

            vs = max(float(vs), V_SEND_FLOOR, DENOM_EPS)

            p0 = float(P_prev[(i, j)])
            q0 = float(Q_prev[(i, j)])
            val = (p0 * p0 + q0 * q0) / vs

            if ELL_CLIP_NONNEG:
                val = max(0.0, val)
            if ELL_CLIP_MAX:
                val = min(val, float(ellmax[(i, j)]))

            ell_raw[(i, j)] = val

        # ------------------------------------------------------------
        # EMA update for ell_fix (ell*) fed to MIQCP
        # ------------------------------------------------------------
        if USE_ELL_EMA_FIX and t >= 2:
            for (i, j) in E:
                ell_fix[(i, j)] = float(BETA_ELL) * float(ell_fix[(i, j)]) + (1.0 - float(BETA_ELL)) * float(ell_raw[(i, j)])
                if ELL_CLIP_NONNEG:
                    ell_fix[(i, j)] = max(0.0, float(ell_fix[(i, j)]))
                if ELL_CLIP_MAX:
                    ell_fix[(i, j)] = min(float(ellmax[(i, j)]), float(ell_fix[(i, j)]))
        elif not USE_ELL_EMA_FIX:
            ell_fix = dict(ell_raw)
        # t==1: keep ell_fix zeros (stable start)

        # --- build proximal references from previous iterate (use warm state) ---
        prox_prev = {
            "Pij": dict(warm["Pij"]),
            "Qij": dict(warm["Qij"]),
            "v": dict(warm["v"]),
        }

        # --- w_loss raw from previous slack P (like IPOPT version) ---
        if slack_gen_idx is None:
            w_raw = 0.0
        else:
            P0_pu_prev = float(warm.get("Pg", {}).get(int(slack_gen_idx), 0.0))
            P0_MW_prev = sn * P0_pu_prev
            w_raw = (slack_c1 + 2.0 * slack_c2 * P0_MW_prev) * float(LOSS_WEIGHT_SCALE)
            if WLOSS_NONNEG:
                w_raw = max(0.0, float(w_raw))

        if t == 1:
            w_loss_sm = float(w_raw)
        else:
            w_loss_sm = float(BETA_WLOSS) * float(w_loss_sm) + (1.0 - float(BETA_WLOSS)) * float(w_raw)
        w_loss = float(w_loss_sm)

        # --- loss proxy coefficients around previous iterate ---
        loss_proxy = compute_loss_proxy_coeffs(
            data=data,
            Pbar=prox_prev["Pij"],
            Qbar=prox_prev["Qij"],
            vbar=prox_prev["v"],
            ubar=warm.get("u_send", None),
            w_loss=w_loss,
        )

        # 1) solve OPF with ell fixed at ell_fix (EMA-smoothed)
        model = build_pyomo_bfmit_model(
            data,
            ell_fix=ell_fix,
            relax_binaries=False,
            warm_start=warm,
            loss_proxy=loss_proxy,
            prox_prev=prox_prev,
        )

        solinfo = solve_miqcp(model, tee=tee)
        if not solinfo["ok"]:
            print(f"[t={t:02d}] MIQCP not solved. solver={solinfo['solver']} "
                  f"termination={solinfo['termination']} status={solinfo['status']} STOP.")
            break

        obj = float(pyo.value(model.obj))

        # 2) read solution
        v_sol = {int(i): float(pyo.value(model.v[i])) for i in buses}
        P_sol = {(i, j): float(pyo.value(model.Pij[i, j])) for (i, j) in E}
        Q_sol = {(i, j): float(pyo.value(model.Qij[i, j])) for (i, j) in E}

        # OLTC u_send (= delta*v_hv) from THIS solution (for theta update)
        u_send_edge = {}
        for (i, j) in E:
            if (i, j) in T_set:
                u_send_edge[(i, j)] = float(pyo.value(model.vsend[i, j]))
            else:
                u_send_edge[(i, j)] = float("nan")

        # binaries and gens (no damping)
        Pg_sol = {int(gg): float(pyo.value(model.Pg[gg])) for gg in model.G}
        Qg_sol = {int(gg): float(pyo.value(model.Qg[gg])) for gg in model.G}
        a_sh_sol = {int(i): float(pyo.value(model.a_sh[i])) for i in model.C}
        beta_sol = {(i, j, tap): float(pyo.value(model.beta[i, j, tap])) for (i, j, tap) in model.BETA_INDEX}

        # dir_flag + send/recv (OLTC forced forward). Use PREVIOUS sign to avoid flip.
        dir_flag = {}
        send_bus = {}
        recv_bus = {}
        for (i, j) in E:
            if (i, j) in T_set:
                dir_flag[(i, j)] = 1
            else:
                dir_flag[(i, j)] = 1 if float(P_prev_sign[(i, j)]) >= 0.0 else -1

            if dir_flag[(i, j)] == 1:
                send_bus[(i, j)] = int(i)
                recv_bus[(i, j)] = int(j)
            else:
                send_bus[(i, j)] = int(j)
                recv_bus[(i, j)] = int(i)

        # physical sending-end flows (consistent with ell_fix used in MIQCP)
        Pphys = {}
        Qphys = {}
        for (i, j) in E:
            rij = float(data["r"][(i, j)])
            xij = float(data["x"][(i, j)])
            if dir_flag[(i, j)] == 1:
                Pphys[(i, j)] = float(P_sol[(i, j)])
                Qphys[(i, j)] = float(Q_sol[(i, j)])
            else:
                # reverse direction uses ell_fix (what the model actually used)
                Pphys[(i, j)] = -float(P_sol[(i, j)]) + rij * float(ell_fix[(i, j)])
                Qphys[(i, j)] = -float(Q_sol[(i, j)]) + xij * float(ell_fix[(i, j)])

        # theta update (BFS propagation)
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

        # ------------------------------------------------------------
        # state damping for iterate references (P_prev,Q_prev,v_prev)
        # ------------------------------------------------------------
        if USE_STATE_DAMPING:
            for b in buses:
                v_prev[int(b)] = (1.0 - float(DAMPING_X)) * float(v_prev[int(b)]) + float(DAMPING_X) * float(v_sol[int(b)])
            for (i, j) in E:
                P_prev[(i, j)] = (1.0 - float(DAMPING_X)) * float(P_prev[(i, j)]) + float(DAMPING_X) * float(P_sol[(i, j)])
                Q_prev[(i, j)] = (1.0 - float(DAMPING_X)) * float(Q_prev[(i, j)]) + float(DAMPING_X) * float(Q_sol[(i, j)])
        else:
            v_prev = dict(v_sol)
            P_prev = dict(P_sol)
            Q_prev = dict(Q_sol)

        # ------------------------------------------------------------
        # rebuild u_send_prev consistently with (beta_sol, v_prev)
        #   u_send_prev(i,j) = delta_eff * v_prev[i]
        # ------------------------------------------------------------
        for (i, j) in E:
            if (i, j) in T_set:
                delta_eff = 0.0
                for tap in K[(i, j)]:
                    delta_eff += float(delta_tap[((i, j), int(tap))]) * float(beta_sol[(i, j, int(tap))])
                u_send_prev[(i, j)] = float(delta_eff) * float(v_prev[int(i)])
            else:
                u_send_prev[(i, j)] = float("nan")

        # ------------------------------------------------------------
        # update ell_state in MODEL ORIENTATION (no dir flip)
        # ------------------------------------------------------------
        sum_diff = 0.0
        max_viol = 0.0

        for (i, j) in E:
            # model sending voltage
            if (i, j) in T_set:
                vs = float(u_send_prev.get((i, j), float("nan")))
                if not np.isfinite(vs):
                    vs = float(v_prev[int(i)])
            else:
                vs = float(v_prev[int(i)])

            vs = max(float(vs), V_SEND_FLOOR, DENOM_EPS)

            ell_calc = (float(P_prev[(i, j)]) ** 2 + float(Q_prev[(i, j)]) ** 2) / vs
            ell_damped = (1.0 - float(damping_ell)) * float(ell_state_prev[(i, j)]) + float(damping_ell) * float(ell_calc)

            if ELL_CLIP_NONNEG:
                ell_damped = max(0.0, float(ell_damped))
            if ELL_CLIP_MAX:
                ell_damped = min(float(ellmax[(i, j)]), float(ell_damped))

            ell_state[(i, j)] = float(ell_damped)

            sum_diff += abs(float(ell_state[(i, j)]) - float(ell_state_prev[(i, j)]))

            viol = 0.0
            if float(ell_state[(i, j)]) < -1e-8:
                viol = -float(ell_state[(i, j)])
            if float(ell_state[(i, j)]) > float(ellmax[(i, j)]) + 1e-8:
                viol = float(ell_state[(i, j)]) - float(ellmax[(i, j)])
            max_viol = max(max_viol, viol)

        # ------------------------------------------------------------
        # Warm-start update (use damped iterate state for continuous vars)
        # ------------------------------------------------------------
        warm["v"] = dict(v_prev)
        warm["Pij"] = dict(P_prev)
        warm["Qij"] = dict(Q_prev)
        warm["Pg"] = dict(Pg_sol)
        warm["Qg"] = dict(Qg_sol)
        warm["a_sh"] = dict(a_sh_sol)
        warm["beta"] = dict(beta_sol)
        warm["u_send"] = dict(u_send_prev)

        # ------------------------------------------------------------
        # Update best solution (FIXED: store consistent ell/theta too)
        # ------------------------------------------------------------
        if obj < best["obj"]:
            best["obj"] = obj
            best["iter"] = t
            best["model"] = model
            best["ell"] = dict(ell_state)
            best["theta"] = dict(theta_new)

        # ------------------------------------------------------------
        # Logging + stopping checks
        # ------------------------------------------------------------
        t_iter1 = time.perf_counter()
        print(f"[t={t:02d}] obj={obj:.6f}  sum|dell|={sum_diff:.3e}  "
              f"max_ell_viol={max_viol:.3e}  w_loss={w_loss:.6f}  iter_time={t_iter1-t_iter0:.2f}s")

        # convergence
        if (sum_diff <= eps) and (max_viol <= 1e-8):
            print(f"[CONVERGED] at t={t} with eps={eps}")
            # best already updated if this is best obj; but return a consistent "best" anyway.
            return best

        # NEW: plateau early-stop on sum|dell|
        sumdiff_hist.append(float(sum_diff))
        obj_hist.append(float(obj))

        if USE_PLATEAU_STOP and (t >= PLATEAU_MIN_ITERS):
            is_plateau, st = _plateau_check_sumdiff(sumdiff_hist, eps)
            if is_plateau:
                print("[EARLY-STOP: PLATEAU] sum|dell| has plateaued.")
                print(f"  window={PLATEAU_WINDOW}, mean_recent={st.get('mean_recent', float('nan')):.3e}, "
                      f"range={st.get('recent_range', float('nan')):.3e} (tol={st.get('range_tol', float('nan')):.3e}), "
                      f"avg_step={st.get('avg_step', float('nan')):.3e} (tol={st.get('step_tol', float('nan')):.3e})")
                # return best solution found so far (consistent model/ell/theta)
                return best

        # update state for next loop
        theta_prev = dict(theta_new)

    # not converged: return best found so far (consistent)
    if best["model"] is None:
        # fallback (shouldn't happen if at least one solve succeeded)
        best["ell"] = dict(ell_state)
        best["theta"] = dict(theta_prev)
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

    print("\n--- Branch flows & ell (state) ---")
    for (i, j) in model.E:
        P = float(pyo.value(model.Pij[i, j]))
        Q = float(pyo.value(model.Qij[i, j]))
        Smag = math.sqrt(max(P * P + Q * Q, 0.0))
        denom = float(pyo.value(model.vsend[i, j])) if (i, j) in T_set else float(pyo.value(model.v[i]))
        denom = max(denom, DENOM_EPS)
        ell_calc = (P * P + Q * Q) / denom
        print(f"({i}->{j}): P={P:+.6f} pu, Q={Q:+.6f} pu, |S|={Smag:.6f} pu, "
              f"ell_calc(naive)={ell_calc:.6f}, ell_state={ell[(i,j)]:.6f}")


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
    print(f"[INFO] ObjEnhance: loss_proxy={USE_LOSS_PROXY}, llin>=0={LOSS_LIN_NONNEG_CONSTRAINT}, proximal={USE_PROXIMAL}, theta_proj={USE_THETA_PROJ_PROX}")
    print(f"[INFO] RHO: P={RHO_P}, Q={RHO_Q}, V={RHO_V}, THETA={RHO_THETA}")
    print(f"[INFO] w_loss smoothing: beta={BETA_WLOSS}, scale={LOSS_WEIGHT_SCALE}, nonneg={WLOSS_NONNEG}")

    print(f"[INFO] Anti-oscillation: ell_ema_fix={USE_ELL_EMA_FIX} (beta_ell={BETA_ELL}), "
          f"state_damp={USE_STATE_DAMPING} (DAMPING_X={DAMPING_X}), theta_wrap_damp={USE_WRAP_THETA_DAMP}")

    print(f"[INFO] Plateau-stop: enabled={USE_PLATEAU_STOP}, min_iters={PLATEAU_MIN_ITERS}, window={PLATEAU_WINDOW}, "
          f"range_tol=max({PLATEAU_ABS_RANGE_TOL}, {PLATEAU_REL_RANGE_TOL}*mean), "
          f"step_tol=max({PLATEAU_ABS_STEP_TOL}, {PLATEAU_REL_STEP_TOL}*mean)")

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

    print("\n[SOLVED] Best iterate summary (consistent model/ell/theta)")
    print(f"  iter  = {sol['iter']}")
    print(f"  obj   = {sol['obj']:.6f}")
    _print_solution(best_model, data, sol["ell"], theta=sol.get("theta", None))

    t1 = time.perf_counter()
    print(f"\n[INFO] Total elapsed time: {t1 - t0:.2f} seconds.")

    print(f"[GEN_COST_ONLY] {sum(pyo.value(best_model.c2[gg])*(data['sn_mva']*pyo.value(best_model.Pg[gg]))**2 + pyo.value(best_model.c1[gg])*(data['sn_mva']*pyo.value(best_model.Pg[gg])) + pyo.value(best_model.c0[gg]) for gg in best_model.G):.6f}")


if __name__ == "__main__":
    main()
