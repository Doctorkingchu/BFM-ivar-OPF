# BFM_ivar_mit.py
# ------------------------------------------------------------
# BFM-ag on IEEE 41-bus mesh with OLTC + shunt
# - Subproblem: MIQCP with ell fixed (same class as upgrade3)
# - Exact-equality vdrop/KCL from upgrade3 base kept by default
# - Bounded-slack window machinery from upgrade3 kept; outer loop
#   controls and tightens it while the BFMag ell update drives
#   coupling residuals to zero.
#
# 41-bus adaptation: every upgrade3 entry point is preserved
# (build_cfg_from_net_metadata, extract_data_fullmesh_branch_table,
# build_subproblem, run_bfm_ag, main) and the network builder is
# still `import ieee41bus as mcase` with `busmeshed39_opf(...)`
# (NOT case300_opf).  The 41-bus data extractors read
# net["line_params_pu_table"], net["fixed_oltc_table"],
# net["fixed_shunt_table"] exactly as before.
#
# New ver3 features spliced in on top of upgrade3:
#
# --- A. In-round MIQCP tightening (free / cheap cuts) ---
#   [A1] Implicit equation Sum_t tv[i,j,t] = v[i] explicitly added.
#   [A2] SOS1 declaration on beta[i,j,*] for solver SOS1 branching.
#   [A3] Receiving-end thermal (Pij-r*ell_fix)^2 + (Qij-x*ell_fix)^2 <= Smax^2.
#   [A4] OBBT preprocessing (off by default on this smaller net).
#   [A5] KVL cycle cuts on a spanning-tree fundamental-cycle basis.
#   [A6] Multi-cut loss under-approximation: l_edge variable with K stored
#        Taylor planes replacing the single-plane linear loss proxy.
#
# --- B. SCIP / solver-side performance ---
#   [B7] Gurobi drop-in first, SCIP NL fallback, SCIP default fallback
#        (degrades gracefully if gurobi/gurobi_persistent is not importable).
#   [B8] MIP warm-start wired through: richer warm dict (beta, a_sh, Pij,
#        Qij, v, Pg, Qg, ell, tv) and NL-writer seeding.
#   [B9] Fix-and-relax: after STABLE_ROUNDS_THRESHOLD consecutive accepted
#        rounds with the same rounded a_sh / beta, those binaries are
#        fixed on subsequent subproblems until a change is detected.
#   [B10] Round-dependent MIP gap schedule.
#
# --- C. Outer BFMag loop acceleration (still MIQCP per round) ---
#   [C11] Anderson type-II acceleration on ell_fix (off by default).
#   [C12] Adaptive trust region on (Pij, Qij, v) replacing constant-rho
#         proximal (off by default).
#   [C13] Optional warm-start of ell from a single BFM (MISOCP) solve
#         (imports BFM_MISOCP if available; off by default).
#   [C14] Theta-reuse: |V_i||V_j|cos(theta_i-theta_j) fed as vS into
#         compute_loss_proxy_coeffs after the first theta LS recovery.
#   [C15] Adaptive over-relaxation (omega) on the ell update.
#
# - adapted to: ieee41bus.py (fixed 41-bus stressed mesh scenario,
#   entry point `busmeshed39_opf(...)`)
# ------------------------------------------------------------

import math
import time
from dataclasses import dataclass
from typing import Dict, Tuple, Any, Optional, List

import numpy as np
import pandas as pd
import pyomo.environ as pyo
from pyomo.opt import TerminationCondition

try:
    import pandapower as pp
except Exception:  # pragma: no cover - optional for PF warm start
    pp = None  # type: ignore

import ieee41bus as mcase


# ============================================================
# Global settings
# ============================================================
TEE_SOLVER_LOG = False

OUTER_MAX_ITERS = 40
OUTER_EPS = 1e-5
DENOM_EPS = 1e-10

ELL_GAMMA = 0.5
THETA_GAMMA = 0.5
CLIP_DTHETA_PREV = math.pi
THETA_RIDGE = 1e-8

# Bounded-slack constraints (removed from objective)
VDROP_SLACK_INIT  = 5.0e-1
VDROP_SLACK_FINAL = 1.5e-1
VDROP_SLACK_DECAY = 0.92

KCL_SLACK_INIT  = 1.0e-2
KCL_SLACK_FINAL = 5.0e-4
KCL_SLACK_DECAY = 0.85

SLACK_RESCUE_MULTS = (1.0, 2.0, 5.0)
VDROP_SLACK_RESCUE_CAP = 1.0
KCL_SLACK_RESCUE_CAP   = 5.0e-2

SLACK_TIGHTEN_TRIGGER = 0.80
SLACK_ENLARGE_ON_FAIL = 1.50
SLACK_ENLARGE_ON_REJECT = 1.25
SLACK_REJECT_MARGIN = 1.05
REJECT_RESCUED_STEPS = False
REJECT_RELAXED_ONLY_STEPS = True
MAX_CONSECUTIVE_REJECTS = 4

SCIP_TIME_LIMIT = 120.0
SCIP_GAP_LIMIT = 1e-4
SCIP_NODE_LIMIT = 300000
SCIP_MEMORY_LIMIT_MB = 8192
SCIP_FEASTOL = 1e-5
SCIP_DUALFEASTOL = 1e-7

NETWORK_BUILD_KWARGS = dict(
    slack_vm_pu=1.0,
    line_max_loading_percent=120.0,
    stress_q_over_p=0.95,
    stress_load_mw_each=300.0,
    r_loss_scale=3.0,
    max_i_ka_base=2.0,
    max_i_ka_stress=1.0,
    line_smax_pu_overrides=None,
)

# ----------------------------
# 1번 변화: objective enhancement knobs
# ----------------------------
USE_LOSS_PROXY = True
LOSS_LIN_NONNEG_CONSTRAINT = True
USE_PROXIMAL = True
USE_PROX_BOUNDS = False
USE_THETA_PROJ_PROX = True

RHO_P = 10.0
RHO_Q = 10.0
RHO_V = 10.0
RHO_THETA = 1.0

PSTEP_FRAC_INIT  = 1.25
PSTEP_FRAC_FINAL = 0.55
PSTEP_FRAC_DECAY = 0.92

QSTEP_FRAC_INIT  = 1.25
QSTEP_FRAC_FINAL = 0.55
QSTEP_FRAC_DECAY = 0.92

VSTEP_ABS_INIT  = 0.25
VSTEP_ABS_FINAL = 0.08
VSTEP_ABS_DECAY = 0.90

BETA_WLOSS = 0.92
LOSS_WEIGHT_SCALE = 1.0
WLOSS_NONNEG = True

EPS_V_LIN = 1e-6
V_SEND_FLOOR = 0.8 ** 2

CLIP_WARMSTART = True

# ----------------------------
# 2번 변화: anti-oscillation knobs
# ----------------------------
USE_ELL_EMA_FIX = False
BETA_ELL = 0.85
ELL_CLIP_NONNEG = True
ELL_CLIP_MAX = True

USE_STATE_DAMPING = True
DAMPING_X = 0.35

USE_WRAP_THETA_DAMP = True

# ----------------------------
# Outer ell-correction knobs
# ----------------------------
USE_OUTER_ELL_HYBRID = True
ELL_HYBRID_I_WEIGHT = 0.50
ELL_HYBRID_V_WEIGHT = 0.50
ELL_BACKTRACK_ALPHAS = (1.0, 0.5, 0.25, 0.10, 0.05, 0.01)
ELL_BACKTRACK_REL_STEP = 0.35
ELL_BACKTRACK_ABS_STEP = 1.0e-4
USE_ACCEPTED_ELL_AS_NEXT_FIX = True
USE_OUTER_ELL_FORCE_FOLLOW = True
USE_ELL_EMA_FIX = False

BEST_VDROP_TOL = VDROP_SLACK_FINAL
BEST_KCL_TOL = KCL_SLACK_FINAL
USE_SLACK_BLOWUP_STOP = False
BLOWUP_AFTER_FEASIBLE_ONLY = False
BLOWUP_VDROP_TOL = VDROP_SLACK_FINAL
BLOWUP_KCL_TOL = KCL_SLACK_FINAL

# ----------------------------
# Plateau early-stop knobs
# ----------------------------
USE_PLATEAU_STOP = True
PLATEAU_MIN_ITERS = 10
PLATEAU_WINDOW = 12

PLATEAU_ABS_RANGE_TOL = 1e-5
PLATEAU_REL_RANGE_TOL = 1e-3

PLATEAU_ABS_STEP_TOL = 1e-6
PLATEAU_REL_STEP_TOL = 1e-3

PLATEAU_REQUIRE_NONCONVERGED = True


# ============================================================
# ver3 new knobs (spliced in; defaults mirror the upstream variant
# except USE_OBBT which is kept off on this smaller network by user
# request — user may flip it on manually)
# ============================================================

# [A1] sum_t tv[i,j,t] = v[i] explicit equation
USE_TV_SUM_EQ = True

# [A2] SOS1 declaration on beta per OLTC edge
USE_BETA_SOS1 = True

# [A3] Receiving-end thermal SOC
USE_RECV_THERMAL = True

# [A4] OBBT preprocessing on Pij/Qij bounds.  Default OFF on 41-bus
# because the network is small and OBBT's 180s+ LP pass is not repaid.
USE_OBBT = False
OBBT_TIME_LIMIT = 5.0
OBBT_OK_RELTOL = 0.02
OBBT_MAX_EDGES = None
OBBT_MIN_SHRINK = 0.02
OBBT_THERMAL_OUTER_BOX_FRAC = 1.0

# [A5] KVL cycle cuts on a spanning-tree basis (off by default; matches ver3)
USE_KVL_CYCLES = False
KVL_MAX_CYCLE_LEN = 12

# [A6] Multi-cut loss under-approximation
USE_MULTI_CUT_LOSS = True
MULTI_CUT_LOSS_HISTORY = 5
MULTI_CUT_LOSS_L_NONNEG = True
LOSS_PLANE_SHIFT_EPS = 1.0e-8

# [B7] Solver order
USE_GUROBI_FIRST = True
GUROBI_MIPGAP_DEFAULT = 1e-4
GUROBI_MIP_GAP = 1e-4
GUROBI_TIME_LIMIT = 120.0

# [B8] MIP warm-start
USE_MIP_WARMSTART = True

# [B9] Fix-and-relax for stable binaries
USE_FIX_AND_RELAX = True
STABLE_ROUNDS_THRESHOLD = 4
FIX_AND_RELAX_STABLE_ROUNDS = STABLE_ROUNDS_THRESHOLD

# [B10] Round-dependent MIP gap schedule
USE_MIPGAP_SCHEDULE = True
USE_ROUND_DEPENDENT_GAP = USE_MIPGAP_SCHEDULE
MIPGAP_EARLY = 5e-3
MIPGAP_MID = 1e-3
MIPGAP_LATE = 1e-5
MIPGAP_MID_START = 5
MIPGAP_LATE_START = 11
# Legacy alias variables kept for compatibility with ver3 internals.
GAP_EARLY = MIPGAP_EARLY
GAP_MID = MIPGAP_MID
GAP_LATE = MIPGAP_LATE
GAP_SWITCH_EARLY = MIPGAP_MID_START - 1
GAP_SWITCH_MID = MIPGAP_LATE_START - 1

# [C11] Anderson acceleration on ell_fix (OFF by default; matches ver3)
USE_ANDERSON_ELL = False
ANDERSON_WINDOW = 5
ANDERSON_M = ANDERSON_WINDOW
ANDERSON_WARMUP_ITERS = 6
ANDERSON_REG = 1e-10
ANDERSON_MIX_BETA = 1.0

# [C12] Adaptive trust region (OFF by default; matches ver3)
USE_ADAPTIVE_TRUST_REGION = False
TR_INIT_P = 0.35
TR_INIT_Q = 0.35
TR_INIT_V = 0.15
TR_EXPAND = 1.5
TR_SHRINK = 0.5
TR_MIN_P = 0.03
TR_MIN_Q = 0.03
TR_MIN_V = 0.01
TR_MAX_P = 1.5
TR_MAX_Q = 1.5
TR_MAX_V = 0.5

# [C13] MISOCP warm-start for ell (off by default; requires BFM_MISOCP)
USE_ELL_WARM_FROM_SOCP = False
ELL_WARM_SOCP_TIME_LIMIT = 60.0

# [C14] Theta-reuse vS in loss proxy
USE_THETA_REUSE_VS = True
THETA_REUSE_START_ITER = 2

# [C15] Adaptive over-relaxation on ell update
USE_ADAPTIVE_OMEGA = True
OMEGA_INIT = 0.85
OMEGA_MIN = 0.8
OMEGA_MAX = 1.5
OMEGA_UP = 1.10
OMEGA_DOWN = 0.80

# PF warm-start
USE_PF_WARMSTART = True


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


def _clip_to_vardata_bounds(vardata, val):
    if val is None:
        return None
    try:
        x = float(val)
    except Exception:
        return None

    lb, ub = vardata.bounds
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


def _extract_warm_from_model(m: pyo.ConcreteModel, data: Dict[str, Any]) -> Dict[str, Dict[Any, float]]:
    warm = {
        "v": {int(i): _val(m.v[i], 1.0) for i in data["buses"]},
        "Pij": {(int(i), int(j)): _val(m.Pij[i, j], 0.0) for (i, j) in data["E"]},
        "Qij": {(int(i), int(j)): _val(m.Qij[i, j], 0.0) for (i, j) in data["E"]},
        "Pg": {int(g): _val(m.Pg[g], 0.0) for g in m.G},
        "Qg": {int(g): _val(m.Qg[g], 0.0) for g in m.G},
        "beta": {(int(i), int(j), int(tap)): _val(m.beta[i, j, tap], 0.0) for (i, j, tap) in m.BETA_INDEX},
        "a_sh": {int(i): _val(m.a_sh[i], 0.0) for i in data["C"]},
        "u_send": {},
    }
    try:
        warm["ell"] = {(int(i), int(j)): float(pyo.value(m.ell_fix[i, j], exception=False) or 0.0) for (i, j) in data["E"]}
    except Exception:
        warm["ell"] = {}
    try:
        warm["tv"] = {(int(i), int(j), int(tap)): _val(m.tv[i, j, tap], 0.0) for (i, j, tap) in m.BETA_INDEX}
    except Exception:
        warm["tv"] = {}
    for (i, j) in data["E"]:
        if (int(i), int(j)) in set(data["T"]):
            warm["u_send"][(int(i), int(j))] = _pval(m.vsend[i, j], warm["v"][int(i)])
        else:
            warm["u_send"][(int(i), int(j))] = float("nan")
    return warm


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


def _effective_vsend_from_solution(
    data: Dict[str, Any],
    model: pyo.ConcreteModel,
    v_sol: Dict[int, float],
    i: int,
    j: int
) -> float:
    if (i, j) in set(data["T"]):
        return _pval(model.vsend[i, j], float(v_sol[i]))
    return float(v_sol[i])


def _wrap_pi(x: float) -> float:
    return (x + math.pi) % (2.0 * math.pi) - math.pi


def _damped_angle(prev: float, new: float, eta: float) -> float:
    return prev + float(eta) * _wrap_pi(new - prev)


def _plateau_check_sumdiff(sum_hist: List[float], eps: float) -> Tuple[bool, Dict[str, float]]:
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
            "mean_recent": mean_recent,
            "recent_range": recent_range,
            "avg_step": avg_step,
            "max_step": max_step,
            "range_tol": range_tol,
            "step_tol": step_tol,
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


def _is_near_feasible_slacks(
    max_vdrop_slack: float,
    max_kcl_slack: float,
    vdrop_tol: float = BEST_VDROP_TOL,
    kcl_tol: float = BEST_KCL_TOL
) -> bool:
    return (float(max_vdrop_slack) <= float(vdrop_tol)) and (float(max_kcl_slack) <= float(kcl_tol))


def _scheduled_slack_bounds(iter_idx: int) -> Tuple[float, float]:
    t = max(1, int(iter_idx))
    vdrop = max(float(VDROP_SLACK_FINAL), float(VDROP_SLACK_INIT) * (float(VDROP_SLACK_DECAY) ** (t - 1)))
    kcl = max(float(KCL_SLACK_FINAL), float(KCL_SLACK_INIT) * (float(KCL_SLACK_DECAY) ** (t - 1)))
    return float(vdrop), float(kcl)


def _scheduled_prox_bounds(iter_idx: int) -> Tuple[float, float, float]:
    t = max(1, int(iter_idx))
    pfrac = max(float(PSTEP_FRAC_FINAL), float(PSTEP_FRAC_INIT) * (float(PSTEP_FRAC_DECAY) ** (t - 1)))
    qfrac = max(float(QSTEP_FRAC_FINAL), float(QSTEP_FRAC_INIT) * (float(QSTEP_FRAC_DECAY) ** (t - 1)))
    vabs = max(float(VSTEP_ABS_FINAL), float(VSTEP_ABS_INIT) * (float(VSTEP_ABS_DECAY) ** (t - 1)))
    return float(pfrac), float(qfrac), float(vabs)


def _rescued_slack_bounds(vdrop_bound: float, kcl_bound: float, mult: float) -> Tuple[float, float]:
    return (
        min(float(VDROP_SLACK_RESCUE_CAP), float(vdrop_bound) * float(mult)),
        min(float(KCL_SLACK_RESCUE_CAP), float(kcl_bound) * float(mult)),
    )


def _rescue_mult_from_tag(tag: str) -> float:
    if "rescued_x5" in str(tag):
        return 5.0
    if "rescued_x2" in str(tag):
        return 2.0
    return 1.0


def _round_dependent_mipgap(iter_idx: int) -> float:
    if not USE_MIPGAP_SCHEDULE:
        return float(SCIP_GAP_LIMIT)
    t = int(iter_idx)
    if t < int(MIPGAP_MID_START):
        return float(MIPGAP_EARLY)
    if t < int(MIPGAP_LATE_START):
        return float(MIPGAP_MID)
    return float(MIPGAP_LATE)


def _gap_for_iter(t: int) -> float:
    return _round_dependent_mipgap(int(t))


def _compute_outer_ell_estimates(
    data: Dict[str, Any],
    v_state: Dict[int, float],
    P_state: Dict[Tuple[int, int], float],
    Q_state: Dict[Tuple[int, int], float],
    u_send_state: Dict[Tuple[int, int], float],
) -> Tuple[Dict[Tuple[int, int], float], Dict[Tuple[int, int], float], Dict[str, float]]:
    E = data["E"]
    T_set = set(data["T"])
    r = data["r"]
    x = data["x"]
    ellmax = data["ellmax"]

    ell_i: Dict[Tuple[int, int], float] = {}
    ell_v: Dict[Tuple[int, int], float] = {}
    mismatch_l1 = 0.0
    mismatch_linf = 0.0

    for (i, j) in E:
        if (i, j) in T_set:
            vs = float(u_send_state.get((i, j), float("nan")))
            if not np.isfinite(vs):
                vs = float(v_state[int(i)])
        else:
            vs = float(v_state[int(i)])
        vs = max(float(vs), V_SEND_FLOOR, DENOM_EPS)

        p = float(P_state[(i, j)])
        q = float(Q_state[(i, j)])
        vj = max(float(v_state[int(j)]), DENOM_EPS)

        ell_i_val = (p * p + q * q) / vs
        if ELL_CLIP_NONNEG:
            ell_i_val = max(0.0, float(ell_i_val))
        if ELL_CLIP_MAX:
            ell_i_val = min(float(ellmax[(i, j)]), float(ell_i_val))

        z2 = float(r[(i, j)]) * float(r[(i, j)]) + float(x[(i, j)]) * float(x[(i, j)])
        if z2 <= 1.0e-12:
            ell_v_val = float(ell_i_val)
        else:
            ell_v_val = (vj - vs + 2.0 * (float(r[(i, j)]) * p + float(x[(i, j)]) * q)) / z2
            if ELL_CLIP_NONNEG:
                ell_v_val = max(0.0, float(ell_v_val))
            if ELL_CLIP_MAX:
                ell_v_val = min(float(ellmax[(i, j)]), float(ell_v_val))

        ell_i[(i, j)] = float(ell_i_val)
        ell_v[(i, j)] = float(ell_v_val)
        dm = abs(float(ell_i_val) - float(ell_v_val))
        mismatch_l1 += dm
        mismatch_linf = max(mismatch_linf, dm)

    return ell_i, ell_v, {"ell_mismatch_l1": float(mismatch_l1), "ell_mismatch_linf": float(mismatch_linf)}


def _outer_hybrid_correct_ell(
    data: Dict[str, Any],
    ell_prev_state: Dict[Tuple[int, int], float],
    v_state: Dict[int, float],
    P_state: Dict[Tuple[int, int], float],
    Q_state: Dict[Tuple[int, int], float],
    u_send_state: Dict[Tuple[int, int], float],
) -> Tuple[Dict[Tuple[int, int], float], Dict[str, float]]:
    ell_i, ell_v, stats = _compute_outer_ell_estimates(data, v_state, P_state, Q_state, u_send_state)
    E = data["E"]
    ellmax = data["ellmax"]

    if not USE_OUTER_ELL_HYBRID:
        out = {}
        sum_diff = 0.0
        max_viol = 0.0
        for (i, j) in E:
            val = float(ell_i[(i, j)])
            if ELL_CLIP_NONNEG:
                val = max(0.0, val)
            if ELL_CLIP_MAX:
                val = min(float(ellmax[(i, j)]), val)
            out[(i, j)] = val
            sum_diff += abs(val - float(ell_prev_state[(i, j)]))
            max_viol = max(max_viol, max(0.0, -val), max(0.0, val - float(ellmax[(i, j)])))
        stats.update({"alpha_used": 1.0, "sum_diff": float(sum_diff), "max_viol": float(max_viol)})
        return out, stats

    target = {}
    nominal = {}
    for (i, j) in E:
        tval = float(ELL_HYBRID_I_WEIGHT) * float(ell_i[(i, j)]) + float(ELL_HYBRID_V_WEIGHT) * float(ell_v[(i, j)])
        if ELL_CLIP_NONNEG:
            tval = max(0.0, tval)
        if ELL_CLIP_MAX:
            tval = min(float(ellmax[(i, j)]), tval)
        target[(i, j)] = float(tval)
        nval = (1.0 - float(ELL_GAMMA)) * float(ell_prev_state[(i, j)]) + float(ELL_GAMMA) * float(tval)
        if ELL_CLIP_NONNEG:
            nval = max(0.0, nval)
        if ELL_CLIP_MAX:
            nval = min(float(ellmax[(i, j)]), nval)
        nominal[(i, j)] = float(nval)

    if USE_OUTER_ELL_FORCE_FOLLOW:
        out = {}
        sum_diff = 0.0
        max_viol = 0.0
        for (i, j) in E:
            val = float(target[(i, j)])
            if ELL_CLIP_NONNEG:
                val = max(0.0, val)
            if ELL_CLIP_MAX:
                val = min(float(ellmax[(i, j)]), val)
            out[(i, j)] = float(val)
            sum_diff += abs(float(val) - float(ell_prev_state[(i, j)]))
            max_viol = max(max_viol, max(0.0, -float(val)), max(0.0, float(val) - float(ellmax[(i, j)])))
        stats.update({"alpha_used": 1.0, "sum_diff": float(sum_diff), "max_viol": float(max_viol), "follow_mode": 1.0})
        return out, stats

    alpha_used = float(ELL_BACKTRACK_ALPHAS[-1])
    for alpha in ELL_BACKTRACK_ALPHAS:
        ok = True
        for (i, j) in E:
            prev = float(ell_prev_state[(i, j)])
            cand = prev + float(alpha) * (float(nominal[(i, j)]) - prev)
            scale = max(abs(prev), abs(float(target[(i, j)])), 1.0e-6)
            step_cap = max(float(ELL_BACKTRACK_ABS_STEP), float(ELL_BACKTRACK_REL_STEP) * scale)
            if abs(cand - prev) > step_cap + 1.0e-12:
                ok = False
                break
        if ok:
            alpha_used = float(alpha)
            break

    out = {}
    sum_diff = 0.0
    max_viol = 0.0
    for (i, j) in E:
        prev = float(ell_prev_state[(i, j)])
        val = prev + float(alpha_used) * (float(nominal[(i, j)]) - prev)
        if ELL_CLIP_NONNEG:
            val = max(0.0, val)
        if ELL_CLIP_MAX:
            val = min(float(ellmax[(i, j)]), val)
        out[(i, j)] = float(val)
        sum_diff += abs(float(val) - prev)
        max_viol = max(max_viol, max(0.0, -float(val)), max(0.0, float(val) - float(ellmax[(i, j)])))

    stats.update({"alpha_used": float(alpha_used), "sum_diff": float(sum_diff), "max_viol": float(max_viol)})
    return out, stats


# ============================================================
# [C11] Anderson type-II acceleration on ell_fix
# ============================================================
def _anderson_step(
    ell_in_hist: List[Dict[Tuple[int, int], float]],
    ell_out_hist: List[Dict[Tuple[int, int], float]],
    ell_curr_in: Dict[Tuple[int, int], float],
    ell_curr_out: Dict[Tuple[int, int], float],
    m_window: int = 5,
    reg: float = 1e-10,
) -> Dict[Tuple[int, int], float]:
    """Compute an Anderson-accelerated combination of past (ell_in, ell_out)
    pairs, returning the next ell_in guess.  Falls back to ell_curr_out if
    the LS system is degenerate.
    """
    keys = sorted(ell_curr_out.keys())
    n = len(keys)
    if n == 0 or len(ell_in_hist) == 0:
        return dict(ell_curr_out)

    window = min(int(m_window), len(ell_in_hist))
    ins = ell_in_hist[-window:] + [ell_curr_in]
    outs = ell_out_hist[-window:] + [ell_curr_out]
    M = len(ins)

    F = np.zeros((n, M), dtype=float)
    for k, (i_dict, o_dict) in enumerate(zip(ins, outs)):
        for r_idx, key in enumerate(keys):
            F[r_idx, k] = float(o_dict.get(key, 0.0)) - float(i_dict.get(key, 0.0))

    if M <= 1:
        return dict(ell_curr_out)
    dF = F[:, 1:] - F[:, :-1]
    fM = F[:, -1]

    try:
        AtA = dF.T @ dF + float(reg) * np.eye(dF.shape[1])
        Atb = dF.T @ fM
        gamma = np.linalg.solve(AtA, Atb)
    except np.linalg.LinAlgError:
        return dict(ell_curr_out)

    out_mat = np.zeros((n, M), dtype=float)
    for k, o_dict in enumerate(outs):
        for r_idx, key in enumerate(keys):
            out_mat[r_idx, k] = float(o_dict.get(key, 0.0))
    dOut = out_mat[:, 1:] - out_mat[:, :-1]
    next_vec = out_mat[:, -1] - dOut @ gamma

    out = {}
    for r_idx, key in enumerate(keys):
        out[key] = float(next_vec[r_idx])
    return out


# ============================================================
# [C13] Warm-start ell from a single BFM (MISOCP) solve (optional)
# ============================================================
def _warm_ell_from_socp(data: Dict[str, Any], time_limit: float) -> Optional[Dict[Tuple[int, int], float]]:
    """Run BFM_MISOCP once and extract ell estimates from it.  Returns None
    if the module is unavailable or the solve fails.  The 41-bus MISOCP build
    function is `build_pyomo_bfm_misocp_model` and the ell variable is m.ell.
    """
    try:
        import BFM_MISOCP as socp_mod  # lazy import; sibling file in this folder
    except Exception:
        return None

    try:
        net = mcase.busmeshed39_opf(**NETWORK_BUILD_KWARGS)
        cfg_local = socp_mod.build_cfg_from_net_metadata(net) if hasattr(socp_mod, "build_cfg_from_net_metadata") else None
        if cfg_local is None:
            cfg_local = build_cfg_from_net_metadata(net)
        # The MISOCP module exposes `extract_bfm41_fullmesh_data` when present.
        extract_fn = getattr(socp_mod, "extract_bfm41_fullmesh_data", None)
        if extract_fn is not None:
            data_local = extract_fn(net, cfg_local)
        else:
            data_local = extract_data_fullmesh_branch_table(net, cfg_local)

        build_fn = getattr(socp_mod, "build_pyomo_bfm_misocp_model", None) \
                   or getattr(socp_mod, "build_pyomo_socp_model", None) \
                   or getattr(socp_mod, "build_model", None)
        if build_fn is None:
            return None
        m = build_fn(data_local, relax_binaries=False) if "relax_binaries" in build_fn.__code__.co_varnames else build_fn(data_local)

        opt = None
        for name in ("gurobi", "scip"):
            cand = pyo.SolverFactory(name)
            if cand is not None and cand.available(exception_flag=False):
                opt = cand
                break
        if opt is None:
            return None
        try:
            opt.options["TimeLimit"] = float(time_limit)
        except Exception:
            pass
        try:
            opt.options["limits/time"] = float(time_limit)
        except Exception:
            pass
        opt.solve(m, tee=False, load_solutions=True)
        ell_attr = getattr(m, "ell", None) or getattr(m, "l", None)
        if ell_attr is None:
            return None
        out = {}
        for (i, j) in data["E"]:
            try:
                out[(int(i), int(j))] = float(pyo.value(ell_attr[int(i), int(j)], exception=False) or 0.0)
            except Exception:
                out[(int(i), int(j))] = 0.0
        return out
    except Exception:
        return None


# ============================================================
# PF-based warm-start (standard iterative-OPF technique)
# ============================================================
def _pf_warm_start(net, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Run pandapower PF on `net` and return BFMag-compatible warm-start
    dictionaries: v (|V|^2), Pij, Qij (pu, sending end), ell, Pg/Qg (pu).
    Returns None when the PF fails or pandapower is unavailable.
    """
    if pp is None:
        return None
    try:
        pp.runpp(net, algorithm="nr", max_iteration=50, init="auto", tolerance_mva=1e-8)
    except Exception:
        try:
            pp.runpp(net, algorithm="nr", max_iteration=100)
        except Exception:
            return None

    if not hasattr(net, "res_bus") or net.res_bus is None or net.res_bus.empty:
        return None

    sn = float(data["sn_mva"])
    E = data["E"]
    buses = data["buses"]
    ellmax = data["ellmax"]
    branch_elements = data["branch_elements"]
    branch_original_dirs = data["branch_original_dirs"]
    gen_records = data["gen_records"]

    v_init: Dict[int, float] = {}
    for i in buses:
        try:
            vm = float(net.res_bus.at[int(i), "vm_pu"])
        except Exception:
            vm = 1.0
        if not np.isfinite(vm) or vm <= 0.0:
            vm = 1.0
        v_init[int(i)] = vm * vm

    P_init: Dict[Tuple[int, int], float] = {(i, j): 0.0 for (i, j) in E}
    Q_init: Dict[Tuple[int, int], float] = {(i, j): 0.0 for (i, j) in E}
    for (i, j) in E:
        elems = branch_elements.get((i, j), [])
        origs = branch_original_dirs.get((i, j), [])
        p_mw = 0.0
        q_mvar = 0.0
        for k, (et, eidx) in enumerate(elems):
            fb0, tb0 = origs[k] if k < len(origs) else (i, j)
            if et == "line" and hasattr(net, "res_line") and int(eidx) in net.res_line.index:
                if (int(fb0), int(tb0)) == (int(i), int(j)):
                    p_mw += float(net.res_line.at[int(eidx), "p_from_mw"])
                    q_mvar += float(net.res_line.at[int(eidx), "q_from_mvar"])
                else:
                    p_mw += float(net.res_line.at[int(eidx), "p_to_mw"])
                    q_mvar += float(net.res_line.at[int(eidx), "q_to_mvar"])
            elif et == "trafo" and hasattr(net, "res_trafo") and int(eidx) in net.res_trafo.index:
                if (int(fb0), int(tb0)) == (int(i), int(j)):
                    p_mw += float(net.res_trafo.at[int(eidx), "p_hv_mw"])
                    q_mvar += float(net.res_trafo.at[int(eidx), "q_hv_mvar"])
                else:
                    p_mw += float(net.res_trafo.at[int(eidx), "p_lv_mw"])
                    q_mvar += float(net.res_trafo.at[int(eidx), "q_lv_mvar"])
        P_init[(i, j)] = p_mw / sn
        Q_init[(i, j)] = q_mvar / sn

    ell_init: Dict[Tuple[int, int], float] = {}
    for (i, j) in E:
        v_send = max(float(v_init[int(i)]), 1e-6)
        p = float(P_init[(i, j)])
        q = float(Q_init[(i, j)])
        val = (p * p + q * q) / v_send
        val = max(0.0, min(val, float(ellmax[(i, j)])))
        ell_init[(i, j)] = val

    Pg_init: Dict[int, float] = {}
    Qg_init: Dict[int, float] = {}
    for g, rec in enumerate(gen_records):
        try:
            if rec.get("type") == "ext_grid":
                eg = int(rec["id"])
                if hasattr(net, "res_ext_grid") and eg in net.res_ext_grid.index:
                    Pg_init[g] = float(net.res_ext_grid.at[eg, "p_mw"]) / sn
                    Qg_init[g] = float(net.res_ext_grid.at[eg, "q_mvar"]) / sn
            elif rec.get("type") == "gen":
                gi = int(rec["id"])
                if hasattr(net, "res_gen") and gi in net.res_gen.index:
                    Pg_init[g] = float(net.res_gen.at[gi, "p_mw"]) / sn
                    Qg_init[g] = float(net.res_gen.at[gi, "q_mvar"]) / sn
        except Exception:
            continue

    return {
        "v": v_init,
        "Pij": P_init,
        "Qij": Q_init,
        "ell": ell_init,
        "Pg": Pg_init,
        "Qg": Qg_init,
    }


# ============================================================
# [C15] Adaptive over-relaxation (omega) on ell update
# ============================================================
def _update_omega(
    omega: float,
    sign_prev: Optional[Dict[Tuple[int, int], int]],
    ell_raw: Dict[Tuple[int, int], float],
    ell_prev: Dict[Tuple[int, int], float],
) -> Tuple[float, Dict[Tuple[int, int], int]]:
    """Adjust omega based on per-edge sign stability of (ell_raw - ell_prev)."""
    sign_new = {}
    for k in ell_raw.keys():
        diff = float(ell_raw[k]) - float(ell_prev.get(k, 0.0))
        if diff > 1e-10:
            sign_new[k] = +1
        elif diff < -1e-10:
            sign_new[k] = -1
        else:
            sign_new[k] = 0

    if sign_prev is None or len(sign_prev) == 0:
        return float(omega), sign_new

    stable = 0
    total = 0
    for k, s in sign_new.items():
        if k in sign_prev and s != 0 and sign_prev[k] != 0:
            total += 1
            if s == sign_prev[k]:
                stable += 1
    if total == 0:
        return float(omega), sign_new
    frac = float(stable) / float(total)

    if frac >= 0.75:
        new_omega = min(float(OMEGA_MAX), float(omega) * float(OMEGA_UP))
    elif frac <= 0.45:
        new_omega = max(float(OMEGA_MIN), float(omega) * float(OMEGA_DOWN))
    else:
        new_omega = float(omega)
    return float(new_omega), sign_new


# ============================================================
# Data extraction (meshed, keep ALL net.line rows)
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
    recommended_taps: Dict[Tuple[int, int], int]
    recommended_shunts: Dict[int, int]
    fix_slack_vm: bool = True


def build_cfg_from_net_metadata(net) -> BuildConfig:
    if "fixed_oltc_table" not in net:
        raise KeyError("Network metadata 'fixed_oltc_table' not found.")
    if "fixed_shunt_table" not in net:
        raise KeyError("Network metadata 'fixed_shunt_table' not found.")

    def _first_present(row, *names):
        for name in names:
            if name in row.index:
                return row[name]
        raise KeyError(f"None of columns {names} found in metadata row.")

    oltc_branches: Dict[Tuple[int, int], OLTCBranchConfig] = {}
    shunt_bcap_pu: Dict[int, float] = {}
    recommended_taps: Dict[Tuple[int, int], int] = {}
    recommended_shunts: Dict[int, int] = {}

    oltc_df = net["fixed_oltc_table"]
    shunt_df = net["fixed_shunt_table"]

    for _, row in oltc_df.iterrows():
        i = int(_first_present(row, "from_bus_pp", "from_bus"))
        j = int(_first_present(row, "to_bus_pp", "to_bus"))
        oltc_branches[(i, j)] = OLTCBranchConfig(
            tap_min=int(row["tap_min"]),
            tap_max=int(row["tap_max"]),
            dV_percent=float(row["dV_percent"]),
        )
        if "recommended_tap" in row.index:
            try:
                rv = float(row["recommended_tap"])
                if not math.isnan(rv):
                    recommended_taps[(i, j)] = int(rv)
            except Exception:
                pass

    for _, row in shunt_df.iterrows():
        b = int(_first_present(row, "bus_pp", "bus"))
        shunt_bcap_pu[b] = float(row["bcap_pu"])

    if "recommended_nonexact_oltc_taps" in net:
        rec_tap_df = net["recommended_nonexact_oltc_taps"]
        if rec_tap_df is not None and len(rec_tap_df.index) > 0:
            for _, row in rec_tap_df.iterrows():
                i = int(_first_present(row, "from_bus_pp", "from_bus"))
                j = int(_first_present(row, "to_bus_pp", "to_bus"))
                recommended_taps[(i, j)] = int(row["tap"])

    if "recommended_nonexact_shunt_status" in net:
        rec_sh_df = net["recommended_nonexact_shunt_status"]
        if rec_sh_df is not None and len(rec_sh_df.index) > 0:
            for _, row in rec_sh_df.iterrows():
                b = int(_first_present(row, "bus_pp", "bus"))
                recommended_shunts[b] = int(row["status"])

    return BuildConfig(
        oltc_branches=oltc_branches,
        shunt_bcap_pu=shunt_bcap_pu,
        recommended_taps=recommended_taps,
        recommended_shunts=recommended_shunts,
        fix_slack_vm=True,
    )


def _merge_parallel_series_equivalent(rows: List[dict]) -> Tuple[float, float, float]:
    """Merge parallel branches with the same directed endpoints into one
    equivalent branch using admittance summation.  Thermal ratings sum.
    """
    if len(rows) == 1:
        r = float(rows[0]["r_pu"])
        x = float(rows[0]["x_pu"])
        s = float(rows[0]["synthetic_smax_mva"])
        return r, x, s

    ysum = 0.0 + 0.0j
    ssum = 0.0
    for row in rows:
        z = complex(float(row["r_pu"]), float(row["x_pu"]))
        if abs(z) <= 1e-12:
            y = complex(1e12, 0.0)
        else:
            y = 1.0 / z
        ysum += y
        ssum += max(0.0, float(row["synthetic_smax_mva"]))

    if abs(ysum) <= 1e-12:
        r = float(sum(float(rw["r_pu"]) for rw in rows) / len(rows))
        x = float(sum(float(rw["x_pu"]) for rw in rows) / len(rows))
        return r, x, ssum

    zeq = 1.0 / ysum
    return float(zeq.real), float(zeq.imag), float(ssum)


def _make_branch_params_df_from_net(net) -> pd.DataFrame:
    """Return a standardized branch parameter table.  Supports either
    net["branch_params_pu_table"] (300-bus style) or
    net["line_params_pu_table"] + net.line (ieee41bus.py style).
    """
    if "branch_params_pu_table" in net and net["branch_params_pu_table"] is not None and not net["branch_params_pu_table"].empty:
        brdf = net["branch_params_pu_table"].copy()
        rename_map = {}
        if "from_bus" in brdf.columns and "from_bus_pp" not in brdf.columns:
            rename_map["from_bus"] = "from_bus_pp"
        if "to_bus" in brdf.columns and "to_bus_pp" not in brdf.columns:
            rename_map["to_bus"] = "to_bus_pp"
        if rename_map:
            brdf = brdf.rename(columns=rename_map)
        if "element_type" not in brdf.columns:
            brdf["element_type"] = "line"
        if "element_index" not in brdf.columns:
            brdf["element_index"] = range(len(brdf))
        if "synthetic_smax_mva" not in brdf.columns:
            if hasattr(net, "line") and len(net.line.index) == len(brdf):
                vals = []
                for idx_row in brdf.index:
                    line_idx = int(brdf.at[idx_row, "element_index"]) if "element_index" in brdf.columns else int(idx_row)
                    fb = int(brdf.at[idx_row, "from_bus_pp"])
                    vn_kv = float(net.bus.at[fb, "vn_kv"])
                    imax = float(net.line.at[line_idx, "max_i_ka"])
                    vals.append(math.sqrt(3.0) * vn_kv * imax)
                brdf["synthetic_smax_mva"] = vals
            else:
                raise KeyError("branch_params_pu_table lacks synthetic_smax_mva and cannot infer it robustly.")
        return brdf

    if "line_params_pu_table" in net and net["line_params_pu_table"] is not None and not net["line_params_pu_table"].empty:
        lpdf = net["line_params_pu_table"].copy().reset_index(drop=True)
        if not hasattr(net, "line") or net.line is None or net.line.empty:
            raise KeyError("line_params_pu_table exists but net.line is empty.")
        ldf = net.line.copy().reset_index().rename(columns={"index": "element_index"})
        if len(ldf) != len(lpdf):
            raise ValueError(
                f"Mismatch between net.line ({len(ldf)}) and line_params_pu_table ({len(lpdf)}) lengths."
            )

        rows = []
        for k in range(len(lpdf)):
            prow = lpdf.iloc[k]
            lrow = ldf.iloc[k]
            fb = int(prow["from_bus"])
            tb = int(prow["to_bus"])
            vn_kv = float(net.bus.at[fb, "vn_kv"])
            imax = float(lrow["max_i_ka"])
            smax_mva = math.sqrt(3.0) * vn_kv * imax
            rows.append({
                "branch_id": int(k),
                "from_bus_pp": fb,
                "to_bus_pp": tb,
                "from_bus_mp": fb + 1,
                "to_bus_mp": tb + 1,
                "r_pu": float(prow["r_pu_used"]),
                "x_pu": float(prow["x_pu_used"]),
                "b_pu": 0.0,
                "ratio_raw": 0.0,
                "angle_deg": 0.0,
                "element_type": "line",
                "element_index": int(lrow["element_index"]),
                "is_transformer_like": bool(prow.get("is_transformer_origin", False)),
                "synthetic_smax_mva": float(smax_mva),
            })
        return pd.DataFrame(rows)

    raise KeyError("Neither 'branch_params_pu_table' nor 'line_params_pu_table' metadata were found.")


def _fundamental_cycles(
    buses: List[int],
    edges: List[Tuple[int, int]],
    slack: int,
    max_cycle_len: int = 12,
) -> List[List[Tuple[Tuple[int, int], int]]]:
    """Build a spanning tree rooted at `slack` and emit fundamental cycles for
    each non-tree edge.  Each cycle is a list of (directed_edge, sign) pairs.
    """
    adj: Dict[int, List[Tuple[int, Tuple[int, int]]]] = {int(b): [] for b in buses}
    for (i, j) in edges:
        adj[int(i)].append((int(j), (int(i), int(j))))
        adj[int(j)].append((int(i), (int(i), int(j))))

    parent: Dict[int, Optional[Tuple[int, Tuple[int, int]]]] = {int(b): None for b in buses}
    tree_edges: set = set()
    visited = {int(slack)}
    stack = [int(slack)]
    while stack:
        u = stack.pop()
        for (v, e) in adj[u]:
            if v in visited:
                continue
            visited.add(v)
            parent[v] = (u, e)
            tree_edges.add(e)
            stack.append(v)

    def _path_to_root(b: int) -> List[int]:
        chain = [int(b)]
        cur = int(b)
        while parent[cur] is not None:
            pu, _ = parent[cur]
            chain.append(int(pu))
            cur = int(pu)
        return chain

    cycles: List[List[Tuple[Tuple[int, int], int]]] = []
    for e in edges:
        if e in tree_edges:
            continue
        i, j = int(e[0]), int(e[1])
        path_i = _path_to_root(i)
        path_j = _path_to_root(j)
        set_i = set(path_i)
        lca = None
        for node in path_j:
            if node in set_i:
                lca = int(node)
                break
        if lca is None:
            continue

        cycle: List[Tuple[Tuple[int, int], int]] = []
        u = i
        while u != lca:
            pu, pe = parent[u]
            sign = +1 if (pe[1] == u) else -1
            cycle.append((pe, -sign))
            u = pu
        cycle.append((e, +1))
        rev: List[Tuple[Tuple[int, int], int]] = []
        u = j
        while u != lca:
            pu, pe = parent[u]
            sign = +1 if (pe[1] == u) else -1
            rev.append((pe, sign))
            u = pu
        cycle.extend(rev[::-1])

        if len(cycle) <= int(max_cycle_len):
            cycles.append(cycle)

    return cycles


def _obbt_tighten_bounds(data: Dict[str, Any], time_per_lp: float = None) -> Optional[Dict[str, Dict[Any, Tuple[float, float]]]]:
    """[A4] LP-based OBBT preprocessing.  Relaxation replaces SOC thermal by
    the rectangle |P|,|Q|<=Smax and treats ell as free in [0, ellmax].  Any
    MIQCP iterate satisfies this LP; min/max on Pij, Qij, v give valid OBBT
    bounds that can tighten the subproblem's variable box.
    """
    if time_per_lp is None:
        time_per_lp = float(OBBT_TIME_LIMIT)
    E = data["E"]
    buses = data["buses"]
    Smax = data["Smax"]
    Vmin = data["Vmin_pu"]
    Vmax = data["Vmax_pu"]
    ellmax = data["ellmax"]
    r = data["r"]
    x = data["x"]
    out_arcs = data["out_arcs"]
    in_arcs = data["in_arcs"]
    slack_bus = int(data["slack_bus"])
    slack_vm_pu = float(data["slack_vm_pu"])
    Pd = data["Pd_pu"]
    Qd = data["Qd_pu"]
    gen_records = data["gen_records"]
    Gset = list(range(len(gen_records)))
    T_set = set(data["T"])
    kvl_cycles = data.get("kvl_cycles", [])

    m = pyo.ConcreteModel("OBBT")
    m.E = pyo.Set(initialize=E, dimen=2, ordered=True)
    m.N = pyo.Set(initialize=buses, ordered=True)
    m.G = pyo.Set(initialize=Gset, ordered=True)

    m.Pij = pyo.Var(m.E, bounds=lambda mm, i, j: (-float(Smax[(i, j)]), float(Smax[(i, j)])))
    m.Qij = pyo.Var(m.E, bounds=lambda mm, i, j: (-float(Smax[(i, j)]), float(Smax[(i, j)])))
    m.v = pyo.Var(m.N, bounds=lambda mm, i: (float(Vmin[int(i)]) ** 2, float(Vmax[int(i)]) ** 2))
    m.ell = pyo.Var(m.E, bounds=lambda mm, i, j: (0.0, float(ellmax[(i, j)])))
    m.Pg = pyo.Var(m.G, bounds=lambda mm, g: (float(gen_records[int(g)]["pmin_pu"]),
                                              float(gen_records[int(g)]["pmax_pu"])))
    m.Qg = pyo.Var(m.G, bounds=lambda mm, g: (float(gen_records[int(g)]["qmin_pu"]),
                                              float(gen_records[int(g)]["qmax_pu"])))

    m.slack_v = pyo.Constraint(expr=m.v[slack_bus] == float(slack_vm_pu) ** 2)

    def _kcl_P(mm, i):
        i = int(i)
        gen = pyo.quicksum(mm.Pg[g] for g in Gset if int(gen_records[g]["bus"]) == i)
        return (
            pyo.quicksum(mm.Pij[a, b] for (a, b) in out_arcs[i])
            - pyo.quicksum(mm.Pij[a, b] - float(r[(a, b)]) * mm.ell[a, b] for (a, b) in in_arcs[i])
            == gen - float(Pd[i])
        )
    m.kcl_P = pyo.Constraint(m.N, rule=_kcl_P)

    def _kcl_Q(mm, i):
        i = int(i)
        gen = pyo.quicksum(mm.Qg[g] for g in Gset if int(gen_records[g]["bus"]) == i)
        return (
            pyo.quicksum(mm.Qij[a, b] for (a, b) in out_arcs[i])
            - pyo.quicksum(mm.Qij[a, b] - float(x[(a, b)]) * mm.ell[a, b] for (a, b) in in_arcs[i])
            == gen - float(Qd[i])
        )
    m.kcl_Q = pyo.Constraint(m.N, rule=_kcl_Q)

    m.vdrop = pyo.ConstraintList()
    for (i, j) in E:
        if (i, j) in T_set:
            continue
        rij = float(r[(int(i), int(j))])
        xij = float(x[(int(i), int(j))])
        z2 = rij * rij + xij * xij
        m.vdrop.add(
            m.v[int(j)] == m.v[int(i)] - 2.0 * (rij * m.Pij[int(i), int(j)] + xij * m.Qij[int(i), int(j)])
            + z2 * m.ell[int(i), int(j)]
        )

    if USE_KVL_CYCLES and len(kvl_cycles) > 0:
        m.kvl = pyo.ConstraintList()
        for cyc in kvl_cycles:
            lhs = 0.0
            for ((ei, ej), sign) in cyc:
                rij = float(r[(int(ei), int(ej))])
                xij = float(x[(int(ei), int(ej))])
                z2 = rij * rij + xij * xij
                lhs = lhs + float(sign) * (
                    2.0 * (rij * m.Pij[int(ei), int(ej)] + xij * m.Qij[int(ei), int(ej)])
                    - z2 * m.ell[int(ei), int(ej)]
                )
            m.kvl.add(lhs == 0.0)

    m.obbt_obj = pyo.Objective(expr=0.0, sense=pyo.minimize)

    opt = None
    for name in ("gurobi", "scip"):
        try:
            cand = pyo.SolverFactory(name)
            if cand is not None and cand.available(exception_flag=False):
                opt = cand
                break
        except Exception:
            continue
    if opt is None:
        return None

    try:
        opt.options["TimeLimit"] = float(time_per_lp)
    except Exception:
        pass
    try:
        opt.options["limits/time"] = float(time_per_lp)
    except Exception:
        pass
    try:
        opt.options["OutputFlag"] = 0
    except Exception:
        pass
    try:
        opt.options["display/verblevel"] = 0
    except Exception:
        pass

    def _opt_val(expr, sense):
        try:
            m.del_component(m.obbt_obj)
        except Exception:
            pass
        m.obbt_obj = pyo.Objective(expr=expr, sense=sense)
        try:
            res = opt.solve(m, tee=False, load_solutions=True)
            tc = res.solver.termination_condition
            if tc == TerminationCondition.infeasible:
                return None
            v = pyo.value(expr, exception=False)
            return float(v) if v is not None else None
        except Exception:
            return None

    P_bounds: Dict[Tuple[int, int], Tuple[float, float]] = {}
    Q_bounds: Dict[Tuple[int, int], Tuple[float, float]] = {}
    v_bounds: Dict[int, Tuple[float, float]] = {}

    edges = list(E)
    if OBBT_MAX_EDGES is not None:
        edges = edges[: int(OBBT_MAX_EDGES)]

    for (i, j) in edges:
        lo = _opt_val(m.Pij[i, j], pyo.minimize)
        hi = _opt_val(m.Pij[i, j], pyo.maximize)
        if (lo is not None) and (hi is not None) and (hi - lo) > 1e-9:
            S = float(Smax[(i, j)])
            if (hi - lo) < (2.0 * S) * (1.0 - float(OBBT_MIN_SHRINK)):
                P_bounds[(int(i), int(j))] = (float(lo), float(hi))
        lo = _opt_val(m.Qij[i, j], pyo.minimize)
        hi = _opt_val(m.Qij[i, j], pyo.maximize)
        if (lo is not None) and (hi is not None) and (hi - lo) > 1e-9:
            S = float(Smax[(i, j)])
            if (hi - lo) < (2.0 * S) * (1.0 - float(OBBT_MIN_SHRINK)):
                Q_bounds[(int(i), int(j))] = (float(lo), float(hi))

    for i in buses:
        lo = _opt_val(m.v[int(i)], pyo.minimize)
        hi = _opt_val(m.v[int(i)], pyo.maximize)
        if (lo is not None) and (hi is not None) and (hi - lo) > 1e-9:
            orig_lo = float(Vmin[int(i)]) ** 2
            orig_hi = float(Vmax[int(i)]) ** 2
            if (hi - lo) < (orig_hi - orig_lo) * (1.0 - float(OBBT_MIN_SHRINK)):
                v_bounds[int(i)] = (float(lo), float(hi))

    return {"P": P_bounds, "Q": Q_bounds, "v": v_bounds}


def extract_data_fullmesh_branch_table(net, cfg: BuildConfig) -> Dict[str, Any]:
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
        gen_records.append(dict(type="ext_grid", id=eg, bus=b,
                                pmin_pu=pmin/sn, pmax_pu=pmax/sn,
                                qmin_pu=qmin/sn, qmax_pu=qmax/sn,
                                c2=c2, c1=c1, c0=c0))

    if hasattr(net, "gen") and len(net.gen.index) > 0:
        for gi in net.gen.index:
            gi = int(gi)
            b = int(net.gen.at[gi, "bus"])
            pmin = float(net.gen.at[gi, "min_p_mw"])
            pmax = float(net.gen.at[gi, "max_p_mw"])
            qmin = float(net.gen.at[gi, "min_q_mvar"])
            qmax = float(net.gen.at[gi, "max_q_mvar"])
            c2, c1, c0 = _find_poly_cost(net, "gen", gi)
            gen_records.append(dict(type="gen", id=gi, bus=b,
                                    pmin_pu=pmin/sn, pmax_pu=pmax/sn,
                                    qmin_pu=qmin/sn, qmax_pu=qmax/sn,
                                    c2=c2, c1=c1, c0=c0))

    for rec in gen_records:
        if float(rec["c2"]) < -1e-12:
            raise ValueError("Nonconvex quadratic cost detected (cp2 < 0).")

    brdf = _make_branch_params_df_from_net(net)
    oltc_dir_set = set(cfg.oltc_branches.keys())

    grouped: Dict[Tuple[int, int], List[dict]] = {}
    element_group: Dict[Tuple[int, int], List[Tuple[str, int]]] = {}
    original_oriented: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}

    for _, row in brdf.iterrows():
        fb0 = int(row["from_bus_pp"])
        tb0 = int(row["to_bus_pp"])
        if (fb0, tb0) in oltc_dir_set:
            fb, tb = fb0, tb0
        elif (tb0, fb0) in oltc_dir_set:
            fb, tb = tb0, fb0
        else:
            fb, tb = fb0, tb0

        key = (fb, tb)
        grouped.setdefault(key, []).append(row)
        element_group.setdefault(key, []).append((str(row["element_type"]), int(row["element_index"])))
        original_oriented.setdefault(key, []).append((fb0, tb0))

    E: List[Tuple[int, int]] = []
    r: Dict[Tuple[int, int], float] = {}
    x: Dict[Tuple[int, int], float] = {}
    Smax: Dict[Tuple[int, int], float] = {}
    ellmax: Dict[Tuple[int, int], float] = {}
    branch_elements: Dict[Tuple[int, int], List[Tuple[str, int]]] = {}
    branch_original_dirs: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}

    for key, rows in grouped.items():
        fb, tb = key
        E.append(key)
        req_pu, xeq_pu, smax_mva = _merge_parallel_series_equivalent(rows)
        r[key] = float(req_pu)
        x[key] = float(xeq_pu)
        smax_pu = float(smax_mva) / sn
        Smax[key] = smax_pu
        vmin_sq = max(Vmin_pu[fb] ** 2, 1e-6)
        ellmax[key] = float((smax_pu ** 2) / vmin_sq)
        branch_elements[key] = list(element_group[key])
        branch_original_dirs[key] = list(original_oriented[key])

    out_arcs = {i: [] for i in buses}
    in_arcs = {i: [] for i in buses}
    for (i, j) in E:
        out_arcs[i].append((i, j))
        in_arcs[j].append((i, j))

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

    slack_gen_idx = None
    slack_c2 = 0.0
    slack_c1 = 0.0
    for idx, rec in enumerate(gen_records):
        if rec["type"] == "ext_grid":
            slack_gen_idx = idx
            slack_c2 = float(rec["c2"])
            slack_c1 = float(rec["c1"])
            break

    # [A5] Pre-compute fundamental cycles on the non-OLTC subgraph so the
    # MIQCP can carry KVL equality cuts around every loop.
    kvl_cycles: List[List[Tuple[Tuple[int, int], int]]] = []
    if USE_KVL_CYCLES:
        oltc_set = set(T)
        non_oltc_edges = [e for e in E if e not in oltc_set]
        if len(non_oltc_edges) > 0:
            all_cycles = _fundamental_cycles(
                buses=buses,
                edges=non_oltc_edges,
                slack=int(slack_bus),
                max_cycle_len=int(KVL_MAX_CYCLE_LEN),
            )
            for cyc in all_cycles:
                if all(e not in oltc_set for (e, _s) in cyc):
                    kvl_cycles.append(cyc)

    return dict(
        sn_mva=sn,
        buses=buses,
        slack_bus=slack_bus,
        slack_vm_pu=slack_vm_pu,
        bus_vn_kv=bus_vn_kv,
        Vmin_pu=Vmin_pu,
        Vmax_pu=Vmax_pu,
        Pd_pu=Pd_pu,
        Qd_pu=Qd_pu,
        gen_records=gen_records,
        slack_gen_idx=slack_gen_idx,
        slack_c2=slack_c2,
        slack_c1=slack_c1,
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
        recommended_taps=dict(cfg.recommended_taps),
        recommended_shunts=dict(cfg.recommended_shunts),
        branch_elements=branch_elements,
        branch_original_dirs=branch_original_dirs,
        fix_slack_vm=cfg.fix_slack_vm,
        kvl_cycles=kvl_cycles,
    )


# ============================================================
# 1번 변화: loss proxy coefficient builder
# ============================================================
def compute_loss_proxy_coeffs(
    *,
    data: Dict[str, Any],
    Pbar: Dict[Tuple[int, int], float],
    Qbar: Dict[Tuple[int, int], float],
    vbar: Dict[int, float],
    ubar: Optional[Dict[Tuple[int, int], float]],
    w_loss: float,
) -> Dict[str, Dict[Any, float]]:
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
        if (i, j) in T_set and (ubar is not None) and ((i, j) in ubar) and np.isfinite(ubar[(i, j)]):
            vS = float(ubar[(i, j)])
        elif (ubar is not None) and ((i, j) in ubar) and np.isfinite(ubar[(i, j)]):
            # [C14] When theta reuse is active ubar carries non-NaN entries for
            # non-OLTC edges too; honor them.
            vS = float(ubar[(i, j)])
        else:
            vS = float(vbar[int(i)])

        vS = max(vS, V_SEND_FLOOR, EPS_V_LIN)

        P0 = float(Pbar[(i, j)])
        Q0 = float(Qbar[(i, j)])

        l0 = (P0 * P0 + Q0 * Q0) / vS

        aP_ = 2.0 * P0 / vS
        aQ_ = 2.0 * Q0 / vS
        aV_ = -(P0 * P0 + Q0 * Q0) / (vS * vS)

        b0_ = l0 - aP_ * P0 - aQ_ * Q0 - aV_ * vS

        aP[(i, j)] = aP_
        aQ[(i, j)] = aQ_
        aV[(i, j)] = aV_
        b0[(i, j)] = b0_

        gamma = float(w_loss) * sn * float(r[(i, j)])
        cP[(i, j)] = gamma * aP_
        cQ[(i, j)] = gamma * aQ_
        cV[(i, j)] = gamma * aV_

    return {
        "aP": aP, "aQ": aQ, "aV": aV, "b0": b0,
        "cP": cP, "cQ": cQ, "cV": cV
    }


# ============================================================
# Subproblem (ell fixed): bounded vdrop/KCL slacks + 1번 변화
# + ver3 additions [A1]-[A6], [B9] fix-and-relax kwargs, [C12] trust region
# ============================================================
def build_subproblem(
    data: Dict[str, Any],
    ell_fix: Dict[Tuple[int, int], float],
    relax_binaries: bool = False,
    warm: Optional[Dict[str, Any]] = None,
    loss_proxy: Optional[Dict[str, Dict[Any, float]]] = None,
    prox_prev: Optional[Dict[str, Dict[Any, float]]] = None,
    vdrop_slack_max: float = VDROP_SLACK_FINAL,
    kcl_slack_max: float = KCL_SLACK_FINAL,
    p_step_frac: float = PSTEP_FRAC_FINAL,
    q_step_frac: float = QSTEP_FRAC_FINAL,
    v_step_abs: float = VSTEP_ABS_FINAL,
    # [A6] Multi-cut loss: list of Taylor planes (aP, aQ, aV, b0 per edge).
    loss_planes: Optional[List[Dict[str, Dict[Tuple[int, int], float]]]] = None,
    w_loss_current: float = 0.0,
    # [B9] Fix-and-relax overrides.
    fix_beta: Optional[Dict[Tuple[int, int, int], int]] = None,
    fix_ash: Optional[Dict[int, int]] = None,
    # [C12] Adaptive trust-region radii.
    tr_P: Optional[float] = None,
    tr_Q: Optional[float] = None,
    tr_V: Optional[float] = None,
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

    m = pyo.ConcreteModel("BFM_ag_subproblem_boundedslack_obj1")

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

    def _init_zero_edge(mm, i, j):
        return 0.0

    m.aP_edge = pyo.Param(m.E, initialize=_init_zero_edge, mutable=True)
    m.aQ_edge = pyo.Param(m.E, initialize=_init_zero_edge, mutable=True)
    m.aV_edge = pyo.Param(m.E, initialize=_init_zero_edge, mutable=True)
    m.b0_edge = pyo.Param(m.E, initialize=_init_zero_edge, mutable=True)

    m.cP_edge = pyo.Param(m.E, initialize=_init_zero_edge, mutable=True)
    m.cQ_edge = pyo.Param(m.E, initialize=_init_zero_edge, mutable=True)
    m.cV_edge = pyo.Param(m.E, initialize=_init_zero_edge, mutable=True)

    m.Pprev = pyo.Param(m.E, initialize=_init_zero_edge, mutable=True)
    m.Qprev = pyo.Param(m.E, initialize=_init_zero_edge, mutable=True)
    m.vprev = pyo.Param(m.N, initialize=lambda mm, i: 1.0, mutable=True)

    m.Pg = pyo.Var(m.G, bounds=lambda mm, g: (mm.Pgmin[g], mm.Pgmax[g]))
    m.Qg = pyo.Var(m.G, bounds=lambda mm, g: (mm.Qgmin[g], mm.Qgmax[g]))
    m.v = pyo.Var(m.N, bounds=lambda mm, i: (mm.Vmin[i] ** 2, mm.Vmax[i] ** 2))
    m.Pij = pyo.Var(m.E, bounds=lambda mm, i, j: (-mm.Smax[i, j], mm.Smax[i, j]))
    m.Qij = pyo.Var(m.E, bounds=lambda mm, i, j: (-mm.Smax[i, j], mm.Smax[i, j]))
    m.Pinj = pyo.Var(m.N)
    m.Qinj = pyo.Var(m.N)

    # [A4] Apply OBBT-tightened bounds if available on the data dict.
    obbt = data.get("obbt", None)
    if USE_OBBT and obbt is not None:
        for (ii, jj), (lo, hi) in obbt.get("P", {}).items():
            if (int(ii), int(jj)) in m.Pij:
                m.Pij[int(ii), int(jj)].setlb(float(lo))
                m.Pij[int(ii), int(jj)].setub(float(hi))
        for (ii, jj), (lo, hi) in obbt.get("Q", {}).items():
            if (int(ii), int(jj)) in m.Qij:
                m.Qij[int(ii), int(jj)].setlb(float(lo))
                m.Qij[int(ii), int(jj)].setub(float(hi))
        for ii, (lo, hi) in obbt.get("v", {}).items():
            if int(ii) in m.v:
                m.v[int(ii)].setlb(float(lo))
                m.v[int(ii)].setub(float(hi))

    if relax_binaries:
        m.a_sh = pyo.Var(m.C, bounds=(0.0, 1.0))
    else:
        m.a_sh = pyo.Var(m.C, within=pyo.Binary)
    m.z = pyo.Var(m.C)

    if relax_binaries:
        m.beta = pyo.Var(m.BETA_INDEX, bounds=(0.0, 1.0))
    else:
        m.beta = pyo.Var(m.BETA_INDEX, within=pyo.Binary)

    def _tv_bounds(mm, i, j, tap):
        vU = float(Vmax[int(i)] ** 2)
        return (0.0, vU)
    m.tv = pyo.Var(m.BETA_INDEX, bounds=_tv_bounds)

    # Exact-equality mode preserved from upgrade3: keep legacy slack symbols for
    # compatibility but fix them to zero so vdrop/KCL residuals must vanish.
    m.s_vdrop = pyo.Var(m.E, bounds=(0.0, 0.0))

    m.sP_pos = pyo.Var(m.N, bounds=(0.0, 0.0))
    m.sP_neg = pyo.Var(m.N, bounds=(0.0, 0.0))
    m.sQ_pos = pyo.Var(m.N, bounds=(0.0, 0.0))
    m.sQ_neg = pyo.Var(m.N, bounds=(0.0, 0.0))

    m.con_bounded_slack = pyo.ConstraintList()
    for (i, j) in E:
        m.con_bounded_slack.add(m.s_vdrop[i, j] <= float(vdrop_slack_max))
    for i in buses:
        m.con_bounded_slack.add(
            m.sP_pos[i] + m.sP_neg[i] + m.sQ_pos[i] + m.sQ_neg[i] <= float(kcl_slack_max)
        )

    if data["fix_slack_vm"]:
        m.slack_v = pyo.Constraint(expr=m.v[slack_bus] == float(slack_vm_pu) ** 2)

    def _Pinj(mm, i):
        gen_sum = pyo.quicksum(mm.Pg[g] for g in mm.G if int(mm.gen_bus[g]) == int(i))
        return mm.Pinj[i] == gen_sum - mm.Pd[i]
    m.Pinj_def = pyo.Constraint(m.N, rule=_Pinj)

    def _qsh_expr(mm, i):
        i = int(i)
        if i not in C_set:
            return 0.0
        return mm.bcap[i] * mm.z[i]
    m.qsh = pyo.Expression(m.N, rule=_qsh_expr)

    def _Qinj(mm, i):
        gen_sum = pyo.quicksum(mm.Qg[g] for g in mm.G if int(mm.gen_bus[g]) == int(i))
        return mm.Qinj[i] == gen_sum - mm.Qd[i] + mm.qsh[i]
    m.Qinj_def = pyo.Constraint(m.N, rule=_Qinj)

    m.con_sh = pyo.ConstraintList()
    for i in C:
        vL = float(Vmin[i] ** 2)
        vU = float(Vmax[i] ** 2)
        m.con_sh.add(m.z[i] <= vU * m.a_sh[i])
        m.con_sh.add(m.z[i] >= vL * m.a_sh[i])
        m.con_sh.add(m.z[i] <= m.v[i] - vL * (1 - m.a_sh[i]))
        m.con_sh.add(m.z[i] >= m.v[i] - vU * (1 - m.a_sh[i]))

    def _onehot(mm, i, j):
        taps = K[(int(i), int(j))]
        return pyo.quicksum(mm.beta[int(i), int(j), int(t)] for t in taps) == 1
    m.onehot = pyo.Constraint(m.T, rule=_onehot)

    m.con_tv = pyo.ConstraintList()
    for (i, j, tap) in m.BETA_INDEX:
        vL = float(Vmin[i] ** 2)
        vU = float(Vmax[i] ** 2)
        m.con_tv.add(m.tv[i, j, tap] <= vU * m.beta[i, j, tap])
        m.con_tv.add(m.tv[i, j, tap] >= vL * m.beta[i, j, tap])
        m.con_tv.add(m.tv[i, j, tap] <= m.v[i] - vL * (1 - m.beta[i, j, tap]))
        m.con_tv.add(m.tv[i, j, tap] >= m.v[i] - vU * (1 - m.beta[i, j, tap]))

    # [A1] Sum_t tv[i,j,t] = v[i] (free cut from McCormick + one-hot).
    if USE_TV_SUM_EQ and len(T) > 0:
        def _tv_sum_rule(mm, i, j):
            taps = K[(int(i), int(j))]
            return pyo.quicksum(mm.tv[int(i), int(j), int(t)] for t in taps) == mm.v[int(i)]
        m.tv_sum_eq = pyo.Constraint(m.T, rule=_tv_sum_rule)

    # [A2] SOS1 on beta per OLTC edge.
    if USE_BETA_SOS1 and len(T) > 0:
        m.beta_sos = pyo.SOSConstraint(
            m.T,
            rule=lambda mm, i, j: [mm.beta[int(i), int(j), int(t)] for t in K[(int(i), int(j))]],
            sos=1,
        )

    def _vsend_rule(mm, i, j):
        if (int(i), int(j)) not in T_set:
            return mm.v[int(i)]
        taps = K[(int(i), int(j))]
        return pyo.quicksum(
            mm.delta_tap[int(i), int(j), int(t)] * mm.tv[int(i), int(j), int(t)]
            for t in taps
        )
    m.vsend = pyo.Expression(m.E, rule=_vsend_rule)

    def _bfmP(mm, i):
        i = int(i)
        out_sum = pyo.quicksum(mm.Pij[a, b] for (a, b) in out_arcs[i])
        in_sum  = pyo.quicksum((mm.Pij[a, b] - mm.r[a, b] * mm.ell_fix[a, b]) for (a, b) in in_arcs[i])
        return out_sum - in_sum + mm.sP_pos[i] - mm.sP_neg[i] == mm.Pinj[i]
    m.BFM_P = pyo.Constraint(m.N, rule=_bfmP)

    def _bfmQ(mm, i):
        i = int(i)
        out_sum = pyo.quicksum(mm.Qij[a, b] for (a, b) in out_arcs[i])
        in_sum  = pyo.quicksum((mm.Qij[a, b] - mm.x[a, b] * mm.ell_fix[a, b]) for (a, b) in in_arcs[i])
        return out_sum - in_sum + mm.sQ_pos[i] - mm.sQ_neg[i] == mm.Qinj[i]
    m.BFM_Q = pyo.Constraint(m.N, rule=_bfmQ)

    m.con_vdrop = pyo.ConstraintList()
    for (i, j) in E:
        i = int(i)
        j = int(j)
        rij = m.r[i, j]
        xij = m.x[i, j]
        z2 = rij * rij + xij * xij
        rhs = m.vsend[i, j] - 2.0 * (rij * m.Pij[i, j] + xij * m.Qij[i, j]) + z2 * m.ell_fix[i, j]
        res = m.v[j] - rhs
        m.con_vdrop.add(res <= m.s_vdrop[i, j])
        m.con_vdrop.add(-res <= m.s_vdrop[i, j])

    # [A5] KVL cycle cuts on non-OLTC subgraph.
    kvl_cycles = data.get("kvl_cycles", [])
    if USE_KVL_CYCLES and len(kvl_cycles) > 0:
        m.con_kvl_cycle = pyo.ConstraintList()
        for cyc in kvl_cycles:
            lhs = 0.0
            for ((ei, ej), sign) in cyc:
                rij = float(r[(int(ei), int(ej))])
                xij = float(x[(int(ei), int(ej))])
                z2 = rij * rij + xij * xij
                term = 2.0 * (rij * m.Pij[int(ei), int(ej)] + xij * m.Qij[int(ei), int(ej)]) \
                       - z2 * m.ell_fix[int(ei), int(ej)]
                lhs = lhs + float(sign) * term
            m.con_kvl_cycle.add(lhs == 0.0)

    def _thermal(mm, i, j):
        i, j = int(i), int(j)
        return mm.Pij[i, j]**2 + mm.Qij[i, j]**2 <= (mm.Smax[i, j]**2)
    m.Thermal = pyo.Constraint(m.E, rule=_thermal)

    # [A3] Receiving-end thermal: (P - r*ell)^2 + (Q - x*ell)^2 <= Smax^2.
    if USE_RECV_THERMAL:
        def _thermal_recv(mm, i, j):
            i, j = int(i), int(j)
            Pto = mm.Pij[i, j] - mm.r[i, j] * mm.ell_fix[i, j]
            Qto = mm.Qij[i, j] - mm.x[i, j] * mm.ell_fix[i, j]
            return Pto * Pto + Qto * Qto <= (mm.Smax[i, j] ** 2)
        m.Thermal_recv = pyo.Constraint(m.E, rule=_thermal_recv)

    def _l_lin_rule(mm, i, j):
        return (
            mm.b0_edge[i, j]
            + mm.aP_edge[i, j] * mm.Pij[i, j]
            + mm.aQ_edge[i, j] * mm.Qij[i, j]
            + mm.aV_edge[i, j] * mm.vsend[i, j]
        )
    m.l_lin = pyo.Expression(m.E, rule=_l_lin_rule)

    if LOSS_LIN_NONNEG_CONSTRAINT and USE_LOSS_PROXY and (not USE_MULTI_CUT_LOSS):
        def _l_lin_nonneg(mm, i, j):
            return mm.l_lin[i, j] >= 0.0
        m.l_lin_nonneg = pyo.Constraint(m.E, rule=_l_lin_nonneg)

    if loss_proxy is not None:
        for (i, j) in E:
            m.aP_edge[i, j] = float(loss_proxy.get("aP", {}).get((i, j), 0.0))
            m.aQ_edge[i, j] = float(loss_proxy.get("aQ", {}).get((i, j), 0.0))
            m.aV_edge[i, j] = float(loss_proxy.get("aV", {}).get((i, j), 0.0))
            m.b0_edge[i, j] = float(loss_proxy.get("b0", {}).get((i, j), 0.0))

            m.cP_edge[i, j] = float(loss_proxy.get("cP", {}).get((i, j), 0.0))
            m.cQ_edge[i, j] = float(loss_proxy.get("cQ", {}).get((i, j), 0.0))
            m.cV_edge[i, j] = float(loss_proxy.get("cV", {}).get((i, j), 0.0))

    # [A6] Multi-cut loss under-approximation.
    m._has_multi_cut_loss = False
    if USE_MULTI_CUT_LOSS and (loss_planes is not None) and (len(loss_planes) > 0):
        if MULTI_CUT_LOSS_L_NONNEG:
            m.l_edge = pyo.Var(m.E, domain=pyo.NonNegativeReals)
        else:
            m.l_edge = pyo.Var(m.E)
        m.con_l_cut = pyo.ConstraintList()
        for plane in loss_planes[-int(MULTI_CUT_LOSS_HISTORY):]:
            aP_p = plane.get("aP", {})
            aQ_p = plane.get("aQ", {})
            aV_p = plane.get("aV", {})
            b0_p = plane.get("b0", {})
            for (i, j) in E:
                m.con_l_cut.add(
                    m.l_edge[i, j] >= (
                        float(b0_p.get((i, j), 0.0)) - float(LOSS_PLANE_SHIFT_EPS)
                        + float(aP_p.get((i, j), 0.0)) * m.Pij[i, j]
                        + float(aQ_p.get((i, j), 0.0)) * m.Qij[i, j]
                        + float(aV_p.get((i, j), 0.0)) * m.vsend[i, j]
                    )
                )
        m._has_multi_cut_loss = True
        m._w_loss_current = float(w_loss_current)

    if prox_prev is not None:
        for (i, j) in E:
            m.Pprev[i, j] = float(prox_prev.get("Pij", {}).get((i, j), 0.0))
            m.Qprev[i, j] = float(prox_prev.get("Qij", {}).get((i, j), 0.0))
        for i in buses:
            m.vprev[i] = float(prox_prev.get("v", {}).get(i, 1.0))

    if USE_PROX_BOUNDS:
        m.con_prox_step = pyo.ConstraintList()
        for (i, j) in E:
            p_bd = max(1.0e-6, float(p_step_frac) * float(Smax[(i, j)]))
            q_bd = max(1.0e-6, float(q_step_frac) * float(Smax[(i, j)]))
            m.con_prox_step.add(m.Pij[i, j] - m.Pprev[i, j] <= p_bd)
            m.con_prox_step.add(m.Pprev[i, j] - m.Pij[i, j] <= p_bd)
            m.con_prox_step.add(m.Qij[i, j] - m.Qprev[i, j] <= q_bd)
            m.con_prox_step.add(m.Qprev[i, j] - m.Qij[i, j] <= q_bd)
        for i in buses:
            v_bd = max(1.0e-6, float(v_step_abs))
            m.con_prox_step.add(m.v[i] - m.vprev[i] <= v_bd)
            m.con_prox_step.add(m.vprev[i] - m.v[i] <= v_bd)

    # [C12] Adaptive trust region (box on P/Q/v deltas).
    if USE_ADAPTIVE_TRUST_REGION and (tr_P is not None) and (tr_Q is not None) and (tr_V is not None):
        m.con_trust_region = pyo.ConstraintList()
        tP = max(float(TR_MIN_P), min(float(TR_MAX_P), float(tr_P)))
        tQ = max(float(TR_MIN_Q), min(float(TR_MAX_Q), float(tr_Q)))
        tV = max(float(TR_MIN_V), min(float(TR_MAX_V), float(tr_V)))
        for (i, j) in E:
            p_bd = max(1.0e-6, tP * float(Smax[(i, j)]))
            q_bd = max(1.0e-6, tQ * float(Smax[(i, j)]))
            m.con_trust_region.add(m.Pij[i, j] - m.Pprev[i, j] <= p_bd)
            m.con_trust_region.add(m.Pprev[i, j] - m.Pij[i, j] <= p_bd)
            m.con_trust_region.add(m.Qij[i, j] - m.Qprev[i, j] <= q_bd)
            m.con_trust_region.add(m.Qprev[i, j] - m.Qij[i, j] <= q_bd)
        for i in buses:
            v_bd = max(1.0e-6, tV)
            m.con_trust_region.add(m.v[i] - m.vprev[i] <= v_bd)
            m.con_trust_region.add(m.vprev[i] - m.v[i] <= v_bd)

    def _obj_rule(mm):
        gen_cost = pyo.quicksum(
            mm.c2[g] * (sn * mm.Pg[g])**2 + mm.c1[g] * (sn * mm.Pg[g]) + mm.c0[g]
            for g in mm.G
        )

        loss_term = 0.0
        if USE_LOSS_PROXY:
            if getattr(mm, "_has_multi_cut_loss", False):
                # [A6] w_loss * sn * r * l_edge, lower-bounded by all Taylor planes.
                wsn = float(mm._w_loss_current) * sn
                loss_term = pyo.quicksum(
                    wsn * float(mm.r[i, j]) * mm.l_edge[i, j]
                    for (i, j) in mm.E
                )
            else:
                loss_term = pyo.quicksum(
                    mm.cP_edge[i, j] * mm.Pij[i, j]
                    + mm.cQ_edge[i, j] * mm.Qij[i, j]
                    + mm.cV_edge[i, j] * mm.vsend[i, j]
                    for (i, j) in mm.E
                )

        prox = 0.0
        if USE_PROXIMAL:
            prox = (
                float(RHO_P) * pyo.quicksum((mm.Pij[i, j] - mm.Pprev[i, j])**2 for (i, j) in mm.E)
                + float(RHO_Q) * pyo.quicksum((mm.Qij[i, j] - mm.Qprev[i, j])**2 for (i, j) in mm.E)
                + float(RHO_V) * pyo.quicksum((mm.v[i] - mm.vprev[i])**2 for i in mm.N)
            )

        theta_proj = 0.0
        if USE_THETA_PROJ_PROX:
            theta_proj = float(RHO_THETA) * pyo.quicksum(
                (mm.x[i, j] * (mm.Pij[i, j] - mm.Pprev[i, j]) - mm.r[i, j] * (mm.Qij[i, j] - mm.Qprev[i, j]))**2
                for (i, j) in mm.E
            )

        return gen_cost + loss_term + prox + theta_proj

    m.obj = pyo.Objective(rule=_obj_rule, sense=pyo.minimize)

    if warm is not None:
        if "v" in warm:
            for i in buses:
                if i in warm["v"]:
                    val = float(warm["v"][i])
                    m.v[i].value = _clip_to_vardata_bounds(m.v[i], val) if CLIP_WARMSTART else val

        if "Pg" in warm:
            for g in Gset:
                if g in warm["Pg"]:
                    val = float(warm["Pg"][g])
                    m.Pg[g].value = _clip_to_vardata_bounds(m.Pg[g], val) if CLIP_WARMSTART else val

        if "Qg" in warm:
            for g in Gset:
                if g in warm["Qg"]:
                    val = float(warm["Qg"][g])
                    m.Qg[g].value = _clip_to_vardata_bounds(m.Qg[g], val) if CLIP_WARMSTART else val

        if "Pij" in warm:
            for (i, j) in E:
                if (i, j) in warm["Pij"]:
                    val = float(warm["Pij"][(i, j)])
                    m.Pij[i, j].value = _clip_to_vardata_bounds(m.Pij[i, j], val) if CLIP_WARMSTART else val

        if "Qij" in warm:
            for (i, j) in E:
                if (i, j) in warm["Qij"]:
                    val = float(warm["Qij"][(i, j)])
                    m.Qij[i, j].value = _clip_to_vardata_bounds(m.Qij[i, j], val) if CLIP_WARMSTART else val

        if "a_sh" in warm:
            for i in C:
                if i in warm["a_sh"]:
                    val = float(warm["a_sh"][i])
                    if relax_binaries:
                        m.a_sh[i].value = _clip_to_vardata_bounds(m.a_sh[i], val)
                    else:
                        m.a_sh[i].value = 1 if val >= 0.5 else 0

        if "beta" in warm:
            if relax_binaries:
                for (i, j, tap) in m.BETA_INDEX:
                    if (i, j, tap) in warm["beta"]:
                        val = float(warm["beta"][(i, j, tap)])
                        m.beta[i, j, tap].value = _clip_to_vardata_bounds(m.beta[i, j, tap], val)
            else:
                rounded_beta = _round_onehot_from_warm(data, warm["beta"])
                for (i, j, tap) in m.BETA_INDEX:
                    m.beta[i, j, tap].value = int(rounded_beta.get((i, j, tap), 0))

        # [B8] Honor tv/ell warm values if provided in a richer warm dict.
        if "tv" in warm:
            for (i, j, tap) in m.BETA_INDEX:
                if (i, j, tap) in warm["tv"]:
                    val = float(warm["tv"][(i, j, tap)])
                    m.tv[i, j, tap].value = _clip_to_vardata_bounds(m.tv[i, j, tap], val)

    # [B9] Fix-and-relax: pin binaries deemed stable across rounds.
    if USE_FIX_AND_RELAX and (not relax_binaries):
        if fix_ash is not None:
            for i, v in fix_ash.items():
                if int(i) in C_set:
                    m.a_sh[int(i)].fix(1.0 if int(v) >= 1 else 0.0)
        if fix_beta is not None:
            for (i, j, tap), v in fix_beta.items():
                if (int(i), int(j), int(tap)) in m.BETA_INDEX:
                    m.beta[int(i), int(j), int(tap)].fix(1.0 if int(v) >= 1 else 0.0)

    m._data = data
    return m


# ============================================================
# [B7] Gurobi drop-in + SCIP fallback + [B8] MIP warm-start
# ============================================================
def _try_gurobi(model: pyo.ConcreteModel, timelimit: float, mipgap: float, tee: bool) -> bool:
    """Attempt a Gurobi solve.  Returns False (degrades gracefully) when
    neither `gurobi_persistent` nor `gurobi` is importable/available.
    """
    if not USE_GUROBI_FIRST:
        return False
    opt = None
    for name in ("gurobi_persistent", "gurobi"):
        try:
            cand = pyo.SolverFactory(name)
            if cand is not None and cand.available(exception_flag=False):
                opt = cand
                break
        except Exception:
            continue
    if opt is None:
        return False

    try:
        opt.options["TimeLimit"] = float(timelimit)
        opt.options["MIPGap"] = float(mipgap)
        opt.options["OutputFlag"] = 1 if tee else 0
        opt.options["NonConvex"] = 2
    except Exception:
        pass

    try:
        if hasattr(opt, "set_instance"):
            opt.set_instance(model)
            res = opt.solve(tee=tee, warmstart=USE_MIP_WARMSTART, load_solutions=True)
        else:
            res = opt.solve(
                model,
                tee=tee,
                load_solutions=True,
                warmstart=USE_MIP_WARMSTART,
            )
    except Exception:
        try:
            res = opt.solve(model, tee=tee, load_solutions=True)
        except Exception:
            return False

    tc = res.solver.termination_condition
    if tc == TerminationCondition.infeasible:
        return False
    return _solution_complete(model, model._data)


def _make_fresh_scip(prefer_nl: bool):
    """Return a freshly constructed SCIP solver instance."""
    if prefer_nl:
        o = pyo.SolverFactory("scip", solver_io="nl")
        if o is not None and o.available(exception_flag=False):
            return o, True
    o = pyo.SolverFactory("scip")
    if o is not None and o.available(exception_flag=False):
        return o, False
    return None, False


def _apply_scip_options(opt, timelimit: float, mipgap: float, tee: bool) -> None:
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


def solve_with_scip(model: pyo.ConcreteModel, timelimit: float, mipgap: float, tee: bool) -> bool:
    """Try Gurobi (if enabled/available) first, then SCIP NL, then SCIP default."""
    if USE_GUROBI_FIRST and _try_gurobi(model, timelimit, mipgap, tee):
        return True

    res = None
    for prefer_nl in (True, False):
        opt, _is_nl = _make_fresh_scip(prefer_nl)
        if opt is None:
            continue
        _apply_scip_options(opt, timelimit, mipgap, tee)
        try:
            res = opt.solve(model, tee=tee, load_solutions=True)
            break
        except Exception as exc:
            if tee:
                print(f"[SCIP] {'NL' if prefer_nl else 'default'} solve raised {type(exc).__name__}: {exc}")
            res = None
            continue

    if res is None:
        return False

    tc = res.solver.termination_condition
    if tc == TerminationCondition.infeasible:
        return False
    return _solution_complete(model, model._data)


def solve_subproblem_robust(
    data: Dict[str, Any],
    ell_fix: Dict[Tuple[int, int], float],
    warm: Optional[Dict[str, Any]],
    loss_proxy: Optional[Dict[str, Dict[Any, float]]],
    prox_prev: Optional[Dict[str, Dict[Any, float]]],
    vdrop_slack_max: float,
    kcl_slack_max: float,
    p_step_frac: float,
    q_step_frac: float,
    v_step_abs: float,
    tee: bool,
    loss_planes: Optional[List[Dict[str, Dict[Tuple[int, int], float]]]] = None,
    w_loss_current: float = 0.0,
    fix_beta: Optional[Dict[Tuple[int, int, int], int]] = None,
    fix_ash: Optional[Dict[int, int]] = None,
    tr_P: Optional[float] = None,
    tr_Q: Optional[float] = None,
    tr_V: Optional[float] = None,
    mipgap: Optional[float] = None,
) -> Tuple[pyo.ConcreteModel, str]:

    last_relaxed_model: Optional[pyo.ConcreteModel] = None
    gap_to_use = float(SCIP_GAP_LIMIT if mipgap is None else mipgap)

    for mult in SLACK_RESCUE_MULTS:
        vb, kb = _rescued_slack_bounds(vdrop_slack_max, kcl_slack_max, float(mult))
        suffix = "" if float(mult) == 1.0 else f"_rescued_x{mult:g}"

        if tee and float(mult) != 1.0:
            print(
                f"[WARN] retrying bounded-slack subproblem with relaxed bounds: "
                f"vdrop<={vb:.3e}, kcl<={kb:.3e}"
            )

        m_bin = build_subproblem(
            data, ell_fix,
            relax_binaries=False,
            warm=warm,
            loss_proxy=loss_proxy,
            prox_prev=prox_prev,
            vdrop_slack_max=vb,
            kcl_slack_max=kb,
            p_step_frac=p_step_frac,
            q_step_frac=q_step_frac,
            v_step_abs=v_step_abs,
            loss_planes=loss_planes,
            w_loss_current=w_loss_current,
            fix_beta=fix_beta,
            fix_ash=fix_ash,
            tr_P=tr_P, tr_Q=tr_Q, tr_V=tr_V,
        )
        if solve_with_scip(m_bin, SCIP_TIME_LIMIT, gap_to_use, tee):
            return m_bin, "binary" + suffix

        if tee:
            print("[WARN] binary infeasible/no incumbent -> RELAXED ...")

        m_relax = build_subproblem(
            data, ell_fix,
            relax_binaries=True,
            warm=warm,
            loss_proxy=loss_proxy,
            prox_prev=prox_prev,
            vdrop_slack_max=vb,
            kcl_slack_max=kb,
            p_step_frac=p_step_frac,
            q_step_frac=q_step_frac,
            v_step_abs=v_step_abs,
            loss_planes=loss_planes,
            w_loss_current=w_loss_current,
            tr_P=tr_P, tr_Q=tr_Q, tr_V=tr_V,
        )
        if solve_with_scip(m_relax, SCIP_TIME_LIMIT, gap_to_use, False):
            last_relaxed_model = m_relax
            tap_choice: Dict[Tuple[int, int], int] = {}
            for (i, j) in data["T"]:
                tap_choice[(i, j)] = _pick_tap_from_beta(m_relax, data, i, j)

            sh_choice: Dict[int, int] = {}
            for i in data["C"]:
                sh_choice[int(i)] = 1 if _val(m_relax.a_sh[i], 0.0) >= 0.5 else 0

            warm_fix = _extract_warm_from_model(m_relax, data)
            warm_fix["beta"] = {
                (i, j, int(tap)): 1.0 if int(tap) == int(tap_choice[(i, j)]) else 0.0
                for (i, j) in data["T"]
                for tap in data["K"][(i, j)]
            }
            warm_fix["a_sh"] = {int(i): float(sh_choice[int(i)]) for i in data["C"]}

            if tee:
                print("[INFO] RELAXED solved -> solving FIXED-discrete MIQCP (from relax->round) ...")

            m_fix = build_subproblem(
                data, ell_fix,
                relax_binaries=False,
                warm=warm_fix,
                loss_proxy=loss_proxy,
                prox_prev=prox_prev,
                vdrop_slack_max=vb,
                kcl_slack_max=kb,
                loss_planes=loss_planes,
                w_loss_current=w_loss_current,
                tr_P=tr_P, tr_Q=tr_Q, tr_V=tr_V,
            )

            for (i, j) in data["T"]:
                pick = int(tap_choice[(i, j)])
                for tap in data["K"][(i, j)]:
                    m_fix.beta[i, j, int(tap)].fix(1.0 if int(tap) == pick else 0.0)

            for i in data["C"]:
                m_fix.a_sh[i].fix(float(sh_choice[int(i)]))

            if solve_with_scip(m_fix, SCIP_TIME_LIMIT, gap_to_use, tee):
                return m_fix, "fixed_from_relax" + suffix

            if tee:
                print("[WARN] Fixed-discrete MIQCP failed; keeping RELAXED solution as fallback.")
            return m_relax, "relaxed_only" + suffix

    if last_relaxed_model is not None:
        return last_relaxed_model, "relaxed_only_rescued"

    raise RuntimeError(
        "RELAXED also infeasible/no-solution under bounded vdrop/KCL slack constraints; "
        "scheduled and rescue bounds were all exhausted."
    )



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
            _pval(model.c2[g]) * (sn * Pg) ** 2
            + _pval(model.c1[g]) * (sn * Pg)
            + _pval(model.c0[g])
        )

    pen_vdrop = 0.0
    pen_kcl = 0.0

    loss_term = 0.0
    if USE_LOSS_PROXY:
        for (i, j) in data["E"]:
            Pij = _val(model.Pij[i, j], 0.0)
            Qij = _val(model.Qij[i, j], 0.0)
            vsend = _pval(model.vsend[i, j], 1.0)
            loss_term += (
                _pval(model.cP_edge[i, j]) * Pij
                + _pval(model.cQ_edge[i, j]) * Qij
                + _pval(model.cV_edge[i, j]) * vsend
            )

    prox_term = 0.0
    if USE_PROXIMAL:
        for (i, j) in data["E"]:
            Pij = _val(model.Pij[i, j], 0.0)
            Qij = _val(model.Qij[i, j], 0.0)
            dP = Pij - _pval(model.Pprev[i, j])
            dQ = Qij - _pval(model.Qprev[i, j])
            prox_term += float(RHO_P) * (dP ** 2) + float(RHO_Q) * (dQ ** 2)
        for i in data["buses"]:
            dv = _val(model.v[i], 1.0) - _pval(model.vprev[i], 1.0)
            prox_term += float(RHO_V) * (dv ** 2)

    theta_proj = 0.0
    if USE_THETA_PROJ_PROX:
        for (i, j) in data["E"]:
            Pij = _val(model.Pij[i, j], 0.0)
            Qij = _val(model.Qij[i, j], 0.0)
            dP = Pij - _pval(model.Pprev[i, j])
            dQ = Qij - _pval(model.Qprev[i, j])
            theta_proj += float(RHO_THETA) * (
                float(data["x"][(i, j)]) * dP - float(data["r"][(i, j)]) * dQ
            ) ** 2

    total = gen_cost + loss_term + prox_term + theta_proj

    return dict(
        gen_cost=gen_cost,
        pen_vdrop=pen_vdrop,
        pen_kcl=pen_kcl,
        loss_term=loss_term,
        prox_term=prox_term,
        theta_proj=theta_proj,
        total=total
    )


# ============================================================
# Outer loop: BFM-ag + 1번 변화 + 2번 변화 + ver3 additions
# ============================================================

def run_bfm_ag(data: Dict[str, Any], max_iters: int, eps: float, tee: bool) -> Dict[str, Any]:
    buses = data["buses"]
    E = data["E"]
    slack = int(data["slack_bus"])
    T_set = set(data["T"])
    ellmax = data["ellmax"]
    K = data["K"]
    delta_tap = data["delta_tap"]

    # [A4] One-shot OBBT to tighten Pij, Qij, v bounds used by every
    # subproblem this run.  Off by default on this smaller network.
    if USE_OBBT and data.get("obbt", None) is None:
        try:
            t_obbt0 = time.perf_counter()
            obbt = _obbt_tighten_bounds(data)
            t_obbt1 = time.perf_counter()
            if obbt is not None:
                n_P = len(obbt.get("P", {}))
                n_Q = len(obbt.get("Q", {}))
                n_V = len(obbt.get("v", {}))
                print(f"[OBBT] tightened P={n_P}, Q={n_Q}, v={n_V} entries in {t_obbt1-t_obbt0:.2f}s")
                data["obbt"] = obbt
        except Exception as exc:
            print(f"[OBBT] preprocessing failed ({exc}); continuing without tighter bounds")

    # PF-based warm-start: seed ell_fix, v_prev, P_prev, Q_prev, Pg, Qg from
    # a pandapower power-flow solve on the same calibrated network.
    net = data.get("net", None)
    pf_seed = None
    if USE_PF_WARMSTART and net is not None:
        t_pf0 = time.perf_counter()
        pf_seed = _pf_warm_start(net, data)
        t_pf1 = time.perf_counter()
        if pf_seed is not None:
            print(
                f"[PF-WARM] seeded ell_fix, v, Pij, Qij, Pg, Qg from pandapower "
                f"PF on the calibrated network in {t_pf1-t_pf0:.2f} s"
            )
        else:
            print("[PF-WARM] pandapower PF failed; falling back to flat warm-start")

    if pf_seed is not None:
        ell_fix = {k: float(pf_seed["ell"].get(k, 0.0)) for k in E}
        ell_state = dict(ell_fix)
        v_prev = {int(b): float(pf_seed["v"].get(int(b), 1.0)) for b in buses}
        P_prev = {k: float(pf_seed["Pij"].get(k, 0.0)) for k in E}
        Q_prev = {k: float(pf_seed["Qij"].get(k, 0.0)) for k in E}
    else:
        ell_fix = {(i, j): 0.0 for (i, j) in E}
        ell_state = {(i, j): 0.0 for (i, j) in E}
        v_prev = {int(b): 1.0 for b in buses}
        P_prev = {(i, j): 0.0 for (i, j) in E}
        Q_prev = {(i, j): 0.0 for (i, j) in E}

    # [C13] Optional MISOCP warm-start for ell (overrides PF seed when enabled).
    if USE_ELL_WARM_FROM_SOCP:
        try:
            seed = _warm_ell_from_socp(data, time_limit=float(ELL_WARM_SOCP_TIME_LIMIT))
            if seed is not None:
                for k in E:
                    v = float(seed.get(k, 0.0))
                    if ELL_CLIP_NONNEG:
                        v = max(0.0, v)
                    if ELL_CLIP_MAX:
                        v = min(float(ellmax[k]), v)
                    ell_fix[k] = v
                    ell_state[k] = v
                print("[SOCP-WARM] seeded ell_fix from MISOCP-BFM41 solve")
        except Exception as exc:
            print(f"[SOCP-WARM] failed ({exc}); starting from ell_fix=0")

    u_send_prev = {(i, j): float("nan") for (i, j) in E}

    theta_prev = {int(b): 0.0 for b in buses}
    theta_prev[slack] = 0.0

    warm_beta = {}
    for (i, j) in data["T"]:
        taps = data["K"][(i, j)]
        rec_tap = int(data.get("recommended_taps", {}).get((i, j), _default_tap_choice([int(t) for t in taps])))
        if rec_tap not in taps:
            rec_tap = _default_tap_choice([int(t) for t in taps])
        for tap in taps:
            warm_beta[(i, j, int(tap))] = 1.0 if int(tap) == int(rec_tap) else 0.0

    warm_ash = {int(i): float(data.get("recommended_shunts", {}).get(int(i), 0)) for i in data["C"]}

    warm_Pg_seed: Dict[int, float] = dict(pf_seed["Pg"]) if (pf_seed is not None) else {}
    warm_Qg_seed: Dict[int, float] = dict(pf_seed["Qg"]) if (pf_seed is not None) else {}

    warm = {
        "v": dict(v_prev),
        "Pij": dict(P_prev),
        "Qij": dict(Q_prev),
        "Pg": warm_Pg_seed,
        "Qg": warm_Qg_seed,
        "beta": warm_beta,
        "a_sh": warm_ash,
        "u_send": dict(u_send_prev),
        "ell": dict(ell_fix),
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
    best_feasible = {
        "iter": 0,
        "gen_cost": float("inf"),
        "total": float("inf"),
        "model": None,
        "ell": None,
        "theta": None,
        "tag": ""
    }
    last = {"iter": 0, "model": None, "ell": None, "theta": None, "tag": ""}
    stopped = {"iter": 0, "gen_cost": float("inf"), "total": float("inf"), "model": None, "ell": None, "theta": None, "tag": ""}

    w_loss_sm = 0.0
    slack_gen_idx = data.get("slack_gen_idx", None)
    slack_c2 = float(data.get("slack_c2", 0.0))
    slack_c1 = float(data.get("slack_c1", 0.0))
    sn = float(data["sn_mva"])

    sumdiff_hist: List[float] = []
    obj_hist: List[float] = []

    curr_vdrop_tol = float(VDROP_SLACK_INIT)
    curr_kcl_tol = float(KCL_SLACK_INIT)
    consecutive_rejects = 0

    # [A6] multi-cut loss plane history.
    loss_planes_hist: List[Dict[str, Dict[Tuple[int, int], float]]] = []

    # [C11] Anderson history of (ell_in, ell_out) pairs.
    ell_in_hist: List[Dict[Tuple[int, int], float]] = []
    ell_out_hist: List[Dict[Tuple[int, int], float]] = []

    # [C12] Adaptive trust-region radii.
    tr_P = float(TR_INIT_P)
    tr_Q = float(TR_INIT_Q)
    tr_V = float(TR_INIT_V)

    # [C15] Adaptive over-relaxation state.
    omega_curr = float(OMEGA_INIT)
    sign_prev_map: Optional[Dict[Tuple[int, int], int]] = None

    # [B9] Stability tracker for fix-and-relax.
    last_beta_rounded: Optional[Dict[Tuple[int, int, int], int]] = None
    last_ash_rounded: Optional[Dict[int, int]] = None
    stable_rounds_beta = 0
    stable_rounds_ash = 0

    for t in range(1, max_iters + 1):
        t0 = time.perf_counter()

        trial_ell_fix = dict(ell_fix)
        ell_state_prev = dict(ell_state)

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

        if USE_ELL_EMA_FIX and t >= 2:
            for (i, j) in E:
                trial_ell_fix[(i, j)] = float(BETA_ELL) * float(ell_fix[(i, j)]) + (1.0 - float(BETA_ELL)) * float(ell_raw[(i, j)])
                if ELL_CLIP_NONNEG:
                    trial_ell_fix[(i, j)] = max(0.0, float(trial_ell_fix[(i, j)]))
                if ELL_CLIP_MAX:
                    trial_ell_fix[(i, j)] = min(float(ellmax[(i, j)]), float(trial_ell_fix[(i, j)]))
        elif not USE_ELL_EMA_FIX:
            trial_ell_fix = dict(ell_raw)

        # [C15] Adaptive over-relaxation on ell update.
        if USE_ADAPTIVE_OMEGA and t >= 2:
            omega_curr, sign_prev_map = _update_omega(omega_curr, sign_prev_map, ell_raw, ell_fix)
            trial_ell_fix = {
                k: float(ell_fix[k]) + float(omega_curr) * (float(trial_ell_fix[k]) - float(ell_fix[k]))
                for k in E
            }
            for k in E:
                if ELL_CLIP_NONNEG:
                    trial_ell_fix[k] = max(0.0, float(trial_ell_fix[k]))
                if ELL_CLIP_MAX:
                    trial_ell_fix[k] = min(float(ellmax[k]), float(trial_ell_fix[k]))
        elif USE_ADAPTIVE_OMEGA:
            _, sign_prev_map = _update_omega(omega_curr, sign_prev_map, ell_raw, ell_fix)

        # [C11] Anderson type-II acceleration on ell_fix (kicks in after warmup).
        if USE_ANDERSON_ELL and t > int(ANDERSON_WARMUP_ITERS):
            ell_fixed_prev = dict(ell_fix)
            ell_raw_now = dict(trial_ell_fix)
            try:
                and_next = _anderson_step(
                    ell_in_hist=ell_in_hist,
                    ell_out_hist=ell_out_hist,
                    ell_curr_in=ell_fixed_prev,
                    ell_curr_out=ell_raw_now,
                    m_window=int(ANDERSON_WINDOW),
                    reg=float(ANDERSON_REG),
                )
                trial_ell_fix = {
                    k: float(ell_fixed_prev[k]) + float(ANDERSON_MIX_BETA) * (float(and_next.get(k, ell_raw_now[k])) - float(ell_fixed_prev[k]))
                    for k in E
                }
                for k in E:
                    if ELL_CLIP_NONNEG:
                        trial_ell_fix[k] = max(0.0, float(trial_ell_fix[k]))
                    if ELL_CLIP_MAX:
                        trial_ell_fix[k] = min(float(ellmax[k]), float(trial_ell_fix[k]))
            except Exception:
                pass
        ell_in_hist.append(dict(ell_fix))
        ell_out_hist.append(dict(trial_ell_fix))
        if len(ell_in_hist) > int(ANDERSON_WINDOW) + 2:
            ell_in_hist.pop(0)
            ell_out_hist.pop(0)

        prox_prev = {
            "Pij": dict(warm["Pij"]),
            "Qij": dict(warm["Qij"]),
            "v": dict(warm["v"]),
        }

        if slack_gen_idx is None:
            w_raw = 0.0
        else:
            P0_pu_prev = float(warm.get("Pg", {}).get(int(slack_gen_idx), 0.0))
            P0_MW_prev = sn * P0_pu_prev
            w_raw = (slack_c1 + 2.0 * slack_c2 * P0_MW_prev) * float(LOSS_WEIGHT_SCALE)
            if WLOSS_NONNEG:
                w_raw = max(0.0, float(w_raw))

        if t == 1:
            trial_w_loss_sm = float(w_raw)
        else:
            trial_w_loss_sm = float(BETA_WLOSS) * float(w_loss_sm) + (1.0 - float(BETA_WLOSS)) * float(w_raw)
        w_loss = float(trial_w_loss_sm)

        # [C14] Build a theta-aware ubar (send-end v) map when theta is known.
        ubar_feed = warm.get("u_send", None)
        if USE_THETA_REUSE_VS and t >= int(THETA_REUSE_START_ITER):
            theta_ubar: Dict[Tuple[int, int], float] = {}
            for (i, j) in E:
                if (i, j) in T_set:
                    theta_ubar[(i, j)] = float(ubar_feed.get((i, j), float("nan"))) if ubar_feed is not None else float("nan")
                    continue
                vi = float(prox_prev["v"].get(int(i), 1.0))
                vj = float(prox_prev["v"].get(int(j), 1.0))
                th_i = float(theta_prev.get(int(i), 0.0))
                th_j = float(theta_prev.get(int(j), 0.0))
                scale = max(0.5, float(math.cos(_wrap_pi(th_i - th_j))))
                # vS here approximates |V_i| * |V_j| * cos(theta_i - theta_j) in
                # the v = |V|^2 basis by using v[i] * scale (matches ver3 form).
                theta_ubar[(i, j)] = max(V_SEND_FLOOR, float(vi) * float(scale))
            ubar_feed = theta_ubar

        loss_proxy = compute_loss_proxy_coeffs(
            data=data,
            Pbar=prox_prev["Pij"],
            Qbar=prox_prev["Qij"],
            vbar=prox_prev["v"],
            ubar=ubar_feed,
            w_loss=w_loss,
        )

        # [A6] Append this plane to the multi-cut history.
        new_plane = {
            "aP": dict(loss_proxy.get("aP", {})),
            "aQ": dict(loss_proxy.get("aQ", {})),
            "aV": dict(loss_proxy.get("aV", {})),
            "b0": dict(loss_proxy.get("b0", {})),
        }
        loss_planes_hist.append(new_plane)
        if len(loss_planes_hist) > int(MULTI_CUT_LOSS_HISTORY):
            loss_planes_hist.pop(0)

        p_step_frac, q_step_frac, v_step_abs = _scheduled_prox_bounds(t)

        # [B9] Prepare fix-and-relax dictionaries when binaries have been stable.
        fix_beta_arg: Optional[Dict[Tuple[int, int, int], int]] = None
        fix_ash_arg: Optional[Dict[int, int]] = None
        if USE_FIX_AND_RELAX:
            if last_beta_rounded is not None and stable_rounds_beta >= int(STABLE_ROUNDS_THRESHOLD):
                fix_beta_arg = dict(last_beta_rounded)
            if last_ash_rounded is not None and stable_rounds_ash >= int(STABLE_ROUNDS_THRESHOLD):
                fix_ash_arg = dict(last_ash_rounded)

        # [B10] Per-round MIP gap.
        mipgap_this_round = _round_dependent_mipgap(t)

        try:
            model, tag = solve_subproblem_robust(
                data,
                trial_ell_fix,
                warm=warm,
                loss_proxy=loss_proxy,
                prox_prev=prox_prev,
                vdrop_slack_max=curr_vdrop_tol,
                kcl_slack_max=curr_kcl_tol,
                p_step_frac=p_step_frac,
                q_step_frac=q_step_frac,
                v_step_abs=v_step_abs,
                tee=tee,
                loss_planes=loss_planes_hist if USE_MULTI_CUT_LOSS else None,
                w_loss_current=float(w_loss),
                fix_beta=fix_beta_arg,
                fix_ash=fix_ash_arg,
                tr_P=tr_P, tr_Q=tr_Q, tr_V=tr_V,
                mipgap=mipgap_this_round,
            )
        except RuntimeError as exc:
            consecutive_rejects += 1
            curr_vdrop_tol = min(float(VDROP_SLACK_RESCUE_CAP), float(curr_vdrop_tol) * float(SLACK_ENLARGE_ON_FAIL))
            curr_kcl_tol = min(float(KCL_SLACK_RESCUE_CAP), float(curr_kcl_tol) * float(SLACK_ENLARGE_ON_FAIL))
            print(
                f"[REJECT t={t:02d}] subproblem infeasible under current bounded-slack window. "
                f"Keeping previous accepted iterate and enlarging bounds to "
                f"vdrop<={curr_vdrop_tol:.3e}, kcl<={curr_kcl_tol:.3e}."
            )
            if tee:
                print(f"  reason: {exc}")
            if consecutive_rejects >= MAX_CONSECUTIVE_REJECTS:
                print("[EARLY-STOP: TOO-MANY-REJECTS] no new accepted iterate could be produced.")
                if stopped["model"] is not None:
                    print(f"[EARLY-STOP: FALLBACK] reporting the most recent solved candidate at iter={stopped['iter']} tag={stopped['tag']}.")
                    if last["model"] is None:
                        last.update(stopped)
                    if best["model"] is None:
                        best.update(stopped)
                return {"best": best, "best_feasible": best_feasible, "last": last, "stopped": stopped}
            continue

        v_sol = {int(i): _val(model.v[i], 1.0) for i in buses}
        P_sol = {(i, j): _val(model.Pij[i, j], 0.0) for (i, j) in E}
        Q_sol = {(i, j): _val(model.Qij[i, j], 0.0) for (i, j) in E}

        u_send_edge = {}
        for (i, j) in E:
            if (i, j) in T_set:
                u_send_edge[(i, j)] = _pval(model.vsend[i, j], float(v_sol[i]))
            else:
                u_send_edge[(i, j)] = float("nan")

        Pg_sol = {int(g): _val(model.Pg[g], 0.0) for g in model.G}
        Qg_sol = {int(g): _val(model.Qg[g], 0.0) for g in model.G}
        a_sh_sol = {int(i): _val(model.a_sh[i], 0.0) for i in data["C"]} if len(data["C"]) > 0 else {}
        beta_sol = {(i, j, tap): _val(model.beta[i, j, tap], 0.0) for (i, j, tap) in model.BETA_INDEX} if len(data["T"]) > 0 else {}

        d_edge = {}
        for (i, j) in E:
            i = int(i)
            j = int(j)
            rij = float(data["r"][(i, j)])
            xij = float(data["x"][(i, j)])

            if (i, j) in T_set:
                vsend = max(float(u_send_edge[(i, j)]), DENOM_EPS)
            else:
                vsend = max(float(v_sol[i]), DENOM_EPS)

            vj = max(float(v_sol[j]), DENOM_EPS)
            denom = math.sqrt(vsend * vj)

            dprev = _wrap_pi(float(theta_prev[i] - theta_prev[j]))
            dprev = float(np.clip(dprev, -CLIP_DTHETA_PREV, CLIP_DTHETA_PREV))

            rhs = (xij * float(P_sol[(i, j)]) - rij * float(Q_sol[(i, j)])) / denom
            d_edge[(i, j)] = float(rhs - math.sin(dprev) + dprev)

        theta_ls = recover_theta_ls(buses=buses, edges=E, d_edge=d_edge, slack=slack, ridge=THETA_RIDGE)
        theta_new = {}
        for b in buses:
            thp = float(theta_prev[int(b)])
            thn = float(theta_ls[int(b)])
            if USE_WRAP_THETA_DAMP:
                theta_new[int(b)] = _damped_angle(thp, thn, float(THETA_GAMMA))
            else:
                theta_new[int(b)] = (1.0 - THETA_GAMMA) * thp + THETA_GAMMA * thn
        theta_new[slack] = 0.0

        cand_v_prev = dict(v_prev)
        cand_P_prev = dict(P_prev)
        cand_Q_prev = dict(Q_prev)
        if USE_STATE_DAMPING:
            for b in buses:
                cand_v_prev[int(b)] = (1.0 - float(DAMPING_X)) * float(v_prev[int(b)]) + float(DAMPING_X) * float(v_sol[int(b)])
            for (i, j) in E:
                cand_P_prev[(i, j)] = (1.0 - float(DAMPING_X)) * float(P_prev[(i, j)]) + float(DAMPING_X) * float(P_sol[(i, j)])
                cand_Q_prev[(i, j)] = (1.0 - float(DAMPING_X)) * float(Q_prev[(i, j)]) + float(DAMPING_X) * float(Q_sol[(i, j)])
        else:
            cand_v_prev = dict(v_sol)
            cand_P_prev = dict(P_sol)
            cand_Q_prev = dict(Q_sol)

        cand_u_send_prev = {}
        for (i, j) in E:
            if (i, j) in T_set:
                delta_eff = 0.0
                for tap in K[(i, j)]:
                    delta_eff += float(delta_tap[((i, j), int(tap))]) * float(beta_sol.get((i, j, int(tap)), 0.0))
                cand_u_send_prev[(i, j)] = float(delta_eff) * float(cand_v_prev[int(i)])
            else:
                cand_u_send_prev[(i, j)] = float("nan")

        cand_ell_state, ell_corr_stats = _outer_hybrid_correct_ell(
            data=data,
            ell_prev_state=ell_state_prev,
            v_state=cand_v_prev,
            P_state=cand_P_prev,
            Q_state=cand_Q_prev,
            u_send_state=cand_u_send_prev,
        )
        ell_new = dict(cand_ell_state)
        sum_diff = float(ell_corr_stats.get("sum_diff", 0.0))
        max_viol = float(ell_corr_stats.get("max_viol", 0.0))
        ell_mismatch_l1 = float(ell_corr_stats.get("ell_mismatch_l1", 0.0))
        ell_mismatch_linf = float(ell_corr_stats.get("ell_mismatch_linf", 0.0))
        ell_alpha_used = float(ell_corr_stats.get("alpha_used", 1.0))

        cand_warm = {
            "v": dict(cand_v_prev),
            "Pij": dict(cand_P_prev),
            "Qij": dict(cand_Q_prev),
            "Pg": dict(Pg_sol),
            "Qg": dict(Qg_sol),
            "a_sh": dict(a_sh_sol),
            "beta": dict(beta_sol),
            "u_send": dict(cand_u_send_prev),
            "ell": dict(ell_new),
        }

        max_vdrop_slack = max(_val(model.s_vdrop[i, j], 0.0) for (i, j) in E)
        max_kcl_slack = max(
            _val(model.sP_pos[i], 0.0) + _val(model.sP_neg[i], 0.0)
            + _val(model.sQ_pos[i], 0.0) + _val(model.sQ_neg[i], 0.0)
            for i in buses
        )

        obj = objective_breakdown(model, data)
        t1 = time.perf_counter()

        rescue_mult = _rescue_mult_from_tag(tag)
        used_vdrop_bound = _rescued_slack_bounds(curr_vdrop_tol, curr_kcl_tol, rescue_mult)[0]
        used_kcl_bound = _rescued_slack_bounds(curr_vdrop_tol, curr_kcl_tol, rescue_mult)[1]

        print(
            f"[t={t:02d}] tag={tag:>16s}  "
            f"gen_cost={obj['gen_cost']:,.6f}  total={obj['total']:,.6f}  "
            f"loss={obj['loss_term']:,.6f}  thetaP={obj['theta_proj']:,.6f}  "
            f"sum|dell|={sum_diff:.3e}  max_ell_viol={max_viol:.3e}  "
            f"ellMisL1={ell_mismatch_l1:.3e}  ellMisInf={ell_mismatch_linf:.3e}  a_ell={ell_alpha_used:.2f}  "
            f"max_vdrop_slack={max_vdrop_slack:.3e}/{curr_vdrop_tol:.3e}->{used_vdrop_bound:.3e}  "
            f"max_kcl_slack={max_kcl_slack:.3e}/{curr_kcl_tol:.3e}->{used_kcl_bound:.3e}  "
            + (f"pstep={p_step_frac:.3f}S qstep={q_step_frac:.3f}S vstep={v_step_abs:.3e}  " if USE_PROX_BOUNDS else "")
            + (f"prox={obj['prox_term']:,.6f}  " if USE_PROXIMAL else "")
            + f"w_loss={w_loss:.6f}  time={t1-t0:.2f}s"
        )

        stopped.update({
            "iter": t,
            "gen_cost": obj["gen_cost"],
            "total": obj["total"],
            "model": model,
            "ell": dict(ell_new),
            "theta": dict(theta_new),
            "tag": f"{tag}_stopping_point"
        })

        ratio_v = max_vdrop_slack / max(used_vdrop_bound, 1.0e-12)
        ratio_k = max_kcl_slack / max(used_kcl_bound, 1.0e-12)
        candidate_uses_rescue = rescue_mult > 1.0 or ("relaxed_only" in str(tag))
        candidate_rejected = False
        reject_reasons = []

        if REJECT_RESCUED_STEPS and rescue_mult > 1.0:
            candidate_rejected = True
            reject_reasons.append(f"rescued-step x{rescue_mult:g}")
        if REJECT_RELAXED_ONLY_STEPS and ("relaxed_only" in str(tag)):
            candidate_rejected = True
            reject_reasons.append("continuous-relaxation fallback")
        if max_vdrop_slack > used_vdrop_bound * float(SLACK_REJECT_MARGIN):
            candidate_rejected = True
            reject_reasons.append("vdrop bound materially exceeded")
        if max_kcl_slack > used_kcl_bound * float(SLACK_REJECT_MARGIN):
            candidate_rejected = True
            reject_reasons.append("kcl bound materially exceeded")

        if candidate_rejected:
            consecutive_rejects += 1
            curr_vdrop_tol = min(
                float(VDROP_SLACK_RESCUE_CAP),
                max(float(curr_vdrop_tol) * float(SLACK_ENLARGE_ON_REJECT), float(max_vdrop_slack) * 1.02)
            )
            curr_kcl_tol = min(
                float(KCL_SLACK_RESCUE_CAP),
                max(float(curr_kcl_tol) * float(SLACK_ENLARGE_ON_REJECT), float(max_kcl_slack) * 1.02)
            )
            # [C12] Shrink trust region on reject.
            if USE_ADAPTIVE_TRUST_REGION:
                tr_P = max(float(TR_MIN_P), float(tr_P) * float(TR_SHRINK))
                tr_Q = max(float(TR_MIN_Q), float(tr_Q) * float(TR_SHRINK))
                tr_V = max(float(TR_MIN_V), float(tr_V) * float(TR_SHRINK))
            print(
                f"[REJECT t={t:02d}] keeping previous accepted iterate because "
                f"{', '.join(reject_reasons)}. "
                f"New adaptive bounds: vdrop<={curr_vdrop_tol:.3e}, kcl<={curr_kcl_tol:.3e}."
            )
            if consecutive_rejects >= MAX_CONSECUTIVE_REJECTS:
                print("[EARLY-STOP: TOO-MANY-REJECTS] reverting to the last accepted iterate.")
                if stopped["model"] is not None:
                    print(f"[EARLY-STOP: FALLBACK] reporting the current stopping-point candidate at iter={stopped['iter']} tag={stopped['tag']}.")
                    if last["model"] is None:
                        last.update(stopped)
                    if best["model"] is None:
                        best.update(stopped)
                return {"best": best, "best_feasible": best_feasible, "last": last, "stopped": stopped}
            continue

        # ---------------------------
        # ACCEPT the candidate step
        # ---------------------------
        consecutive_rejects = 0
        ell_fix = dict(cand_ell_state) if USE_ACCEPTED_ELL_AS_NEXT_FIX else dict(trial_ell_fix)
        ell_state = dict(cand_ell_state)
        v_prev = dict(cand_v_prev)
        P_prev = dict(cand_P_prev)
        Q_prev = dict(cand_Q_prev)
        u_send_prev = dict(cand_u_send_prev)
        theta_prev = dict(theta_new)
        warm = cand_warm
        w_loss_sm = float(trial_w_loss_sm)

        # [C12] Expand trust region on accept (bounded by TR_MAX_*).
        if USE_ADAPTIVE_TRUST_REGION:
            tr_P = min(float(TR_MAX_P), float(tr_P) * float(TR_EXPAND))
            tr_Q = min(float(TR_MAX_Q), float(tr_Q) * float(TR_EXPAND))
            tr_V = min(float(TR_MAX_V), float(tr_V) * float(TR_EXPAND))

        # [B9] Update stability tracker for fix-and-relax.
        if USE_FIX_AND_RELAX:
            beta_rounded = {}
            for (i, j) in data["T"]:
                pick = _pick_tap_from_beta(model, data, i, j)
                for tap in data["K"][(i, j)]:
                    beta_rounded[(int(i), int(j), int(tap))] = 1 if int(tap) == int(pick) else 0
            ash_rounded = {
                int(i): (1 if _val(model.a_sh[i], 0.0) >= 0.5 else 0)
                for i in data["C"]
            }
            if last_beta_rounded is not None and beta_rounded == last_beta_rounded:
                stable_rounds_beta += 1
            else:
                stable_rounds_beta = 0
            if last_ash_rounded is not None and ash_rounded == last_ash_rounded:
                stable_rounds_ash += 1
            else:
                stable_rounds_ash = 0
            last_beta_rounded = beta_rounded
            last_ash_rounded = ash_rounded

        last.update({"iter": t, "model": model, "ell": dict(ell_new), "theta": dict(theta_new), "tag": tag})

        if obj["gen_cost"] < best["gen_cost"]:
            best.update({
                "iter": t,
                "gen_cost": obj["gen_cost"],
                "total": obj["total"],
                "model": model,
                "ell": dict(ell_new),
                "theta": dict(theta_new),
                "tag": tag
            })

        is_near_feasible = _is_near_feasible_slacks(
            max_vdrop_slack,
            max_kcl_slack,
            vdrop_tol=used_vdrop_bound,
            kcl_tol=used_kcl_bound,
        )
        if is_near_feasible:
            if obj["gen_cost"] < best_feasible["gen_cost"]:
                best_feasible.update({
                    "iter": t,
                    "gen_cost": obj["gen_cost"],
                    "total": obj["total"],
                    "model": model,
                    "ell": dict(ell_new),
                    "theta": dict(theta_new),
                    "tag": tag
                })

        if (sum_diff <= eps) and (max_viol <= 1e-8):
            print(f"[CONVERGED] t={t}, eps={eps}")
            return {"best": best, "best_feasible": best_feasible, "last": last, "stopped": stopped}

        sumdiff_hist.append(float(sum_diff))
        obj_hist.append(float(obj["total"]))

        curr_vdrop_tol = min(float(VDROP_SLACK_RESCUE_CAP), max(float(curr_vdrop_tol), float(max_vdrop_slack) * 1.02))
        curr_kcl_tol = min(float(KCL_SLACK_RESCUE_CAP), max(float(curr_kcl_tol), float(max_kcl_slack) * 1.02))
        if (ratio_v <= float(SLACK_TIGHTEN_TRIGGER)) and (ratio_k <= float(SLACK_TIGHTEN_TRIGGER)):
            curr_vdrop_tol = max(float(VDROP_SLACK_FINAL), float(curr_vdrop_tol) * float(VDROP_SLACK_DECAY))
            curr_kcl_tol = max(float(KCL_SLACK_FINAL), float(curr_kcl_tol) * float(KCL_SLACK_DECAY))

        if USE_PLATEAU_STOP and (t >= PLATEAU_MIN_ITERS):
            is_plateau, st = _plateau_check_sumdiff(sumdiff_hist, eps)
            if is_plateau:
                print("[EARLY-STOP: PLATEAU] sum|dell| has plateaued.")
                print(
                    f"  window={PLATEAU_WINDOW}, mean_recent={st.get('mean_recent', float('nan')):.3e}, "
                    f"range={st.get('recent_range', float('nan')):.3e} (tol={st.get('range_tol', float('nan')):.3e}), "
                    f"avg_step={st.get('avg_step', float('nan')):.3e} (tol={st.get('step_tol', float('nan')):.3e})"
                )
                return {"best": best, "best_feasible": best_feasible, "last": last, "stopped": stopped}

    if best["model"] is None and stopped["model"] is not None:
        best.update(stopped)
    if last["model"] is None and stopped["model"] is not None:
        last.update(stopped)

    return {"best": best, "best_feasible": best_feasible, "last": last, "stopped": stopped}


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
    print(f"  gen_cost   = {obj['gen_cost']:,.6f}")
    print(f"  pen_vdrop  = {obj['pen_vdrop']:,.6f}")
    print(f"  pen_kcl    = {obj['pen_kcl']:,.6f}")
    print(f"  loss_term  = {obj['loss_term']:,.6f}")
    print(f"  prox_term  = {obj['prox_term']:,.6f}")
    print(f"  theta_proj = {obj['theta_proj']:,.6f}")
    print(f"  total      = {obj['total']:,.6f}")

    max_vdrop_slack = max(_val(model.s_vdrop[i, j], 0.0) for (i, j) in E)
    max_kcl_slack = max(
        _val(model.sP_pos[i], 0.0) + _val(model.sP_neg[i], 0.0)
        + _val(model.sQ_pos[i], 0.0) + _val(model.sQ_neg[i], 0.0)
        for i in buses
    )
    sum_vdrop_slack = sum(_val(model.s_vdrop[i, j], 0.0) for (i, j) in E)
    sum_kcl_slack = sum(
        _val(model.sP_pos[i], 0.0) + _val(model.sP_neg[i], 0.0)
        + _val(model.sQ_pos[i], 0.0) + _val(model.sQ_neg[i], 0.0)
        for i in buses
    )
    print(f"  sum_vdrop_slack = {sum_vdrop_slack:.6e}")
    print(f"  sum_kcl_slack   = {sum_kcl_slack:.6e}")
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
        print("\n--- Switched Shunts (computed qsh = a*bcap*v) ---")
        for i in data["C"]:
            a = _val(model.a_sh[i], 0.0)
            v = _val(model.v[i], 1.0)
            qpu = float(data["bcap"][i]) * float(a) * float(v)
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

    print("\n--- Branch flows (P,Q,|S|) and ell_state ---")
    for (i, j) in E:
        P = _val(model.Pij[i, j], 0.0)
        Q = _val(model.Qij[i, j], 0.0)
        Smag = math.sqrt(max(P * P + Q * Q, 0.0))

        if (i, j) in T_set:
            denom = max(_pval(model.vsend[i, j], _val(model.v[i], 1.0)), DENOM_EPS)
        else:
            denom = max(_val(model.v[i], 1.0), DENOM_EPS)

        ell_calc = (P * P + Q * Q) / denom
        s_vdrop = _val(model.s_vdrop[i, j], 0.0)

        print(
            f"  ({i:2d}->{j:2d})  P={P:+.6f} pu  Q={Q:+.6f} pu  |S|={Smag:.6f} pu  "
            f"ell_calc={ell_calc:.6f}  ell_state={ell[(i,j)]:.6f}  vdrop_slack={s_vdrop:.3e}"
        )


# ============================================================
# Main
# ============================================================
def main():
    t0 = time.perf_counter()

    net = mcase.busmeshed39_opf(**NETWORK_BUILD_KWARGS)

    cfg = build_cfg_from_net_metadata(net)
    data = extract_data_fullmesh_branch_table(net, cfg)
    # Attach the live pandapower net so run_bfm_ag can derive a PF warm-start.
    data["net"] = net

    print("[INFO] Data summary (IEEE 41-bus, BFM-ivar with mitigation w/ A1-A6, B7-B10, C11-C15)")
    print("  network = ieee41bus.busmeshed39_opf (standalone build kwargs)")
    print(
        f"  A-knobs: tv_sum={USE_TV_SUM_EQ}, sos1={USE_BETA_SOS1}, recv_thermal={USE_RECV_THERMAL}, "
        f"obbt={USE_OBBT}, kvl_cycles={USE_KVL_CYCLES}, multi_cut_loss={USE_MULTI_CUT_LOSS}"
    )
    print(
        f"  B-knobs: gurobi_first={USE_GUROBI_FIRST}, warmstart={USE_MIP_WARMSTART}, "
        f"fix_and_relax={USE_FIX_AND_RELAX} (stable>={STABLE_ROUNDS_THRESHOLD}), "
        f"gap_schedule={USE_MIPGAP_SCHEDULE} ({MIPGAP_EARLY:.0e}/{MIPGAP_MID:.0e}/{MIPGAP_LATE:.0e})"
    )
    print(
        f"  C-knobs: anderson={USE_ANDERSON_ELL} (m={ANDERSON_WINDOW}, warmup={ANDERSON_WARMUP_ITERS}), "
        f"trust_region={USE_ADAPTIVE_TRUST_REGION}, socp_warm={USE_ELL_WARM_FROM_SOCP}, "
        f"theta_reuse={USE_THETA_REUSE_VS}, omega={USE_ADAPTIVE_OMEGA} (init={OMEGA_INIT})"
    )
    print(f"  #buses  = {len(data['buses'])}")
    print(f"  #branches  = {len(data['E'])}  (directed, aggregated from network branch metadata)")
    print(f"  #gens   = {len(data['gen_records'])}")
    print(f"  #OLTC   = {len(data['T'])}  (beta vars = {sum(len(data['K'][ij]) for ij in data['T'])})")
    print(f"  #shunts = {len(data['C'])}")
    print(f"  outer: max_iters={OUTER_MAX_ITERS}, eps={OUTER_EPS}")
    print(f"  ell_gamma={ELL_GAMMA}, theta_gamma={THETA_GAMMA}, ridge={THETA_RIDGE}")
    print("  exact vdrop/KCL mode: legacy slack variables are fixed to zero, so both equations are enforced as equalities.")
    print(f"  legacy slack-control machinery remains in the file for compatibility, but is inactive in exact-equality mode.")
    print(
        f"  obj-enhance: loss_proxy={USE_LOSS_PROXY}, llin>=0={LOSS_LIN_NONNEG_CONSTRAINT}, "
        f"prox_obj={USE_PROXIMAL}, prox_bounds={USE_PROX_BOUNDS}, theta_proj={USE_THETA_PROJ_PROX}"
    )
    if USE_PROX_BOUNDS:
        print(
            f"  prox-bounds schedule: P {PSTEP_FRAC_INIT:.2f}->{PSTEP_FRAC_FINAL:.2f} *Smax (decay={PSTEP_FRAC_DECAY}), "
            f"Q {QSTEP_FRAC_INIT:.2f}->{QSTEP_FRAC_FINAL:.2f} *Smax (decay={QSTEP_FRAC_DECAY}), "
            f"v {VSTEP_ABS_INIT:.2e}->{VSTEP_ABS_FINAL:.2e} on |V|^2 (decay={VSTEP_ABS_DECAY})"
        )
    if USE_PROXIMAL:
        print(
            f"  prox-objective weights: RHO_P={RHO_P}, RHO_Q={RHO_Q}, RHO_V={RHO_V}"
        )
    print(
        f"  theta-proj weight: RHO_THETA={RHO_THETA}"
    )
    print(
        f"  w_loss smoothing: beta={BETA_WLOSS}, scale={LOSS_WEIGHT_SCALE}, nonneg={WLOSS_NONNEG}"
    )
    print(
        f"  anti-oscillation: ell_ema_fix={USE_ELL_EMA_FIX} (beta_ell={BETA_ELL}), "
        f"state_damp={USE_STATE_DAMPING} (DAMPING_X={DAMPING_X}), theta_wrap_damp={USE_WRAP_THETA_DAMP}"
    )
    print(
        f"  outer ell-correction: enabled={USE_OUTER_ELL_HYBRID}, "
        f"target={ELL_HYBRID_I_WEIGHT:.2f}*ell_i + {ELL_HYBRID_V_WEIGHT:.2f}*ell_v, "
        f"force_follow={USE_OUTER_ELL_FORCE_FOLLOW}, backtrack_alphas={ELL_BACKTRACK_ALPHAS}, rel_step={ELL_BACKTRACK_REL_STEP}, abs_step={ELL_BACKTRACK_ABS_STEP}, "
        f"accepted_ell_as_next_fix={USE_ACCEPTED_ELL_AS_NEXT_FIX}"
    )
    print(
        f"  feasible-best: accepted iterates are judged against the CURRENT adaptive bounds; "
        f"final reference vdrop_tol={BEST_VDROP_TOL:.1e}, kcl_tol={BEST_KCL_TOL:.1e}"
    )
    print(
        f"  slack-blowup stop: enabled={USE_SLACK_BLOWUP_STOP}, after_feasible={BLOWUP_AFTER_FEASIBLE_ONLY}, "
        f"vdrop_tol={BLOWUP_VDROP_TOL:.2e}, kcl_tol={BLOWUP_KCL_TOL:.2e}"
    )
    print(
        f"  plateau-stop: enabled={USE_PLATEAU_STOP}, min_iters={PLATEAU_MIN_ITERS}, window={PLATEAU_WINDOW}, "
        f"range_tol=max({PLATEAU_ABS_RANGE_TOL}, {PLATEAU_REL_RANGE_TOL}*mean), "
        f"step_tol=max({PLATEAU_ABS_STEP_TOL}, {PLATEAU_REL_STEP_TOL}*mean)"
    )
    print(
        "  warm-start: pandapower PF on the calibrated network seeds ell_fix, v, "
        "Pij, Qij, Pg, Qg (iterative-OPF standard practice, gated by USE_PF_WARMSTART)."
    )

    sol = run_bfm_ag(data, max_iters=OUTER_MAX_ITERS, eps=OUTER_EPS, tee=TEE_SOLVER_LOG)

    best = sol["best"]
    best_feasible = sol["best_feasible"]
    last = sol["last"]
    stopped = sol.get("stopped", {"model": None, "iter": 0, "tag": ""})

    print("\n[SOLVED] Summary")
    print(f"  best(iter={best['iter']})  gen_cost={best['gen_cost']:.6f}  total={best['total']:.6f}  tag={best['tag']}")
    if best_feasible["model"] is not None:
        print(
            f"  best_feasible(iter={best_feasible['iter']})  "
            f"gen_cost={best_feasible['gen_cost']:.6f}  total={best_feasible['total']:.6f}  tag={best_feasible['tag']}"
        )
        if best_feasible["tag"] == "relaxed_only":
            print("  best_feasible_note = continuous relaxation fallback; not integer-feasible")
        elif best_feasible["tag"] == "fixed_from_relax":
            print("  best_feasible_note = rounded/fixed discrete recovery from relaxation")
    else:
        print("  best_feasible = none")
    print(f"  last(iter={last['iter']})  tag={last['tag']}")

    if best_feasible["model"] is not None:
        print_report(
            title=f"BEST FEASIBLE @ iter={best_feasible['iter']}  tag={best_feasible['tag']}",
            model=best_feasible["model"],
            data=data,
            ell=best_feasible["ell"],
            theta=best_feasible["theta"],
        )

    if best["model"] is not None and (
        best_feasible["model"] is None or best["iter"] != best_feasible["iter"]
    ):
        print_report(
            title=f"BEST (by generation cost) @ iter={best['iter']}  tag={best['tag']}",
            model=best["model"],
            data=data,
            ell=best["ell"],
            theta=best["theta"],
        )

    if last["model"] is not None and last["iter"] != best["iter"] and (
        best_feasible["model"] is None or last["iter"] != best_feasible["iter"]
    ):
        print_report(
            title=f"LAST ITERATE @ iter={last['iter']}  tag={last['tag']}",
            model=last["model"],
            data=data,
            ell=last["ell"],
            theta=last["theta"],
        )

    if stopped.get("model", None) is not None and (
        (best.get("model", None) is None) or (best.get("iter", 0) != stopped.get("iter", 0)) or (best.get("tag", "") != stopped.get("tag", ""))
    ) and (
        (last.get("model", None) is None) or (last.get("iter", 0) != stopped.get("iter", 0)) or (last.get("tag", "") != stopped.get("tag", ""))
    ) and (
        (best_feasible.get("model", None) is None) or (best_feasible.get("iter", 0) != stopped.get("iter", 0)) or (best_feasible.get("tag", "") != stopped.get("tag", ""))
    ):
        print_report(
            title=f"STOPPING-POINT CANDIDATE @ iter={stopped['iter']}  tag={stopped['tag']}",
            model=stopped["model"],
            data=data,
            ell=stopped["ell"],
            theta=stopped["theta"],
        )

    t1 = time.perf_counter()
    print(f"\n[INFO] Total elapsed time: {t1 - t0:.2f} seconds.")


if __name__ == "__main__":
    main()
