# BFM_ivar_mit.py
# ------------------------------------------------------------
# BFM-ivar with Sec. 4.2 mitigations on IEEE 3012-bus mesh with OLTC + shunt.
# Self-contained (no separate base file): T4 cycle-KVL penalty, W3 adaptive
# hybrid weight, H1 top-K KCL hard buses, distributed slack, capacity
# release, schedule tuned for 3012-bus stability.
#
# - main() auto-generates bfmag_warmstart_snapshot.npz when missing,
#   by invoking make_warmstart_snapshot.py with the latest DCOPF
#   results, so iter 1 no longer falls back silently to the
#   (alpha*smax)^2 / vmean^2 heuristic.
#
# Subproblem: MIQCP with ell fixed; bounded vdrop/KCL slack penalized in
# the objective; outer ell-correction with EMA smoothing and backtracking.
# adapted to ieee3012bus.py (IEEE 3012-bus MATPOWER case3012wp).
# ------------------------------------------------------------

import datetime as _dt
import math
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, Any, Optional, List

import numpy as np
import pyomo.environ as pyo
from pyomo.opt import TerminationCondition

import ieee3012bus as mcase


RESULT_LOG_PATH = Path(__file__).resolve().with_name("BFM_ivar_mit_results.txt")


# ============================================================
# Global settings
# ============================================================
# 2026-04-28: TEE turned ON temporarily so iter-1 SCIP progress is
# visible in the terminal.  Set back to False once the log cadence is
# verified -- TEE adds ~5-10% overhead and clutters the archive log.
TEE_SOLVER_LOG = True

OUTER_MAX_ITERS = 20  # mit: lets the slow ell EMA (BETA_ELL=0.08) fully converge.
OUTER_EPS = 1e-5
DENOM_EPS = 1e-10

ELL_GAMMA = 0.5
THETA_GAMMA = 0.5
CLIP_DTHETA_PREV = math.pi
THETA_RIDGE = 1e-8

# Bounded-slack constraints: iter 1 uses a very wide window so that the cold
# start (Pg, ell_fix all zero) can land on a feasible incumbent quickly.
# Subsequent iterations tighten via decay and/or the adaptive bound tracker.
# The window must stay wide enough to absorb transient ell/ell_fix mismatch,
# but the slack is then heavily penalized in the objective (RHO_KCL, RHO_VDROP
# below) so the solver never uses it unless the problem truly forces it.
VDROP_SLACK_INIT  = 20.0
VDROP_SLACK_FINAL = 3.0e-1
VDROP_SLACK_DECAY = 0.92  # mit: 0.85 -> 0.92 milder contraction so iter 3 stays feasible

# KCL slack window: 50 pu/bus is wide enough to absorb the heuristic
# warm-start's uniformly-distributed phantom reactive losses without the
# stiff buses (239, 3010, 3007, ...) saturating into a REJECT cascade.
# RHO_KCL=1e4 makes 1 pu of slack cost ~1000 EUR/MW, above any Polish
# generator marginal cost, so the solver minimizes usage. Shrinking
# below this reproduces the iter-2 REJECT loop.
KCL_SLACK_INIT  = 50.0
KCL_SLACK_FINAL = 3.0
KCL_SLACK_DECAY = 0.92  # mit: 0.75 -> 0.92 milder contraction so iter 3 stays feasible

# If the current scheduled bounds are still too tight for a particular outer
# iteration, the robust subproblem solver may retry with slightly relaxed
# bounded-slack windows before declaring infeasibility.
SLACK_RESCUE_MULTS = (1.0, 2.0, 5.0)  # mit: keep x5 tier as the rescue ladder needs an extra retry under the milder schedule.
VDROP_SLACK_RESCUE_CAP = 50.0
KCL_SLACK_RESCUE_CAP   = 100.0

# Adaptive bounded-slack control:
# - start from the current accepted bounds
# - tighten only when the accepted step uses comfortable interior slack
# - if the subproblem fails or needs rescue bounds, reject that candidate,
#   keep the previous accepted iterate, and enlarge/freeze the bounds
SLACK_TIGHTEN_TRIGGER = 0.80
SLACK_ENLARGE_ON_FAIL = 1.50
SLACK_ENLARGE_ON_REJECT = 1.25
SLACK_REJECT_MARGIN = 1.05
REJECT_RESCUED_STEPS = False
REJECT_RELAXED_ONLY_STEPS = True
MAX_CONSECUTIVE_REJECTS = 6  # mit: gives the rescue ladder one extra chance per iter so iter 3-4 doesn't hit EARLY-STOP under the milder schedule.

SCIP_TIME_LIMIT = 600.0         # 2026-04-29: 180 was too aggressive. Iter 2 of run 03:26 was 60K LP iter at 150s with dualbound improving, just 30s short of incumbent. Binary fixing leaves a continuous QP that still needs ~400-700s on the 26k-var root LP. Per single SCIP solve.
SCIP_GAP_LIMIT = 1e-4
# iter 1 is the hardest MIP (no warm state from previous iters, and as of
# 2026-04-26 also the first iter to carry the full loss_proxy + multi-cut +
# proximal load -- WARMUP_ITERS=0). Give it a materially larger budget and
# looser gap so it always returns an incumbent instead of triggering the
# REJECT path. Raised from 900s to 1500s when warmup-disable was activated.
SCIP_TIME_LIMIT_ITER1 = 1500.0
SCIP_GAP_LIMIT_ITER1 = 5e-3

# ----------------------------
# Warmup iterations
# ----------------------------
# Iters 1..WARMUP_ITERS skip loss_proxy / proximal / multi-cut and run on
# Pg-warm + ell_fix only. WARMUP_ITERS=2 is the proven-feasible setting:
# attempts at WARMUP_ITERS=0 made the iter-1 LP relaxation infeasible at
# presolve because warm["Pij"|"Qij"|"v"] are zero/1.0 at iter 1, so the
# loss-proxy plane gets built at a degenerate (Pbar=0, Qbar=0, vbar=1.0)
# anchor and RHO_P=10 proximal pulls every flow toward zero. Lifting the
# WARMUP_ITERS=0 path safely would require seeding Pij/Qij/v from the
# warm-state snapshot (today the snapshot loader only fills ell_pf).
WARMUP_ITERS = 2
SCIP_NODE_LIMIT = 300000
SCIP_MEMORY_LIMIT_MB = 8192
SCIP_FEASTOL = 1e-5
SCIP_DUALFEASTOL = 1e-7

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
    expose_global_recommendations=True,
    # mit: capacity-released + distributed slack settings
    release_ext_grid_cap=True,
    disable_pump_mode=True,
    add_distributed_slack_gens=True,
    distributed_slack_count=5,
    distributed_slack_pmax_each=2000.0,
    distributed_slack_c1=80.0,
    distributed_slack_c2=0.01,
)

# ----------------------------
# Python warm-start snapshot
# ----------------------------
# The (alpha*smax)^2 / vmean^2 heuristic fallback (_compute_ell_warm_start)
# spreads ~650 MW active and ~3.5 GVAr reactive of fictitious losses
# uniformly across all 3566 BFM edges. At 3012-bus scale this saturates
# the per-bus KCL slack budget on a few stiff buses and triggers an iter-2
# REJECT cascade even with the rescue window widened to 20 pu/bus.
#
# To avoid that, make_warmstart_snapshot.py produces a Python-only snapshot
# (.npz) seeded from a converged DCOPF (default) or SOCP relaxation: it
# carries (P_pu, Q_pu, vm_pu, theta_deg) per BFM edge / bus so iter 1 sees
# spatially-correct flows. main() auto-runs make_warmstart_snapshot.py
# when the .npz is missing and a DCOPF results file is available.
#
# Cascade (in order): Python snapshot -> pandapower runpp -> heuristic.
USE_PYTHON_WARMSTART_SNAPSHOT = True
PYTHON_SNAPSHOT_PATH = Path(__file__).resolve().with_name(
    "bfmag_warmstart_snapshot.npz"
)

# ----------------------------
# 1번 변화: objective enhancement knobs
# ----------------------------
USE_LOSS_PROXY = True
# LOSS_LIN_NONNEG_CONSTRAINT=False: the per-edge l_lin >= 0 cut adds the
# only structural constraint delta between iter 1 (always feasible) and
# iter 2; under the heuristic warm-start that cut combined with the
# proximal pull made iter 2 infeasible. The loss proxy still contributes
# through the objective coefficients, so dispatch remains loss-aware.
LOSS_LIN_NONNEG_CONSTRAINT = False
USE_PROXIMAL = True
USE_PROX_BOUNDS = False
USE_THETA_PROJ_PROX = True

# Proximal weights: 10/10/10 initial → 0.1/0.1/0.1 final via 0.85 decay.
# Warmup iters (1..WARMUP_ITERS) bypass proximal entirely; iter 3+ anchors
# against iter-2's solution. An attempt at 1/0.1/1 paired with WARMUP=0
# made iter 1 LP-infeasible at presolve, so the original values stayed.
RHO_P_INIT = 10.0
RHO_P_FINAL = 0.1
RHO_P_DECAY = 0.85
RHO_Q_INIT = 10.0
RHO_Q_FINAL = 0.1
RHO_Q_DECAY = 0.85
RHO_V_INIT = 10.0
RHO_V_FINAL = 0.1
RHO_V_DECAY = 0.85
RHO_THETA_INIT = 1.0
RHO_THETA_FINAL = 1.0e-2
RHO_THETA_DECAY = 0.85

# Legacy constants (kept so diagnostic prints don't break). The actual
# effective weights are computed per outer iteration by _scheduled_rho().
RHO_P = RHO_P_INIT
RHO_Q = RHO_Q_INIT
RHO_V = RHO_V_INIT
RHO_THETA = RHO_THETA_INIT

# ----------------------------
# Bounded-slack mode for vdrop / KCL
# ----------------------------
# On 3012-bus the strict-equality mode (slacks fixed to zero, used in
# upgrade4.py) makes iter 2 infeasible because iter 1's ell_fix=0 produces
# a big ell mismatch that the strict equalities cannot absorb. Bounded-
# slack lets vdrop/KCL slacks take values in [0, *_slack_max] but
# penalizes them in the objective so the solver drives them toward zero
# whenever a near-exact solution exists.
USE_EXACT_EQUALITY_SLACK = False  # True => upgrade4.py mode; False => bounded+penalized
# Slack penalty weights. The 300-bus values (2e5/2e7) wreck SCIP LP
# numerics at this scale, and weights <=5e2 are below Polish generator
# marginal costs (KCL slack becomes "cheaper than real generation",
# inflating apparent cost-optimality with non-physical injections). The
# 1e3/1e4 values keep numerics clean while pricing 1 pu of KCL slack at
# ~1000 EUR/MW (well above any case3012wp marginal cost).
RHO_VDROP = 1.0e5  # mit: 1e3 -> 1e5 strong slack penalty (port from 300-bus calibration)
RHO_KCL   = 1.0e6  # mit: 1e4 -> 1e6 strong slack penalty (port from 300-bus calibration)

# Legacy step-bound schedule (kept for compatibility/logging).
# With USE_PROX_BOUNDS=False, P/Q/v proximal regularization is handled in the objective.
PSTEP_FRAC_INIT  = 1.25
PSTEP_FRAC_FINAL = 0.55
PSTEP_FRAC_DECAY = 0.92

QSTEP_FRAC_INIT  = 1.25
QSTEP_FRAC_FINAL = 0.55
QSTEP_FRAC_DECAY = 0.92

# Voltage-step bound is applied to v = |V|^2. Keep it fairly loose early on so
# the first accepted iterate can move away from the flat start, then tighten.
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
# EMA smoothing on the ell_fix fed into the inner MIQCP. The iter1->iter2
# transition is violent at this scale (ell can jump 0 -> ~85 pu^2 on some
# edges); BETA_ELL=0.5 tempers it. EMA must stay ON: under the heuristic
# warm-start, BETA=0.9 keeps the wrong spatial distribution sticky on the
# ~90% of edges that carry near-zero flow.
USE_ELL_EMA_FIX = True
BETA_ELL = 0.08  # mit: slow EMA (port from 300-bus ver2_BFMag_claude calibration)
ELL_CLIP_NONNEG = True
ELL_CLIP_MAX = True
# Per-edge ell_fix cap: ellmax_e = max(ELL_FIX_HARD_CAP_PU2,
#                                      (smax_e * ELL_HARD_CAP_HEADROOM / vmin_fr)^2)
# clipped above by raw_ellmax = (smax/vmin)^2. The scalar 5 pu^2 floor is
# tuned for ~1 pu lines; the smax-derived term lets major transformers
# (3008->182, 145->3007, ...) keep their natural ell ~= 27 pu^2 at thermal,
# which iter 3+ needs for reactive KCL closure. 1.20 headroom = 144% of
# thermal — leaves room for short-term overload, still bounds the
# pathological iter-1 tail.
ELL_FIX_HARD_CAP_PU2 = 5.0
ELL_HARD_CAP_HEADROOM = 1.20

USE_STATE_DAMPING = True
DAMPING_X = 0.35

USE_WRAP_THETA_DAMP = True

# ----------------------------
# Outer ell-correction knobs (accuracy improvement without changing inner problem)
# ----------------------------
USE_OUTER_ELL_HYBRID = True
# Blend current-law ell estimate and vdrop-rearranged ell estimate.
ELL_HYBRID_I_WEIGHT = 0.50
ELL_HYBRID_V_WEIGHT = 0.50
# Trust-step backtracking from previous ell-state toward the hybrid target.
ELL_BACKTRACK_ALPHAS = (1.0, 0.5, 0.25, 0.10, 0.05, 0.01)
ELL_BACKTRACK_REL_STEP = 0.35
ELL_BACKTRACK_ABS_STEP = 1.0e-4
# After an accepted step, use the corrected ell-state as the next inner fixed ell.
USE_ACCEPTED_ELL_AS_NEXT_FIX = True
# Force the inner ell_fix to follow the outer-computed ell target directly.
# Surrogate loss remains in the objective; only the ell feed/update policy changes.
USE_OUTER_ELL_FORCE_FOLLOW = True
# EMA must stay ON: the ACCEPT block commits ell_fix as a (1-BETA_ELL)-
# blended step, not a direct cand_ell_state assignment. Bypassing EMA at
# commit reproduces the "REJECT every iterate from t=2" loop because the
# unsmoothed outer-hybrid target lands at ~50 pu^2 on low-impedance edges.

# With bounded-slack, every accepted iterate satisfies the current
# feasibility window. Feasible-best is judged against the scheduled bounds
# and the separate blow-up stop is disabled.
BEST_VDROP_TOL = VDROP_SLACK_FINAL
BEST_KCL_TOL = KCL_SLACK_FINAL
USE_SLACK_BLOWUP_STOP = False
BLOWUP_AFTER_FEASIBLE_ONLY = False
BLOWUP_VDROP_TOL = VDROP_SLACK_FINAL
BLOWUP_KCL_TOL = KCL_SLACK_FINAL

# ----------------------------
# 2번 변화: plateau early-stop knobs
# ----------------------------
USE_PLATEAU_STOP = True
PLATEAU_MIN_ITERS = 5  # 2026-04-29 cut from 10: gen_cost flattens by iter 2-3 on this case, so plateau detection should engage earlier.
PLATEAU_WINDOW = 6  # 2026-04-29 cut from 12: matched to the smaller MIN_ITERS so the rolling range/step check is over a comparable horizon.

PLATEAU_ABS_RANGE_TOL = 1e-5
PLATEAU_REL_RANGE_TOL = 1e-3

PLATEAU_ABS_STEP_TOL = 1e-6
PLATEAU_REL_STEP_TOL = 1e-3

PLATEAU_REQUIRE_NONCONVERGED = True


# ============================================================
# Ported from ver3_BFMag_final.py (300-bus): structural upgrades to break
# the iter-2 REJECT cascade that survived the EMA + ELL_FIX_HARD_CAP_PU2 +
# heuristic warm-start fixes.
#
# - [A6] Multi-cut loss: replace the single-plane loss linearization with
#   an envelope of K=5 stored Taylor planes.  Each plane is a tangent
#   under-approximation of (P^2+Q^2)/v at a previous iterate's flows;
#   the envelope tightens the loss representation iter over iter without
#   making the inner problem non-convex.  This is the key fix for the
#   iter-2 infeasibility -- with one plane the loss surface is locally
#   exact at iter-1's flows but blows up the moment the inner MIQCP wants
#   to move (and on 3012-bus iter 2 MUST move because iter-1 saturates
#   KCL slack on a few stiff buses).
# - [B7]+[B8] Solver order: try Gurobi (fast on convex MIQCP), fall back
#   to SCIP NL, then SCIP default.  Each attempt uses a fresh
#   SolverFactory so previous .nl writer state cannot leak.  USE_GUROBI_
#   FIRST=True is safe even without a Gurobi license -- _try_gurobi is a
#   graceful no-op then.
# - [B10] Round-dependent gap schedule: warmup iters keep the existing
#   loose gap (5e-3); mid iters tighten to 1e-3; late iters to 1e-4.
#   Subsumes the binary WARMUP-vs-normal gap split previously hardcoded
#   in run_bfm_ag.  Time limits remain governed by the existing
#   SCIP_TIME_LIMIT_ITER1 / SCIP_TIME_LIMIT split.
# - [C14] Theta-reuse vS in loss proxy: already wired via
#   warm["u_send"] -> compute_loss_proxy_coeffs(ubar=...).  Flag added
#   for documentation; flipping False would force vS = vbar[i].
#
# NOT ported (ver3 itself disables them, and they are even riskier at
# 3012-bus scale):
#   [A4] OBBT (3566 branches x 2 bounds x 2 directions = ~14k LPs => prohibitive)
#   [A5] KVL cycle cuts (asymmetrically tighten under bounded slack)
#   [C11] Anderson type-II on ell (overshoots when ell-mismatch stalls,
#         which it does on this network's low-z tail)
#   [C12] Adaptive trust region (cuts feasible region under flat warm-start)
#   [C13] MISOCP-BFM warm-start for ell (heavy, requires an external SOCP
#         solve up front; the SOCP solver is not included in this upload).
#         Re-enable only if iter-1 cold start remains stubborn after
#         Tier-1 fixes.
# ============================================================

# [A6] Multi-cut loss under-approximation.  When True, build_subproblem
# adds an l_edge variable (one per edge) constrained by all stored
# Taylor planes; the objective uses w_loss * sn * r * l_edge in place
# of the single-plane (cP*P + cQ*Q + cV*v) term.  History size = K
# planes kept (most recent K used).
USE_MULTI_CUT_LOSS = False  # mit: not in BFMivar prescription (T4+H1 carry feasibility instead).
MULTI_CUT_LOSS_HISTORY = 3  # 2026-04-29 cut from 5: smaller envelope keeps LP overhead manageable (~10K cuts vs 17K) while still preventing single-plane gaming.
MULTI_CUT_LOSS_L_NONNEG = True

# [B7] Solver order.  _try_gurobi() is a graceful no-op when Gurobi is
# not on PATH, so flipping this True without a Gurobi license is safe
# (every solve falls through to SCIP at zero cost).  The risk is when
# Gurobi IS available but cannot find an incumbent within its budget:
# the rescue ladder calls solve_with_scip up to 9x per outer iter
# (binary -> relaxed -> fixed_from_relax x 3 mults), so a failing Gurobi
# at 120s/attempt burns up to 9 * 120 = 1080s before any SCIP attempt
# starts.  We default to False on 3012-bus because the project's CLAUDE.md
# documents the env as SCIP-only; flip to True if Gurobi is genuinely
# installed and you want the speedup on iter 1's hard MIQCP.
USE_GUROBI_FIRST = False
GUROBI_TIME_LIMIT = 120.0
GUROBI_MIPGAP_DEFAULT = 1e-4

# [B8] MIP warm-start.  For SCIP NL, initial variable values written to
# the .nl file by pyomo's writer are always consumed -- no kwarg needed.
# This flag now only controls whether _try_gurobi passes warmstart=True
# to Gurobi (the Gurobi wrapper honors it).
USE_MIP_WARMSTART = True

# [B10] Round-dependent MIP gap schedule.  Subsumes the existing
# WARMUP_ITERS-vs-normal binary split: GAP_EARLY for warmup iters
# (1..GAP_SWITCH_EARLY), GAP_MID for the convergence phase
# (early+1..GAP_SWITCH_MID), GAP_LATE for the tail.  When False,
# falls back to the legacy SCIP_GAP_LIMIT_ITER1/SCIP_GAP_LIMIT split.
USE_ROUND_DEPENDENT_GAP = True
GAP_EARLY = 5e-3                # iter 1..GAP_SWITCH_EARLY (default == WARMUP_ITERS)
GAP_MID   = 1e-3                # iter (early+1)..GAP_SWITCH_MID
GAP_LATE  = 1e-4                # iter (mid+1)+
GAP_SWITCH_EARLY = 2            # match WARMUP_ITERS
GAP_SWITCH_MID = 10

# [C14] Theta-reuse vS in loss proxy.  Already wired in this file via
# warm["u_send"] -> compute_loss_proxy_coeffs(ubar=warm["u_send"]).
# Flag kept for documentation only.
USE_THETA_REUSE_VS = True


# ============================================================
# 3번 변화: convergence fixes for the iter-2->iter-3 cliff (2026-04-28)
# ------------------------------------------------------------
# Diagnosed from earlier solver logs:
# iter 1 ACCEPTed with max_kcl_slack=5.21 pu (huge), iter 2 ACCEPTed with
# max_kcl_slack=1.77 pu (still huge), iter 3+ all REJECT even at relax/x5
# (kcl<=140). Root cause: at iter 3 prox+multi-cut+tight-gap turn ON
# simultaneously, anchored on iter-2's slack-saturated non-physical state.
# Each flag below is independently togglable; defaults are tuned for the
# 3012-bus pathology described in branchflowmodel_3012bus/CLAUDE.md.
# ============================================================

# (A) Anchor-cleanliness gate.  Disable proximal + multi-cut + theta-proj
# when the previous accepted iter's slacks indicate the anchor is non-
# physical.  Without this, iter 3 anchors prox/loss-cuts on a state where
# 1.77 pu of KCL is artificially absorbed, which is the iter-3 cliff.
USE_ANCHOR_CLEAN_GATE = True
ANCHOR_CLEAN_KCL_PU = 3.0
ANCHOR_CLEAN_VDROP_PU = 0.30

# (B) Plane-feasibility guard on multi-cut history.  Reject a new Taylor
# plane if it would force l_edge < 0 at any stored anchor (within tol),
# i.e. if the new plane is inconsistent with prior planes' shared region.
USE_PLANE_FEAS_GUARD = False  # mit: paired with USE_MULTI_CUT_LOSS=False
PLANE_GUARD_TOL = 1.0e-3

# (C) Trust-region on ell_fix step.  Wrap EMA with per-edge bound
#   |ell_t - ell_{t-1}| <= max(TR_RADIUS_ABS, TR_RADIUS_REL * |ell_{t-1}|).
# Adapt: enlarge by TR_GROW on ACCEPT, shrink by TR_SHRINK on REJECT,
# clamp to [TR_MIN, TR_MAX].  Synergistic with EMA -- EMA gives shape,
# TR caps magnitude.  When OFF the original EMA path is unchanged.
USE_ELL_TRUST_REGION = True
TR_RADIUS_REL_INIT = 0.50
TR_RADIUS_ABS_INIT = 0.20
TR_GROW = 1.5
TR_SHRINK = 0.5
TR_RADIUS_REL_MIN = 0.05
TR_RADIUS_REL_MAX = 1.50
TR_RADIUS_ABS_MIN = 1.0e-3
TR_RADIUS_ABS_MAX = 1.00

# (D) Augmented Lagrangian on KCL/vdrop slack.  Maintain dual variables
# lambda_kcl[bus], lambda_vdrop[edge].  Objective gains a linear lambda*s
# term on top of the existing quadratic-equivalent linear penalty.  At
# ACCEPT time the duals climb proportional to slack used, so feasibility
# is enforced even if RHO is moderate.  Deltas are clipped via AL_LAM_MAX
# to prevent runaway.
#
# 2026-04-28: turned OFF after the all-flags-on iter 1 run hit the 1500s
# SCIP budget without producing an incumbent. AL adds a per-bus linear
# lambda*slack term that, when paired with E's down-scaled gen_cost,
# leaves SCIP without a clear LP-relaxation gradient -- the optimizer
# finds many "almost-feasible" candidates and stalls in B&B. Re-enable
# only AFTER iter-3 cliff is resolved by (A)+(B), since AL is orthogonal
# to that fix and only matters for tail-iter feasibility tightening.
USE_AUG_LAGRANGIAN = False
AL_ETA_KCL = 5.0e2
AL_ETA_VDROP = 5.0e1
AL_LAM_MAX_KCL = 1.0e5
AL_LAM_MAX_VDROP = 1.0e4

# (E) Phase-I / Phase-II split.  For first PHASE_I_ITERS iterations,
# down-weight gen_cost and up-weight slack penalties so the solver chases
# feasibility first; flip back to normal weights afterwards.  Combines
# with WARMUP_ITERS but is orthogonal -- WARMUP turns features off,
# Phase-I rebalances objective.
#
# 2026-04-28: turned OFF.  cost_scale=1e-3 makes the iter-1 MIQCP a
# "minimize slack" LP-relaxation with a near-zero gen_cost gradient, and
# SCIP loses the cost-based branching guidance that makes the 216
# OLTC + 9 shunt binary tree tractable.  WARMUP_ITERS=2 already provides
# the same feasibility-first effect by turning prox/loss/multi-cut OFF,
# so Phase-I was redundant and counterproductive for iter 1's MIP search.
# Re-enable only with PHASE_I_COST_SCALE >= 0.1 if the problem actually
# needs more feasibility pressure than WARMUP alone supplies.
USE_PHASE_I_SPLIT = False
PHASE_I_ITERS = 2
PHASE_I_COST_SCALE = 1.0e-3
PHASE_I_RHO_BOOST = 5.0

# (F) Binary fixing after stabilization.  After F_BIN_FIX_ITER, fix OLTC
# tap binaries and shunt step variables to the incumbent's values, so
# every subsequent iter is a continuous QP in SCIP terms (much faster).
# Set F_BIN_FIX_ITER very high to disable.
USE_BINARY_FIXING = True
F_BIN_FIX_ITER = 2  # 2026-04-28 cut from 4: in run 19:20 iter 1's binary incumbent (gen_cost=1.664M, gap=0.25%) matched all subsequent iters' gen_cost to integer pu-cents. Locking it from iter 2 onward turns iter 3+ into a continuous QP and eliminates the root-LP MIP tree entirely.
F_BIN_FIX_OLTC = True
F_BIN_FIX_SHUNT = True

# (G) Anderson Acceleration on the ell fixed-point map.  Uses last
# AA_HISTORY iterates; safeguard: if the AA step's residual norm exceeds
# AA_SAFEGUARD_FACTOR * the EMA step's residual norm, fall back to EMA.
# Disabled by default while WARMUP_ITERS <= AA_MIN_HISTORY.
#
# 2026-04-28: turned OFF until the (A)+(B)+(C) trio is verified.  AA is a
# CONVERGENCE-RATE accelerator, not a convergence FIX -- it does nothing
# helpful before the iterates converge in shape.  With (C) trust-region
# already capping per-iter ell movement and (A) gating prox/multi-cut,
# AA is decoupled from the cliff fix and adds 9 LinAlg-solve risk per
# outer iter for no clear benefit until the basic alternation works.
USE_ANDERSON_ELL = False
AA_HISTORY = 3
AA_MIN_HISTORY = 2
AA_REG = 1.0e-6
AA_SAFEGUARD_FACTOR = 2.0

# (H) Stratified KCL penalty (load vs gen).  Buses with no local
# generation get RHO_KCL_LOAD; buses with local generation get
# RHO_KCL_GEN (heavier).  Reasoning: slack on a load bus is "physically
# OK to absorb" (no Q reserve at that bus), while slack on a gen bus
# means ell_fix is likely wrong nearby.
USE_STRATIFIED_KCL_PENALTY = True
# 2026-04-29 reversed (was 0.5 / 2.0): the original logic ("load buses have
# no Q reserve, so absorbing slack there is physical") applies to Q-side,
# but the dominant slack in case3012wp iter 2 is on the P-side. P-slack at a
# load bus is unmet load (most unphysical), while P-slack at a gen bus can
# be absorbed by ramping a not-yet-saturated generator. Diagnostic: in run
# 07:38 all top-10 KCL-slack buses (92, 63, 215, 2989, 62, ...) were load
# buses precisely because the previous factors made slack 4x cheaper there.
# Q-side is not materially affected (max kcl_slackQ ~ 2e-8 is numerical noise).
RHO_KCL_LOAD_FACTOR = 2.0
RHO_KCL_GEN_FACTOR = 0.5

# (I) IIS-guided slack widening.  When relax/x1 fails, identify top-K
# offending buses/edges from the relaxed solution and widen ONLY their
# bounds for the next attempt instead of multiplying every bound.
USE_IIS_GUIDED_WIDENING = True
IIS_WIDEN_TOPK_BUSES = 20
IIS_WIDEN_TOPK_EDGES = 50
IIS_WIDEN_FACTOR = 5.0

# (J) Slack-aware proximal anchor.  When forming prox_prev, subtract a
# fraction of the absorbed slack from (P_prev, Q_prev) so the anchor
# reflects a more physical operating point than the raw solution.
USE_SLACK_AWARE_ANCHOR = False  # mit: not in BFMivar prescription
SLACK_AWARE_FRACTION = 0.5


# ============================================================
# BFM-ivar mitigation features (T4 / W3 / H1)
# ============================================================
# T4: Loop residual penalty -- meshed KVL consistency directly enforced
# inside the inner MIQCP. When data["loop_basis_T4"] holds a list of
# fundamental cycles (each = list of (u, v, sign) tuples) and
# USE_T4_LOOP_PENALTY is True, the objective gains
#     RHO_LOOP_T4 * sum_C ( sum_{(u,v) in C} sign*(x_uv P_uv - r_uv Q_uv)/sqrt(vbar_u*vbar_v) )^2.
# This is a quadratic-in-(P,Q) penalty so the inner problem stays MIQCP.
USE_T4_LOOP_PENALTY = True  # mit: enables meshed-KVL consistency inside the inner MIQCP
RHO_LOOP_T4 = 1.0e4  # mit: 1e2 -> 1e4 stronger loop-residual penalty

# W3: Adaptive per-edge hybrid weight for the outer ell update. When True,
# replaces the static (ELL_HYBRID_I_WEIGHT, ELL_HYBRID_V_WEIGHT) with a
# per-edge weight derived from the impedance condition r^2 + x^2: small-
# impedance edges weight ell_I more because the vdrop-rearranged estimator
# becomes ill-conditioned on low-z lines (its denominator is r^2 + x^2).
# Reference scale is data["_zsq_median_T4"] -- the median r^2+x^2 across
# the network, cached during extract_data_fullmesh_branch_table().
USE_W3_ADAPTIVE_HYBRID = True  # mit: per-edge hybrid weight using r^2+x^2 scaled by network median

# H1 (BFMivar hard constraint): top-K KCL slack absorbing buses get
# strict equality (sP_pos = sP_neg = sQ_pos = sQ_neg = 0). This forces
# the inner MIQCP to find a dispatch that closes KCL exactly at those
# buses, rather than absorbing the imbalance via slack. The bus list is
# extracted from a prior result's "KCL slack (top N)" section during
# extract_data_fullmesh_branch_table() and cached on data["kcl_hard_buses"].
# Only the binary (non-relaxed) attempts impose the hard equality;
# relaxed/rescue attempts retain the soft slack so the rescue ladder
# can still recover from a hard-equality infeasibility.
USE_KCL_HARD_BUSES = True  # mit: top-K KCL absorbing buses pinned to strict equality

# H1 source: top-K KCL slack buses are extracted from the previous run's
# result log (this file's own RESULT_LOG_PATH).  Because the __main__
# teeing TRUNCATES that file before main() runs, the extraction has to
# happen BEFORE the teeing — we read the prior content in the __main__
# block and stash the bus list in _H1_KCL_HARD_BUSES, which
# extract_data_fullmesh_branch_table reads when populating
# data["kcl_hard_buses"].  First run finds no file → empty list → H1
# no-op.  Second run sees the previous run's top-K → H1 active.
KCL_HARD_TOPK = 3
KCL_HARD_SOURCE_FILE = "BFM_ivar_mit_results.txt"
KCL_HARD_MIN_SLACK_PU = 0.5
_H1_KCL_HARD_BUSES: List[int] = []  # populated in __main__ before teeing

# Verification tolerances (Sec. 3.2.3 / Prop. 2)
VERIFY_KCL_TOL = 5.0e-2
VERIFY_VDROP_TOL = 1.0e-3


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


def _scheduled_rho(iter_idx: int) -> Tuple[float, float, float, float]:
    t = max(1, int(iter_idx))
    # When DECAY < 1 the schedule decays from INIT toward FINAL (FINAL is the
    # lower cap, original BFMag behavior). When DECAY >= 1 the schedule ramps
    # up from INIT toward FINAL (FINAL is the upper cap, BFMivar v1 mode --
    # contraction-guaranteeing growth of proximal regularization to ensure
    # sum|dell| -> 0 per Sec. 4.2.1 of the writeup).
    def _branch(init_, final_, decay_):
        raw = float(init_) * (float(decay_) ** (t - 1))
        if float(decay_) >= 1.0:
            return min(float(final_), raw)
        return max(float(final_), raw)
    rp = _branch(RHO_P_INIT, RHO_P_FINAL, RHO_P_DECAY)
    rq = _branch(RHO_Q_INIT, RHO_Q_FINAL, RHO_Q_DECAY)
    rv = _branch(RHO_V_INIT, RHO_V_FINAL, RHO_V_DECAY)
    rt = _branch(RHO_THETA_INIT, RHO_THETA_FINAL, RHO_THETA_DECAY)
    return float(rp), float(rq), float(rv), float(rt)


def _proportional_pg_warmstart(data: Dict[str, Any]) -> Dict[str, Dict[int, float]]:
    """Distribute total P/Q load among generators in proportion to their
    positive pmax/qmax. Gives iter 1 a physically reasonable Pg starting
    point so the MIP quickly finds an incumbent instead of triggering REJECT."""
    total_load_p = sum(float(data["Pd_pu"][i]) for i in data["buses"])
    total_load_q = sum(float(data["Qd_pu"][i]) for i in data["buses"])

    sum_pmax_pos = sum(max(0.0, float(rec["pmax_pu"])) for rec in data["gen_records"])
    sum_qmax_pos = sum(max(0.0, float(rec["qmax_pu"])) for rec in data["gen_records"])

    Pg_warm: Dict[int, float] = {}
    Qg_warm: Dict[int, float] = {}
    for idx, rec in enumerate(data["gen_records"]):
        pmax_rec = float(rec["pmax_pu"])
        pmin_rec = float(rec["pmin_pu"])
        pmax_pos = max(0.0, pmax_rec)
        share_p = pmax_pos / max(sum_pmax_pos, 1.0e-9)
        Pg_warm[int(idx)] = _clip(share_p * float(total_load_p), pmin_rec, pmax_rec)

        qmax_rec = float(rec["qmax_pu"])
        qmin_rec = float(rec["qmin_pu"])
        qmax_pos = max(0.0, qmax_rec)
        share_q = qmax_pos / max(sum_qmax_pos, 1.0e-9)
        Qg_warm[int(idx)] = _clip(share_q * float(total_load_q), qmin_rec, qmax_rec)

    return {"Pg": Pg_warm, "Qg": Qg_warm}


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
    # W3: per-edge adaptive weight from impedance conditioning (BFMivar v1).
    # zsq_med_T4 is cached on data by BFM_ivar_mit; if absent we fall back
    # to the static (ELL_HYBRID_I_WEIGHT, ELL_HYBRID_V_WEIGHT).
    _zsq_med_T4 = float(data.get("_zsq_median_T4", 0.0)) if USE_W3_ADAPTIVE_HYBRID else 0.0
    _r_for_W3 = data.get("r", {}) if USE_W3_ADAPTIVE_HYBRID else {}
    _x_for_W3 = data.get("x", {}) if USE_W3_ADAPTIVE_HYBRID else {}
    for (i, j) in E:
        if USE_W3_ADAPTIVE_HYBRID and _zsq_med_T4 > 0.0:
            r_e = float(_r_for_W3.get((i, j), 0.0))
            x_e = float(_x_for_W3.get((i, j), 0.0))
            zsq_e = r_e * r_e + x_e * x_e
            ratio = zsq_e / max(_zsq_med_T4 * 0.1, 1e-12)
            w_V_eff = float(ELL_HYBRID_V_WEIGHT) * (ratio / (1.0 + ratio))
            w_I_eff = 1.0 - w_V_eff
            tval = w_I_eff * float(ell_i[(i, j)]) + w_V_eff * float(ell_v[(i, j)])
        else:
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
# 3번 변화 helpers (2026-04-28 convergence fixes)
# ============================================================

def _is_anchor_clean(max_kcl_prev: float, max_vdrop_prev: float) -> bool:
    """(A) Whether the previous accepted iter's slacks are below the gates
    that qualify it as a physical anchor for prox / multi-cut / theta-proj.
    A non-clean anchor distorts those terms toward the slacked operating
    point (which is what produced the iter-3 cliff at 2026-04-27 21:38)."""
    if not USE_ANCHOR_CLEAN_GATE:
        return True
    return (
        float(max_kcl_prev) <= float(ANCHOR_CLEAN_KCL_PU)
        and float(max_vdrop_prev) <= float(ANCHOR_CLEAN_VDROP_PU)
    )


def _push_loss_plane_guarded(
    history: List[Dict[str, Dict[Tuple[int, int], float]]],
    new_plane: Dict[str, Dict[Tuple[int, int], float]],
    anchor_history: List[Dict[str, Dict[Any, float]]],
    max_history: int,
) -> Tuple[List[Dict[str, Dict[Tuple[int, int], float]]], int]:
    """(B) Push new_plane onto history only if it is satisfied (within
    PLANE_GUARD_TOL) by every stored anchor's (P,Q,v) reference -- i.e.
    the plane does not force l < 0 at a past iterate. anchor_history
    parallels history: anchor_history[k] = {"Pij":..., "Qij":..., "v":...}
    captured at the same iter that produced history[k+1].

    Returns (updated_history, n_skipped) where n_skipped is the number of
    edges whose plane coefficient was zeroed because they violated the
    guard at one or more anchors. We zero per-edge instead of dropping
    the whole plane, since usually only a few pathological edges fail.
    """
    if (not USE_PLANE_FEAS_GUARD) or (not anchor_history):
        history = (history + [new_plane])[-int(max_history):]
        return history, 0

    aP = dict(new_plane.get("aP", {}))
    aQ = dict(new_plane.get("aQ", {}))
    aV = dict(new_plane.get("aV", {}))
    b0 = dict(new_plane.get("b0", {}))

    n_skipped = 0
    for edge in list(b0.keys()):
        for past in anchor_history:
            P_past = past.get("Pij", {}).get(edge, 0.0)
            Q_past = past.get("Qij", {}).get(edge, 0.0)
            i, _j = edge
            v_past = past.get("v", {}).get(int(i), 1.0)
            l_pred = (
                float(b0.get(edge, 0.0))
                + float(aP.get(edge, 0.0)) * float(P_past)
                + float(aQ.get(edge, 0.0)) * float(Q_past)
                + float(aV.get(edge, 0.0)) * float(v_past)
            )
            if l_pred < -float(PLANE_GUARD_TOL):
                b0[edge] = 0.0
                aP[edge] = 0.0
                aQ[edge] = 0.0
                aV[edge] = 0.0
                n_skipped += 1
                break
    cleaned = {"aP": aP, "aQ": aQ, "aV": aV, "b0": b0}
    history = (history + [cleaned])[-int(max_history):]
    return history, n_skipped


def _trust_region_clip_ell(
    ell_prev: Dict[Tuple[int, int], float],
    ell_target: Dict[Tuple[int, int], float],
    radius_rel: float,
    radius_abs: float,
    ellmax: Dict[Tuple[int, int], float],
) -> Dict[Tuple[int, int], float]:
    """(C) Per-edge trust-region clip on ell step.  Limits |ell - ell_prev|
    to max(radius_abs, radius_rel * |ell_prev|).  Re-clips into
    [0, ellmax]."""
    if not USE_ELL_TRUST_REGION:
        return dict(ell_target)
    out = {}
    for k in ell_prev:
        prev = float(ell_prev[k])
        tgt = float(ell_target.get(k, prev))
        max_step = max(float(radius_abs), float(radius_rel) * abs(prev))
        delta = tgt - prev
        if abs(delta) > max_step:
            delta = math.copysign(max_step, delta)
        val = prev + delta
        if ELL_CLIP_NONNEG:
            val = max(0.0, val)
        cap = float(ellmax.get(k, val))
        if ELL_CLIP_MAX and val > cap:
            val = cap
        out[k] = float(val)
    return out


def _adapt_trust_region(
    radius_rel: float, radius_abs: float, accepted: bool
) -> Tuple[float, float]:
    """(C) ACCEPT -> grow, REJECT -> shrink, clamped to TR_RADIUS_*_MIN/MAX."""
    if not USE_ELL_TRUST_REGION:
        return float(radius_rel), float(radius_abs)
    if accepted:
        radius_rel = min(float(TR_RADIUS_REL_MAX), float(radius_rel) * float(TR_GROW))
        radius_abs = min(float(TR_RADIUS_ABS_MAX), float(radius_abs) * float(TR_GROW))
    else:
        radius_rel = max(float(TR_RADIUS_REL_MIN), float(radius_rel) * float(TR_SHRINK))
        radius_abs = max(float(TR_RADIUS_ABS_MIN), float(radius_abs) * float(TR_SHRINK))
    return float(radius_rel), float(radius_abs)


def _anderson_step(
    ell_history: List[Dict[Tuple[int, int], float]],
    target_history: List[Dict[Tuple[int, int], float]],
    ell_emaprime: Dict[Tuple[int, int], float],
) -> Optional[Dict[Tuple[int, int], float]]:
    """(G) Anderson Acceleration on ell-update.
    Given iterates {ell_k}, fixed-point map outputs {T_k = T(ell_k)},
    we solve min ||sum_k alpha_k * F_k||^2 s.t. sum alpha_k = 1
    where F_k = T_k - ell_k. Then ell_next = sum_k alpha_k * T_k.

    Returns the Anderson step, or None if history too short or LS ill-
    conditioned. Caller must apply safeguard against EMA-step residual.
    """
    if not USE_ANDERSON_ELL:
        return None
    K = min(len(ell_history), len(target_history), int(AA_HISTORY))
    if K < int(AA_MIN_HISTORY) + 1:
        return None
    ell_hist = ell_history[-K:]
    tgt_hist = target_history[-K:]
    keys = list(ell_hist[0].keys())
    n = len(keys)
    F = np.zeros((n, K), dtype=float)
    Tm = np.zeros((n, K), dtype=float)
    for k in range(K):
        for r, key in enumerate(keys):
            ev = float(ell_hist[k].get(key, 0.0))
            tv = float(tgt_hist[k].get(key, ev))
            F[r, k] = tv - ev
            Tm[r, k] = tv
    # Solve (F^T F + reg I) alpha = (F^T F + reg I) [..constraint sum=1..]
    # via Lagrange: minimize alpha^T G alpha s.t. 1^T alpha = 1
    G = F.T @ F + float(AA_REG) * np.eye(K)
    try:
        ones = np.ones(K)
        Ginv1 = np.linalg.solve(G, ones)
        denom = float(ones @ Ginv1)
        if not np.isfinite(denom) or abs(denom) < 1.0e-12:
            return None
        alpha = Ginv1 / denom
    except np.linalg.LinAlgError:
        return None
    if not np.all(np.isfinite(alpha)):
        return None
    ell_aa = Tm @ alpha
    out = {keys[r]: float(ell_aa[r]) for r in range(n)}
    return out


def _anderson_residual_norm(
    ell_proposed: Dict[Tuple[int, int], float],
    ell_target: Dict[Tuple[int, int], float],
) -> float:
    s = 0.0
    for k in ell_proposed:
        d = float(ell_target.get(k, ell_proposed[k])) - float(ell_proposed[k])
        s += d * d
    return math.sqrt(s)


def _build_kcl_penalty_per_bus(
    data: Dict[str, Any],
) -> Dict[int, float]:
    """(H) Stratified KCL penalty: gen buses get RHO_KCL * RHO_KCL_GEN_FACTOR,
    pure load buses get RHO_KCL * RHO_KCL_LOAD_FACTOR."""
    buses = data["buses"]
    gen_records = data.get("gen_records", [])
    gen_buses = set(int(g.get("bus", -1)) for g in gen_records)
    if not USE_STRATIFIED_KCL_PENALTY:
        return {int(b): float(RHO_KCL) for b in buses}
    return {
        int(b): float(RHO_KCL) * (
            float(RHO_KCL_GEN_FACTOR) if int(b) in gen_buses else float(RHO_KCL_LOAD_FACTOR)
        )
        for b in buses
    }


def _identify_top_violators(
    model: pyo.ConcreteModel,
    data: Dict[str, Any],
    topk_buses: int,
    topk_edges: int,
) -> Tuple[List[int], List[Tuple[int, int]]]:
    """(I) Read the relaxed solution's slack distribution and return the
    top-K offending buses (KCL) and edges (vdrop)."""
    buses = data["buses"]
    E = data["E"]
    bus_slack: List[Tuple[int, float]] = []
    for i in buses:
        s = (
            _val(model.sP_pos[i], 0.0) + _val(model.sP_neg[i], 0.0)
            + _val(model.sQ_pos[i], 0.0) + _val(model.sQ_neg[i], 0.0)
        )
        bus_slack.append((int(i), float(s)))
    bus_slack.sort(key=lambda kv: kv[1], reverse=True)
    top_buses = [kv[0] for kv in bus_slack[:int(topk_buses)] if kv[1] > 1.0e-9]

    edge_slack: List[Tuple[Tuple[int, int], float]] = []
    for (i, j) in E:
        s = _val(model.s_vdrop[i, j], 0.0)
        edge_slack.append(((int(i), int(j)), float(s)))
    edge_slack.sort(key=lambda kv: kv[1], reverse=True)
    top_edges = [kv[0] for kv in edge_slack[:int(topk_edges)] if kv[1] > 1.0e-9]
    return top_buses, top_edges


def _slack_aware_anchor(
    P_raw: Dict[Tuple[int, int], float],
    Q_raw: Dict[Tuple[int, int], float],
    v_raw: Dict[int, float],
    sP_pos: Dict[int, float],
    sP_neg: Dict[int, float],
    sQ_pos: Dict[int, float],
    sQ_neg: Dict[int, float],
    in_arcs: Dict[int, List[Tuple[int, int]]],
    out_arcs: Dict[int, List[Tuple[int, int]]],
    fraction: float,
) -> Tuple[Dict[Tuple[int, int], float], Dict[Tuple[int, int], float], Dict[int, float]]:
    """(J) Heuristically distribute absorbed slack back across the bus's
    incident edges so the prox anchor reflects a more physical state.
    For bus i: residual_P = sP_pos[i] - sP_neg[i] (the artificial active
    injection that closed KCL).  We attribute -fraction * residual_P
    evenly across the bus's incident edges (out-arcs sender side).  Same
    for Q.
    Voltages are not adjusted -- they are exact within vdrop slack at
    most one pu, which is small enough to be ignored.
    """
    P_anchor = dict(P_raw)
    Q_anchor = dict(Q_raw)
    if not USE_SLACK_AWARE_ANCHOR or float(fraction) <= 0.0:
        return P_anchor, Q_anchor, dict(v_raw)
    for i, arcs_out in out_arcs.items():
        n_inc = len(arcs_out) + len(in_arcs.get(i, []))
        if n_inc == 0:
            continue
        residual_P = float(sP_pos.get(int(i), 0.0)) - float(sP_neg.get(int(i), 0.0))
        residual_Q = float(sQ_pos.get(int(i), 0.0)) - float(sQ_neg.get(int(i), 0.0))
        if abs(residual_P) < 1.0e-9 and abs(residual_Q) < 1.0e-9:
            continue
        delta_P = float(fraction) * residual_P / float(n_inc)
        delta_Q = float(fraction) * residual_Q / float(n_inc)
        for (a, b) in arcs_out:
            P_anchor[(a, b)] = float(P_anchor.get((a, b), 0.0)) + delta_P
            Q_anchor[(a, b)] = float(Q_anchor.get((a, b), 0.0)) + delta_Q
        for (a, b) in in_arcs.get(int(i), []):
            P_anchor[(a, b)] = float(P_anchor.get((a, b), 0.0)) - delta_P
            Q_anchor[(a, b)] = float(Q_anchor.get((a, b), 0.0)) - delta_Q
    return P_anchor, Q_anchor, dict(v_raw)


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

    oltc_branches: Dict[Tuple[int, int], OLTCBranchConfig] = {}
    shunt_bcap_pu: Dict[int, float] = {}
    recommended_taps: Dict[Tuple[int, int], int] = {}
    recommended_shunts: Dict[int, int] = {}

    oltc_df = net["fixed_oltc_table"]
    shunt_df = net["fixed_shunt_table"]

    for _, row in oltc_df.iterrows():
        i = int(row["from_bus_pp"])
        j = int(row["to_bus_pp"])
        oltc_branches[(i, j)] = OLTCBranchConfig(
            tap_min=int(row["tap_min"]),
            tap_max=int(row["tap_max"]),
            dV_percent=float(row["dV_percent"]),
        )
        if "recommended_tap" in row:
            try:
                rv = float(row["recommended_tap"])
                if not math.isnan(rv):
                    recommended_taps[(i, j)] = int(rv)
            except Exception:
                pass

    for _, row in shunt_df.iterrows():
        b = int(row["bus_pp"])
        shunt_bcap_pu[b] = float(row["bcap_pu"])

    if "recommended_nonexact_oltc_taps" in net:
        rec_tap_df = net["recommended_nonexact_oltc_taps"]
        if rec_tap_df is not None and len(rec_tap_df.index) > 0:
            for _, row in rec_tap_df.iterrows():
                recommended_taps[(int(row["from_bus_pp"]), int(row["to_bus_pp"]))] = int(row["tap"])

    if "recommended_nonexact_shunt_status" in net:
        rec_sh_df = net["recommended_nonexact_shunt_status"]
        if rec_sh_df is not None and len(rec_sh_df.index) > 0:
            for _, row in rec_sh_df.iterrows():
                recommended_shunts[int(row["bus_pp"])] = int(row["status"])

    return BuildConfig(
        oltc_branches=oltc_branches,
        shunt_bcap_pu=shunt_bcap_pu,
        recommended_taps=recommended_taps,
        recommended_shunts=recommended_shunts,
        fix_slack_vm=True,
    )


def _load_full_snapshot_from_npz(
    snapshot_path: Path,
    E: List[Tuple[int, int]],
    ellmax: Dict[Tuple[int, int], float],
    sn_mva: float,
    buses: List[int],
) -> Optional[Dict[str, Any]]:
    """Load a Python warm-start snapshot produced by make_warmstart_snapshot.py.

    Returns the full warm state -- ell_pf AND per-edge (P, Q) AND per-bus
    (v, theta) -- so the outer loop can seed prox_prev["Pij"]/["Qij"]/["v"]
    and warm["Pij"]/["Qij"]/["v"] with physically meaningful values instead
    of zero/1.0. (The seed is what makes safe WARMUP_ITERS=0 operation
    possible in principle; see the WARMUP_ITERS comment block.)

    Returns None silently if the file is missing, so the caller can chain
    fallbacks. Logs a single line on success.

    The snapshot file is a numpy .npz with arrays:
        bus_idx, theta_deg, vm_pu                       (per bus)
        edge_from, edge_to, P_pu, Q_pu                  (per BFM edge)
        sn_mva                                          (scalar)
    Edges absent from the snapshot get (P, Q) = (0, 0) and ell_pf = 0.
    """
    if not snapshot_path.is_file():
        return None
    try:
        snap = np.load(str(snapshot_path), allow_pickle=False)
    except Exception as exc:  # corrupted / wrong format
        print(f"[INFO] Python snapshot at {snapshot_path.name} unreadable: {exc}")
        return None

    required = ("bus_idx", "theta_deg", "vm_pu", "edge_from", "edge_to", "P_pu", "Q_pu")
    if not all(k in snap.files for k in required):
        missing = [k for k in required if k not in snap.files]
        print(f"[INFO] Python snapshot at {snapshot_path.name} missing fields: {missing}.")
        return None

    bus_idx = np.asarray(snap["bus_idx"], dtype=np.int64)
    theta_deg = np.asarray(snap["theta_deg"], dtype=np.float64)
    vm_pu = np.asarray(snap["vm_pu"], dtype=np.float64)
    edge_from = np.asarray(snap["edge_from"], dtype=np.int64)
    edge_to = np.asarray(snap["edge_to"], dtype=np.int64)
    P_pu = np.asarray(snap["P_pu"], dtype=np.float64)
    Q_pu = np.asarray(snap["Q_pu"], dtype=np.float64)

    if not (bus_idx.shape == theta_deg.shape == vm_pu.shape):
        print(f"[INFO] Python snapshot bus arrays have inconsistent shapes; skipping.")
        return None
    if not (edge_from.shape == edge_to.shape == P_pu.shape == Q_pu.shape):
        print(f"[INFO] Python snapshot edge arrays have inconsistent shapes; skipping.")
        return None

    # Build per-bus dicts (default vm=1.0, theta=0.0 for any unmapped bus).
    vm_by_bus: Dict[int, float] = {int(b): 1.0 for b in buses}
    theta_by_bus: Dict[int, float] = {int(b): 0.0 for b in buses}
    for k in range(len(bus_idx)):
        b = int(bus_idx[k])
        if b in vm_by_bus:
            v_val = float(vm_pu[k])
            if math.isfinite(v_val) and v_val > 0.0:
                vm_by_bus[b] = v_val
            t_val = float(theta_deg[k])
            if math.isfinite(t_val):
                theta_by_bus[b] = t_val

    # Build per-edge dicts. The snapshot's (edge_from, edge_to) is the
    # snapshot's directed convention; if it differs from BFM's (i, j) we
    # flip the sign on the matched key. Unmatched BFM edges get 0.
    Pij_by_edge: Dict[Tuple[int, int], float] = {key: 0.0 for key in E}
    Qij_by_edge: Dict[Tuple[int, int], float] = {key: 0.0 for key in E}
    snap_lookup: Dict[Tuple[int, int], Tuple[float, float]] = {}
    for k in range(len(edge_from)):
        a = int(edge_from[k])
        b = int(edge_to[k])
        snap_lookup[(a, b)] = (float(P_pu[k]), float(Q_pu[k]))

    matched = 0
    for (i, j) in E:
        if (i, j) in snap_lookup:
            p, q = snap_lookup[(i, j)]
            Pij_by_edge[(i, j)] = p
            Qij_by_edge[(i, j)] = q
            matched += 1
        elif (j, i) in snap_lookup:
            p, q = snap_lookup[(j, i)]
            Pij_by_edge[(i, j)] = -p
            Qij_by_edge[(i, j)] = -q
            matched += 1

    if matched == 0:
        print(
            f"[INFO] Python snapshot {snapshot_path.name} has 0/{len(E)} BFM edges matched; "
            f"skipping (likely a different network)."
        )
        return None

    # Compute ell_pf = (P^2 + Q^2) / vm_fr^2.
    #
    # Three caps stack:
    #   - per-edge ellmax (= min(raw_ellmax, max(ELL_FIX_HARD_CAP_PU2,
    #     per_edge_cap))). Lets high-smax tie lines carry their physical ell.
    #   - scalar ELL_FIX_HARD_CAP_PU2 (default 5.0). The 20260427_22 run
    #     proved that even one edge with ell~80 is enough to make iter 1's
    #     KCL+vdrop system infeasible at presolve (the snapshot's high-flow
    #     tie lines like edge 97-98 with smax=15.93 pu carry P=8.94 pu in
    #     DCOPF, giving raw ell_pf = 80; with that magnitude on the
    #     receiving bus's KCL_Q, the bounded slack window cannot close
    #     the bus balance even at relax/x5).
    #   - the heuristic warm-start applies the same scalar cap (see
    #     _compute_ell_warm_start). To match its proven-feasible behavior
    #     at iter 1-2, we apply the scalar cap here too.
    # The scalar cap is conservative -- snapshot edges that carry physical
    # ell above 5 pu^2 get clipped, losing some spatial fidelity. But this
    # is strictly better than the uniform heuristic because the SHAPE
    # (which edges carry flow) remains correct, and the cap only attenuates
    # the highest-flow tail. Iter 2+'s ell_state is no longer pinned by
    # ell_fix, so the outer loop can recover the high-flow tail from
    # iter 2 onward via the EMA / force-follow update.
    ell_pf: Dict[Tuple[int, int], float] = {}
    for (i, j) in E:
        p = Pij_by_edge[(i, j)]
        q = Qij_by_edge[(i, j)]
        v = vm_by_bus.get(int(i), 1.0)
        denom = max(v * v, 1.0e-6)
        raw = (p * p + q * q) / denom
        edge_cap = float(ellmax.get((i, j), raw))
        ell_pf[(i, j)] = float(max(0.0, min(raw, edge_cap, float(ELL_FIX_HARD_CAP_PU2))))

    source = "unknown"
    if "source" in snap.files:
        try:
            source = str(snap["source"])
        except Exception:
            pass
    print(
        f"[INFO] Python warm-start snapshot loaded from {snapshot_path.name} "
        f"(source={source}); matched {matched}/{len(E)} edges, "
        f"{len(bus_idx)} buses."
    )

    return {
        "ell_pf": ell_pf,
        "Pij": Pij_by_edge,
        "Qij": Qij_by_edge,
        "vm": vm_by_bus,
        "theta_deg": theta_by_bus,
        "source": source,
        "matched_edges": matched,
    }


# Sanity gates on the ell_pf produced by pandapower runpp.
# An edge running at 200% emergency thermal loading sits at ell ~ 4 pu^2;
# values above ~100 pu^2 indicate a numerically degenerate PF solution
# (e.g. case3012wp's flat-start NR sometimes converges to a state with
# 50+ GVAr circulating on a single 220 kV line, which is a Jacobian
# artefact, not a physical operating point). When the PF result trips
# either gate we reject the runpp-derived warm-start entirely and fall
# back to the heuristic so iter 1 doesn't get fed garbage ell_fix.
#   MAX_SANE_ELL_PF_PU2: per-edge soft gate (count edges over this).
#   MAX_SANE_ELL_PF_FRAC_OVER: trip if more than this fraction of
#       edges exceed the soft gate.
#   MAX_SANE_ELL_PF_HARD_PU2: per-edge hard gate. ANY single edge above
#       this triggers immediate rejection (catches 50+ GVAr circulating
#       currents that only show up on a few edges and would not exceed
#       MAX_SANE_ELL_PF_FRAC_OVER on their own).
MAX_SANE_ELL_PF_PU2 = 100.0
MAX_SANE_ELL_PF_FRAC_OVER = 0.001
MAX_SANE_ELL_PF_HARD_PU2 = 1.0e3


def _compute_ell_pf_from_runpp(
    net,
    E: List[Tuple[int, int]],
    ellmax: Dict[Tuple[int, int], float],
    sn_mva: float,
    branch_elements: Dict[Tuple[int, int], List[Tuple[str, int]]],
    branch_original_dirs: Dict[Tuple[int, int], List[Tuple[int, int]]],
) -> Optional[Dict[Tuple[int, int], float]]:
    """Spatial PF-based ell_pf warm-start from pandapower runpp results.

    Only usable after _harmonize_voltage_setpoints has been applied (which
    fixes the historical "Voltage controlling elements at the same bus
    have different setpoints" failure on case3012wp). Returns None if PF
    has not been run on `net` (no res_line / res_trafo / res_bus tables),
    or if the result trips the MAX_SANE_ELL_PF_PU2 gate (pandapower's
    flat-start NR on case3012wp can converge to a degenerate state with
    50+ GVAr circulating on a single line; that's a numerical artefact,
    not a usable warm-start).

    For each BFM directed edge (i, j):
      1. Walk the branch_elements list (may contain multiple parallel
         lines / trafos that BFM merged into one equivalent edge).
      2. Sum the per-element complex power flows, flipping sign as needed
         when the pp-element's native orientation differs from BFM's
         (i, j) orientation.
      3. Read the BFM-from-bus voltage magnitude from net.res_bus.
      4. ell_pf = (P_sum_pu^2 + Q_sum_pu^2) / vm_fr^2, clipped to ellmax.
    """
    if not (hasattr(net, "res_line") and hasattr(net, "res_bus")):
        return None
    if len(net.res_bus.index) == 0:
        return None

    ell_pf: Dict[Tuple[int, int], float] = {}
    for key in E:
        i, j = key
        elements = branch_elements.get(key, [])
        original_dirs = branch_original_dirs.get(key, [])
        if not elements or len(elements) != len(original_dirs):
            return None
        P_sum_mw = 0.0
        Q_sum_mw = 0.0
        contributors = 0
        for (et, eidx), (orig_fb, orig_tb) in zip(elements, original_dirs):
            try:
                if et == "line":
                    if eidx not in net.res_line.index:
                        continue
                    Pf = float(net.res_line.at[eidx, "p_from_mw"])
                    Qf = float(net.res_line.at[eidx, "q_from_mvar"])
                elif et == "trafo":
                    if eidx not in net.res_trafo.index:
                        continue
                    Pf = float(net.res_trafo.at[eidx, "p_hv_mw"])
                    Qf = float(net.res_trafo.at[eidx, "q_hv_mvar"])
                else:
                    continue
            except Exception:
                continue
            if not (math.isfinite(Pf) and math.isfinite(Qf)):
                continue
            if (orig_fb, orig_tb) == (i, j):
                P_sum_mw += Pf
                Q_sum_mw += Qf
            else:
                P_sum_mw -= Pf
                Q_sum_mw -= Qf
            contributors += 1
        if contributors == 0:
            return None
        try:
            vm_fr = float(net.res_bus.at[i, "vm_pu"])
        except Exception:
            return None
        if not math.isfinite(vm_fr) or vm_fr <= 0.0:
            return None
        denom = max(vm_fr * vm_fr, 1.0e-6)
        P_pu = P_sum_mw / float(sn_mva)
        Q_pu = Q_sum_mw / float(sn_mva)
        raw = (P_pu * P_pu + Q_pu * Q_pu) / denom
        cap = float(ellmax.get(key, raw))
        ell_pf[key] = float(max(0.0, min(raw, cap)))

    # Sanity gate: reject the entire PF if too many edges land in the
    # numerical-artefact regime. ell_pf has already been clipped to ellmax,
    # so this check operates on the un-clipped raw values via a separate
    # pass to catch the degenerate runpp state on case3012wp.
    over_gate = 0
    raw_max = 0.0
    for key in E:
        i, _j = key
        elements = branch_elements.get(key, [])
        original_dirs = branch_original_dirs.get(key, [])
        P_sum_mw = 0.0
        Q_sum_mw = 0.0
        for (et, eidx), (orig_fb, orig_tb) in zip(elements, original_dirs):
            try:
                if et == "line":
                    if eidx not in net.res_line.index:
                        continue
                    Pf = float(net.res_line.at[eidx, "p_from_mw"])
                    Qf = float(net.res_line.at[eidx, "q_from_mvar"])
                elif et == "trafo":
                    if eidx not in net.res_trafo.index:
                        continue
                    Pf = float(net.res_trafo.at[eidx, "p_hv_mw"])
                    Qf = float(net.res_trafo.at[eidx, "q_hv_mvar"])
                else:
                    continue
            except Exception:
                continue
            if not (math.isfinite(Pf) and math.isfinite(Qf)):
                continue
            sgn = 1.0 if (orig_fb, orig_tb) == key else -1.0
            P_sum_mw += sgn * Pf
            Q_sum_mw += sgn * Qf
        try:
            vm_fr = float(net.res_bus.at[i, "vm_pu"])
        except Exception:
            vm_fr = 1.0
        denom = max(vm_fr * vm_fr, 1.0e-6)
        P_pu = P_sum_mw / float(sn_mva)
        Q_pu = Q_sum_mw / float(sn_mva)
        raw = (P_pu * P_pu + Q_pu * Q_pu) / denom
        if raw > MAX_SANE_ELL_PF_PU2:
            over_gate += 1
        if raw > raw_max:
            raw_max = raw
    nE = max(1, len(E))
    soft_trip = over_gate / nE > MAX_SANE_ELL_PF_FRAC_OVER
    hard_trip = raw_max > MAX_SANE_ELL_PF_HARD_PU2
    if soft_trip or hard_trip:
        reason = []
        if hard_trip:
            reason.append(f"max raw ell_pf = {raw_max:.1f} pu^2 > hard gate {MAX_SANE_ELL_PF_HARD_PU2:.0f}")
        if soft_trip:
            reason.append(
                f"{over_gate}/{nE} ({100.0 * over_gate / nE:.2f}%) edges exceed soft gate "
                f"{MAX_SANE_ELL_PF_PU2:.0f} pu^2 (limit {100.0 * MAX_SANE_ELL_PF_FRAC_OVER:.2f}%)"
            )
        print(
            f"[INFO] ell_pf warm-start: pandapower runpp converged but result is a "
            f"numerical artefact, not a usable warm-start; rejecting "
            f"({'; '.join(reason)})."
        )
        return None
    return ell_pf


def _compute_ell_warm_start(
    Smax: Dict[Tuple[int, int], float],
    ellmax: Dict[Tuple[int, int], float],
    alpha: float = 0.3,
    vmean: float = 1.0,
) -> Dict[Tuple[int, int], float]:
    """Heuristic per-edge ell warm-start when PF data is unavailable.

    Used as the final fallback when neither the Python snapshot nor the
    pandapower runpp-based warm-start can produce a usable ell_pf.
    Assumes every branch carries roughly `alpha` fraction of its thermal
    rating at |V| ~ vmean, which gives
        ell_warm[key] = (alpha * Smax[key])^2 / vmean^2
    This is conservative (a real system is typically less loaded on
    average) but sufficient to seed iter 1's KCL with realistic loss
    magnitudes instead of the ell_fix=0 cold start.

    The seed is clipped both by ellmax and by the scalar ELL_FIX_HARD_CAP_PU2.
    The scalar cap is load-bearing because ellmax was changed (2026-04-26) to
    a per-edge value with floor ELL_FIX_HARD_CAP_PU2, otherwise
    (smax * ELL_HARD_CAP_HEADROOM / vmin)^2 -- which can reach ~10000 pu^2 on
    high-smax tie lines, where raw (alpha*smax)^2 already exceeds 900 pu^2.
    Without the scalar cap, iter 1's linearized x*ell injects GVAr of phantom
    reactive loss that saturates the bounded KCL slack (REJECT cascade).
    """
    a2 = float(alpha) * float(alpha)
    v2 = max(float(vmean) * float(vmean), 1.0e-6)
    out: Dict[Tuple[int, int], float] = {}
    for key, smax_pu in Smax.items():
        raw = a2 * float(smax_pu) * float(smax_pu) / v2
        out[key] = float(min(raw, float(ellmax[key]), float(ELL_FIX_HARD_CAP_PU2)))
    return out


def _merge_parallel_series_equivalent(rows: List[dict]) -> Tuple[float, float, float]:
    """
    Merge parallel branches with the same directed endpoints into one equivalent branch.
    Uses admittance summation for z_eq and sums thermal ratings.
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
        # Fallback: keep arithmetic mean impedance if data is degenerate.
        r = float(sum(float(rw["r_pu"]) for rw in rows) / len(rows))
        x = float(sum(float(rw["x_pu"]) for rw in rows) / len(rows))
        return r, x, ssum

    zeq = 1.0 / ysum
    return float(zeq.real), float(zeq.imag), float(ssum)


def _compute_loop_basis_T4(buses, edges) -> List[List[Tuple[int, int, int]]]:
    """Fundamental cycle basis (T4 helper).  Returns a list of cycles, each
    a list of (u, v, sign) tuples where (u, v) is an edge in `edges` and
    sign is +1 if traversed forward (matches the stored edge orientation)
    or -1 if backward.  Empty list if networkx is unavailable.
    """
    try:
        import networkx as nx
    except Exception:
        return []
    G = nx.Graph()
    G.add_nodes_from(list(buses))
    for (i, j) in edges:
        G.add_edge(int(i), int(j))
    edge_set = set((int(i), int(j)) for (i, j) in edges)
    cycles: List[List[Tuple[int, int, int]]] = []
    for cycle_nodes in nx.cycle_basis(G):
        if len(cycle_nodes) < 3:
            continue
        cycle_edges: List[Tuple[int, int, int]] = []
        valid = True
        for k in range(len(cycle_nodes)):
            u = int(cycle_nodes[k])
            v = int(cycle_nodes[(k + 1) % len(cycle_nodes)])
            if (u, v) in edge_set:
                cycle_edges.append((u, v, +1))
            elif (v, u) in edge_set:
                cycle_edges.append((v, u, -1))
            else:
                valid = False
                break
        if valid and cycle_edges:
            cycles.append(cycle_edges)
    return cycles


def _extract_top_kcl_buses(
    result_file_path: Path,
    top_k: int = KCL_HARD_TOPK,
    min_slack_pu: float = KCL_HARD_MIN_SLACK_PU,
) -> List[int]:
    """H1 helper.  Parse a prior result file's "KCL slack (top N)" section
    and return the buses with absorbed |slackP| above `min_slack_pu`, sorted
    descending and trimmed to `top_k`.  Returns [] if the file is missing
    or no eligible bus is found (so the feature degrades to a no-op on the
    first run).
    """
    if not result_file_path.is_file():
        return []
    text = result_file_path.read_text(encoding="utf-8", errors="replace")
    pat = re.compile(
        r"^\s*bus\s+(\d+):\s+slackP=([+\-]?[\d.eE+\-]+),\s+slackQ=",
        re.MULTILINE,
    )
    in_section = False
    bus_slacks: List[Tuple[int, float]] = []
    for line in text.splitlines():
        if "KCL slack (top" in line:
            in_section = True
            continue
        if in_section:
            if line.strip().startswith("---"):
                break
            m = pat.match(line)
            if m:
                bus_slacks.append((int(m.group(1)), abs(float(m.group(2)))))
    bus_slacks.sort(key=lambda t: t[1], reverse=True)
    return [b for (b, s) in bus_slacks if s > float(min_slack_pu)][: int(top_k)]


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

    if "branch_params_pu_table" not in net or net["branch_params_pu_table"] is None or net["branch_params_pu_table"].empty:
        raise KeyError("Network metadata 'branch_params_pu_table' not found or empty.")

    brdf = net["branch_params_pu_table"].copy()
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
        raw_ellmax = float((smax_pu ** 2) / vmin_sq)
        # Per-edge ell cap (방안 2, 2026-04-26): the historical scalar
        # ELL_FIX_HARD_CAP_PU2=5.0 was too tight for high-smax transformer
        # edges (smax~5 pu => natural operating ell ~ 27 pu^2), starving the
        # KCL closure on iter 3+ and producing the REJECT cascade. The new
        # per-edge cap allows each edge enough headroom for its own rating
        # while still bounding low-smax lines at the original 5.0 floor.
        floor = float(ELL_FIX_HARD_CAP_PU2)
        head = float(ELL_HARD_CAP_HEADROOM)
        per_edge_cap = ((smax_pu * head) ** 2) / vmin_sq if smax_pu > 0.0 else 0.0
        if floor > 0.0:
            cap = max(floor, per_edge_cap)
        else:
            cap = per_edge_cap if per_edge_cap > 0.0 else raw_ellmax
        ellmax[key] = float(min(raw_ellmax, cap))
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

    # Warm-start ell_fix. Priority (best -> fallback):
    #   1. Python snapshot (.npz) from make_warmstart_snapshot.py — carries
    #      full (P, Q, v, theta) seeded from a converged DCOPF (or SOCP).
    #   2. pandapower runpp — uses _harmonize_voltage_setpoints from
    #      ieee3012bus.py to work around case3012wp's conflicting Vg setpoints.
    #   3. Heuristic ell = (alpha*smax)^2 / vmean^2 — distributes phantom
    #      losses uniformly across all 3566 edges; saturates per-bus KCL
    #      slack budget on stiff buses and triggers the iter-2 REJECT
    #      cascade. Strictly a last-resort fallback.
    ell_pf: Optional[Dict[Tuple[int, int], float]] = None
    ell_pf_source = "heuristic"
    warm_state_snapshot: Optional[Dict[str, Any]] = None
    if USE_PYTHON_WARMSTART_SNAPSHOT:
        snap_full = _load_full_snapshot_from_npz(
            PYTHON_SNAPSHOT_PATH, E, ellmax, sn, buses
        )
        if snap_full is not None:
            ell_pf = snap_full["ell_pf"]
            ell_pf_source = f"python_snapshot[{snap_full.get('source', '?')}]"
            warm_state_snapshot = snap_full
        else:
            print(
                f"[INFO] ell_pf warm-start: Python snapshot unavailable at "
                f"{PYTHON_SNAPSHOT_PATH.name}. Trying pandapower runpp next."
            )

    if ell_pf is None:
        try:
            from ieee3012bus import _runpp_calibration as _runpp_helper
            pf_ok = _runpp_helper(net) if not (hasattr(net, "res_line") and len(net.res_line.index) > 0) else True
        except Exception:
            pf_ok = False
        if pf_ok:
            ell_pf_runpp = _compute_ell_pf_from_runpp(
                net=net,
                E=E,
                ellmax=ellmax,
                sn_mva=sn,
                branch_elements=branch_elements,
                branch_original_dirs=branch_original_dirs,
            )
            if ell_pf_runpp is not None:
                ell_pf = ell_pf_runpp
                ell_pf_source = "pandapower_runpp"
                print(
                    f"[INFO] ell_pf warm-start: derived from pandapower runpp; "
                    f"covers all {len(E)} BFM edges."
                )

    if ell_pf is None:
        print(
            "[INFO] ell_pf warm-start: falling back to heuristic "
            "(alpha*smax)^2 / vmean^2 with alpha=0.3, vmean=1.0."
        )
        ell_pf = _compute_ell_warm_start(Smax=Smax, ellmax=ellmax, alpha=0.3, vmean=1.0)

    data = dict(
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
        ell_pf=ell_pf,
        ell_pf_source=ell_pf_source,
        warm_state_snapshot=warm_state_snapshot,
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
    )

    # ------------------------------------------------------------------
    # BFM-ivar mitigation data (T4 cycle basis / W3 zsq median / H1 hard buses)
    # ------------------------------------------------------------------
    # T4: cycle basis for the loop-residual penalty
    data["loop_basis_T4"] = _compute_loop_basis_T4(buses, E)
    n_edges = len(E)
    n_buses = len(buses)
    print(f"  [mit] T4 cycle basis: {len(data['loop_basis_T4'])} cycles  "
          f"(theoretical |E|-|N|+1 = {n_edges - n_buses + 1})")

    # W3: median r^2 + x^2 across the network (scale for adaptive hybrid weight)
    _zsq_list = sorted(
        float(r.get((i, j), 0.0)) ** 2 + float(x.get((i, j), 0.0)) ** 2
        for (i, j) in E
    )
    data["_zsq_median_T4"] = float(_zsq_list[len(_zsq_list) // 2]) if _zsq_list else 1.0
    print(f"  [mit] W3 zsq median: {data['_zsq_median_T4']:.4e}")

    # H1: top-K KCL hard-equality buses pre-extracted by __main__ (before
    # teeing truncated the result file).  Module-level `_H1_KCL_HARD_BUSES`
    # is empty on first run / when called outside __main__.
    data["kcl_hard_buses"] = list(_H1_KCL_HARD_BUSES)
    if data["kcl_hard_buses"]:
        print(f"  [mit] H1 hard-equality KCL buses: {data['kcl_hard_buses']}")
    else:
        print("  [mit] H1 hard-equality KCL buses: NONE (no prior result file)")

    return data


# ============================================================
# 1번 변화: loss proxy coefficient builder
# ============================================================
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
    rho_p: float = RHO_P,
    rho_q: float = RHO_Q,
    rho_v: float = RHO_V,
    rho_theta: float = RHO_THETA,
    # [A6] Multi-cut loss: list of stored Taylor planes keyed by edge.
    #      Each entry is {"aP":{...}, "aQ":{...}, "aV":{...}, "b0":{...}}.
    #      When non-empty AND USE_MULTI_CUT_LOSS, build_subproblem adds an
    #      m.l_edge variable lower-bounded by every plane and the objective
    #      replaces the single-plane (cP*P + cQ*Q + cV*v) term with
    #      w_loss_current * sn * r * l_edge.  When None / empty, the
    #      legacy single-plane proxy is used (backwards-compatible).
    loss_planes: Optional[List[Dict[str, Dict[Tuple[int, int], float]]]] = None,
    # [A6] current w_loss for the l_edge contribution to the objective.
    #      Only used when multi-cut is active.
    w_loss_current: float = 0.0,
    # (D) Augmented Lagrangian dual multipliers.  When non-None, an
    #     additional linear term sum_b lambda_kcl[b] * sP/sQ slacks +
    #     sum_e lambda_vdrop[e] * s_vdrop[e] is added to the objective.
    lambda_kcl: Optional[Dict[int, float]] = None,
    lambda_vdrop: Optional[Dict[Tuple[int, int], float]] = None,
    # (E) Phase-I/Phase-II: gen_cost is multiplied by cost_scale.  In
    #     Phase-I (warmup) we set cost_scale=PHASE_I_COST_SCALE so the
    #     objective is dominated by slack penalties (find feasibility
    #     first), and the slack penalty multipliers are scaled up via
    #     rho_slack_boost.
    cost_scale: float = 1.0,
    rho_slack_boost: float = 1.0,
    # (H) Per-bus KCL penalty.  When non-None, replaces the global RHO_KCL
    #     for slack penalty in the objective (load buses get RHO_KCL_LOAD,
    #     gen buses get RHO_KCL_GEN).  Falls back to global RHO_KCL when
    #     None.
    rho_kcl_per_bus: Optional[Dict[int, float]] = None,
    # (I) Per-bus / per-edge slack max.  When non-None, OVERRIDES the
    #     scalar kcl_slack_max / vdrop_slack_max for those specific buses /
    #     edges (other buses/edges get the scalar bound).  Used by the
    #     IIS-guided rescue ladder so only top-K offenders are widened.
    kcl_slack_max_per_bus: Optional[Dict[int, float]] = None,
    vdrop_slack_max_per_edge: Optional[Dict[Tuple[int, int], float]] = None,
    # (F) Binary fixing.  When non-None, OLTC tap binaries are fixed to
    #     fix_oltc[(i,j,tap)] in {0,1} and shunt binaries to fix_shunt[i].
    fix_oltc: Optional[Dict[Tuple[int, int, int], int]] = None,
    fix_shunt: Optional[Dict[int, int]] = None,
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

    # Slack variables for bounded vdrop / KCL residuals.
    # With USE_EXACT_EQUALITY_SLACK=True the slacks are hardbound to zero and
    # both equations are enforced as strict equalities (original 300-bus mode).
    # With USE_EXACT_EQUALITY_SLACK=False the slacks may take values in
    # [0, vdrop_slack_max] / [0, kcl_slack_max] and are penalized in the
    # objective; this lets iter 2+ absorb transient ell-mismatch instead of
    # returning infeasible, which is what causes the REJECT loop on larger
    # networks (e.g. IEEE 3012-bus).
    # (I) IIS-guided per-bus / per-edge slack windows.  Lookup with
    # fallback to the scalar bound for buses/edges not explicitly listed.
    def _vdrop_ub_for(e: Tuple[int, int]) -> float:
        if vdrop_slack_max_per_edge is not None and e in vdrop_slack_max_per_edge:
            return float(max(vdrop_slack_max_per_edge[e], 0.0))
        return float(max(vdrop_slack_max, 0.0))

    def _kcl_ub_for(b: int) -> float:
        if kcl_slack_max_per_bus is not None and int(b) in kcl_slack_max_per_bus:
            return float(max(kcl_slack_max_per_bus[int(b)], 0.0))
        return float(max(kcl_slack_max, 0.0))

    if USE_EXACT_EQUALITY_SLACK:
        m.s_vdrop = pyo.Var(m.E, bounds=(0.0, 0.0))
        m.sP_pos  = pyo.Var(m.N, bounds=(0.0, 0.0))
        m.sP_neg  = pyo.Var(m.N, bounds=(0.0, 0.0))
        m.sQ_pos  = pyo.Var(m.N, bounds=(0.0, 0.0))
        m.sQ_neg  = pyo.Var(m.N, bounds=(0.0, 0.0))
    else:
        m.s_vdrop = pyo.Var(m.E, bounds=lambda mm, i, j: (0.0, _vdrop_ub_for((int(i), int(j)))))
        m.sP_pos  = pyo.Var(m.N, bounds=lambda mm, i: (0.0, _kcl_ub_for(int(i))))
        m.sP_neg  = pyo.Var(m.N, bounds=lambda mm, i: (0.0, _kcl_ub_for(int(i))))
        m.sQ_pos  = pyo.Var(m.N, bounds=lambda mm, i: (0.0, _kcl_ub_for(int(i))))
        m.sQ_neg  = pyo.Var(m.N, bounds=lambda mm, i: (0.0, _kcl_ub_for(int(i))))

    m.con_bounded_slack = pyo.ConstraintList()
    for (i, j) in E:
        m.con_bounded_slack.add(m.s_vdrop[i, j] <= _vdrop_ub_for((int(i), int(j))))
    for i in buses:
        m.con_bounded_slack.add(
            m.sP_pos[i] + m.sP_neg[i] + m.sQ_pos[i] + m.sQ_neg[i] <= _kcl_ub_for(int(i))
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

    def _thermal(mm, i, j):
        i, j = int(i), int(j)
        return mm.Pij[i, j]**2 + mm.Qij[i, j]**2 <= (mm.Smax[i, j]**2)
    m.Thermal = pyo.Constraint(m.E, rule=_thermal)

    def _l_lin_rule(mm, i, j):
        return (
            mm.b0_edge[i, j]
            + mm.aP_edge[i, j] * mm.Pij[i, j]
            + mm.aQ_edge[i, j] * mm.Qij[i, j]
            + mm.aV_edge[i, j] * mm.vsend[i, j]
        )
    m.l_lin = pyo.Expression(m.E, rule=_l_lin_rule)

    # The l_lin_nonneg gate is only meaningful with the legacy single-plane
    # proxy.  With multi-cut active, l_edge has its own NonNegativeReals
    # domain (when MULTI_CUT_LOSS_L_NONNEG=True) and serves the same role.
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

    # [A6] Multi-cut loss under-approximation.  Each stored Taylor plane
    # (aP_k, aQ_k, aV_k, b0_k) is a tangent under-estimate of the convex
    # function ell(P,Q,v) = (P^2+Q^2)/v at a previous reference point.
    # We introduce l_edge >= max_k(plane_k); the objective then uses
    # w_loss * sn * r * l_edge in place of the single-plane term, so the
    # solver picks (P,Q,v) that minimize an envelope of K planes -- which
    # tightens iter over iter as more planes accumulate -- instead of a
    # single linearization that is locally tight at one previous iterate
    # but blows up the moment the inner MIQCP wants to move.  This is
    # what allows iter-2 on 3012-bus to escape iter-1's KCL-saturated
    # operating point without going infeasible under the ell_fix update.
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
                        float(b0_p.get((i, j), 0.0))
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

    # (D), (H): cache resolved penalty weights as constants captured in
    # _obj_rule so the scope is explicit and we avoid re-allocations per
    # iteration of pyo.quicksum.
    _rho_vdrop_eff = float(RHO_VDROP) * float(rho_slack_boost)
    if rho_kcl_per_bus is not None:
        _rho_kcl_eff = {int(b): float(rho_kcl_per_bus[int(b)]) * float(rho_slack_boost) for b in buses}
    else:
        _rho_kcl_eff = {int(b): float(RHO_KCL) * float(rho_slack_boost) for b in buses}
    _lambda_kcl = (
        {int(b): float(lambda_kcl.get(int(b), 0.0)) for b in buses}
        if lambda_kcl is not None else {int(b): 0.0 for b in buses}
    )
    _lambda_vdrop = (
        {(int(i), int(j)): float(lambda_vdrop.get((int(i), int(j)), 0.0)) for (i, j) in E}
        if lambda_vdrop is not None else {(int(i), int(j)): 0.0 for (i, j) in E}
    )

    def _obj_rule(mm):
        gen_cost = float(cost_scale) * pyo.quicksum(
            mm.c2[g] * (sn * mm.Pg[g])**2 + mm.c1[g] * (sn * mm.Pg[g]) + mm.c0[g]
            for g in mm.G
        )

        loss_term = 0.0
        if USE_LOSS_PROXY:
            if getattr(mm, "_has_multi_cut_loss", False):
                # [A6] w_loss * sn * r * l_edge, where l_edge is lower-bounded
                # by every stored Taylor plane (envelope under-approximation).
                wsn = float(mm._w_loss_current) * sn
                loss_term = pyo.quicksum(
                    wsn * float(mm.r[i, j]) * mm.l_edge[i, j]
                    for (i, j) in mm.E
                )
            else:
                # Legacy single-plane linearization (kept for the warmup-iter
                # branch where loss_planes is None).
                loss_term = pyo.quicksum(
                    mm.cP_edge[i, j] * mm.Pij[i, j]
                    + mm.cQ_edge[i, j] * mm.Qij[i, j]
                    + mm.cV_edge[i, j] * mm.vsend[i, j]
                    for (i, j) in mm.E
                )

        prox = 0.0
        if USE_PROXIMAL:
            prox = (
                float(rho_p) * pyo.quicksum((mm.Pij[i, j] - mm.Pprev[i, j])**2 for (i, j) in mm.E)
                + float(rho_q) * pyo.quicksum((mm.Qij[i, j] - mm.Qprev[i, j])**2 for (i, j) in mm.E)
                + float(rho_v) * pyo.quicksum((mm.v[i] - mm.vprev[i])**2 for i in mm.N)
            )

        theta_proj = 0.0
        if USE_THETA_PROJ_PROX:
            theta_proj = float(rho_theta) * pyo.quicksum(
                (mm.x[i, j] * (mm.Pij[i, j] - mm.Pprev[i, j]) - mm.r[i, j] * (mm.Qij[i, j] - mm.Qprev[i, j]))**2
                for (i, j) in mm.E
            )

        pen_vdrop_term = 0.0
        pen_kcl_term = 0.0
        al_kcl_term = 0.0
        al_vdrop_term = 0.0
        if not USE_EXACT_EQUALITY_SLACK:
            if _rho_vdrop_eff > 0.0:
                pen_vdrop_term = _rho_vdrop_eff * pyo.quicksum(
                    mm.s_vdrop[i, j] for (i, j) in mm.E
                )
            # (H) per-bus stratified KCL penalty.
            pen_kcl_term = pyo.quicksum(
                _rho_kcl_eff[int(i)] * (
                    mm.sP_pos[i] + mm.sP_neg[i] + mm.sQ_pos[i] + mm.sQ_neg[i]
                )
                for i in mm.N
            )
            # (D) Augmented Lagrangian linear-in-slack term.  At iter 0
            # all lambdas are zero so this is a no-op; after each ACCEPT
            # the duals climb proportional to slack used.
            al_kcl_term = pyo.quicksum(
                _lambda_kcl[int(i)] * (
                    mm.sP_pos[i] + mm.sP_neg[i] + mm.sQ_pos[i] + mm.sQ_neg[i]
                )
                for i in mm.N
            )
            al_vdrop_term = pyo.quicksum(
                _lambda_vdrop[(int(i), int(j))] * mm.s_vdrop[i, j]
                for (i, j) in mm.E
            )

        # T4: BFM-ivar loop residual penalty for meshed KVL consistency.
        # See module-level USE_T4_LOOP_PENALTY for the formulation. The
        # penalty is rho_loop_T4 * sum_C (cycle angle-residual)^2 with the
        # angle approximation x_uv P_uv - r_uv Q_uv) / sqrt(vbar_u vbar_v).
        loop_term = 0.0
        if USE_T4_LOOP_PENALTY:
            _loop_basis = data.get("loop_basis_T4", None)
            if _loop_basis is not None and prox_prev is not None and "v" in prox_prev:
                _vbar = prox_prev["v"]
                _r_T4 = data.get("r", {})
                _x_T4 = data.get("x", {})
                _cycle_terms = []
                for _cycle in _loop_basis:
                    _cycle_sum = 0.0
                    for (_u, _v, _sgn) in _cycle:
                        _vu = max(1e-9, float(_vbar.get(int(_u), 1.0)))
                        _vv = max(1e-9, float(_vbar.get(int(_v), 1.0)))
                        _sqrtuv = math.sqrt(_vu * _vv)
                        _r_uv = float(_r_T4.get((int(_u), int(_v)), 0.0))
                        _x_uv = float(_x_T4.get((int(_u), int(_v)), 0.0))
                        _cycle_sum = _cycle_sum + float(_sgn) * (
                            _x_uv * mm.Pij[_u, _v] - _r_uv * mm.Qij[_u, _v]
                        ) / _sqrtuv
                    _cycle_terms.append(_cycle_sum * _cycle_sum)
                if _cycle_terms:
                    loop_term = float(RHO_LOOP_T4) * pyo.quicksum(_cycle_terms)

        return (
            gen_cost + loss_term + prox + theta_proj
            + pen_vdrop_term + pen_kcl_term
            + al_kcl_term + al_vdrop_term
            + loop_term
        )

    m.obj = pyo.Objective(rule=_obj_rule, sense=pyo.minimize)

    # Store the effective proximal weights on the model so that
    # objective_breakdown (called later with only `model, data`) can
    # recompute each term with the same rho values used in the solve.
    m._rho_p = float(rho_p)
    m._rho_q = float(rho_q)
    m._rho_v = float(rho_v)
    m._rho_theta = float(rho_theta)

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

    # (F) Binary fixing.  After F_BIN_FIX_ITER, run_bfm_ag passes the
    # incumbent's OLTC tap and shunt step values; we fix the corresponding
    # binaries here so SCIP only solves a continuous QP from this iter on.
    if (not relax_binaries) and fix_oltc is not None and F_BIN_FIX_OLTC and USE_BINARY_FIXING:
        for (i, j, tap) in m.BETA_INDEX:
            if (i, j, tap) in fix_oltc:
                m.beta[i, j, tap].fix(int(fix_oltc[(i, j, tap)]))
    if (not relax_binaries) and fix_shunt is not None and F_BIN_FIX_SHUNT and USE_BINARY_FIXING:
        for i in C:
            if int(i) in fix_shunt:
                m.a_sh[i].fix(int(fix_shunt[int(i)]))

    # H1 (BFMivar v1): hard-equality KCL on top-K offending buses.  Only
    # active for binary attempts; relaxed/rescue solves keep the soft
    # slack so an over-strict hard set can still recover via the existing
    # rescue ladder.  The top-K bus list is provided by BFM_ivar_mit.py
    # via data["kcl_hard_buses"] (list of bus indices in pp's 0-indexed
    # numbering).
    if (not relax_binaries) and USE_KCL_HARD_BUSES:
        _hard_buses = data.get("kcl_hard_buses", []) or []
        for _b in _hard_buses:
            _bi = int(_b)
            if _bi in buses:
                m.sP_pos[_bi].fix(0.0)
                m.sP_neg[_bi].fix(0.0)
                m.sQ_pos[_bi].fix(0.0)
                m.sQ_neg[_bi].fix(0.0)

    m._data = data
    return m


# ============================================================
# [B10] Round-dependent MIP gap schedule
# ============================================================
def _gap_for_iter(t: int) -> float:
    """Return the MIP gap to use for outer iteration t.

    Subsumes the legacy WARMUP_ITERS-vs-normal binary split.  The ramp
    shape (loose -> mid -> tight) lets iters 1..2 land an incumbent
    quickly while still reaching solver-tight optimality on the tail.
    """
    if not USE_ROUND_DEPENDENT_GAP:
        # Fallback to the legacy split (matches what run_bfm_ag did before).
        return float(SCIP_GAP_LIMIT_ITER1) if int(t) <= int(WARMUP_ITERS) else float(SCIP_GAP_LIMIT)
    if int(t) <= int(GAP_SWITCH_EARLY):
        return float(GAP_EARLY)
    if int(t) <= int(GAP_SWITCH_MID):
        return float(GAP_MID)
    return float(GAP_LATE)


# ============================================================
# [B7] Gurobi drop-in with SCIP fallback + [B8] MIP warm-start
# ============================================================
def _try_gurobi(
    model: pyo.ConcreteModel, timelimit: float, mipgap: float, tee: bool
) -> Tuple[bool, str]:
    """Best-effort Gurobi solve.  Returns (ok, reason).

    ok=False with reason="unavailable" means Gurobi could not be loaded
    (no license, not installed, or wrapper raised on import) -- the
    caller should fall through to SCIP.
    """
    if not USE_GUROBI_FIRST:
        return False, "disabled"
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
        return False, "unavailable"

    try:
        opt.options["TimeLimit"] = float(timelimit)
        opt.options["MIPGap"] = float(mipgap)
        opt.options["OutputFlag"] = 1 if tee else 0
        # The MIQCP we build is convex (SOC thermal + linear KCL/vdrop +
        # convex quadratic objective), so NonConvex=2 is safe and
        # aggressive -- Gurobi will accept the SOC and quadratic
        # constraints without complaining about non-convexity.
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
    except Exception as exc1:
        # Some Gurobi/Pyomo combos reject warmstart=True; retry without it.
        try:
            res = opt.solve(model, tee=tee, load_solutions=True)
        except Exception as exc2:
            return False, f"exception:{type(exc2).__name__}"

    tc = res.solver.termination_condition
    tc_str = str(tc)
    if tc == TerminationCondition.infeasible:
        return False, "infeasible"
    if not _solution_complete(model, model._data):
        return False, f"{tc_str}(no_incumbent)"
    return True, f"gurobi:{tc_str}"


def _make_fresh_scip(prefer_nl: bool):
    """Return a freshly constructed SCIP solver instance (and a flag for
    the NL interface).  Always recreated so that a previous solve's
    writer state -- .nl buffers, cached variable ids, etc. -- cannot
    leak into the next call.
    """
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


# ============================================================
# Solve with SCIP (Gurobi first, then SCIP NL, then SCIP default)
# ============================================================
def solve_with_scip(
    model: pyo.ConcreteModel, timelimit: float, mipgap: float, tee: bool
) -> Tuple[bool, str]:
    """Solve `model` with Gurobi (if available) then SCIP.  Returns (ok, reason).

    ok=True  -> a solver returned a usable incumbent (values loaded).
    ok=False -> infeasible, OR no incumbent within time/node budget
                (the two cases are distinguished by `reason`):
                  "infeasible"              -> mathematically infeasible
                  "<tc>(no_incumbent)"      -> tc says optimal/feasible/
                                               maxTimeLimit/... but load
                                               failed -- typically the
                                               solver timed out before
                                               finding any integer-
                                               feasible point.
                  "solver_unavailable"      -> neither Gurobi nor SCIP
                                               on PATH.
                  "exception:<...>"         -> exception during solve.
    Callers log `reason` so the outer REJECT message can distinguish
    genuine infeasibility from time-limit exhaustion.
    """
    # [B7] Try Gurobi first.  Returns (False, "unavailable"|"disabled")
    # if Gurobi isn't usable, in which case we fall through to SCIP.
    ok_g, reason_g = _try_gurobi(model, min(float(timelimit), float(GUROBI_TIME_LIMIT)), float(mipgap), tee)
    if ok_g:
        return True, reason_g
    if reason_g == "infeasible":
        # Gurobi proved infeasibility -- no need to retry on SCIP.
        return False, "infeasible"

    # SCIP path.  Try NL first (richer solver options + solver_io support),
    # then default Pyomo SCIP wrapper.  Each attempt uses a fresh
    # SolverFactory so a previous solve's state cannot leak.
    last_reason = "solver_unavailable"
    for prefer_nl in (True, False):
        opt, _is_nl = _make_fresh_scip(prefer_nl)
        if opt is None:
            continue
        _apply_scip_options(opt, timelimit, mipgap, tee)
        try:
            res = opt.solve(model, tee=tee, load_solutions=True)
        except Exception as exc:
            last_reason = f"exception:{type(exc).__name__}"
            continue

        tc = res.solver.termination_condition
        tc_str = str(tc)
        if tc == TerminationCondition.infeasible:
            return False, "infeasible"
        if _solution_complete(model, model._data):
            return True, tc_str
        last_reason = f"{tc_str}(no_incumbent)"

    return False, last_reason


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
    rho_p: float = RHO_P,
    rho_q: float = RHO_Q,
    rho_v: float = RHO_V,
    rho_theta: float = RHO_THETA,
    scip_time_limit: float = SCIP_TIME_LIMIT,
    scip_gap_limit: float = SCIP_GAP_LIMIT,
    # [A6] Multi-cut loss: forwarded to every build_subproblem call below
    # so the rescue ladder (binary -> relaxed -> fixed-from-relax) all use
    # the same envelope.
    loss_planes: Optional[List[Dict[str, Dict[Tuple[int, int], float]]]] = None,
    w_loss_current: float = 0.0,
    # 3번 변화 forwarded knobs
    lambda_kcl: Optional[Dict[int, float]] = None,
    lambda_vdrop: Optional[Dict[Tuple[int, int], float]] = None,
    cost_scale: float = 1.0,
    rho_slack_boost: float = 1.0,
    rho_kcl_per_bus: Optional[Dict[int, float]] = None,
    fix_oltc: Optional[Dict[Tuple[int, int, int], int]] = None,
    fix_shunt: Optional[Dict[int, int]] = None,
) -> Tuple[pyo.ConcreteModel, str]:

    last_relaxed_model: Optional[pyo.ConcreteModel] = None
    attempt_reasons: List[str] = []

    # (I) IIS-guided per-bus / per-edge widening.  Filled in after the
    # first relax/x1 attempt fails -- captures the relaxed solution's top
    # offenders so the next x2/x5 attempts widen ONLY those, instead of
    # blanket widening every slack.
    iis_kcl_overrides: Optional[Dict[int, float]] = None
    iis_vdrop_overrides: Optional[Dict[Tuple[int, int], float]] = None

    def _build_kwargs(vb: float, kb: float, relax: bool, warm_ovr=None,
                      kcl_per_bus_override: Optional[Dict[int, float]] = None,
                      vdrop_per_edge_override: Optional[Dict[Tuple[int, int], float]] = None):
        return dict(
            relax_binaries=relax,
            warm=(warm if warm_ovr is None else warm_ovr),
            loss_proxy=loss_proxy,
            prox_prev=prox_prev,
            vdrop_slack_max=vb,
            kcl_slack_max=kb,
            p_step_frac=p_step_frac,
            q_step_frac=q_step_frac,
            v_step_abs=v_step_abs,
            rho_p=rho_p, rho_q=rho_q, rho_v=rho_v, rho_theta=rho_theta,
            loss_planes=loss_planes,
            w_loss_current=w_loss_current,
            lambda_kcl=lambda_kcl,
            lambda_vdrop=lambda_vdrop,
            cost_scale=cost_scale,
            rho_slack_boost=rho_slack_boost,
            rho_kcl_per_bus=rho_kcl_per_bus,
            kcl_slack_max_per_bus=kcl_per_bus_override,
            vdrop_slack_max_per_edge=vdrop_per_edge_override,
            fix_oltc=fix_oltc,
            fix_shunt=fix_shunt,
        )

    for mult in SLACK_RESCUE_MULTS:
        vb, kb = _rescued_slack_bounds(vdrop_slack_max, kcl_slack_max, float(mult))
        suffix = "" if float(mult) == 1.0 else f"_rescued_x{mult:g}"
        mult_tag = f"x{mult:g}"

        # (I) After x1 has produced a relaxed-but-infeasible signal, switch
        # subsequent x2/x5 attempts to per-bus/per-edge IIS-guided widening
        # so we widen only top offenders instead of every slack.
        kcl_override = iis_kcl_overrides if (USE_IIS_GUIDED_WIDENING and float(mult) > 1.0) else None
        vdrop_override = iis_vdrop_overrides if (USE_IIS_GUIDED_WIDENING and float(mult) > 1.0) else None

        if tee and float(mult) != 1.0:
            print(
                f"[WARN] retrying bounded-slack subproblem with relaxed bounds: "
                f"vdrop<={vb:.3e}, kcl<={kb:.3e}"
                + (f" (IIS-guided: {len(kcl_override or {})} buses, {len(vdrop_override or {})} edges)"
                   if (kcl_override or vdrop_override) else "")
            )

        m_bin = build_subproblem(
            data, ell_fix,
            **_build_kwargs(vb, kb, relax=False,
                            kcl_per_bus_override=kcl_override,
                            vdrop_per_edge_override=vdrop_override),
        )
        ok, reason = solve_with_scip(m_bin, scip_time_limit, scip_gap_limit, tee)
        if ok:
            return m_bin, "binary" + suffix
        attempt_reasons.append(f"bin/{mult_tag}:{reason}")

        if tee:
            print(f"[WARN] binary {reason} -> RELAXED ...")

        m_relax = build_subproblem(
            data, ell_fix,
            **_build_kwargs(vb, kb, relax=True,
                            kcl_per_bus_override=kcl_override,
                            vdrop_per_edge_override=vdrop_override),
        )
        ok, reason = solve_with_scip(m_relax, scip_time_limit, scip_gap_limit, False)
        if ok:
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
                **_build_kwargs(vb, kb, relax=False, warm_ovr=warm_fix,
                                kcl_per_bus_override=kcl_override,
                                vdrop_per_edge_override=vdrop_override),
            )

            for (i, j) in data["T"]:
                pick = int(tap_choice[(i, j)])
                for tap in data["K"][(i, j)]:
                    m_fix.beta[i, j, int(tap)].fix(1.0 if int(tap) == pick else 0.0)

            for i in data["C"]:
                m_fix.a_sh[i].fix(float(sh_choice[int(i)]))

            ok_fix, reason_fix = solve_with_scip(m_fix, scip_time_limit, scip_gap_limit, tee)
            if ok_fix:
                return m_fix, "fixed_from_relax" + suffix
            attempt_reasons.append(f"fix/{mult_tag}:{reason_fix}")

            if tee:
                print(f"[WARN] Fixed-discrete MIQCP {reason_fix}; keeping RELAXED solution as fallback.")
            return m_relax, "relaxed_only" + suffix
        attempt_reasons.append(f"relax/{mult_tag}:{reason}")

        # (I) The x1 relaxed solve failed too.  If we have a stored
        # last_relaxed_model from a prior outer iter, harvest top offenders
        # from it; otherwise rebuild with the relaxed binary set OPEN at
        # an even-larger window just to identify violators.
        if USE_IIS_GUIDED_WIDENING and float(mult) == 1.0 and last_relaxed_model is None:
            # Try a wide-open relax solve solely to identify top offenders.
            wide_v, wide_k = _rescued_slack_bounds(vdrop_slack_max, kcl_slack_max, float(SLACK_RESCUE_MULTS[-1]))
            try:
                m_probe = build_subproblem(
                    data, ell_fix,
                    **_build_kwargs(wide_v, wide_k, relax=True),
                )
                ok_p, _ = solve_with_scip(m_probe, scip_time_limit, scip_gap_limit, False)
                if ok_p:
                    top_buses, top_edges = _identify_top_violators(
                        m_probe, data, IIS_WIDEN_TOPK_BUSES, IIS_WIDEN_TOPK_EDGES
                    )
                    iis_kcl_overrides = {int(b): float(kcl_slack_max) * float(IIS_WIDEN_FACTOR) for b in top_buses}
                    iis_vdrop_overrides = {e: float(vdrop_slack_max) * float(IIS_WIDEN_FACTOR) for e in top_edges}
                    if tee:
                        print(
                            f"[INFO] IIS probe identified {len(top_buses)} offending buses, "
                            f"{len(top_edges)} offending edges; widening only those on next attempt."
                        )
            except Exception:
                pass
        elif USE_IIS_GUIDED_WIDENING and last_relaxed_model is not None and iis_kcl_overrides is None:
            top_buses, top_edges = _identify_top_violators(
                last_relaxed_model, data, IIS_WIDEN_TOPK_BUSES, IIS_WIDEN_TOPK_EDGES
            )
            iis_kcl_overrides = {int(b): float(kcl_slack_max) * float(IIS_WIDEN_FACTOR) for b in top_buses}
            iis_vdrop_overrides = {e: float(vdrop_slack_max) * float(IIS_WIDEN_FACTOR) for e in top_edges}

    if last_relaxed_model is not None:
        return last_relaxed_model, "relaxed_only_rescued"

    raise RuntimeError(
        "RELAXED also infeasible/no-solution under bounded vdrop/KCL slack constraints; "
        "scheduled and rescue bounds were all exhausted. "
        f"attempts=[{', '.join(attempt_reasons)}]"
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
    if not USE_EXACT_EQUALITY_SLACK:
        if float(RHO_VDROP) > 0.0:
            pen_vdrop = float(RHO_VDROP) * sum(
                _val(model.s_vdrop[i, j], 0.0) for (i, j) in data["E"]
            )
        if float(RHO_KCL) > 0.0:
            pen_kcl = float(RHO_KCL) * sum(
                _val(model.sP_pos[i], 0.0) + _val(model.sP_neg[i], 0.0)
                + _val(model.sQ_pos[i], 0.0) + _val(model.sQ_neg[i], 0.0)
                for i in data["buses"]
            )

    loss_term = 0.0
    if USE_LOSS_PROXY:
        if getattr(model, "_has_multi_cut_loss", False):
            # [A6] mirror what build_subproblem put in the objective:
            # w_loss * sn * r * l_edge, summed over edges.
            wsn = float(getattr(model, "_w_loss_current", 0.0)) * float(data["sn_mva"])
            for (i, j) in data["E"]:
                loss_term += wsn * float(data["r"][(i, j)]) * _val(model.l_edge[i, j], 0.0)
        else:
            for (i, j) in data["E"]:
                Pij = _val(model.Pij[i, j], 0.0)
                Qij = _val(model.Qij[i, j], 0.0)
                vsend = _pval(model.vsend[i, j], 1.0)
                loss_term += (
                    _pval(model.cP_edge[i, j]) * Pij
                    + _pval(model.cQ_edge[i, j]) * Qij
                    + _pval(model.cV_edge[i, j]) * vsend
                )

    # Use the rho values that were actually embedded in the subproblem (stored
    # on the model when build_subproblem created it). Falling back to the
    # initial/legacy constants preserves behavior for models built externally.
    eff_rho_p = float(getattr(model, "_rho_p", RHO_P))
    eff_rho_q = float(getattr(model, "_rho_q", RHO_Q))
    eff_rho_v = float(getattr(model, "_rho_v", RHO_V))
    eff_rho_theta = float(getattr(model, "_rho_theta", RHO_THETA))

    prox_term = 0.0
    if USE_PROXIMAL:
        for (i, j) in data["E"]:
            Pij = _val(model.Pij[i, j], 0.0)
            Qij = _val(model.Qij[i, j], 0.0)
            dP = Pij - _pval(model.Pprev[i, j])
            dQ = Qij - _pval(model.Qprev[i, j])
            prox_term += eff_rho_p * (dP ** 2) + eff_rho_q * (dQ ** 2)
        for i in data["buses"]:
            dv = _val(model.v[i], 1.0) - _pval(model.vprev[i], 1.0)
            prox_term += eff_rho_v * (dv ** 2)

    theta_proj = 0.0
    if USE_THETA_PROJ_PROX:
        for (i, j) in data["E"]:
            Pij = _val(model.Pij[i, j], 0.0)
            Qij = _val(model.Qij[i, j], 0.0)
            dP = Pij - _pval(model.Pprev[i, j])
            dQ = Qij - _pval(model.Qprev[i, j])
            theta_proj += eff_rho_theta * (
                float(data["x"][(i, j)]) * dP - float(data["r"][(i, j)]) * dQ
            ) ** 2

    total = gen_cost + loss_term + prox_term + theta_proj + pen_vdrop + pen_kcl

    # Effective (fair) physical cost: actual generation cost plus the penalty
    # the solver paid to violate the BFM/KCL equations. An iterate that
    # reduces gen_cost only by pushing residual into the slack variables
    # will therefore NOT look cheap under eff_cost.
    eff_cost = gen_cost + pen_vdrop + pen_kcl

    return dict(
        gen_cost=gen_cost,
        pen_vdrop=pen_vdrop,
        pen_kcl=pen_kcl,
        loss_term=loss_term,
        prox_term=prox_term,
        theta_proj=theta_proj,
        total=total,
        eff_cost=eff_cost,
    )


# ============================================================
# Outer loop: BFM-ag + 1번 변화 + 2번 변화
# ============================================================

def run_bfm_ag(data: Dict[str, Any], max_iters: int, eps: float, tee: bool) -> Dict[str, Any]:
    buses = data["buses"]
    E = data["E"]
    slack = int(data["slack_bus"])
    T_set = set(data["T"])
    ellmax = data["ellmax"]
    K = data["K"]
    delta_tap = data["delta_tap"]

    # Warm-start ell_fix. data["ell_pf"] is set by extract_data's cascade
    # (Python snapshot -> pandapower runpp -> heuristic). Without a warm
    # start, iter 1 sees ell_fix = 0 and is forced to absorb line losses
    # in the KCL slack variables, which saturates the per-bus budget and
    # produces pathological ell_raw that breaks iter 2.
    ell_pf = dict(data.get("ell_pf") or {})
    if ell_pf and all((key in ell_pf) for key in E):
        ell_fix = {(i, j): float(ell_pf[(i, j)]) for (i, j) in E}
        ell_state = {(i, j): float(ell_pf[(i, j)]) for (i, j) in E}
        _r = data["r"]
        _x = data["x"]
        _sn = float(data["sn_mva"])
        _ploss_mw   = _sn * sum(float(_r[(i, j)]) * float(ell_fix[(i, j)]) for (i, j) in E)
        _qloss_mvar = _sn * sum(float(_x[(i, j)]) * float(ell_fix[(i, j)]) for (i, j) in E)
        _ell_max = max(ell_fix.values()) if ell_fix else 0.0
        _ell_mean = (sum(ell_fix.values()) / len(ell_fix)) if ell_fix else 0.0
        _src = str(data.get("ell_pf_source") or "heuristic")
        _src_label = {
            "heuristic": "heuristic alpha=0.3 of thermal",
            "pandapower_runpp": "pandapower runpp (ell_pf = (P^2+Q^2)/v^2)",
            "python_snapshot[dcopf]": "Python DCOPF snapshot (P from DCOPF, Q=0)",
            "python_snapshot[socp]": "Python SOCP snapshot (P, Q from SOCP relaxation)",
        }.get(_src, _src)
        print(
            f"[INFO] Iter-1 warm-start ({_src_label}): "
            f"mean ell_warm={_ell_mean:.3e}, max ell_warm={_ell_max:.3e} pu^2, "
            f"implied P-loss ~ {_ploss_mw:.1f} MW, Q-loss ~ {_qloss_mvar:.1f} MVAr."
        )
    else:
        ell_fix = {(i, j): 0.0 for (i, j) in E}
        ell_state = {(i, j): 0.0 for (i, j) in E}
        print("[WARN] heuristic ell warm-start unavailable; falling back to ell_fix=0 cold start.")

    # The Python warm-start snapshot's value at iter 1 is solely its
    # spatially-correct ell_pf (consumed via data["ell_pf"] in the
    # ell_fix block above). We deliberately DO NOT seed warm["Pij"] /
    # warm["Qij"] / warm["v"] / theta_prev from the snapshot here, because:
    #   - The snapshot's P_pu values come from DCOPF's optimal generator
    #     dispatch. BFMag iter 1 receives a *different* generator dispatch
    #     (the proportional-pmax warm-start at line ~2839), so feeding
    #     snapshot Pij as an MIP incumbent simultaneously with proportional
    #     Pg breaks KCL by O(GW) -- the bus injection from proportional
    #     dispatch does not match the line flows the snapshot's Pij implies.
    #   - The snapshot was generated assuming Q=0 (DCOPF lossless); Q
    #     dispatch on Polish 3012-bus is genuinely large, so the snapshot
    #     under-anchors the Q profile.
    #   - Warmup iters (iter 1..WARMUP_ITERS) bypass proximal/loss_proxy/
    #     multi-cut anyway, so warm["Pij"] does not anchor anything via
    #     the prox term -- it only seeds SCIP's MIP incumbent. A bad
    #     incumbent forces SCIP to repair via slack, hurting feasibility.
    # The 20260427_22 run set warm["Pij"]=snapshot AND
    # effective_warmup_iters=0 (proximal anchored on snapshot directly)
    # and reproduced the iter-1 presolve infeasibility cascade that the
    # WARMUP_ITERS comment block at the top of this file predicted.
    P_prev = {(i, j): 0.0 for (i, j) in E}
    Q_prev = {(i, j): 0.0 for (i, j) in E}
    v_prev = {int(b): 1.0 for b in buses}
    u_send_prev = {(i, j): float("nan") for (i, j) in E}

    theta_prev = {int(b): 0.0 for b in buses}
    theta_prev[slack] = 0.0

    snap_warm = data.get("warm_state_snapshot")
    if snap_warm is not None:
        print(
            f"[INFO] Python snapshot active: ell_fix at iter 1 is spatially "
            f"correct ({snap_warm.get('matched_edges', 0)}/{len(E)} edges); "
            f"warm['Pij']/['Qij']/['v']/theta_prev intentionally left at "
            f"0/0/1.0/0 to match BFMag's proportional-Pg warmup dispatch "
            f"(snapshot's P assumes DCOPF's dispatch, would break KCL if "
            f"paired with the proportional Pg seed)."
        )

    # WARMUP_ITERS is honored even with a snapshot. Iter 1-2 bypass
    # loss_proxy / multi-cut / proximal (these need a converged anchor,
    # which the snapshot's Q=0 profile is not). Iter 3 picks up
    # proximal/loss_proxy with iter 2's solution as anchor -- by then,
    # iter 2 has run on the snapshot-derived spatially-correct ell_fix,
    # so the iter-3 anchor is physical.
    effective_warmup_iters = int(WARMUP_ITERS)

    warm_beta = {}
    for (i, j) in data["T"]:
        taps = data["K"][(i, j)]
        rec_tap = int(data.get("recommended_taps", {}).get((i, j), _default_tap_choice([int(t) for t in taps])))
        if rec_tap not in taps:
            rec_tap = _default_tap_choice([int(t) for t in taps])
        for tap in taps:
            warm_beta[(i, j, int(tap))] = 1.0 if int(tap) == int(rec_tap) else 0.0

    warm_ash = {int(i): float(data.get("recommended_shunts", {}).get(int(i), 0)) for i in data["C"]}

    # Iter 1 warm-start: distribute total load among generators in proportion
    # to their pmax/qmax. Without this the MIP starts from Pg=0 and has to
    # allocate ~27 GW of load from scratch, which is what triggered the
    # iter 1 REJECT path on 3012-bus.
    pg_warm0 = _proportional_pg_warmstart(data)

    warm = {
        "v": dict(v_prev),
        "Pij": dict(P_prev),
        "Qij": dict(Q_prev),
        "Pg": dict(pg_warm0["Pg"]),
        "Qg": dict(pg_warm0["Qg"]),
        "beta": warm_beta,
        "a_sh": warm_ash,
        "u_send": dict(u_send_prev),
    }

    best = {
        "iter": 0,
        "gen_cost": float("inf"),
        "eff_cost": float("inf"),
        "total": float("inf"),
        "model": None,
        "ell": None,
        "theta": None,
        "tag": ""
    }
    best_feasible = {
        "iter": 0,
        "gen_cost": float("inf"),
        "eff_cost": float("inf"),
        "total": float("inf"),
        "model": None,
        "ell": None,
        "theta": None,
        "tag": ""
    }
    last = {"iter": 0, "model": None, "ell": None, "theta": None, "tag": ""}
    stopped = {"iter": 0, "gen_cost": float("inf"), "eff_cost": float("inf"), "total": float("inf"), "model": None, "ell": None, "theta": None, "tag": ""}

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

    # History of feasible-best gen_cost, used for cost-based early stopping.
    feasible_cost_hist: List[float] = []
    GEN_COST_PLATEAU_MIN_ITERS = 10
    GEN_COST_PLATEAU_WINDOW = 5
    GEN_COST_PLATEAU_REL_TOL = 1.0e-5

    # [A6] Multi-cut loss plane history.  Each entry is a snapshot of the
    # Taylor-plane coefficients (aP, aQ, aV, b0) computed at iter t's
    # reference point.  After warmup, build_subproblem uses the most
    # recent MULTI_CUT_LOSS_HISTORY planes as a piecewise-linear under-
    # approximation envelope of the convex loss (P^2+Q^2)/v.
    loss_planes_hist: List[Dict[str, Dict[Tuple[int, int], float]]] = []
    # (B) Anchor history paired with loss_planes_hist: anchor_history[k]
    # is the (P, Q, v) reference point at which loss_planes_hist[k] was
    # generated.  Used by _push_loss_plane_guarded to verify a new plane
    # does not force l_edge < 0 at a prior anchor.
    plane_anchor_hist: List[Dict[str, Dict[Any, float]]] = []

    # (C) Trust-region radius for ell_fix step.
    tr_radius_rel = float(TR_RADIUS_REL_INIT)
    tr_radius_abs = float(TR_RADIUS_ABS_INIT)

    # (D) Augmented Lagrangian dual variables.  Both start at zero so the
    # first iter sees only the quadratic-equivalent linear penalty; after
    # each ACCEPT they climb proportional to the slack used.
    lambda_kcl_dict: Dict[int, float] = {int(b): 0.0 for b in buses}
    lambda_vdrop_dict: Dict[Tuple[int, int], float] = {(int(i), int(j)): 0.0 for (i, j) in E}

    # (G) Anderson Acceleration history: paired (ell_k, T(ell_k)) where
    # T is the fixed-point map ell_t -> hybrid-corrected ell estimate.
    # Maintained by appending after every ACCEPT.
    aa_ell_hist: List[Dict[Tuple[int, int], float]] = []
    aa_target_hist: List[Dict[Tuple[int, int], float]] = []

    # (H) Stratified per-bus KCL penalty.  Computed once from the bus
    # type (load vs gen) and forwarded to every build_subproblem call.
    rho_kcl_per_bus = _build_kcl_penalty_per_bus(data)

    # (F) Binary fixing state.  After the incumbent stabilizes (iter >=
    # F_BIN_FIX_ITER) and we have an accepted iterate, take the OLTC tap
    # selection and shunt step values from the incumbent and fix them.
    fix_oltc_dict: Optional[Dict[Tuple[int, int, int], int]] = None
    fix_shunt_dict: Optional[Dict[int, int]] = None

    # Slacks of the last ACCEPTed iterate, used for (A) anchor-cleanliness
    # and (J) slack-aware anchor.
    prev_accepted_max_kcl = float("inf")
    prev_accepted_max_vdrop = float("inf")
    prev_accepted_sP_pos: Dict[int, float] = {int(b): 0.0 for b in buses}
    prev_accepted_sP_neg: Dict[int, float] = {int(b): 0.0 for b in buses}
    prev_accepted_sQ_pos: Dict[int, float] = {int(b): 0.0 for b in buses}
    prev_accepted_sQ_neg: Dict[int, float] = {int(b): 0.0 for b in buses}

    for t in range(1, max_iters + 1):
        t0 = time.perf_counter()

        # ------------------------------------------------------------------
        # Build a TRIAL ell_fix and loss-weight smoothing from the CURRENT
        # accepted iterate. These are only committed after the new candidate
        # step is accepted.
        # ------------------------------------------------------------------
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

        # (G) Anderson Acceleration on the ell fixed-point map: combine the
        # last AA_HISTORY (ell, target) pairs into a single accelerated step.
        # If the AA step's residual norm exceeds AA_SAFEGUARD_FACTOR * EMA's,
        # fall back to the EMA result.
        if USE_ANDERSON_ELL and t >= int(AA_MIN_HISTORY) + 1:
            aa_step = _anderson_step(aa_ell_hist, aa_target_hist, trial_ell_fix)
            if aa_step is not None:
                ema_resid = _anderson_residual_norm(trial_ell_fix, ell_raw)
                aa_resid = _anderson_residual_norm(aa_step, ell_raw)
                if aa_resid <= float(AA_SAFEGUARD_FACTOR) * max(ema_resid, 1.0e-12):
                    for (i, j) in E:
                        v = float(aa_step.get((int(i), int(j)), trial_ell_fix[(i, j)]))
                        if ELL_CLIP_NONNEG:
                            v = max(0.0, v)
                        if ELL_CLIP_MAX:
                            v = min(float(ellmax[(i, j)]), v)
                        trial_ell_fix[(i, j)] = v

        # (C) Trust-region clip: |trial_ell_fix - ell_fix| <= radius.
        # Wrapped after EMA+AA so the radius governs the final magnitude
        # of the step regardless of which mechanism produced it.
        if USE_ELL_TRUST_REGION and t >= 2:
            trial_ell_fix = _trust_region_clip_ell(
                ell_fix, trial_ell_fix, tr_radius_rel, tr_radius_abs, ellmax
            )

        # (J) Slack-aware proximal anchor: subtract a fraction of the
        # absorbed slack from (P_prev, Q_prev) so the anchor reflects a
        # more physical operating point than the slacked solution.
        if USE_SLACK_AWARE_ANCHOR and t >= 2:
            P_anchor, Q_anchor, v_anchor = _slack_aware_anchor(
                P_raw=warm["Pij"],
                Q_raw=warm["Qij"],
                v_raw=warm["v"],
                sP_pos=prev_accepted_sP_pos,
                sP_neg=prev_accepted_sP_neg,
                sQ_pos=prev_accepted_sQ_pos,
                sQ_neg=prev_accepted_sQ_neg,
                in_arcs=data["in_arcs"],
                out_arcs=data["out_arcs"],
                fraction=float(SLACK_AWARE_FRACTION),
            )
            prox_prev = {
                "Pij": P_anchor,
                "Qij": Q_anchor,
                "v": v_anchor,
            }
        else:
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

        loss_proxy = compute_loss_proxy_coeffs(
            data=data,
            Pbar=prox_prev["Pij"],
            Qbar=prox_prev["Qij"],
            vbar=prox_prev["v"],
            ubar=warm.get("u_send", None),
            w_loss=w_loss,
        )

        # [A6] Append this iter's plane to the history (degenerate iter-1
        # plane is harmless under MULTI_CUT_LOSS_L_NONNEG since it reduces
        # to l_edge >= 0, which is already implied).
        # (B) When USE_PLANE_FEAS_GUARD=True, plane coefficients that
        # would force l_edge < 0 at any prior anchor are zeroed.
        if USE_MULTI_CUT_LOSS:
            new_plane = {
                "aP": dict(loss_proxy.get("aP", {})),
                "aQ": dict(loss_proxy.get("aQ", {})),
                "aV": dict(loss_proxy.get("aV", {})),
                "b0": dict(loss_proxy.get("b0", {})),
            }
            loss_planes_hist, n_skipped = _push_loss_plane_guarded(
                loss_planes_hist,
                new_plane,
                plane_anchor_hist,
                int(MULTI_CUT_LOSS_HISTORY),
            )
            if n_skipped > 0 and tee:
                print(
                    f"[INFO] plane-feas guard zeroed {n_skipped} edge coeffs "
                    f"that would force l_edge<0 at a stored anchor."
                )
            plane_anchor_hist.append({
                "Pij": dict(prox_prev.get("Pij", {})),
                "Qij": dict(prox_prev.get("Qij", {})),
                "v": dict(prox_prev.get("v", {})),
            })
            if len(plane_anchor_hist) > int(MULTI_CUT_LOSS_HISTORY):
                plane_anchor_hist.pop(0)

        p_step_frac, q_step_frac, v_step_abs = _scheduled_prox_bounds(t)

        # Scheduled proximal weights (decay to RHO_*_FINAL so that the final
        # iterates approximate the unbiased cost optimum).
        iter_rho_p, iter_rho_q, iter_rho_v, iter_rho_theta = _scheduled_rho(t)

        # Warmup iters (iter 1 .. iter effective_warmup_iters) are the
        # hardest MIPs: no warm ell or flat v_prev at iter 1, and iter-1's
        # non-physical flows contaminating the loss_proxy / proximal anchor
        # at iter 2. Deactivating loss_proxy and proximal for these iters
        # removes two anchor-biased objective components so SCIP can find an
        # incumbent from the proportional-warm Pg much faster, and grants
        # the larger SCIP budget + looser gap. See WARMUP_ITERS comment
        # block for why iter 2 benefits from the same treatment as iter 1
        # on 3012-bus. Multi-cut loss is also disabled during warmup -- the
        # iter-1 / iter-2 planes are computed at flat or non-physical
        # reference flows and only become meaningful once we have an
        # accepted iterate.
        # NOTE: effective_warmup_iters is set at the top of run_bfm_ag
        # based on whether a Python warm-start snapshot is available
        # (snapshot present -> 0; otherwise WARMUP_ITERS default).
        # (A) Anchor-cleanliness gate.  Even past warmup, if the previous
        # accepted iter's slacks were large, treating its (P,Q,v) as a
        # prox/multi-cut anchor distorts those terms toward the slacked
        # state.  Disable them until the anchor cleans up.
        anchor_clean = _is_anchor_clean(prev_accepted_max_kcl, prev_accepted_max_vdrop)
        if t <= effective_warmup_iters or (USE_ANCHOR_CLEAN_GATE and not anchor_clean):
            iter_loss_proxy = None
            iter_prox_prev = None
            iter_loss_planes = None
            iter_w_loss_current = 0.0
            if t > effective_warmup_iters and (not anchor_clean) and tee:
                print(
                    f"[INFO] anchor-clean gate active at t={t}: prev_max_kcl="
                    f"{prev_accepted_max_kcl:.3e}, prev_max_vdrop="
                    f"{prev_accepted_max_vdrop:.3e}; skipping "
                    f"prox/multi-cut/theta-proj this iter."
                )
        else:
            iter_loss_proxy = loss_proxy
            iter_prox_prev = prox_prev
            iter_loss_planes = (loss_planes_hist if USE_MULTI_CUT_LOSS else None)
            iter_w_loss_current = float(w_loss)

        # (E) Phase-I / Phase-II cost scaling.  In Phase-I down-weight
        # gen_cost so the solver chases feasibility first; in Phase-II
        # restore gen_cost weight 1.0.  Slack penalties are simultaneously
        # boosted by PHASE_I_RHO_BOOST during Phase-I.
        if USE_PHASE_I_SPLIT and t <= int(PHASE_I_ITERS):
            iter_cost_scale = float(PHASE_I_COST_SCALE)
            iter_rho_slack_boost = float(PHASE_I_RHO_BOOST)
        else:
            iter_cost_scale = 1.0
            iter_rho_slack_boost = 1.0

        # (F) Binary fixing payload (only effective from F_BIN_FIX_ITER+).
        if USE_BINARY_FIXING and t >= int(F_BIN_FIX_ITER):
            iter_fix_oltc = fix_oltc_dict
            iter_fix_shunt = fix_shunt_dict
        else:
            iter_fix_oltc = None
            iter_fix_shunt = None
        # Iter 1 always gets the enlarged SCIP budget regardless of warmup
        # gating: with WARMUP_ITERS=0 it carries the full loss_proxy +
        # multi-cut + proximal load on the first solve and starts with no
        # previous-iter state.
        iter_scip_time = float(SCIP_TIME_LIMIT_ITER1) if t == 1 else float(SCIP_TIME_LIMIT)
        # [B10] Round-dependent gap schedule.  Subsumes the legacy binary
        # iter-warmup-vs-normal split when USE_ROUND_DEPENDENT_GAP=True.
        iter_scip_gap = _gap_for_iter(t)

        # ------------------------------------------------------------------
        # Solve subproblem with the CURRENT adaptive bounds.
        # If it fails, keep the previous accepted iterate and enlarge bounds.
        # ------------------------------------------------------------------
        try:
            model, tag = solve_subproblem_robust(
                data,
                trial_ell_fix,
                warm=warm,
                loss_proxy=iter_loss_proxy,
                prox_prev=iter_prox_prev,
                vdrop_slack_max=curr_vdrop_tol,
                kcl_slack_max=curr_kcl_tol,
                p_step_frac=p_step_frac,
                q_step_frac=q_step_frac,
                v_step_abs=v_step_abs,
                tee=tee,
                rho_p=iter_rho_p,
                rho_q=iter_rho_q,
                rho_v=iter_rho_v,
                rho_theta=iter_rho_theta,
                scip_time_limit=iter_scip_time,
                scip_gap_limit=iter_scip_gap,
                loss_planes=iter_loss_planes,
                w_loss_current=iter_w_loss_current,
                # (D), (E), (F), (H) forwarded to inner build_subproblem.
                lambda_kcl=(lambda_kcl_dict if USE_AUG_LAGRANGIAN else None),
                lambda_vdrop=(lambda_vdrop_dict if USE_AUG_LAGRANGIAN else None),
                cost_scale=iter_cost_scale,
                rho_slack_boost=iter_rho_slack_boost,
                rho_kcl_per_bus=rho_kcl_per_bus,
                fix_oltc=iter_fix_oltc,
                fix_shunt=iter_fix_shunt,
            )
        except RuntimeError as exc:
            consecutive_rejects += 1
            curr_vdrop_tol = min(float(VDROP_SLACK_RESCUE_CAP), float(curr_vdrop_tol) * float(SLACK_ENLARGE_ON_FAIL))
            curr_kcl_tol = min(float(KCL_SLACK_RESCUE_CAP), float(curr_kcl_tol) * float(SLACK_ENLARGE_ON_FAIL))
            # (C) Shrink trust region on REJECT so the next ell_fix step
            # is smaller, increasing the odds of feasibility.
            tr_radius_rel, tr_radius_abs = _adapt_trust_region(
                tr_radius_rel, tr_radius_abs, accepted=False
            )
            print(
                f"[REJECT t={t:02d}] subproblem returned no usable solution under current bounded-slack window. "
                f"Keeping previous accepted iterate and enlarging bounds to "
                f"vdrop<={curr_vdrop_tol:.3e}, kcl<={curr_kcl_tol:.3e}; "
                f"TR radius rel={tr_radius_rel:.3e}, abs={tr_radius_abs:.3e}."
            )
            # Always print the attempt breakdown so "infeasible" vs
            # "maxTimeLimit(no_incumbent)" is visible at a glance (the
            # outer loop treats both as REJECT, but they require
            # different remediation -- true infeasibility needs wider
            # slacks or a better warm-start, no_incumbent needs more
            # SCIP time).
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

        # ------------------------------------------------------------------
        # Build CANDIDATE next-state quantities. These are only committed if
        # the candidate is accepted.
        # ------------------------------------------------------------------
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
            + ("" if USE_EXACT_EQUALITY_SLACK else f"penV={obj['pen_vdrop']:,.3e}  penK={obj['pen_kcl']:,.3e}  ")
            + f"w_loss={w_loss:.6f}  time={t1-t0:.2f}s"
        )

        stopped.update({
            "iter": t,
            "gen_cost": obj["gen_cost"],
            "eff_cost": obj["eff_cost"],
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
            # (C) Shrink TR on candidate-REJECT so next ell-step is smaller.
            tr_radius_rel, tr_radius_abs = _adapt_trust_region(
                tr_radius_rel, tr_radius_abs, accepted=False
            )
            print(
                f"[REJECT t={t:02d}] keeping previous accepted iterate because "
                f"{', '.join(reject_reasons)}. "
                f"New adaptive bounds: vdrop<={curr_vdrop_tol:.3e}, kcl<={curr_kcl_tol:.3e}; "
                f"TR radius rel={tr_radius_rel:.3e}, abs={tr_radius_abs:.3e}."
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

        # Capture per-bus slack distribution for (J) slack-aware anchor
        # at the next iter, plus the scalar maxes for (A) anchor-cleanliness.
        prev_accepted_max_kcl = float(max_kcl_slack)
        prev_accepted_max_vdrop = float(max_vdrop_slack)
        prev_accepted_sP_pos = {int(b): _val(model.sP_pos[b], 0.0) for b in buses}
        prev_accepted_sP_neg = {int(b): _val(model.sP_neg[b], 0.0) for b in buses}
        prev_accepted_sQ_pos = {int(b): _val(model.sQ_pos[b], 0.0) for b in buses}
        prev_accepted_sQ_neg = {int(b): _val(model.sQ_neg[b], 0.0) for b in buses}

        # (D) Augmented Lagrangian dual updates.  lambda climbs proportional
        # to the slack used at this iter, with a per-component cap to
        # prevent runaway and saturation of the linear term.
        if USE_AUG_LAGRANGIAN:
            for b in buses:
                slack_b = (
                    prev_accepted_sP_pos[int(b)] + prev_accepted_sP_neg[int(b)]
                    + prev_accepted_sQ_pos[int(b)] + prev_accepted_sQ_neg[int(b)]
                )
                lam = float(lambda_kcl_dict.get(int(b), 0.0)) + float(AL_ETA_KCL) * float(slack_b)
                lambda_kcl_dict[int(b)] = max(0.0, min(float(AL_LAM_MAX_KCL), lam))
            for (i, j) in E:
                s_e = _val(model.s_vdrop[i, j], 0.0)
                lam = float(lambda_vdrop_dict.get((int(i), int(j)), 0.0)) + float(AL_ETA_VDROP) * float(s_e)
                lambda_vdrop_dict[(int(i), int(j))] = max(0.0, min(float(AL_LAM_MAX_VDROP), lam))

        # (G) Anderson Acceleration history maintenance: pair (ell_pre_step,
        # ell_target) so the next iter's _anderson_step has a well-defined
        # fixed-point map.
        if USE_ANDERSON_ELL:
            aa_ell_hist.append(dict(ell_fix))           # ell_t (input to T)
            aa_target_hist.append(dict(cand_ell_state)) # T(ell_t)
            if len(aa_ell_hist) > int(AA_HISTORY) + 1:
                aa_ell_hist.pop(0)
                aa_target_hist.pop(0)

        # (C) Trust-region: ACCEPT -> grow.
        tr_radius_rel, tr_radius_abs = _adapt_trust_region(
            tr_radius_rel, tr_radius_abs, accepted=True
        )

        # (F) Binary fixing snapshot: at exactly F_BIN_FIX_ITER, capture
        # the incumbent's OLTC tap selection and shunt step so subsequent
        # iters skip the integer search.
        if USE_BINARY_FIXING and t == int(F_BIN_FIX_ITER) - 1:
            if F_BIN_FIX_OLTC and len(data["T"]) > 0:
                fix_oltc_dict = {}
                for (i, j) in data["T"]:
                    pick = _pick_tap_from_beta(model, data, int(i), int(j))
                    for tap in data["K"][(int(i), int(j))]:
                        fix_oltc_dict[(int(i), int(j), int(tap))] = 1 if int(tap) == int(pick) else 0
            if F_BIN_FIX_SHUNT and len(data["C"]) > 0:
                fix_shunt_dict = {
                    int(i): (1 if _val(model.a_sh[i], 0.0) >= 0.5 else 0)
                    for i in data["C"]
                }
            if tee:
                print(
                    f"[INFO] (F) binary fixing snapshot taken at t={t}: "
                    f"{len(fix_oltc_dict or {})} OLTC entries, "
                    f"{len(fix_shunt_dict or {})} shunt entries; "
                    f"iter {F_BIN_FIX_ITER}+ will solve with binaries fixed."
                )

        # ell_fix commit for iter_{t+1}:
        #   USE_ACCEPTED_ELL_AS_NEXT_FIX=True path feeds cand_ell_state (the
        #     outer-hybrid target). With USE_OUTER_ELL_FORCE_FOLLOW=True that
        #     target is un-damped (0.5*ell_i + 0.5*ell_v clipped only). On
        #     3012-bus the very first target can be ~50 pu^2 on low-impedance
        #     edges, which makes iter_{t+1}'s KCL reactive balance infeasible
        #     (x*ell_fix ~ 5 pu per branch of induced reactive loss). The
        #     pre-2026-04 code relied on trial_ell_fix's EMA for damping but
        #     this path BYPASSES trial_ell_fix entirely -- so we re-apply EMA
        #     here on the commit to honour the intent in the "2번 변화" and
        #     USE_OUTER_ELL_FORCE_FOLLOW comment blocks above.
        #   USE_ACCEPTED_ELL_AS_NEXT_FIX=False path uses trial_ell_fix which
        #     is already EMA'd at lines 1870-1876, so no re-application here.
        # (C) Trust-region clip is also re-applied on commit so the committed
        #     ell_fix never moves more than the current TR radius from the
        #     pre-commit ell_fix (i.e. the value the just-solved MIQCP saw).
        ell_fix_pre_commit = dict(ell_fix)
        if USE_ACCEPTED_ELL_AS_NEXT_FIX:
            if USE_ELL_EMA_FIX:
                _ell_fix_next = {}
                for _k in cand_ell_state:
                    _ell_prev = float(ell_fix[_k])
                    _ell_tgt  = float(cand_ell_state[_k])
                    _ell_val  = float(BETA_ELL) * _ell_prev + (1.0 - float(BETA_ELL)) * _ell_tgt
                    if ELL_CLIP_NONNEG:
                        _ell_val = max(0.0, _ell_val)
                    if ELL_CLIP_MAX:
                        _ell_val = min(float(ellmax[_k]), _ell_val)
                    _ell_fix_next[_k] = _ell_val
                ell_fix = _ell_fix_next
            else:
                ell_fix = dict(cand_ell_state)
        else:
            ell_fix = dict(trial_ell_fix)
        if USE_ELL_TRUST_REGION:
            ell_fix = _trust_region_clip_ell(
                ell_fix_pre_commit, ell_fix, tr_radius_rel, tr_radius_abs, ellmax,
            )
        ell_state = dict(cand_ell_state)
        v_prev = dict(cand_v_prev)
        P_prev = dict(cand_P_prev)
        Q_prev = dict(cand_Q_prev)
        u_send_prev = dict(cand_u_send_prev)
        theta_prev = dict(theta_new)
        warm = cand_warm
        w_loss_sm = float(trial_w_loss_sm)

        last.update({"iter": t, "model": model, "ell": dict(ell_new), "theta": dict(theta_new), "tag": tag})

        # Select `best` by effective cost (gen_cost + slack penalties), NOT by
        # raw gen_cost. Picking by gen_cost alone would prefer iterates that
        # reduced gen_cost only by pushing residual into KCL/vdrop slacks,
        # which are physically infeasible.
        if obj["eff_cost"] < best["eff_cost"]:
            best.update({
                "iter": t,
                "gen_cost": obj["gen_cost"],
                "eff_cost": obj["eff_cost"],
                "total": obj["total"],
                "model": model,
                "ell": dict(ell_new),
                "theta": dict(theta_new),
                "tag": tag
            })

        # Use the absolute BEST_*_TOL defaults (not the current relaxed bound)
        # so that `best_feasible` means "physically near-feasible," not just
        # "within the currently-permitted slack window."
        is_near_feasible = _is_near_feasible_slacks(
            max_vdrop_slack,
            max_kcl_slack,
        )
        if is_near_feasible:
            # Among physically near-feasible iterates, gen_cost is a fair
            # ranking criterion (penalties are tiny by definition).
            if obj["gen_cost"] < best_feasible["gen_cost"]:
                best_feasible.update({
                    "iter": t,
                    "gen_cost": obj["gen_cost"],
                    "eff_cost": obj["eff_cost"],
                    "total": obj["total"],
                    "model": model,
                    "ell": dict(ell_new),
                    "theta": dict(theta_new),
                    "tag": tag
                })
            feasible_cost_hist.append(float(obj["gen_cost"]))

        if (sum_diff <= eps) and (max_viol <= 1e-8):
            print(f"[CONVERGED] t={t}, eps={eps}")
            return {"best": best, "best_feasible": best_feasible, "last": last, "stopped": stopped}

        sumdiff_hist.append(float(sum_diff))
        obj_hist.append(float(obj["total"]))

        # Tighten the adaptive bounds only when the accepted step uses
        # comfortable interior slack; otherwise keep the current window.
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

        # Cost-based early stop: once we have enough near-feasible iterates and
        # their gen_cost has flat-lined within relative tolerance, keep going
        # is a waste of SCIP budget. Relies on best_feasible being a truly
        # physical near-feasible tracker (see _is_near_feasible_slacks above).
        if (t >= GEN_COST_PLATEAU_MIN_ITERS) and (len(feasible_cost_hist) >= GEN_COST_PLATEAU_WINDOW):
            recent_costs = feasible_cost_hist[-GEN_COST_PLATEAU_WINDOW:]
            mean_cost = sum(recent_costs) / len(recent_costs)
            cost_range = max(recent_costs) - min(recent_costs)
            scale = max(abs(mean_cost), 1.0)
            if cost_range <= GEN_COST_PLATEAU_REL_TOL * scale:
                print(
                    f"[EARLY-STOP: GEN-COST-PLATEAU] feasible gen_cost has stabilized. "
                    f"window={GEN_COST_PLATEAU_WINDOW}, mean={mean_cost:,.4f}, "
                    f"range={cost_range:.3e} (rel_tol={GEN_COST_PLATEAU_REL_TOL:.1e})."
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
def _auto_generate_warmstart_snapshot() -> None:
    """Ensure bfmag_warmstart_snapshot.npz exists before main() builds the net.

    Cascade (cheapest -> most expensive):
      1. .npz already there -> nothing to do.
      2. .npz.bak present -> restore (last good snapshot from a prior run).
      3. DCOPF_results.txt present -> generate via
         make_warmstart_snapshot.make_dcopf_snapshot().
      4. Nothing available -> log an actionable message and let the BFMag
         runpp -> heuristic fallback chain take over.

    Any failure is logged and swallowed so the run still proceeds.
    """
    snapshot_path = Path(__file__).resolve().with_name("bfmag_warmstart_snapshot.npz")
    if snapshot_path.is_file():
        return

    backup_path = snapshot_path.parent / (snapshot_path.name + ".bak")
    if backup_path.is_file():
        try:
            import shutil
            shutil.copy2(str(backup_path), str(snapshot_path))
            print(f"[INFO] Auto-restored warm-start snapshot from {backup_path.name}.")
            return
        except Exception as exc:
            print(f"[WARN] Auto-restore from {backup_path.name} failed: {exc}.")

    dcopf_results = snapshot_path.with_name("DCOPF_results.txt")
    if not dcopf_results.is_file():
        print(
            f"[INFO] Warm-start snapshot {snapshot_path.name} not found and "
            f"no {dcopf_results.name} available. To get a spatially-correct "
            f"seed, run 'python -B DCOPF.py' then "
            f"'python -B make_warmstart_snapshot.py' before this driver. "
            f"Proceeding with runpp -> heuristic fallback chain."
        )
        return

    print(
        f"[INFO] Warm-start snapshot {snapshot_path.name} not found — "
        f"generating from {dcopf_results.name} via make_warmstart_snapshot.py."
    )
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import make_warmstart_snapshot as mws  # type: ignore[import-not-found]
        mws.make_dcopf_snapshot(dcopf_results, snapshot_path)
    except Exception as exc:
        print(f"[WARN] Auto-generate snapshot failed: {exc}. Continuing with fallback chain.")


def verify_kkt() -> Optional[bool]:
    """Inspect the converged BFMivar iterate's slack levels (Sec. 3.2.3 / Prop. 2).

    A strict-interior iterate (max KCL slack < VERIFY_KCL_TOL and max vdrop slack
    < VERIFY_VDROP_TOL) satisfies the Prop. 2 premise that the converged BFMivar
    fixed point coincides with the ACOPF KKT point.  Reads the latest result
    log written by this run's tee and reports PASS / PARTIAL.
    """
    if not RESULT_LOG_PATH.is_file():
        print("[VERIFY] No result file at", RESULT_LOG_PATH)
        return None
    text = RESULT_LOG_PATH.read_text(encoding="utf-8", errors="replace")
    m_kcl = re.search(r"max_kcl_slack\s*=\s*([+\-]?[\d.eE+\-]+)", text)
    m_vd = re.search(r"max_vdrop_slack\s*=\s*([+\-]?[\d.eE+\-]+)", text)
    if not (m_kcl and m_vd):
        print("[VERIFY] Could not parse summary")
        return None
    max_kcl = float(m_kcl.group(1))
    max_vd = float(m_vd.group(1))
    print(f"[VERIFY] max_kcl_slack    = {max_kcl:.4e}  (tol = {VERIFY_KCL_TOL:.0e})")
    print(f"[VERIFY] max_vdrop_slack  = {max_vd:.4e}  (tol = {VERIFY_VDROP_TOL:.0e})")
    interior = (abs(max_kcl) < VERIFY_KCL_TOL) and (abs(max_vd) < VERIFY_VDROP_TOL)
    if interior:
        print("[VERIFY] [PASS] Strict-interior; Prop. 2 KKT premise satisfied.")
    else:
        print("[VERIFY] [PARTIAL] Slacks not strictly interior.")
    return interior


def main():
    t0 = time.perf_counter()

    _auto_generate_warmstart_snapshot()

    net = mcase.case3012_opf(**NETWORK_BUILD_KWARGS)

    cfg = build_cfg_from_net_metadata(net)
    data = extract_data_fullmesh_branch_table(net, cfg)

    print("[INFO] Data summary (IEEE 3012-bus, BFM-ivar with mitigation: T4+W3+H1)")
    print("  network = ieee3012bus (standalone build kwargs)")
    print(f"  #buses  = {len(data['buses'])}")
    print(f"  #branches  = {len(data['E'])}  (directed, aggregated from branch_params_pu_table)")
    print(f"  #gens   = {len(data['gen_records'])}")
    print(f"  #OLTC   = {len(data['T'])}  (beta vars = {sum(len(data['K'][ij]) for ij in data['T'])})")
    print(f"  #shunts = {len(data['C'])}")
    print(f"  outer: max_iters={OUTER_MAX_ITERS}, eps={OUTER_EPS}")
    print(f"  ell_gamma={ELL_GAMMA}, theta_gamma={THETA_GAMMA}, ridge={THETA_RIDGE}")
    if USE_EXACT_EQUALITY_SLACK:
        print("  exact vdrop/KCL mode: slack variables are hardbound to zero, so both equations are enforced as equalities.")
        print(f"  legacy slack-control machinery remains in the file for compatibility, but is inactive in exact-equality mode.")
    else:
        print(
            f"  bounded-slack mode: s_vdrop in [0,{VDROP_SLACK_INIT:.2e}] (per edge), "
            f"sum(sP+sQ) in [0,{KCL_SLACK_INIT:.2e}] (per bus), penalty RHO_VDROP={RHO_VDROP:.1e}, RHO_KCL={RHO_KCL:.1e}."
        )
        print(
            f"  slack schedule: vdrop {VDROP_SLACK_INIT:.1e}->{VDROP_SLACK_FINAL:.1e} (decay={VDROP_SLACK_DECAY}), "
            f"kcl {KCL_SLACK_INIT:.1e}->{KCL_SLACK_FINAL:.1e} (decay={KCL_SLACK_DECAY}); "
            f"adaptive bounds enlarge on failure/reject up to caps vdrop<={VDROP_SLACK_RESCUE_CAP:.1e}, kcl<={KCL_SLACK_RESCUE_CAP:.1e}."
        )
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
            f"  prox-objective weights (scheduled): "
            f"RHO_P {RHO_P_INIT}->{RHO_P_FINAL} (decay={RHO_P_DECAY}), "
            f"RHO_Q {RHO_Q_INIT}->{RHO_Q_FINAL} (decay={RHO_Q_DECAY}), "
            f"RHO_V {RHO_V_INIT}->{RHO_V_FINAL} (decay={RHO_V_DECAY})"
        )
    print(
        f"  theta-proj weight (scheduled): RHO_THETA {RHO_THETA_INIT}->{RHO_THETA_FINAL} (decay={RHO_THETA_DECAY})"
    )
    if WARMUP_ITERS > 0:
        print(
            f"  warmup (iter 1..{WARMUP_ITERS}): loss_proxy=OFF, multi-cut loss=OFF, proximal=OFF, Pg warm-start from proportional dispatch, "
            f"SCIP time<={SCIP_TIME_LIMIT_ITER1:.0f}s; "
            f"normal (iter {WARMUP_ITERS+1}+): SCIP time<={SCIP_TIME_LIMIT:.0f}s"
        )
    else:
        print(
            f"  warmup disabled (WARMUP_ITERS=0): loss_proxy/multi-cut/proximal active from iter 1; "
            f"iter 1 SCIP time<={SCIP_TIME_LIMIT_ITER1:.0f}s, iter 2+ SCIP time<={SCIP_TIME_LIMIT:.0f}s"
        )
    if USE_ROUND_DEPENDENT_GAP:
        print(
            f"  [B10] gap schedule (round-dependent): "
            f"iter 1..{GAP_SWITCH_EARLY} -> {GAP_EARLY:.1e}, "
            f"iter {GAP_SWITCH_EARLY+1}..{GAP_SWITCH_MID} -> {GAP_MID:.1e}, "
            f"iter {GAP_SWITCH_MID+1}+ -> {GAP_LATE:.1e}"
        )
    else:
        print(
            f"  gap schedule (legacy binary): warmup -> {SCIP_GAP_LIMIT_ITER1:.1e}, normal -> {SCIP_GAP_LIMIT:.1e}"
        )
    _multi_cut_warmup_note = (
        f"warmup iters 1..{WARMUP_ITERS} still skip the loss term"
        if WARMUP_ITERS > 0
        else "active from iter 1"
    )
    print(
        f"  [A6] multi-cut loss: enabled={USE_MULTI_CUT_LOSS}, "
        f"history={MULTI_CUT_LOSS_HISTORY}, l_edge>=0={MULTI_CUT_LOSS_L_NONNEG} "
        f"(replaces single-plane loss linearization with envelope of K stored Taylor planes; "
        f"{_multi_cut_warmup_note})"
    )
    print(
        f"  [B7] solver order: gurobi_first={USE_GUROBI_FIRST} (graceful no-op if Gurobi unavailable), "
        f"then SCIP NL, then SCIP default; warmstart={USE_MIP_WARMSTART}"
    )
    print(
        f"  [C14] theta-reuse vS in loss proxy: enabled={USE_THETA_REUSE_VS} "
        f"(wired via warm['u_send'] -> compute_loss_proxy_coeffs(ubar=...))"
    )
    print(
        f"  best selection: by eff_cost = gen_cost + pen_vdrop + pen_kcl (NOT gen_cost alone)"
    )
    print(
        f"  best_feasible gate: |vdrop_slack|<={BEST_VDROP_TOL:.2e}, |kcl_slack|<={BEST_KCL_TOL:.2e} (absolute, not adaptive-window)"
    )
    print(
        f"  w_loss smoothing: beta={BETA_WLOSS}, scale={LOSS_WEIGHT_SCALE}, nonneg={WLOSS_NONNEG}"
    )
    print(
        f"  anti-oscillation: ell_ema_fix={USE_ELL_EMA_FIX} (beta_ell={BETA_ELL}), "
        f"state_damp={USE_STATE_DAMPING} (DAMPING_X={DAMPING_X}), theta_wrap_damp={USE_WRAP_THETA_DAMP}"
    )
    print(
        f"  ell_fix cap (per-edge): max(floor={ELL_FIX_HARD_CAP_PU2:.2f}, "
        f"(smax_e * {ELL_HARD_CAP_HEADROOM:.2f} / vmin_fr)^2), "
        f"clipped above by raw_ellmax = (smax_e / vmin_fr)^2"
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
    # ---- 3번 변화 (2026-04-28 convergence fixes) ----
    print(
        f"  [3-A] anchor-clean gate: enabled={USE_ANCHOR_CLEAN_GATE}, "
        f"kcl_thr={ANCHOR_CLEAN_KCL_PU:.2e} pu, vdrop_thr={ANCHOR_CLEAN_VDROP_PU:.2e} pu "
        f"(disables prox/multi-cut/theta-proj when anchor is non-physical)"
    )
    print(
        f"  [3-B] plane-feas guard on multi-cut: enabled={USE_PLANE_FEAS_GUARD}, tol={PLANE_GUARD_TOL:.1e}"
    )
    print(
        f"  [3-C] trust-region on ell_fix: enabled={USE_ELL_TRUST_REGION}, "
        f"radius_rel init={TR_RADIUS_REL_INIT}, abs init={TR_RADIUS_ABS_INIT}, "
        f"grow={TR_GROW}, shrink={TR_SHRINK}, "
        f"rel_clamp=[{TR_RADIUS_REL_MIN}, {TR_RADIUS_REL_MAX}], "
        f"abs_clamp=[{TR_RADIUS_ABS_MIN}, {TR_RADIUS_ABS_MAX}]"
    )
    print(
        f"  [3-D] augmented Lagrangian on slack: enabled={USE_AUG_LAGRANGIAN}, "
        f"eta_kcl={AL_ETA_KCL:.1e}, eta_vdrop={AL_ETA_VDROP:.1e}, "
        f"lam_max_kcl={AL_LAM_MAX_KCL:.1e}, lam_max_vdrop={AL_LAM_MAX_VDROP:.1e}"
    )
    print(
        f"  [3-E] Phase-I/II split: enabled={USE_PHASE_I_SPLIT}, phase_I iters={PHASE_I_ITERS}, "
        f"cost_scale_phase_I={PHASE_I_COST_SCALE}, rho_boost_phase_I={PHASE_I_RHO_BOOST}"
    )
    print(
        f"  [3-F] binary fixing: enabled={USE_BINARY_FIXING}, fix_after_iter={F_BIN_FIX_ITER}, "
        f"fix_oltc={F_BIN_FIX_OLTC}, fix_shunt={F_BIN_FIX_SHUNT}"
    )
    print(
        f"  [3-G] Anderson acceleration on ell: enabled={USE_ANDERSON_ELL}, "
        f"history={AA_HISTORY}, min_history={AA_MIN_HISTORY}, "
        f"reg={AA_REG}, safeguard_factor={AA_SAFEGUARD_FACTOR}"
    )
    print(
        f"  [3-H] stratified KCL penalty: enabled={USE_STRATIFIED_KCL_PENALTY}, "
        f"load_factor={RHO_KCL_LOAD_FACTOR}, gen_factor={RHO_KCL_GEN_FACTOR}"
    )
    print(
        f"  [3-I] IIS-guided slack widening: enabled={USE_IIS_GUIDED_WIDENING}, "
        f"topk_buses={IIS_WIDEN_TOPK_BUSES}, topk_edges={IIS_WIDEN_TOPK_EDGES}, "
        f"factor={IIS_WIDEN_FACTOR}"
    )
    print(
        f"  [3-J] slack-aware proximal anchor: enabled={USE_SLACK_AWARE_ANCHOR}, "
        f"fraction={SLACK_AWARE_FRACTION}"
    )

    sol = run_bfm_ag(data, max_iters=OUTER_MAX_ITERS, eps=OUTER_EPS, tee=TEE_SOLVER_LOG)

    best = sol["best"]
    best_feasible = sol["best_feasible"]
    last = sol["last"]
    stopped = sol.get("stopped", {"model": None, "iter": 0, "tag": ""})

    print("\n[SOLVED] Summary")
    print(
        f"  best(iter={best['iter']})  gen_cost={best['gen_cost']:.6f}  "
        f"eff_cost={best.get('eff_cost', float('nan')):.6f}  total={best['total']:.6f}  tag={best['tag']}"
    )
    if best_feasible["model"] is not None:
        print(
            f"  best_feasible(iter={best_feasible['iter']})  "
            f"gen_cost={best_feasible['gen_cost']:.6f}  "
            f"eff_cost={best_feasible.get('eff_cost', float('nan')):.6f}  "
            f"total={best_feasible['total']:.6f}  tag={best_feasible['tag']}"
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

    print()
    print("=" * 78)
    print("Verification (Sec. 3.2.3 / Prop. 2)")
    print("=" * 78)
    verify_kkt()


if __name__ == "__main__":
    class _Tee:
        """stdout/stderr multiplexer that flushes after every write.

        Console output can be truncated by a small terminal window, so we
        also mirror everything to a persistent log file. Flushing on every
        write means the file stays up-to-date even if the process is
        killed mid-iteration (e.g. user Ctrl-C during a long SCIP solve),
        so partial logs are never lost.
        """

        def __init__(self, *streams):
            self._streams = streams

        def write(self, data):
            for stream in self._streams:
                try:
                    stream.write(data)
                    stream.flush()
                except Exception:
                    pass
            return len(data)

        def flush(self):
            for stream in self._streams:
                try:
                    stream.flush()
                except Exception:
                    pass

        def isatty(self):
            return any(getattr(stream, "isatty", lambda: False)() for stream in self._streams)

    # Pre-extract H1 hard-equality buses from the prior run's result file
    # BEFORE the teeing block below truncates it.  This is the only chance
    # to read previous-run state out of the file.
    _H1_KCL_HARD_BUSES = _extract_top_kcl_buses(
        Path(__file__).resolve().with_name(KCL_HARD_SOURCE_FILE),
        top_k=KCL_HARD_TOPK,
        min_slack_pu=KCL_HARD_MIN_SLACK_PU,
    )

    # Two log files per run:
    #   - RESULT_LOG_PATH (BFM_ivar_mit_results.txt): always overwritten;
    #     canonical "latest" snapshot.  Read by H1 on the next run.
    #   - RESULT_LOG_ARCHIVE: timestamped copy under logs/ so previous runs are
    #     never clobbered.
    _ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    _archive_dir = RESULT_LOG_PATH.parent / "logs"
    try:
        _archive_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        _archive_dir = RESULT_LOG_PATH.parent
    RESULT_LOG_ARCHIVE = _archive_dir / f"BFM_ivar_mit_results_{_ts}.txt"

    with RESULT_LOG_PATH.open("w", encoding="utf-8") as _fh_latest, \
         RESULT_LOG_ARCHIVE.open("w", encoding="utf-8") as _fh_archive:
        _stdout0, _stderr0 = sys.stdout, sys.stderr
        sys.stdout = _Tee(sys.stdout, _fh_latest, _fh_archive)
        sys.stderr = _Tee(sys.stderr, _fh_latest, _fh_archive)
        try:
            print(f"[INFO] Run timestamp: {_ts}")
            print(f"[INFO] Writing result log (latest, overwritten each run): {RESULT_LOG_PATH}")
            print(f"[INFO] Writing result log (archive, per-run): {RESULT_LOG_ARCHIVE}")
            main()
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            sys.stdout = _stdout0
            sys.stderr = _stderr0
