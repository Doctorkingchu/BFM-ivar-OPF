# BFM_it.py
# ------------------------------------------------------------
# Pyomo MIQCP BFM-it for IEEE 300-bus system
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
#   ieee300bus.py
# ------------------------------------------------------------

import math
import time
from dataclasses import dataclass
from typing import Dict, Tuple, Any, Optional, List

import pandapower as pp
import pyomo.environ as pyo
from pyomo.opt import TerminationCondition

import ieee300bus as mcase


# ============================================================
# User options
# ============================================================
FIX_SLACK_VM = True

# If None -> use all metadata-defined OLTC/shunts from the network
ACTIVE_OLTC_EDGES: Optional[List[Tuple[int, int]]] = None
ACTIVE_SHUNT_BUSES: Optional[List[int]] = None

# Standalone network build settings used directly in this script.
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
    expose_global_recommendations=False,
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
    """
    Return True only when *all* active variables carry values.

    A weaker "any variable has a value" check is unsafe here because the BFMit
    model is heavily warm-started. After an infeasible solve, many variables can
    still retain initializer / warm-start values, which would make the model
    look solved even though SCIP loaded no incumbent.
    """
    saw_var = False
    for v in model.component_data_objects(pyo.Var, active=True, descend_into=True):
        saw_var = True
        if v.value is None:
            return False
    return saw_var


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


def _best_effort_runpp(net) -> bool:
    attempts = [
        dict(algorithm="nr", init="results", calculate_voltage_angles=True, enforce_q_lims=False, numba=False),
        dict(algorithm="nr", init="flat", calculate_voltage_angles=True, enforce_q_lims=False, numba=False),
        dict(algorithm="nr", init="auto", calculate_voltage_angles=True, enforce_q_lims=False, numba=False),
        dict(algorithm="bfsw", init="flat", calculate_voltage_angles=False, enforce_q_lims=False, numba=False),
    ]
    for kw in attempts:
        try:
            pp.runpp(net, **kw)
            return True
        except Exception:
            continue
    return False


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


@dataclass
class BranchTableBuildConfig:
    oltc_branches: Dict[Tuple[int, int], OLTCBranchConfig]
    shunt_bcap_pu: Dict[int, float]
    recommended_taps: Dict[Tuple[int, int], int]
    recommended_shunts: Dict[int, int]
    fix_slack_vm: bool = True


def build_branch_table_cfg_from_net_metadata(net) -> BranchTableBuildConfig:
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

    return BranchTableBuildConfig(
        oltc_branches=oltc_branches,
        shunt_bcap_pu=shunt_bcap_pu,
        recommended_taps=recommended_taps,
        recommended_shunts=recommended_shunts,
        fix_slack_vm=True,
    )


def _merge_parallel_series_equivalent(rows: List[dict]) -> Tuple[float, float, float]:
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


def extract_data_fullmesh_branch_table_local(net, cfg: BranchTableBuildConfig) -> Dict[str, Any]:
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
    )


def _extract_branch_pf_injection_pu(net, element_type: str, element_index: int, from_bus: int, to_bus: int) -> Tuple[Optional[float], Optional[float]]:
    try:
        if element_type == "line" and hasattr(net, "res_line") and net.res_line is not None and element_index in net.res_line.index:
            line = net.line.loc[element_index]
            rf = int(line["from_bus"])
            rt = int(line["to_bus"])
            if from_bus == rf and to_bus == rt:
                return (
                    float(net.res_line.at[element_index, "p_from_mw"]) / float(net.sn_mva),
                    float(net.res_line.at[element_index, "q_from_mvar"]) / float(net.sn_mva),
                )
            if from_bus == rt and to_bus == rf:
                return (
                    float(net.res_line.at[element_index, "p_to_mw"]) / float(net.sn_mva),
                    float(net.res_line.at[element_index, "q_to_mvar"]) / float(net.sn_mva),
                )
        elif element_type == "trafo" and hasattr(net, "res_trafo") and net.res_trafo is not None and element_index in net.res_trafo.index:
            trafo = net.trafo.loc[element_index]
            hv = int(trafo["hv_bus"])
            lv = int(trafo["lv_bus"])
            if from_bus == hv and to_bus == lv:
                return (
                    float(net.res_trafo.at[element_index, "p_hv_mw"]) / float(net.sn_mva),
                    float(net.res_trafo.at[element_index, "q_hv_mvar"]) / float(net.sn_mva),
                )
            if from_bus == lv and to_bus == hv:
                return (
                    float(net.res_trafo.at[element_index, "p_lv_mw"]) / float(net.sn_mva),
                    float(net.res_trafo.at[element_index, "q_lv_mvar"]) / float(net.sn_mva),
                )
    except Exception:
        return None, None

    return None, None


def _get_private_recommendations(net) -> Tuple[Dict[Tuple[int, int], int], Dict[int, int]]:
    store_key = None
    if "ver5_scenario_profile" in net and isinstance(net["ver5_scenario_profile"], dict):
        store_key = net["ver5_scenario_profile"].get("private_recommendation_store")
    if not store_key and "build_profile" in net and isinstance(net["build_profile"], dict):
        store_key = net["build_profile"].get("private_recommendation_store")
    if not store_key:
        store_key = "ver5_bfmag_private_recommendations"

    private = net.get(store_key, {})
    if not isinstance(private, dict):
        return {}, {}

    oltc = private.get("oltc_taps", {})
    shunt = private.get("shunt_status", {})

    if not isinstance(oltc, dict):
        oltc = {}
    if not isinstance(shunt, dict):
        shunt = {}

    return (
        {tuple(map(int, k)): int(v) for k, v in oltc.items()},
        {int(k): int(v) for k, v in shunt.items()},
    )


def build_pf_guided_bfmit_initialization(net, data: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[Tuple[int, int], float], bool]:
    pf_ok = _best_effort_runpp(net)
    buses = data["buses"]
    E = data["E"]
    T_set = set(data["T"])
    has_bus_results = hasattr(net, "res_bus") and net.res_bus is not None and not net.res_bus.empty
    has_ext_results = hasattr(net, "res_ext_grid") and net.res_ext_grid is not None and not net.res_ext_grid.empty
    has_gen_results = hasattr(net, "res_gen") and net.res_gen is not None and not net.res_gen.empty

    warm = {
        "v": {int(i): 1.0 for i in buses},
        "Pij": {(int(i), int(j)): 0.0 for (i, j) in E},
        "Qij": {(int(i), int(j)): 0.0 for (i, j) in E},
        "Pg": {},
        "Qg": {},
        "beta": {},
        "a_sh": {},
    }

    private_taps, private_shunts = _get_private_recommendations(net)
    rec_taps = dict(data.get("recommended_taps", {}))
    rec_shunts = dict(data.get("recommended_shunts", {}))
    if not rec_taps:
        rec_taps = dict(private_taps)
    if not rec_shunts:
        rec_shunts = dict(private_shunts)

    if has_bus_results:
        for i in buses:
            if int(i) in net.res_bus.index:
                vm = float(net.res_bus.at[int(i), "vm_pu"])
                warm["v"][int(i)] = max(vm * vm, DENOM_EPS)

    for g_idx, rec in enumerate(data["gen_records"]):
        et = str(rec["type"])
        rid = int(rec["id"])
        p_val = None
        q_val = None
        try:
            if et == "ext_grid" and has_ext_results and rid in net.res_ext_grid.index:
                p_val = float(net.res_ext_grid.at[rid, "p_mw"]) / float(net.sn_mva)
                q_val = float(net.res_ext_grid.at[rid, "q_mvar"]) / float(net.sn_mva)
            elif et == "gen" and has_gen_results and rid in net.res_gen.index:
                p_val = float(net.res_gen.at[rid, "p_mw"]) / float(net.sn_mva)
                q_val = float(net.res_gen.at[rid, "q_mvar"]) / float(net.sn_mva)
        except Exception:
            p_val = None
            q_val = None

        if p_val is None:
            p_val = float(rec["pmin_pu"])
        if q_val is None:
            q_val = min(max(0.0, float(rec["qmin_pu"])), float(rec["qmax_pu"]))

        warm["Pg"][int(g_idx)] = float(p_val)
        warm["Qg"][int(g_idx)] = float(q_val)

    for i in data["C"]:
        warm["a_sh"][int(i)] = float(rec_shunts.get(int(i), 0))

    for (i, j) in data["T"]:
        taps = [int(t) for t in data["K"][(i, j)]]
        pick = int(rec_taps.get((int(i), int(j)), _default_tap_choice(taps)))
        if pick not in taps:
            pick = _default_tap_choice(taps)
        for t in taps:
            warm["beta"][(int(i), int(j), int(t))] = 1.0 if int(t) == int(pick) else 0.0

    for (i, j) in E:
        p_sum = 0.0
        q_sum = 0.0
        any_flow = False
        elements = data.get("branch_elements", {}).get((int(i), int(j)), [])
        original_dirs = data.get("branch_original_dirs", {}).get((int(i), int(j)), [])
        for (et, eidx), (fb0, tb0) in zip(elements, original_dirs):
            p_pu, q_pu = _extract_branch_pf_injection_pu(net, str(et), int(eidx), int(i), int(j))
            if p_pu is None or q_pu is None:
                p_pu, q_pu = _extract_branch_pf_injection_pu(net, str(et), int(eidx), int(fb0), int(tb0))
            if p_pu is None or q_pu is None:
                continue
            p_sum += float(p_pu)
            q_sum += float(q_pu)
            any_flow = True
        if any_flow:
            warm["Pij"][(int(i), int(j))] = float(p_sum)
            warm["Qij"][(int(i), int(j))] = float(q_sum)

    ell_init: Dict[Tuple[int, int], float] = {}
    for (i, j) in E:
        p = float(warm["Pij"][(int(i), int(j))])
        q = float(warm["Qij"][(int(i), int(j))])
        if (int(i), int(j)) in T_set:
            taps = [int(t) for t in data["K"][(int(i), int(j))]]
            pick = _default_tap_choice(taps)
            for t in taps:
                if warm["beta"].get((int(i), int(j), int(t)), 0.0) >= 0.5:
                    pick = int(t)
                    break
            delta = float(data["delta_tap"][((int(i), int(j)), int(pick))])
            denom = delta * float(warm["v"][int(i)])
        else:
            denom = float(warm["v"][int(i)])
        denom = max(denom, DENOM_EPS)
        ell_val = (p * p + q * q) / denom
        ell_init[(int(i), int(j))] = min(max(float(ell_val), 0.0), float(data["ellmax"][(int(i), int(j))]))

    return warm, ell_init, bool(pf_ok or has_bus_results or has_ext_results or has_gen_results)


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

    for i in buses:
        v0 = _clip(1.0, float(Vmin[i] ** 2), float(Vmax[i] ** 2))
        m.v[i].value = float(v0)

    for g in Gset:
        p0 = float(gen_records[int(g)]["pmin_pu"])
        q0 = _clip(0.0, float(gen_records[int(g)]["qmin_pu"]), float(gen_records[int(g)]["qmax_pu"]))
        m.Pg[g].value = float(p0)
        m.Qg[g].value = float(q0)

    for (i, j) in E:
        m.Pij[i, j].value = 0.0
        m.Qij[i, j].value = 0.0

    # Seed discrete variables with a valid default configuration so SCIP's
    # candidate storage does not start from an obviously bound-violating point.
    for i in C:
        m.a_sh[i].value = 0.0

    for (i, j) in T:
        default_tap = _default_tap_choice([int(t) for t in K[(i, j)]])
        for tap in K[(i, j)]:
            m.beta[i, j, int(tap)].value = 1.0 if int(tap) == int(default_tap) else 0.0

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

    for i in buses:
        if m.qsh[i].value is None:
            m.qsh[i].value = 0.0

    for i in C:
        a_val = _safe_value(m.a_sh[i], 0.0)
        v_val = _safe_value(m.v[i], max(float(Vmin[i] ** 2), 1.0))
        z_val = a_val * v_val
        m.z[i].value = float(z_val)
        m.qsh[i].value = float(bcap[i] * z_val)

    for (i, j, tap) in m.BETA_INDEX:
        beta_val = _safe_value(m.beta[i, j, tap], 0.0)
        v_val = _safe_value(m.v[i], max(float(Vmin[i] ** 2), 1.0))
        m.tv[i, j, tap].value = float(beta_val * v_val)

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


def _discrete_choices_from_warm(
    data: Dict[str, Any],
    warm_start: Optional[Dict[str, Any]],
) -> Tuple[Dict[Tuple[int, int], int], Dict[int, int]]:
    tap_choice: Dict[Tuple[int, int], int] = {}
    sh_choice: Dict[int, int] = {}

    beta_ws = {}
    a_sh_ws = {}
    if isinstance(warm_start, dict):
        beta_ws = warm_start.get("beta", {}) or {}
        a_sh_ws = warm_start.get("a_sh", {}) or {}

    for (i, j) in data["T"]:
        taps = [int(t) for t in data["K"][(i, j)]]
        pick = None
        best_v = -1e100
        for t in taps:
            v = float(beta_ws.get((int(i), int(j), int(t)), 0.0))
            if v > best_v:
                best_v = v
                pick = int(t)
        if pick is None and (i, j) in data.get("recommended_taps", {}):
            pick = int(data["recommended_taps"][(i, j)])
        if pick is None or pick not in taps:
            pick = _default_tap_choice(taps)
        tap_choice[(int(i), int(j))] = int(pick)

    for i in data["C"]:
        if int(i) in a_sh_ws:
            sh_choice[int(i)] = 1 if float(a_sh_ws[int(i)]) >= 0.5 else 0
        else:
            sh_choice[int(i)] = int(data.get("recommended_shunts", {}).get(int(i), 0))

    return tap_choice, sh_choice


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
        if tee:
            print("[WARN] Relaxed solve also failed; trying warm-start-fixed QCQP ...")

        tap_choice, sh_choice = _discrete_choices_from_warm(data, warm_start)
        m_seed = build_pyomo_bfmit_model(data, ell_fix, relax_binaries=False, warm_start=warm_start)
        for (i, j) in data["T"]:
            pick = tap_choice[(i, j)]
            for t in data["K"][(i, j)]:
                m_seed.beta[i, j, int(t)].fix(1.0 if int(t) == int(pick) else 0.0)
        for i in data["C"]:
            m_seed.a_sh[i].fix(float(sh_choice[int(i)]))

        ok_seed = _solve_with_scip(m_seed, TIME_LIMIT_FIXED, MIP_GAP, tee)
        if ok_seed:
            return m_seed, "fixed_from_seed"

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
def run_bfmit(
    data: Dict[str, Any],
    max_iters: int,
    eps: float,
    tee: bool,
    initial_warm: Optional[Dict[str, Any]] = None,
    initial_ell: Optional[Dict[Tuple[int, int], float]] = None,
) -> Dict[str, Any]:
    buses = data["buses"]
    E = data["E"]
    T_set = set(data["T"])
    ellmax = data["ellmax"]

    ell_prev = dict(initial_ell) if initial_ell is not None else {(i, j): 0.0 for (i, j) in E}
    ell_last_ok = dict(ell_prev)

    warm = initial_warm if initial_warm is not None else {
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
        ell_anchor = dict(ell_last_ok) if best["model"] is not None else {(i, j): 0.0 for (i, j) in E}
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
                e: (1.0 - BACKTRACK_BLEND) * float(ell_anchor[e]) + BACKTRACK_BLEND * float(ell_try[e])
                for e in E
            }
            if tee:
                print(f"[WARN] Backtracking ell_fix (bt={bt}/{BACKTRACK_MAX_TRIES}) -> blending toward feasible reference ell.")

        if not solved or model is None or (not _has_loaded_solution(model)):
            print(f"[FAIL] t={t}: could not obtain an incumbent after backtracking. Stopping.")
            if (t == 1) and initial_warm is not None:
                seed_ell = dict(initial_ell) if initial_ell is not None else {(i, j): 0.0 for (i, j) in E}
                seed_model = build_pyomo_bfmit_model(data, seed_ell, relax_binaries=False, warm_start=initial_warm)
                seed_obj = float(pyo.value(seed_model.obj))
                print("[WARN] Returning PF-guided seed snapshot as fallback result.")
                return {
                    "iter": 0,
                    "obj": seed_obj,
                    "model": seed_model,
                    "ell": seed_ell,
                    "tag": "pf_seed_fallback",
                }
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
    print("BFM-it MIQCP RESULT (IEEE 300-bus)")
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

    net = mcase.case300_opf(**NETWORK_BUILD_KWARGS)
    cfg = build_branch_table_cfg_from_net_metadata(net)
    data = extract_data_fullmesh_branch_table_local(net, cfg)
    initial_warm, initial_ell, pf_seed_ok = build_pf_guided_bfmit_initialization(net, data)

    print("[INFO] Data summary (IEEE 300-bus, BFM-it)")
    print("  network = ieee300bus (standalone build kwargs)")
    print(f"  #buses   = {len(data['buses'])}")
    print(f"  #branches = {len(data['E'])}   (directed, aggregated from branch_params_pu_table)")
    print(f"  #gens    = {len(data['gen_records'])}")
    print(f"  #OLTC    = {len(data['T'])}   (beta vars = {sum(len(data['K'][ij]) for ij in data['T'])})")
    print(f"  #shunts  = {len(data['C'])}")
    print(f"  outer: max_iters={OUTER_MAX_ITERS}, eps={OUTER_EPS}, ell_gamma={ELL_DAMPING_GAMMA}")
    print(f"  backtrack: max_tries={BACKTRACK_MAX_TRIES}, blend={BACKTRACK_BLEND}")
    print(f"  SCIP: time(bin/relax/fix)={TIME_LIMIT_BINARY}/{TIME_LIMIT_RELAX}/{TIME_LIMIT_FIXED}s, gap={MIP_GAP}, feastol={SCIP_FEASTOL}")
    print(f"  PF-guided initialization = {'enabled' if pf_seed_ok else 'fallback-defaults'}")

    sol = run_bfmit(
        data,
        max_iters=OUTER_MAX_ITERS,
        eps=OUTER_EPS,
        tee=TEE_SOLVER_LOG,
        initial_warm=initial_warm,
        initial_ell=initial_ell,
    )

    best_model = sol["model"]
    if best_model is None:
        print("[FAIL] No solution produced.")
        return

    print("\n[SOLVED] Best/Last iterate summary")
    print(f"  iter = {sol['iter']}")
    print(f"  obj  = {sol['obj']:.6f}")
    print(f"  tag  = {sol.get('tag','')}")
    if sol.get("tag") == "relaxed_only":
        print("  note = continuous relaxation fallback; not integer-feasible")
    elif sol.get("tag") == "fixed_from_relax":
        print("  note = rounded/fixed discrete fallback")
    elif sol.get("tag") == "fixed_from_seed":
        print("  note = warm-start discrete seed fixed and solved")
    elif sol.get("tag") == "pf_seed_fallback":
        print("  note = optimizer found no incumbent; reporting PF-guided seed snapshot")
    print_solution(best_model, data, sol["ell"])

    t1 = time.perf_counter()
    print(f"\n[INFO] Total elapsed time: {t1 - t0:.2f} seconds.")


if __name__ == "__main__":
    main()
