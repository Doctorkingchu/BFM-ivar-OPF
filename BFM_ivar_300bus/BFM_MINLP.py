# BFM_MINLP.py
# ------------------------------------------------------------
# Pyomo Branch-Flow Model (BFM) AC-OPF / MINLP
# for the ver3 IEEE 300-bus mesh network
#
#   - Uses aggregated directed branches from branch_params_pu_table as E
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
#   ieee300bus.py
# ------------------------------------------------------------

import math
import time
import copy
from dataclasses import dataclass
from typing import Dict, Tuple, List, Any, Optional

import numpy as np
import pyomo.environ as pyo
import pandapower as pp

import ieee300bus as mcase


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


# ----------------------------
# Metadata readers
# ----------------------------
def read_network_device_metadata(net) -> Dict[str, Any]:
    cfg300 = build_branch_table_cfg_from_net_metadata(net)
    oltc_branches: Dict[Tuple[int, int], OLTCBranchConfig] = {}
    oltc_edges_ordered: List[Tuple[int, int]] = []

    for (i, j), tcfg in cfg300.oltc_branches.items():
        oltc_edges_ordered.append((int(i), int(j)))
        oltc_branches[(int(i), int(j))] = OLTCBranchConfig(
            tap_min=int(tcfg.tap_min),
            tap_max=int(tcfg.tap_max),
            dV_percent=float(tcfg.dV_percent),
        )

    shunts: Dict[int, ShuntConfig] = {}
    for b, val in cfg300.shunt_bcap_pu.items():
        shunts[int(b)] = ShuntConfig(bcap_pu=float(val), v_rated_sq=1.0)

    return {
        "oltc_edges_ordered": oltc_edges_ordered,
        "oltc_branches": oltc_branches,
        "shunts": shunts,
    }


def extract_bfm_fullmesh_data_ver2(net, cfg: BuildConfig) -> Dict[str, Any]:
    cfg300 = build_branch_table_cfg_from_net_metadata(net)
    data300 = extract_data_fullmesh_branch_table_local(net, cfg300)

    Vmin = {int(i): float(v) for i, v in data300["Vmin_pu"].items()}
    Vmax = {int(i): float(v) for i, v in data300["Vmax_pu"].items()}
    bcap_pu = {int(i): float(v) for i, v in data300["bcap"].items()}
    v_rated_sq = {int(i): float(cfg.shunts[int(i)].v_rated_sq) for i in data300["C"]}
    Mq = {}
    for i in data300["C"]:
        vU = float(Vmax[int(i)] ** 2)
        Mq[int(i)] = abs(bcap_pu[int(i)]) * (vU / max(v_rated_sq[int(i)], 1e-9)) + 1e-6

    return {
        "sn_mva": data300["sn_mva"],
        "buses": list(data300["buses"]),
        "slack_bus": int(data300["slack_bus"]),
        "slack_vm_pu": float(data300["slack_vm_pu"]),
        "Pd_pu": dict(data300["Pd_pu"]),
        "Qd_pu": dict(data300["Qd_pu"]),
        "Vmin": Vmin,
        "Vmax": Vmax,
        "gen_records": list(data300["gen_records"]),
        "E": list(data300["E"]),
        "r": dict(data300["r"]),
        "x": dict(data300["x"]),
        "ellmax": {(i, j): float(data300["Smax"][(i, j)] ** 2) for (i, j) in data300["E"]},
        "T": list(data300["T"]),
        "K": dict(data300["K"]),
        "alpha_tap": dict(data300["alpha_tap"]),
        "delta_tap": dict(data300["delta_tap"]),
        "C": list(data300["C"]),
        "bcap_pu": bcap_pu,
        "v_rated_sq": v_rated_sq,
        "Mq": Mq,
        "fix_slack_vm": bool(cfg.fix_slack_vm),
        "recommended_taps": dict(data300.get("recommended_taps", {})),
        "recommended_shunts": dict(data300.get("recommended_shunts", {})),
        "branch_elements": dict(data300.get("branch_elements", {})),
        "branch_original_dirs": dict(data300.get("branch_original_dirs", {})),
        "line_id_of_edge": {tuple(edge): idx for idx, edge in enumerate(data300["E"])},
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


def _clip_to_var_bounds(vardata, val: float) -> float:
    x = float(val)
    lb, ub = vardata.bounds
    if lb is not None:
        try:
            x = max(x, float(pyo.value(lb)))
        except Exception:
            pass
    if ub is not None:
        try:
            x = min(x, float(pyo.value(ub)))
        except Exception:
            pass
    return x


def _recommended_tap_or_default(data: Dict[str, Any], i: int, j: int) -> int:
    taps = [int(t) for t in data["K"][(int(i), int(j))]]
    rec = data.get("recommended_taps", {}).get((int(i), int(j)))
    if rec is not None and int(rec) in taps:
        return int(rec)
    return _default_tap(taps)


def _recommended_shunt_or_default(data: Dict[str, Any], i: int) -> int:
    return 1 if int(data.get("recommended_shunts", {}).get(int(i), 0)) else 0


def _pf_branch_pq_mva_in_orientation(
    net,
    element_type: str,
    element_index: int,
    ori_from: int,
    ori_to: int,
) -> Tuple[float, float]:
    try:
        if element_type == "line":
            if element_index not in net.line.index or element_index not in net.res_line.index:
                return 0.0, 0.0
            fb = int(net.line.at[element_index, "from_bus"])
            tb = int(net.line.at[element_index, "to_bus"])
            if (int(ori_from), int(ori_to)) == (fb, tb):
                return (
                    float(net.res_line.at[element_index, "p_from_mw"]),
                    float(net.res_line.at[element_index, "q_from_mvar"]),
                )
            if (int(ori_from), int(ori_to)) == (tb, fb):
                return (
                    float(net.res_line.at[element_index, "p_to_mw"]),
                    float(net.res_line.at[element_index, "q_to_mvar"]),
                )
            return 0.0, 0.0

        if element_type == "trafo":
            if element_index not in net.trafo.index or element_index not in net.res_trafo.index:
                return 0.0, 0.0
            hv = int(net.trafo.at[element_index, "hv_bus"])
            lv = int(net.trafo.at[element_index, "lv_bus"])
            if (int(ori_from), int(ori_to)) == (hv, lv):
                return (
                    float(net.res_trafo.at[element_index, "p_hv_mw"]),
                    float(net.res_trafo.at[element_index, "q_hv_mvar"]),
                )
            if (int(ori_from), int(ori_to)) == (lv, hv):
                return (
                    float(net.res_trafo.at[element_index, "p_lv_mw"]),
                    float(net.res_trafo.at[element_index, "q_lv_mvar"]),
                )
            return 0.0, 0.0
    except Exception:
        return 0.0, 0.0

    return 0.0, 0.0


def _has_complete_primal_solution(model: pyo.ConcreteModel) -> bool:
    for g in model.G:
        if model.Pg[g].value is None or model.Qg[g].value is None:
            return False
    for i in model.N:
        if (
            model.v[i].value is None
            or model.Pinj[i].value is None
            or model.Qinj[i].value is None
            or model.qsh[i].value is None
        ):
            return False
    for (i, j) in model.E:
        if (
            model.Pij[i, j].value is None
            or model.Qij[i, j].value is None
            or model.ell[i, j].value is None
        ):
            return False
    for (i, j) in model.T:
        if model.delta[i, j].value is None:
            return False
    for (i, j, tap) in model.BETA_INDEX:
        if model.beta[i, j, tap].value is None:
            return False
    for i in model.C:
        if model.a_sh[i].value is None:
            return False
    return True


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

    try:
        # Align PF warm start with the metadata-backed recommended OLTC positions when available.
        branch_elements = data.get("branch_elements", {})
        for (i, j) in data.get("T", []):
            tap_pick = _recommended_tap_or_default(data, int(i), int(j))
            for (et, eidx) in branch_elements.get((int(i), int(j)), []):
                if et == "trafo" and eidx in net.trafo.index and "tap_pos" in net.trafo.columns:
                    net.trafo.at[eidx, "tap_pos"] = int(tap_pick)

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
            pick = _recommended_tap_or_default(data, int(i), int(j))
            for tap in data["K"][(i, j)]:
                model.beta[i, j, tap].set_value(1.0 if tap == pick else 0.0)
            model.delta[i, j].set_value(data["delta_tap"][((i, j), pick)])

        for i in model.C:
            ash = _recommended_shunt_or_default(data, int(i))
            model.a_sh[i].set_value(float(ash))
        for i in model.N:
            if int(i) in set(data["C"]):
                if _recommended_shunt_or_default(data, int(i)):
                    q_target = float(data["bcap_pu"][int(i)]) * (float(model.v[int(i)].value) / max(float(data["v_rated_sq"][int(i)]), 1e-9))
                    model.qsh[int(i)].set_value(_clip_to_var_bounds(model.qsh[int(i)], q_target))
                else:
                    model.qsh[int(i)].set_value(0.0)
            else:
                model.qsh[int(i)].set_value(0.0)

        for i in model.N:
            pinj = -float(data["Pd_pu"][int(i)])
            qinj = -float(data["Qd_pu"][int(i)]) + float(model.qsh[int(i)].value or 0.0)
            for gg in model.G:
                if int(model.gen_bus[gg]) == int(i):
                    pinj += float(model.Pg[gg].value or 0.0)
                    qinj += float(model.Qg[gg].value or 0.0)
            model.Pinj[int(i)].set_value(pinj)
            model.Qinj[int(i)].set_value(qinj)

        branch_original_dirs = data.get("branch_original_dirs", {})
        for (i, j) in model.E:
            p_mw = 0.0
            q_mvar = 0.0
            elems = list(branch_elements.get((int(i), int(j)), []))
            dirs = list(branch_original_dirs.get((int(i), int(j)), []))
            for idx, (et, eidx) in enumerate(elems):
                ori = dirs[idx] if idx < len(dirs) else (int(i), int(j))
                pmw_k, qmvar_k = _pf_branch_pq_mva_in_orientation(net, str(et), int(eidx), int(ori[0]), int(ori[1]))
                p_mw += pmw_k
                q_mvar += qmvar_k

            p_pu = p_mw / sn
            q_pu = q_mvar / sn
            delta_e = float(model.delta[int(i), int(j)].value) if (int(i), int(j)) in set(data["T"]) else 1.0
            vsend = max(delta_e * float(model.v[int(i)].value), 1e-8)
            ell0 = max((p_pu * p_pu + q_pu * q_pu) / vsend, 0.0)

            model.Pij[i, j].set_value(_clip_to_var_bounds(model.Pij[i, j], p_pu))
            model.Qij[i, j].set_value(_clip_to_var_bounds(model.Qij[i, j], q_pu))
            model.ell[i, j].set_value(_clip_to_var_bounds(model.ell[i, j], ell0))

        return True

    except Exception:
        return False


def set_default_discrete_and_fix(model: pyo.ConcreteModel, data: Dict[str, Any]):
    for (i, j) in model.T:
        taps = data["K"][(i, j)]
        pick = _recommended_tap_or_default(data, int(i), int(j))
        for t in taps:
            model.beta[i, j, t].set_value(1 if t == pick else 0)
            model.beta[i, j, t].fix(1 if t == pick else 0)
        model.delta[i, j].set_value(data["delta_tap"][((i, j), pick)])
        model.delta[i, j].fix(data["delta_tap"][((i, j), pick)])
    for i in model.C:
        ash = _recommended_shunt_or_default(data, int(i))
        model.a_sh[i].set_value(ash)
        model.a_sh[i].fix(ash)
        if ash:
            q_target = float(data["bcap_pu"][int(i)]) * (float(model.v[int(i)].value or 1.0) / max(float(data["v_rated_sq"][int(i)]), 1e-9))
            model.qsh[int(i)].set_value(_clip_to_var_bounds(model.qsh[int(i)], q_target))
        else:
            model.qsh[int(i)].set_value(0.0)


# ----------------------------
# Diagnostics
# ----------------------------
def evaluate_quality(model: pyo.ConcreteModel, data: Dict[str, Any]) -> Dict[str, float]:
    if not _has_complete_primal_solution(model):
        return {
            "max_bfm_p": math.inf,
            "max_bfm_q": math.inf,
            "max_vdrop": math.inf,
            "max_irel": math.inf,
            "max_onehot": math.inf,
            "max_delta": math.inf,
            "max_shunt": math.inf,
            "max_frac": math.inf,
            "max_resid": math.inf,
        }

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

    if not _has_complete_primal_solution(model):
        return {
            "ok": False,
            "reason": f"no loaded feasible solution ({term})",
            "obj": obj,
            "quality": None,
            "termination": term,
            "elapsed": elapsed
        }

    try:
        q = evaluate_quality(model, model._data_for_check)
    except Exception as e:
        return {
            "ok": False,
            "reason": f"quality check failed: {e}",
            "obj": obj,
            "quality": None,
            "termination": term,
            "elapsed": elapsed
        }
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


def _build_ver2_net():
    return mcase.case300_opf(**NETWORK_BUILD_KWARGS)


def _tap_choice_from_model(model: Optional[pyo.ConcreteModel], data: Dict[str, Any]) -> Dict[Tuple[int, int], int]:
    out: Dict[Tuple[int, int], int] = {}
    if model is None:
        return out
    for (i, j) in data["T"]:
        best_tap = None
        best_val = -1e100
        for tap in data["K"][(i, j)]:
            val = pyo.value(model.beta[i, j, tap], exception=False)
            if val is None:
                continue
            fval = float(val)
            if fval > best_val:
                best_val = fval
                best_tap = int(tap)
        if best_tap is not None:
            out[(int(i), int(j))] = int(best_tap)
    return out


def _shunt_choice_from_model(model: Optional[pyo.ConcreteModel], data: Dict[str, Any]) -> Dict[int, int]:
    out: Dict[int, int] = {}
    if model is None:
        return out
    for i in data["C"]:
        val = pyo.value(model.a_sh[int(i)], exception=False)
        if val is None:
            continue
        out[int(i)] = 1 if float(val) >= 0.5 else 0
    return out


def _recommended_tap_choice(data: Dict[str, Any]) -> Dict[Tuple[int, int], int]:
    out: Dict[Tuple[int, int], int] = {}
    for (i, j) in data["T"]:
        out[(int(i), int(j))] = _recommended_tap_or_default(data, int(i), int(j))
    return out


def _recommended_shunt_choice(data: Dict[str, Any]) -> Dict[int, int]:
    out: Dict[int, int] = {}
    for i in data["C"]:
        out[int(i)] = _recommended_shunt_or_default(data, int(i))
    return out


def _apply_discrete_choice_to_net(net, tap_choice: Dict[Tuple[int, int], int], sh_choice: Dict[int, int]):
    brdf = net["branch_params_pu_table"] if "branch_params_pu_table" in net else None
    if brdf is not None and not brdf.empty:
        for (u, v), tap in tap_choice.items():
            hit = brdf[
                (((brdf["from_bus_pp"] == int(u)) & (brdf["to_bus_pp"] == int(v))) |
                 ((brdf["from_bus_pp"] == int(v)) & (brdf["to_bus_pp"] == int(u))))
                & (brdf["element_type"] == "trafo")
            ]
            for _, row in hit.iterrows():
                ti = int(row["element_index"])
                if ti in net.trafo.index and "tap_pos" in net.trafo.columns:
                    net.trafo.at[ti, "tap_pos"] = int(tap)

    if "fixed_shunt_table" in net:
        sn = float(net.sn_mva)
        for _, row in net["fixed_shunt_table"].iterrows():
            bus = int(row["bus_pp"]) if "bus_pp" in row.index else int(row["bus"])
            if int(sh_choice.get(bus, 0)) < 1:
                continue
            pp.create_shunt(
                net,
                bus=bus,
                p_mw=0.0,
                q_mvar=-float(row["bcap_pu"]) * sn,
                in_service=True,
                name=f"MINLP_BFM_SW_SHUNT@bus{bus}",
            )


def _apply_model_dispatch_to_net(net, model: Optional[pyo.ConcreteModel], data: Dict[str, Any]):
    if model is None:
        return

    sn = float(data["sn_mva"])
    for gg in model.G:
        rec = data["gen_records"][int(gg)]
        bus = int(rec["bus"])
        pg = pyo.value(model.Pg[gg], exception=False)
        vg2 = pyo.value(model.v[bus], exception=False)
        vm = math.sqrt(max(float(vg2), 0.0)) if vg2 is not None else None

        if rec["type"] == "ext_grid":
            idx = int(rec["id"])
            if idx in net.ext_grid.index and vm is not None and "vm_pu" in net.ext_grid.columns:
                net.ext_grid.at[idx, "vm_pu"] = float(vm)
        elif rec["type"] == "gen":
            idx = int(rec["id"])
            if idx in net.gen.index:
                if pg is not None and "p_mw" in net.gen.columns:
                    net.gen.at[idx, "p_mw"] = sn * float(pg)
                if vm is not None and "vm_pu" in net.gen.columns:
                    net.gen.at[idx, "vm_pu"] = float(vm)


def _run_pp_attempts(net):
    attempts = [
        ("nr", "results", True),
        ("nr", "dc", True),
        ("nr", "flat", True),
        ("nr", "flat", False),
        ("bfsw", "flat", False),
    ]

    last_error = None
    for algo, init, calc_va in attempts:
        trial = copy.deepcopy(net)
        try:
            pp.runpp(
                trial,
                algorithm=algo,
                init=init,
                calculate_voltage_angles=calc_va,
                enforce_q_lims=False,
                numba=False,
            )
            if bool(getattr(trial, "converged", False)):
                return trial, {"algorithm": algo, "init": init, "angles": calc_va}
            last_error = f"runpp did not converge with algorithm={algo}, init={init}"
        except Exception as exc:
            last_error = f"{algo}/{init}: {exc}"

    return None, {"error": last_error or "runpp failed"}


def _compute_pf_generation_cost(net) -> float:
    total = 0.0
    if not hasattr(net, "poly_cost") or net.poly_cost is None or net.poly_cost.empty:
        return total

    def _poly_cost(et: str, idx: int, p_mw: float) -> float:
        row = net.poly_cost[(net.poly_cost.et == et) & (net.poly_cost.element == idx)]
        if row.empty:
            return 0.0
        rec = row.iloc[0]
        return float(rec.cp2_eur_per_mw2) * p_mw * p_mw + float(rec.cp1_eur_per_mw) * p_mw + float(rec.cp0_eur)

    for eg in net.ext_grid.index:
        total += _poly_cost("ext_grid", int(eg), float(net.res_ext_grid.at[eg, "p_mw"]))
    for gi in net.gen.index:
        total += _poly_cost("gen", int(gi), float(net.res_gen.at[gi, "p_mw"]))
    return float(total)


def _print_pf_solution(net, title: str, pf_cost: float, pf_info: Dict[str, Any], tap_choice: Dict[Tuple[int, int], int], sh_choice: Dict[int, int]):
    print(f"\n[SOLVED] {title}")
    if "algorithm" in pf_info:
        print(
            f"[INFO] PF fallback used algorithm={pf_info['algorithm']}, "
            f"init={pf_info['init']}, angles={int(bool(pf_info['angles']))}"
        )

    print("\n--- Objective ---")
    print(f"PF recomputed generation cost (EUR): {pf_cost:.10f}")

    print("\n--- Bus Voltages ---")
    slack_bus = int(net.ext_grid.at[int(net.ext_grid.index[0]), "bus"]) if len(net.ext_grid.index) else -1
    for i in net.bus.index:
        vm = float(net.res_bus.at[i, "vm_pu"])
        va = float(net.res_bus.at[i, "va_degree"]) if "va_degree" in net.res_bus.columns else 0.0
        tag = " [slack]" if int(i) == slack_bus else ""
        print(f"Bus {int(i):3d}: V={vm:.6f} pu, theta={va:+.6f} deg{tag}")

    print("\n--- Generator Dispatch ---")
    for eg in net.ext_grid.index:
        bus = int(net.ext_grid.at[eg, "bus"])
        p_mw = float(net.res_ext_grid.at[eg, "p_mw"])
        q_mvar = float(net.res_ext_grid.at[eg, "q_mvar"])
        print(f"ext_grid[{int(eg)}] @ bus {bus:3d}: P={p_mw:.6f} MW, Q={q_mvar:.6f} MVAr")
    for gi in net.gen.index:
        bus = int(net.gen.at[gi, "bus"])
        p_mw = float(net.res_gen.at[gi, "p_mw"])
        q_mvar = float(net.res_gen.at[gi, "q_mvar"])
        print(f"gen[{int(gi)}] @ bus {bus:3d}: P={p_mw:.6f} MW, Q={q_mvar:.6f} MVAr")

    if tap_choice:
        print("\n--- OLTC Selected Taps ---")
        for (u, v) in sorted(tap_choice):
            print(f"OLTC ({int(u)},{int(v)}): tap={int(tap_choice[(u, v)]):>3d}")

    if sh_choice:
        print("\n--- Switched Shunt Status ---")
        for bus in sorted(sh_choice):
            print(f"Shunt @ bus {int(bus):3d}: a_sh={int(sh_choice[bus])}")


def _solve_pf_fallback(
    data: Dict[str, Any],
    source_model: Optional[pyo.ConcreteModel] = None,
    tap_choice: Optional[Dict[Tuple[int, int], int]] = None,
    sh_choice: Optional[Dict[int, int]] = None,
    warm_pf_net=None,
):
    net = _build_ver2_net()

    if tap_choice is None or not tap_choice:
        tap_choice = _tap_choice_from_model(source_model, data)
    if sh_choice is None or not sh_choice:
        sh_choice = _shunt_choice_from_model(source_model, data)

    if not tap_choice:
        tap_choice = _recommended_tap_choice(data)
    if not sh_choice:
        sh_choice = _recommended_shunt_choice(data)

    try:
        _apply_discrete_choice_to_net(net, tap_choice, sh_choice)
    except Exception:
        pass

    try:
        _apply_model_dispatch_to_net(net, source_model, data)
    except Exception:
        pass

    trial, info = _run_pp_attempts(net)
    if trial is not None:
        return {
            "ok": True,
            "net": trial,
            "info": info,
            "pf_cost": _compute_pf_generation_cost(trial),
            "tap_choice": tap_choice,
            "sh_choice": sh_choice,
        }

    if warm_pf_net is not None and bool(getattr(warm_pf_net, "converged", False)):
        return {
            "ok": True,
            "net": warm_pf_net,
            "info": {"algorithm": "nr", "init": "flat", "angles": True, "note": "warm-start PF from base net"},
            "pf_cost": _compute_pf_generation_cost(warm_pf_net),
            "tap_choice": tap_choice,
            "sh_choice": sh_choice,
        }

    bare_net = _build_ver2_net()
    bare_trial, bare_info = _run_pp_attempts(bare_net)
    if bare_trial is not None:
        return {
            "ok": True,
            "net": bare_trial,
            "info": bare_info,
            "pf_cost": _compute_pf_generation_cost(bare_trial),
            "tap_choice": {},
            "sh_choice": {},
        }

    return {
        "ok": False,
        "reason": bare_info.get("error", info.get("error", "PF fallback failed")),
    }


# ----------------------------
# Main
# ----------------------------
def main():
    t_all0 = time.perf_counter()

    # 1) Build ver2 IEEE 300-bus mesh
    net = _build_ver2_net()

    # 2) Read metadata and build config
    meta = read_network_device_metadata(net)
    cfg = BuildConfig(
        oltc_branches=meta["oltc_branches"],
        shunts=meta["shunts"],
        fix_slack_vm=True,
    )

    # 3) Extract data
    data = extract_bfm_fullmesh_data_ver2(net, cfg)

    print(f"[INFO] #buses = {len(data['buses'])}")
    print("[INFO] network = ieee300bus (standalone build kwargs)")
    print(f"[INFO] #branches (directed, aggregated) = {len(data['E'])}")
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
    print("\n[INFO] Solving fixed-discrete NLP (recommended/default taps+shunts) with IPOPT ...")
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
            return

    print("\n[INFO] Switching to guaranteed PF fallback on ieee300bus ...")
    pf_source_model = None
    if _has_complete_primal_solution(model):
        pf_source_model = model
    elif fixed_ok:
        pf_source_model = fixed_nlp

    pf_fb = _solve_pf_fallback(
        data,
        source_model=pf_source_model,
        warm_pf_net=net if pf_ok else None,
    )
    if pf_fb["ok"]:
        _print_pf_solution(
            pf_fb["net"],
            "best-effort PF fallback result",
            pf_fb["pf_cost"],
            pf_fb["info"],
            pf_fb["tap_choice"],
            pf_fb["sh_choice"],
        )
    else:
        print(f"[FAIL] PF fallback also failed: {pf_fb['reason']}")

    t_all1 = time.perf_counter()
    print(f"\n[INFO] Total elapsed time: {t_all1 - t_all0:.2f} seconds")


if __name__ == "__main__":
    main()
