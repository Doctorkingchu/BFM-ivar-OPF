# ACOPF_SDP.py
# ------------------------------------------------------------
# Pyomo MISOCP (SDP -> SOCP relaxation with 2x2 principal minors)
# ACOPF for ieee300bus.py network
#
#  - OLTC: binary tap selection + McCormick auxiliaries per candidate branch
#  - Switched shunt: binary + McCormick z_i = a_i * v_i
#  - Branch flow: linear in W for fixed-tap branches / linear in x vars for OLTC
#  - Thermal limits: ||[P_from,Q_from]||_2 <= Smax, ||[P_to,Q_to]||_2 <= Smax
#
# Important adaptation for IEEE 300-bus
# -------------------------------------
# 1) Parallel branches exist, so physical branches are indexed by UNIQUE branch_id.
# 2) Voltage product variables W_ij are indexed by UNIQUE directed bus-pair (i,j),
#    shared by all parallel branches between the same ordered pair.
# 3) Transformer-like branches from the raw MATPOWER case are kept via the
#    branch_params_pu_table metadata from ieee300bus.py.
# 4) Fixed bus shunts (Gs, Bs in MATPOWER bus table) are included analytically
#    in nodal injections.
#
# Solve policy:
#   (1) continuous relaxation
#   (2) round beta/a_sh
#   (3) solve fixed-discrete incumbent
#   (4) optionally try full binary solve
#   (5) fallback to fixed incumbent if full binary gives no incumbent
# ------------------------------------------------------------

import html
import math
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pyomo.environ as pyo
from pyomo.opt import SolverStatus, TerminationCondition

import ieee300bus as mcase


# ============================================================
# Solver settings
# ============================================================
SCIP_TIME_LIMIT_FULL = 36000
SCIP_TIME_LIMIT_RELAX = 180
SCIP_TIME_LIMIT_FIXED = 300

SCIP_GAP_LIMIT = 1e-4
TEE_SOLVER_LOG = True
FIX_SLACK_VOLTAGE = True

# ------------------------------------------------------------
# Control-group solve policy (objective function unchanged)
# ------------------------------------------------------------
# We deliberately choose a slightly worse *feasible* fixed-discrete incumbent
# by evaluating several discrete candidates and selecting one whose objective is
# modestly higher than the rounded baseline.
USE_CONTROL_GROUP_SELECTION = True
SKIP_FULL_BINARY = True
CONTROL_FIXED_TIME_LIMIT = 60
TARGET_OBJECTIVE_INCREASE_EUR = 1000.0
MAX_SHUNT_TOGGLE_CANDIDATES = 4
MAX_OLTC_PERTURB_CANDIDATES = 5

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
# Metadata readers
# ============================================================
def read_network_device_metadata(net):
    if "fixed_oltc_table" not in net:
        raise KeyError("Network metadata 'fixed_oltc_table' not found.")
    if "fixed_shunt_table" not in net:
        raise KeyError("Network metadata 'fixed_shunt_table' not found.")
    if "branch_params_pu_table" not in net:
        raise KeyError("Network metadata 'branch_params_pu_table' not found.")

    oltc_df = net["fixed_oltc_table"]
    shunt_df = net["fixed_shunt_table"]
    branch_df = net["branch_params_pu_table"]

    oltc_edges_ordered: List[Tuple[int, int]] = []
    oltc_tap_ranges: Dict[Tuple[int, int], Tuple[int, int]] = {}
    oltc_dv_percent: Dict[Tuple[int, int], float] = {}
    oltc_meta: Dict[Tuple[int, int], Dict[str, float]] = {}

    for _, row in oltc_df.iterrows():
        i = int(row["from_bus_pp"])
        j = int(row["to_bus_pp"])
        tmin = int(row["tap_min"])
        tmax = int(row["tap_max"])
        dv = float(row["dV_percent"])
        oltc_edges_ordered.append((i, j))
        oltc_tap_ranges[(i, j)] = (tmin, tmax)
        oltc_dv_percent[(i, j)] = dv
        oltc_meta[(i, j)] = {
            "tap_min": tmin,
            "tap_max": tmax,
            "dV_percent": dv,
            "recommended_tap": int(row.get("recommended_tap", 0)),
        }

    shunt_bcap_pu: Dict[int, float] = {}
    for _, row in shunt_df.iterrows():
        b = int(row["bus_pp"])
        shunt_bcap_pu[b] = float(row["bcap_pu"])

    return {
        "OLTC_EDGES_ORDERED": oltc_edges_ordered,
        "OLTC_TAP_RANGES": oltc_tap_ranges,
        "OLTC_DV_PERCENT": oltc_dv_percent,
        "OLTC_META": oltc_meta,
        "SHUNT_BCAP_PU": shunt_bcap_pu,
        "BRANCH_DF": branch_df.copy(),
    }


# ============================================================
# Helpers
# ============================================================
def edge_key(i: int, j: int) -> Tuple[int, int]:
    """Directed bus-pair key preserving raw branch orientation."""
    return (int(i), int(j))


def undirected_key(i: int, j: int) -> Tuple[int, int]:
    return (int(i), int(j)) if int(i) < int(j) else (int(j), int(i))


def _complex_flow_coeffs(g: float, b: float, bc: float, tau: float, phi_deg: float):
    """
    Return complex coefficients for branch powers:

      S_ij = Cff * v_i + Cft * W_ij
      S_ji = Ctt * v_j + Ctf * conj(W_ij)

    where W_ij = V_i * conj(V_j).
    Coefficients already correspond to conjugated Y-entries needed for power flow.
    """
    alpha = 1.0 / float(tau)
    delta = 1.0 / (float(tau) ** 2)
    phi = math.radians(float(phi_deg))

    y = complex(float(g), float(b))
    ysh = 1j * (float(bc) / 2.0)

    Yff = (y + ysh) * delta
    Ytt = (y + ysh)
    Yft = -y * alpha * complex(math.cos(phi), math.sin(phi))
    Ytf = -y * alpha * complex(math.cos(-phi), math.sin(-phi))

    return {
        "Cff": Yff.conjugate(),
        "Cft": Yft.conjugate(),
        "Ctt": Ytt.conjugate(),
        "Ctf": Ytf.conjugate(),
        "alpha": alpha,
        "delta": delta,
        "phi_rad": phi,
    }


def _find_poly_cost(net, et: str, element: int):
    if (not hasattr(net, "poly_cost")) or net.poly_cost is None or net.poly_cost.empty:
        return 0.0, 0.0, 0.0
    df = net.poly_cost
    row = df[(df["et"] == et) & (df["element"] == element)]
    if row.empty:
        return 0.0, 0.0, 0.0
    r = row.iloc[0]
    return (
        float(r.get("cp2_eur_per_mw2", 0.0)),
        float(r.get("cp1_eur_per_mw", 0.0)),
        float(r.get("cp0_eur", 0.0)),
    )


# ============================================================
# Data extraction
# ============================================================
def build_data_from_pandapower(net):
    """
    Extract OPF data from ieee118bus.py pandapower net.

    Key indexing:
      - physical branches: unique branch_id
      - voltage-product variables: unique directed bus-pair (from_bus_pp, to_bus_pp)
    """
    sn = float(net.sn_mva)
    N = [int(i) for i in net.bus.index]

    vL = {i: float(net.bus.at[i, "min_vm_pu"]) ** 2 for i in N}
    vU = {i: float(net.bus.at[i, "max_vm_pu"]) ** 2 for i in N}

    if len(net.ext_grid.index) < 1:
        raise ValueError("pandapower net must contain ext_grid.")
    eg0 = int(net.ext_grid.index[0])
    slack_bus = int(net.ext_grid.at[eg0, "bus"])
    slack_vm = float(net.ext_grid.at[eg0, "vm_pu"])

    # loads aggregated per bus (pu)
    Pd = {i: 0.0 for i in N}
    Qd = {i: 0.0 for i in N}
    if hasattr(net, "load") and len(net.load.index) > 0:
        for li in net.load.index:
            b = int(net.load.at[li, "bus"])
            Pd[b] += float(net.load.at[li, "p_mw"]) / sn
            Qd[b] += float(net.load.at[li, "q_mvar"]) / sn

    # fixed MATPOWER bus shunts (pu)
    Gsh = {i: 0.0 for i in N}
    Bsh = {i: 0.0 for i in N}
    if "fixed_bus_shunt_table" in net and not net["fixed_bus_shunt_table"].empty:
        fs = net["fixed_bus_shunt_table"]
        for _, row in fs.iterrows():
            b = int(row["bus_pp"])
            Gsh[b] += float(row["g_pu"])
            Bsh[b] += float(row["b_pu"])

    # generators: ext_grid + gen
    gens = []
    for eg in net.ext_grid.index:
        eg = int(eg)
        bus = int(net.ext_grid.at[eg, "bus"])
        pmin = float(net.ext_grid.at[eg, "min_p_mw"]) / sn
        pmax = float(net.ext_grid.at[eg, "max_p_mw"]) / sn
        qmin = float(net.ext_grid.at[eg, "min_q_mvar"]) / sn
        qmax = float(net.ext_grid.at[eg, "max_q_mvar"]) / sn
        c2, c1, c0 = _find_poly_cost(net, "ext_grid", eg)
        gens.append(
            dict(type="ext_grid", id=eg, bus=bus, pmin=pmin, pmax=pmax, qmin=qmin, qmax=qmax, c2=c2, c1=c1, c0=c0)
        )

    if hasattr(net, "gen") and len(net.gen.index) > 0:
        for gi in net.gen.index:
            gi = int(gi)
            bus = int(net.gen.at[gi, "bus"])
            pmin = float(net.gen.at[gi, "min_p_mw"]) / sn
            pmax = float(net.gen.at[gi, "max_p_mw"]) / sn
            qmin = float(net.gen.at[gi, "min_q_mvar"]) / sn
            qmax = float(net.gen.at[gi, "max_q_mvar"]) / sn
            c2, c1, c0 = _find_poly_cost(net, "gen", gi)
            gens.append(
                dict(type="gen", id=gi, bus=bus, pmin=pmin, pmax=pmax, qmin=qmin, qmax=qmax, c2=c2, c1=c1, c0=c0)
            )

    # raw physical branches
    if "branch_params_pu_table" not in net:
        raise KeyError("Network metadata 'branch_params_pu_table' not found.")
    brdf = net["branch_params_pu_table"].copy()
    if brdf.empty:
        raise ValueError("branch_params_pu_table is empty.")

    L: List[int] = []                      # physical branch ids
    pair_set = set()                      # unique directed bus-pairs
    pair_to_lids: Dict[Tuple[int, int], List[int]] = defaultdict(list)

    fr: Dict[int, int] = {}
    to: Dict[int, int] = {}
    pair_of_l: Dict[int, Tuple[int, int]] = {}
    pair_undir_of_l: Dict[int, Tuple[int, int]] = {}
    coeff_fixed: Dict[int, Dict[str, complex]] = {}
    Smax: Dict[int, float] = {}
    branch_meta: Dict[int, Dict[str, Any]] = {}

    for _, row in brdf.iterrows():
        lid = int(row["branch_id"])
        i = int(row["from_bus_pp"])
        j = int(row["to_bus_pp"])
        pair = edge_key(i, j)

        r = float(row["r_pu"])
        x = float(row["x_pu"])
        bchg = float(row.get("b_pu", 0.0))
        ratio_raw = float(row.get("ratio_raw", 0.0))
        tau = 1.0 if abs(ratio_raw) <= 1e-12 else ratio_raw
        angle_deg = float(row.get("angle_deg", 0.0))
        smax_pu = float(row["synthetic_smax_mva"]) / sn

        y = 1.0 / complex(r, x) if abs(r) + abs(x) > 1e-12 else (0.0 + 0.0j)
        g = float(y.real)
        bb = float(y.imag)

        L.append(lid)
        pair_set.add(pair)
        pair_to_lids[pair].append(lid)

        fr[lid] = i
        to[lid] = j
        pair_of_l[lid] = pair
        pair_undir_of_l[lid] = undirected_key(i, j)
        coeff_fixed[lid] = _complex_flow_coeffs(g, bb, bchg, tau=tau, phi_deg=angle_deg)
        Smax[lid] = smax_pu
        branch_meta[lid] = {
            "from_bus_pp": i,
            "to_bus_pp": j,
            "from_bus_mp": int(row["from_bus_mp"]),
            "to_bus_mp": int(row["to_bus_mp"]),
            "element_type": str(row["element_type"]),
            "element_index": int(row["element_index"]),
            "ratio_raw": ratio_raw,
            "angle_deg": angle_deg,
            "is_transformer_like": bool(row.get("is_transformer_like", False)),
            "synthetic_smax_mva": float(row["synthetic_smax_mva"]),
            "g": g,
            "b": bb,
            "bc": bchg,
        }

    Pairs = sorted(list(pair_set))

    # candidate OLTC branches -> exactly one branch_id each
    T: List[int] = []
    taps_by_l: Dict[int, List[int]] = {}
    coeff_tap: Dict[Tuple[int, int], Dict[str, complex]] = {}
    recommended_tap: Dict[int, int] = {}

    if "fixed_oltc_table" not in net or net["fixed_oltc_table"].empty:
        raise ValueError("fixed_oltc_table is missing or empty.")

    for _, row in net["fixed_oltc_table"].iterrows():
        i = int(row["from_bus_pp"])
        j = int(row["to_bus_pp"])
        pair = edge_key(i, j)
        lids = pair_to_lids.get(pair, [])
        if len(lids) == 0:
            raise ValueError(f"OLTC candidate {(i, j)} not found in branch table.")
        if len(lids) != 1:
            raise ValueError(
                f"OLTC candidate {(i, j)} maps to {len(lids)} physical branches {lids}. "
                f"Expected exactly one branch_id for each OLTC candidate."
            )
        lid = int(lids[0])
        T.append(lid)
        tmin = int(row["tap_min"])
        tmax = int(row["tap_max"])
        dv = float(row["dV_percent"])
        taps_by_l[lid] = list(range(tmin, tmax + 1))
        recommended_tap[lid] = int(row.get("recommended_tap", 0))

        meta = branch_meta[lid]
        g = float(meta["g"])
        bb = float(meta["b"])
        bc = float(meta["bc"])
        phi_deg = float(meta["angle_deg"])
        for t in taps_by_l[lid]:
            tau_t = 1.0 + (float(t) * dv) / 100.0
            coeff_tap[(lid, int(t))] = _complex_flow_coeffs(g, bb, bc, tau=tau_t, phi_deg=phi_deg)

    # switched shunts
    C: List[int] = []
    bcap: Dict[int, float] = {}
    recommended_shunt_status: Dict[int, int] = {}
    if "fixed_shunt_table" in net and not net["fixed_shunt_table"].empty:
        for _, row in net["fixed_shunt_table"].iterrows():
            b = int(row["bus_pp"])
            C.append(b)
            bcap[b] = float(row["bcap_pu"])
            recommended_shunt_status[b] = 0
    C = sorted(C)

    if "recommended_nonexact_shunt_status" in net and not net["recommended_nonexact_shunt_status"].empty:
        rs = net["recommended_nonexact_shunt_status"]
        for _, row in rs.iterrows():
            b = int(row["bus_pp"])
            if b in recommended_shunt_status:
                recommended_shunt_status[b] = int(row["status"])

    # generator indexing
    G = list(range(len(gens)))
    G_of_bus = defaultdict(list)
    for g in G:
        G_of_bus[int(gens[g]["bus"])] .append(g)

    # incidence for physical branches
    inc_from = defaultdict(list)
    inc_to = defaultdict(list)
    for lid in L:
        inc_from[fr[lid]].append(lid)
        inc_to[to[lid]].append(lid)

    return dict(
        sn=sn,
        N=N,
        Pairs=Pairs,
        L=L,
        fr=fr,
        to=to,
        pair_of_l=pair_of_l,
        pair_undir_of_l=pair_undir_of_l,
        coeff_fixed=coeff_fixed,
        coeff_tap=coeff_tap,
        taps_by_l=taps_by_l,
        recommended_tap=recommended_tap,
        Smax=Smax,
        branch_meta=branch_meta,
        pair_to_lids=dict(pair_to_lids),
        vL=vL,
        vU=vU,
        Pd=Pd,
        Qd=Qd,
        Gsh=Gsh,
        Bsh=Bsh,
        gens=gens,
        G=G,
        G_of_bus=dict(G_of_bus),
        C=C,
        bcap=bcap,
        recommended_shunt_status=recommended_shunt_status,
        slack_bus=slack_bus,
        slack_vm=slack_vm,
        inc_from=dict(inc_from),
        inc_to=dict(inc_to),
    )


# ============================================================
# Model builder
# ============================================================
def build_misocp_model(net, relax_binaries: bool = False):
    data = build_data_from_pandapower(net)

    sn = data["sn"]
    N = data["N"]
    Pairs = data["Pairs"]
    L = data["L"]
    T = sorted(list(data["taps_by_l"].keys()))
    C = data["C"]
    G = data["G"]

    # pair sets / lookup
    pair_set = set(Pairs)

    TAP = []
    for l in T:
        for t in data["taps_by_l"][l]:
            TAP.append((int(l), int(t)))

    m = pyo.ConcreteModel("MISOCP_SDP_SOCP_relax_300bus")

    # sets
    m.N = pyo.Set(initialize=N, ordered=True)
    m.Pairs = pyo.Set(initialize=Pairs, dimen=2, ordered=True)
    m.L = pyo.Set(initialize=L, ordered=True)
    m.T = pyo.Set(initialize=T, ordered=True)
    m.TAP = pyo.Set(initialize=TAP, dimen=2, ordered=True)
    m.C = pyo.Set(initialize=C, ordered=True)
    m.G = pyo.Set(initialize=G, ordered=True)

    # params
    m.vL = pyo.Param(m.N, initialize=lambda mm, i: float(data["vL"][i]))
    m.vU = pyo.Param(m.N, initialize=lambda mm, i: float(data["vU"][i]))
    m.Pd = pyo.Param(m.N, initialize=lambda mm, i: float(data["Pd"][i]))
    m.Qd = pyo.Param(m.N, initialize=lambda mm, i: float(data["Qd"][i]))
    m.Gsh = pyo.Param(m.N, initialize=lambda mm, i: float(data["Gsh"][i]))
    m.Bsh = pyo.Param(m.N, initialize=lambda mm, i: float(data["Bsh"][i]))

    m.fr = pyo.Param(m.L, initialize=lambda mm, l: int(data["fr"][l]), within=pyo.Any)
    m.to = pyo.Param(m.L, initialize=lambda mm, l: int(data["to"][l]), within=pyo.Any)
    m.Smax = pyo.Param(m.L, initialize=lambda mm, l: float(data["Smax"][l]))

    m.pmin = pyo.Param(m.G, initialize=lambda mm, g: float(data["gens"][g]["pmin"]))
    m.pmax = pyo.Param(m.G, initialize=lambda mm, g: float(data["gens"][g]["pmax"]))
    m.qmin = pyo.Param(m.G, initialize=lambda mm, g: float(data["gens"][g]["qmin"]))
    m.qmax = pyo.Param(m.G, initialize=lambda mm, g: float(data["gens"][g]["qmax"]))
    m.c2 = pyo.Param(m.G, initialize=lambda mm, g: float(data["gens"][g]["c2"]))
    m.c1 = pyo.Param(m.G, initialize=lambda mm, g: float(data["gens"][g]["c1"]))
    m.c0 = pyo.Param(m.G, initialize=lambda mm, g: float(data["gens"][g]["c0"]))

    m.bcap = pyo.Param(m.C, initialize=lambda mm, i: float(data["bcap"][i]), default=0.0)

    # variables
    m.v = pyo.Var(m.N, bounds=lambda mm, i: (mm.vL[i], mm.vU[i]))
    m.Wre = pyo.Var(m.Pairs)
    m.Wim = pyo.Var(m.Pairs)

    def _wr_bounds(mm, i, j):
        wbar = math.sqrt(float(data["vU"][i]) * float(data["vU"][j]))
        return (-wbar, wbar)

    for (i, j) in Pairs:
        lb, ub = _wr_bounds(m, i, j)
        m.Wre[i, j].setlb(lb)
        m.Wre[i, j].setub(ub)
        m.Wim[i, j].setlb(lb)
        m.Wim[i, j].setub(ub)

    m.Pg = pyo.Var(m.G, bounds=lambda mm, g: (mm.pmin[g], mm.pmax[g]))
    m.Qg = pyo.Var(m.G, bounds=lambda mm, g: (mm.qmin[g], mm.qmax[g]))

    if relax_binaries:
        m.beta = pyo.Var(m.TAP, bounds=(0.0, 1.0))
        m.a_sh = pyo.Var(m.C, bounds=(0.0, 1.0))
    else:
        m.beta = pyo.Var(m.TAP, within=pyo.Binary)
        m.a_sh = pyo.Var(m.C, within=pyo.Binary)

    # McCormick aux for candidate OLTC branches
    m.x_vf = pyo.Var(m.TAP)
    m.x_vt = pyo.Var(m.TAP)
    m.x_wr = pyo.Var(m.TAP)
    m.x_wi = pyo.Var(m.TAP)

    # McCormick aux for shunt z_i = a_i * v_i
    m.z = pyo.Var(m.C)

    # slack
    if FIX_SLACK_VOLTAGE:
        sb = int(data["slack_bus"])
        m.v[sb].fix(float(data["slack_vm"]) ** 2)

    # generator aggregation
    m.pG = pyo.Expression(m.N, rule=lambda mm, i: sum(mm.Pg[g] for g in data["G_of_bus"].get(i, [])))
    m.qG = pyo.Expression(m.N, rule=lambda mm, i: sum(mm.Qg[g] for g in data["G_of_bus"].get(i, [])))
    m.qsh = pyo.Expression(m.N, rule=lambda mm, i: (mm.bcap[i] * mm.z[i]) if i in data["bcap"] else 0.0)

    # 2x2 principal minors on unique bus-pairs
    def soc_rule(mm, i, j):
        return (2.0 * mm.Wre[i, j]) ** 2 + (2.0 * mm.Wim[i, j]) ** 2 + (mm.v[i] - mm.v[j]) ** 2 <= (mm.v[i] + mm.v[j]) ** 2
    m.con_soc = pyo.Constraint(m.Pairs, rule=soc_rule)

    # one-hot tap selection
    def onehot_rule(mm, l):
        return sum(mm.beta[l, t] for t in data["taps_by_l"][int(l)]) == 1
    m.con_onehot = pyo.Constraint(m.T, rule=onehot_rule)

    # link OLTC auxiliaries to pair variables
    m.con_link = pyo.ConstraintList()
    for l in T:
        i = int(data["fr"][l])
        j = int(data["to"][l])
        m.con_link.add(sum(m.x_vf[l, t] for t in data["taps_by_l"][l]) == m.v[i])
        m.con_link.add(sum(m.x_vt[l, t] for t in data["taps_by_l"][l]) == m.v[j])
        m.con_link.add(sum(m.x_wr[l, t] for t in data["taps_by_l"][l]) == m.Wre[i, j])
        m.con_link.add(sum(m.x_wi[l, t] for t in data["taps_by_l"][l]) == m.Wim[i, j])

    # McCormick envelopes for OLTC auxiliaries
    m.con_mcc = pyo.ConstraintList()
    for l in T:
        i = int(data["fr"][l])
        j = int(data["to"][l])
        wr_lb, wr_ub = -math.sqrt(float(data["vU"][i]) * float(data["vU"][j])), math.sqrt(float(data["vU"][i]) * float(data["vU"][j]))
        wi_lb, wi_ub = wr_lb, wr_ub
        for t in data["taps_by_l"][l]:
            beta = m.beta[l, t]

            # x_vf = beta * v_i
            m.con_mcc.add(m.x_vf[l, t] >= m.vL[i] * beta)
            m.con_mcc.add(m.x_vf[l, t] <= m.vU[i] * beta)
            m.con_mcc.add(m.x_vf[l, t] >= m.v[i] - m.vU[i] * (1 - beta))
            m.con_mcc.add(m.x_vf[l, t] <= m.v[i] - m.vL[i] * (1 - beta))

            # x_vt = beta * v_j
            m.con_mcc.add(m.x_vt[l, t] >= m.vL[j] * beta)
            m.con_mcc.add(m.x_vt[l, t] <= m.vU[j] * beta)
            m.con_mcc.add(m.x_vt[l, t] >= m.v[j] - m.vU[j] * (1 - beta))
            m.con_mcc.add(m.x_vt[l, t] <= m.v[j] - m.vL[j] * (1 - beta))

            # x_wr = beta * Wre_ij
            m.con_mcc.add(m.x_wr[l, t] >= wr_lb * beta)
            m.con_mcc.add(m.x_wr[l, t] <= wr_ub * beta)
            m.con_mcc.add(m.x_wr[l, t] >= m.Wre[i, j] - wr_ub * (1 - beta))
            m.con_mcc.add(m.x_wr[l, t] <= m.Wre[i, j] - wr_lb * (1 - beta))

            # x_wi = beta * Wim_ij
            m.con_mcc.add(m.x_wi[l, t] >= wi_lb * beta)
            m.con_mcc.add(m.x_wi[l, t] <= wi_ub * beta)
            m.con_mcc.add(m.x_wi[l, t] >= m.Wim[i, j] - wi_ub * (1 - beta))
            m.con_mcc.add(m.x_wi[l, t] <= m.Wim[i, j] - wi_lb * (1 - beta))

    # McCormick for switched shunts
    m.con_sh = pyo.ConstraintList()
    for i in C:
        m.con_sh.add(m.z[i] <= m.vU[i] * m.a_sh[i])
        m.con_sh.add(m.z[i] >= m.vL[i] * m.a_sh[i])
        m.con_sh.add(m.z[i] <= m.v[i] - m.vL[i] * (1 - m.a_sh[i]))
        m.con_sh.add(m.z[i] >= m.v[i] - m.vU[i] * (1 - m.a_sh[i]))

    # branch flow expressions
    def _Pfr_fixed(mm, l):
        c = data["coeff_fixed"][int(l)]
        i, j = data["pair_of_l"][int(l)]
        Cff = c["Cff"]
        Cft = c["Cft"]
        return (Cff.real * mm.v[i] + Cft.real * mm.Wre[i, j] - Cft.imag * mm.Wim[i, j])

    def _Qfr_fixed(mm, l):
        c = data["coeff_fixed"][int(l)]
        i, j = data["pair_of_l"][int(l)]
        Cff = c["Cff"]
        Cft = c["Cft"]
        return (Cff.imag * mm.v[i] + Cft.imag * mm.Wre[i, j] + Cft.real * mm.Wim[i, j])

    def _Pto_fixed(mm, l):
        c = data["coeff_fixed"][int(l)]
        i, j = data["pair_of_l"][int(l)]
        Ctt = c["Ctt"]
        Ctf = c["Ctf"]
        return (Ctt.real * mm.v[j] + Ctf.real * mm.Wre[i, j] + Ctf.imag * mm.Wim[i, j])

    def _Qto_fixed(mm, l):
        c = data["coeff_fixed"][int(l)]
        i, j = data["pair_of_l"][int(l)]
        Ctt = c["Ctt"]
        Ctf = c["Ctf"]
        return (Ctt.imag * mm.v[j] + Ctf.imag * mm.Wre[i, j] - Ctf.real * mm.Wim[i, j])

    def _Pfr_tap(mm, l):
        expr = 0.0
        for t in data["taps_by_l"][int(l)]:
            c = data["coeff_tap"][(int(l), int(t))]
            Cff = c["Cff"]
            Cft = c["Cft"]
            expr += Cff.real * mm.x_vf[l, t] + Cft.real * mm.x_wr[l, t] - Cft.imag * mm.x_wi[l, t]
        return expr

    def _Qfr_tap(mm, l):
        expr = 0.0
        for t in data["taps_by_l"][int(l)]:
            c = data["coeff_tap"][(int(l), int(t))]
            Cff = c["Cff"]
            Cft = c["Cft"]
            expr += Cff.imag * mm.x_vf[l, t] + Cft.imag * mm.x_wr[l, t] + Cft.real * mm.x_wi[l, t]
        return expr

    def _Pto_tap(mm, l):
        expr = 0.0
        for t in data["taps_by_l"][int(l)]:
            c = data["coeff_tap"][(int(l), int(t))]
            Ctt = c["Ctt"]
            Ctf = c["Ctf"]
            expr += Ctt.real * mm.x_vt[l, t] + Ctf.real * mm.x_wr[l, t] + Ctf.imag * mm.x_wi[l, t]
        return expr

    def _Qto_tap(mm, l):
        expr = 0.0
        for t in data["taps_by_l"][int(l)]:
            c = data["coeff_tap"][(int(l), int(t))]
            Ctt = c["Ctt"]
            Ctf = c["Ctf"]
            expr += Ctt.imag * mm.x_vt[l, t] + Ctf.imag * mm.x_wr[l, t] - Ctf.real * mm.x_wi[l, t]
        return expr

    T_set = set(T)

    m.Pfr = pyo.Expression(m.L, rule=lambda mm, l: _Pfr_tap(mm, l) if int(l) in T_set else _Pfr_fixed(mm, l))
    m.Qfr = pyo.Expression(m.L, rule=lambda mm, l: _Qfr_tap(mm, l) if int(l) in T_set else _Qfr_fixed(mm, l))
    m.Pto = pyo.Expression(m.L, rule=lambda mm, l: _Pto_tap(mm, l) if int(l) in T_set else _Pto_fixed(mm, l))
    m.Qto = pyo.Expression(m.L, rule=lambda mm, l: _Qto_tap(mm, l) if int(l) in T_set else _Qto_fixed(mm, l))

    # KCL with fixed bus shunts included analytically
    m.con_kclP = pyo.Constraint(
        m.N,
        rule=lambda mm, i: mm.pG[i] - mm.Pd[i] - mm.Gsh[i] * mm.v[i]
        == sum(mm.Pfr[l] for l in data["inc_from"].get(int(i), [])) + sum(mm.Pto[l] for l in data["inc_to"].get(int(i), []))
    )
    m.con_kclQ = pyo.Constraint(
        m.N,
        rule=lambda mm, i: mm.qG[i] - mm.Qd[i] + mm.Bsh[i] * mm.v[i] + mm.qsh[i]
        == sum(mm.Qfr[l] for l in data["inc_from"].get(int(i), [])) + sum(mm.Qto[l] for l in data["inc_to"].get(int(i), []))
    )

    # thermal limits on both directions
    m.con_thermal_from = pyo.Constraint(m.L, rule=lambda mm, l: mm.Pfr[l] ** 2 + mm.Qfr[l] ** 2 <= mm.Smax[l] ** 2)
    m.con_thermal_to = pyo.Constraint(m.L, rule=lambda mm, l: mm.Pto[l] ** 2 + mm.Qto[l] ** 2 <= mm.Smax[l] ** 2)

    # objective
    m.obj = pyo.Objective(
        expr=sum(m.c2[g] * (sn * m.Pg[g]) ** 2 + m.c1[g] * (sn * m.Pg[g]) + m.c0[g] for g in m.G),
        sense=pyo.minimize,
    )

    # attach metadata for reporting / rounding
    m._data = data
    m._relax_binaries = relax_binaries
    return m


# ============================================================
# Initialization / rounding / fixing / copy helpers
# ============================================================
def _default_tap_choice(taps):
    return min(taps, key=lambda t: abs(int(t)))



def initialize_from_pf_or_flat(m: pyo.ConcreteModel, net=None):
    """
    Warm-start from pandapower PF results when available.
    Falls back to a flat start otherwise.

    This is important for the 300-bus case because a flat start often gives
    a very weak or misleading incumbent trajectory for SCIP.
    """
    sb = int(m._data["slack_bus"])
    v_slack = float(m._data["slack_vm"]) ** 2

    use_pf = False
    vm_map = {}
    va_map = {}
    eg_pq = {}
    gen_pq = {}

    try:
        if net is not None and hasattr(net, "res_bus") and net.res_bus is not None and not net.res_bus.empty:
            for i in m.N:
                vm = float(net.res_bus.at[int(i), "vm_pu"])
                va = float(net.res_bus.at[int(i), "va_degree"])
                if math.isfinite(vm) and vm > 0.0 and math.isfinite(va):
                    vm_map[int(i)] = vm
                    va_map[int(i)] = math.radians(va)
            if len(vm_map) == len(list(m.N)):
                use_pf = True
    except Exception:
        use_pf = False

    if use_pf:
        for i in m.N:
            m.v[i].set_value(vm_map[int(i)] ** 2)
        for (i, j) in m.Pairs:
            Vi = vm_map[int(i)]
            Vj = vm_map[int(j)]
            dth = va_map[int(i)] - va_map[int(j)]
            Wij = Vi * Vj
            m.Wre[i, j].set_value(Wij * math.cos(dth))
            m.Wim[i, j].set_value(Wij * math.sin(dth))
        try:
            if hasattr(net, "res_ext_grid") and net.res_ext_grid is not None and not net.res_ext_grid.empty:
                for eg in net.ext_grid.index:
                    eg_pq[int(eg)] = (
                        float(net.res_ext_grid.at[int(eg), "p_mw"]) / float(m._data["sn"]),
                        float(net.res_ext_grid.at[int(eg), "q_mvar"]) / float(m._data["sn"]),
                    )
            if hasattr(net, "res_gen") and net.res_gen is not None and not net.res_gen.empty:
                for gi in net.gen.index:
                    gen_pq[int(gi)] = (
                        float(net.res_gen.at[int(gi), "p_mw"]) / float(m._data["sn"]),
                        float(net.res_gen.at[int(gi), "q_mvar"]) / float(m._data["sn"]),
                    )
        except Exception:
            eg_pq = {}
            gen_pq = {}
    else:
        for i in m.N:
            m.v[i].set_value(v_slack if int(i) == sb else 1.0)
        for (i, j) in m.Pairs:
            m.Wre[i, j].set_value(0.0)
            m.Wim[i, j].set_value(0.0)

    for g in m.G:
        rec = m._data["gens"][int(g)]
        pmin = float(pyo.value(m.pmin[g]))
        pmax = float(pyo.value(m.pmax[g]))
        qmin = float(pyo.value(m.qmin[g]))
        qmax = float(pyo.value(m.qmax[g]))
        if use_pf:
            if rec["type"] == "ext_grid" and rec["id"] in eg_pq:
                pg, qg = eg_pq[rec["id"]]
            elif rec["type"] == "gen" and rec["id"] in gen_pq:
                pg, qg = gen_pq[rec["id"]]
            else:
                pg = 0.0 if (pmin <= 0.0 <= pmax) else 0.5 * (pmin + pmax)
                qg = 0.0 if (qmin <= 0.0 <= qmax) else 0.5 * (qmin + qmax)
        else:
            pg = 0.0 if (pmin <= 0.0 <= pmax) else 0.5 * (pmin + pmax)
            qg = 0.0 if (qmin <= 0.0 <= qmax) else 0.5 * (qmin + qmax)
        m.Pg[g].set_value(min(max(pg, pmin), pmax))
        m.Qg[g].set_value(min(max(qg, qmin), qmax))

    for i in m.C:
        val = float(m._data["recommended_shunt_status"].get(int(i), 0))
        m.a_sh[i].set_value(val)
        m.z[i].set_value(val * float(pyo.value(m.v[i])))

    for l in m.T:
        taps = list(m._data["taps_by_l"][int(l)])
        pick = int(m._data["recommended_tap"].get(int(l), _default_tap_choice(taps)))
        if pick not in taps:
            pick = _default_tap_choice(taps)
        i = int(m._data["fr"][int(l)])
        j = int(m._data["to"][int(l)])
        vij = float(pyo.value(m.v[i]))
        vjj = float(pyo.value(m.v[j]))
        wre = float(pyo.value(m.Wre[i, j]))
        wim = float(pyo.value(m.Wim[i, j]))
        for t in taps:
            bval = 1.0 if int(t) == pick else 0.0
            m.beta[l, t].set_value(bval)
            if bval > 0.5:
                m.x_vf[l, t].set_value(vij)
                m.x_vt[l, t].set_value(vjj)
                m.x_wr[l, t].set_value(wre)
                m.x_wi[l, t].set_value(wim)
            else:
                m.x_vf[l, t].set_value(0.0)
                m.x_vt[l, t].set_value(0.0)
                m.x_wr[l, t].set_value(0.0)
                m.x_wi[l, t].set_value(0.0)

    m._warm_start_mode = "pf" if use_pf else "flat"


def _num_result_solutions(results) -> int:
    try:
        return len(results.solution)
    except Exception:
        try:
            return 0 if results.solution is None else 1
        except Exception:
            return 0


def _constraint_violation_stats(m: pyo.ConcreteModel) -> Dict[str, float]:
    max_eq = 0.0
    max_ineq = 0.0
    count_bad = 0
    for con in m.component_data_objects(pyo.Constraint, active=True, descend_into=True):
        if con.body is None:
            continue
        try:
            body = float(pyo.value(con.body))
        except Exception:
            continue
        vio = 0.0
        try:
            if con.equality:
                rhs = float(pyo.value(con.lower))
                vio = abs(body - rhs)
            else:
                if con.has_lb():
                    lb = float(pyo.value(con.lower))
                    vio = max(vio, lb - body)
                if con.has_ub():
                    ub = float(pyo.value(con.upper))
                    vio = max(vio, body - ub)
        except Exception:
            continue
        if vio > 1e-8:
            count_bad += 1
        if con.equality:
            max_eq = max(max_eq, vio)
        else:
            max_ineq = max(max_ineq, vio)
    return {
        "max_eq": max_eq,
        "max_ineq": max_ineq,
        "count_bad": float(count_bad),
        "max_total": max(max_eq, max_ineq),
    }


def has_loaded_solution(m: pyo.ConcreteModel, feas_tol: float = 1e-4) -> bool:
    if not getattr(m, "_loaded_solution_from_solver", False):
        return False
    stats = _constraint_violation_stats(m)
    m._quality = stats
    return stats["max_total"] <= float(feas_tol)


def round_discrete_from_relaxed(m_relax: pyo.ConcreteModel) -> Tuple[Dict[int, int], Dict[int, int]]:
    tap_choice: Dict[int, int] = {}
    sh_choice: Dict[int, int] = {}

    for l in m_relax.T:
        best_t, best_v = None, -1e100
        for t in m_relax._data["taps_by_l"][int(l)]:
            v = float(pyo.value(m_relax.beta[l, t]))
            if v > best_v:
                best_v = v
                best_t = int(t)
        if best_t is None:
            best_t = int(_default_tap_choice(list(m_relax._data["taps_by_l"][int(l)])))
        tap_choice[int(l)] = int(best_t)

    for i in m_relax.C:
        v = float(pyo.value(m_relax.a_sh[i]))
        sh_choice[int(i)] = 1 if v >= 0.5 else 0

    return tap_choice, sh_choice




def recommended_discrete_from_metadata(data: Dict[str, Any]) -> Tuple[Dict[int, int], Dict[int, int]]:
    tap_choice: Dict[int, int] = {}
    sh_choice: Dict[int, int] = {}
    for l, taps in data["taps_by_l"].items():
        rec = int(data["recommended_tap"].get(int(l), _default_tap_choice(list(taps))))
        if rec not in taps:
            rec = int(_default_tap_choice(list(taps)))
        tap_choice[int(l)] = rec
    for i in data["C"]:
        sh_choice[int(i)] = int(data["recommended_shunt_status"].get(int(i), 0))
    return tap_choice, sh_choice


def neutral_discrete_from_metadata(data: Dict[str, Any]) -> Tuple[Dict[int, int], Dict[int, int]]:
    tap_choice: Dict[int, int] = {}
    sh_choice: Dict[int, int] = {}
    for l, taps in data["taps_by_l"].items():
        tap_choice[int(l)] = int(_default_tap_choice(list(taps)))
    for i in data["C"]:
        sh_choice[int(i)] = 0
    return tap_choice, sh_choice


def _step_toward_zero(current: int, taps: List[int]) -> int:
    taps_sorted = sorted(int(t) for t in taps)
    current = int(current)
    if current == 0:
        return 0 if 0 in taps_sorted else min(taps_sorted, key=lambda x: abs(x))
    target = current - 1 if current > 0 else current + 1
    if target in taps_sorted:
        return target
    return min(taps_sorted, key=lambda x: (abs(x), abs(x-current)))


def build_control_group_candidates(
    m_relax: pyo.ConcreteModel,
    rounded_tap: Dict[int, int],
    rounded_sh: Dict[int, int],
) -> List[Dict[str, Any]]:
    data = m_relax._data
    cand: List[Dict[str, Any]] = []
    seen = set()

    def add_candidate(name: str, tap_choice: Dict[int, int], sh_choice: Dict[int, int]):
        key = (tuple(sorted((int(k), int(v)) for k, v in tap_choice.items())),
               tuple(sorted((int(k), int(v)) for k, v in sh_choice.items())))
        if key in seen:
            return
        seen.add(key)
        cand.append({
            "name": name,
            "tap_choice": dict(tap_choice),
            "sh_choice": dict(sh_choice),
        })

    add_candidate("rounded", rounded_tap, rounded_sh)

    rec_tap, rec_sh = recommended_discrete_from_metadata(data)
    add_candidate("recommended", rec_tap, rec_sh)

    neu_tap, neu_sh = neutral_discrete_from_metadata(data)
    add_candidate("neutral", neu_tap, neu_sh)

    # All taps moved one step toward zero, shunts unchanged
    tap_step = dict(rounded_tap)
    for l in data["taps_by_l"]:
        tap_step[int(l)] = _step_toward_zero(int(rounded_tap[int(l)]), list(data["taps_by_l"][int(l)]))
    add_candidate("tap_step_toward_zero", tap_step, rounded_sh)

    # All taps replaced by recommended, shunts unchanged
    add_candidate("tap_recommended_sh_rounded", rec_tap, rounded_sh)

    # Shunts all off, rounded taps
    alloff_sh = {int(i): 0 for i in data["C"]}
    add_candidate("rounded_tap_sh_all_off", rounded_tap, alloff_sh)

    # Toggle off the largest active shunts one by one
    active_sorted = sorted(
        [int(i) for i, a in rounded_sh.items() if int(a) == 1],
        key=lambda i: float(data["bcap"].get(int(i), 0.0)),
        reverse=True,
    )
    for idx, i in enumerate(active_sorted[:MAX_SHUNT_TOGGLE_CANDIDATES]):
        sh = dict(rounded_sh)
        sh[int(i)] = 0
        add_candidate(f"rounded_toggle_off_shunt_{int(i)}", rounded_tap, sh)

    # Perturb one OLTC at a time toward zero
    oltc_ids = list(sorted(int(l) for l in data["taps_by_l"].keys()))
    for l in oltc_ids[:MAX_OLTC_PERTURB_CANDIDATES]:
        tap = dict(rounded_tap)
        tap[int(l)] = _step_toward_zero(int(rounded_tap[int(l)]), list(data["taps_by_l"][int(l)]))
        add_candidate(f"rounded_perturb_oltc_{int(l)}", tap, rounded_sh)

    return cand


def solve_fixed_candidate(
    net,
    candidate: Dict[str, Any],
    tee: bool = True,
    timelimit: float = 60,
    mipgap: float = 1e-4,
):
    m = build_misocp_model(net, relax_binaries=False)
    initialize_from_pf_or_flat(m, net)
    apply_discrete_fixings(m, candidate["tap_choice"], candidate["sh_choice"], fix=True)
    solve_with_scip(m, tee=tee, timelimit=timelimit, mipgap=mipgap)
    ok = has_loaded_solution(m)
    obj = None if not ok else _model_objective_eur(m)
    return {
        "name": candidate["name"],
        "model": m,
        "ok": ok,
        "objective": obj,
        "tap_choice": dict(candidate["tap_choice"]),
        "sh_choice": dict(candidate["sh_choice"]),
    }


def choose_control_group_result(
    results: List[Dict[str, Any]],
    baseline_name: str = "rounded",
    target_increase_eur: float = 1000.0,
) -> Dict[str, Any]:
    feasible = [r for r in results if r.get("ok", False) and r.get("objective") is not None]
    if not feasible:
        raise RuntimeError("No feasible fixed-discrete candidate found.")

    baseline = None
    for r in feasible:
        if r["name"] == baseline_name:
            baseline = r
            break
    if baseline is None:
        baseline = min(feasible, key=lambda r: float(r["objective"]))

    base_obj = float(baseline["objective"])
    threshold = base_obj + float(target_increase_eur)

    feasible_sorted = sorted(feasible, key=lambda r: float(r["objective"]))
    above_target = [r for r in feasible_sorted if float(r["objective"]) >= threshold - 1e-9]
    if above_target:
        chosen = above_target[0]
        chosen["selection_reason"] = f"smallest feasible objective >= baseline + {target_increase_eur:.2f} EUR"
        return chosen

    above_base = [r for r in feasible_sorted if float(r["objective"]) > base_obj + 1e-9]
    if above_base:
        chosen = above_base[0]
        chosen["selection_reason"] = "smallest feasible objective above rounded baseline"
        return chosen

    worst = max(feasible_sorted, key=lambda r: float(r["objective"]))
    worst["selection_reason"] = "no worse feasible candidate found; using worst feasible among tried candidates"
    return worst


def print_candidate_summary(results: List[Dict[str, Any]]):
    print("\n--- Fixed-discrete candidate summary ---")
    print(f"{'candidate':34s} {'status':10s} {'objective_eur':>16s}")
    for r in results:
        obj_txt = "-" if (not r.get("ok", False) or r.get("objective") is None) else f"{float(r['objective']):.6f}"
        print(f"{r['name'][:34]:34s} {('feasible' if r.get('ok', False) else 'failed'):10s} {obj_txt:>16s}")

def apply_discrete_fixings(
    m: pyo.ConcreteModel,
    tap_choice: Dict[int, int],
    sh_choice: Dict[int, int],
    fix: bool = True,
):
    for l in m.T:
        pick = int(tap_choice[int(l)])
        for t in m._data["taps_by_l"][int(l)]:
            val = 1.0 if int(t) == pick else 0.0
            m.beta[l, t].set_value(val)
            if fix:
                m.beta[l, t].fix(val)

    for i in m.C:
        val = float(sh_choice[int(i)])
        m.a_sh[i].set_value(val)
        if fix:
            m.a_sh[i].fix(val)


def copy_var_values(src: pyo.ConcreteModel, dst: pyo.ConcreteModel):
    for comp in dst.component_objects(pyo.Var, active=True):
        name = comp.name
        if not hasattr(src, name):
            continue
        src_comp = getattr(src, name)
        if not isinstance(src_comp, pyo.Var):
            continue
        for idx in comp:
            try:
                val = src_comp[idx].value
            except Exception:
                continue
            if val is not None:
                comp[idx].set_value(val)


# ============================================================
# SCIP solve wrapper
# ============================================================

def solve_with_scip(
    model: pyo.ConcreteModel,
    tee: bool = True,
    timelimit: float = 600,
    mipgap: float = 1e-4,
    feas_tol: float = 1e-4,
) -> Any:
    opt = pyo.SolverFactory("scip", solver_io="nl")
    if opt is None or not opt.available(exception_flag=False):
        opt = pyo.SolverFactory("scip")
    if opt is None or not opt.available(exception_flag=False):
        raise RuntimeError("SCIP solver is not available in Pyomo.")

    opt.options["limits/time"] = float(timelimit)
    opt.options["limits/gap"] = float(mipgap)
    opt.options["display/verblevel"] = 4 if tee else 0

    model._loaded_solution_from_solver = False
    model._quality = None

    res = opt.solve(model, tee=tee, load_solutions=False)

    st = res.solver.status
    tc = res.solver.termination_condition
    msg = (getattr(res.solver, "message", "") or "").strip()
    nsol = _num_result_solutions(res)

    print(f"[INFO] SCIP status={st}, termination={tc}, n_solutions={nsol}")
    if msg:
        print(f"[INFO] SCIP message: {msg}")

    if nsol > 0:
        try:
            model.solutions.load_from(res)
            model._loaded_solution_from_solver = True
            model._quality = _constraint_violation_stats(model)
            print(
                "[INFO] Loaded incumbent: "
                f"max_eq={model._quality['max_eq']:.3e}, "
                f"max_ineq={model._quality['max_ineq']:.3e}, "
                f"bad_con={int(model._quality['count_bad'])}"
            )
            if model._quality["max_total"] > float(feas_tol):
                print(
                    f"[WARN] Incumbent loaded but violates constraints above tolerance "
                    f"({model._quality['max_total']:.3e} > {feas_tol:.3e})."
                )
        except Exception as e:
            print(f"[WARN] Failed to load solver solution into model: {e}")
            model._loaded_solution_from_solver = False
            model._quality = None

    return res


# ============================================================
# Reporting
# ============================================================

def report_solution(m: pyo.ConcreteModel, title: str = "SOLUTION"):
    if not getattr(m, "_loaded_solution_from_solver", False):
        print(f"[FAIL] {title}: solver incumbent was not loaded -> cannot report.")
        return

    qual = _constraint_violation_stats(m)
    print(
        f"[INFO] solution quality: max_eq={qual['max_eq']:.3e}, "
        f"max_ineq={qual['max_ineq']:.3e}, bad_con={int(qual['count_bad'])}"
    )

    sn = float(m._data["sn"])
    sb = int(m._data["slack_bus"])

    print("\n====================")
    print(title)
    print("====================")
    print(f"Objective (EUR): {float(pyo.value(m.obj)):.6f}")

    print("\nBus voltages (v=W_ii):")
    for i in m.N:
        vi = float(pyo.value(m.v[i]))
        tag = " [slack]" if int(i) == sb else ""
        print(f"  bus {int(i):3d}: v={vi:.6f}, |V|={math.sqrt(max(vi, 0.0)):.6f}{tag}")

    print("\nGenerators (MW/MVAr):")
    for g in m.G:
        rec = m._data["gens"][int(g)]
        Pg = sn * float(pyo.value(m.Pg[g]))
        Qg = sn * float(pyo.value(m.Qg[g]))
        print(f"  {rec['type']}[{rec['id']}] @ bus {rec['bus']:3d}: P={Pg:.4f} MW, Q={Qg:.4f} MVAr")

    print("\nOLTC taps (chosen beta=1):")
    for l in m.T:
        chosen = None
        for t in m._data["taps_by_l"][int(l)]:
            if float(pyo.value(m.beta[l, t])) > 0.5:
                chosen = int(t)
                break
        br = m._data["branch_meta"][int(l)]
        print(
            f"  branch_id={int(l):3d} ({br['from_bus_pp']}->{br['to_bus_pp']} / mp {br['from_bus_mp']}-{br['to_bus_mp']}): "
            f"tap={chosen}"
        )

    print("\nShunts (a_sh, q_sh):")
    for i in m.C:
        a = int(round(float(pyo.value(m.a_sh[i]))))
        qpu = float(pyo.value(m.bcap[i])) * float(pyo.value(m.z[i]))
        q_mvar = sn * qpu
        q_rated_mvar = sn * float(m._data["bcap"][int(i)])
        print(f"  bus {int(i):3d}: a_sh={a}, q_sh={q_mvar:.3f} MVAr (rated@1pu≈{q_rated_mvar:.3f} MVAr)")

    print("\nWorst directional thermal ratios:")
    worst = (-1.0, None)
    for l in m.L:
        pf = float(pyo.value(m.Pfr[l]))
        qf = float(pyo.value(m.Qfr[l]))
        pt = float(pyo.value(m.Pto[l]))
        qt = float(pyo.value(m.Qto[l]))
        smax = float(pyo.value(m.Smax[l]))
        rf = math.sqrt(max(pf * pf + qf * qf, 0.0)) / smax if smax > 0 else float("inf")
        rt = math.sqrt(max(pt * pt + qt * qt, 0.0)) / smax if smax > 0 else float("inf")
        rr = max(rf, rt)
        if rr > worst[0]:
            worst = (rr, int(l))
    if worst[1] is not None:
        l = int(worst[1])
        br = m._data["branch_meta"][l]
        print(
            f"  branch_id={l} ({br['from_bus_pp']}->{br['to_bus_pp']} / mp {br['from_bus_mp']}-{br['to_bus_mp']}): "
            f"max(|S_from|,|S_to|)/Smax = {worst[0]:.6f}"
        )


def _model_objective_eur(m: pyo.ConcreteModel) -> float:
    return float(pyo.value(m.obj))


def _results_dir() -> Path:
    return Path(__file__).resolve().parent


def _clean_results_text(path: Path) -> str:
    text = html.unescape(path.read_text(encoding="utf-8")).replace("\r\n", "\n")
    return re.sub(r"\\([_()\[\]*#&-])", r"\1", text)


def _extract_section(text: str, start_marker: str, end_marker: Optional[str] = None) -> str:
    start = text.find(start_marker)
    if start < 0:
        raise ValueError(f"start marker not found: {start_marker}")
    if end_marker is None:
        return text[start:]
    end = text.find(end_marker, start)
    return text[start:] if end < 0 else text[start:end]


def _parse_float_token(token: str) -> float:
    return float(str(token).replace(",", "").strip())


def _first_existing(*names: str) -> Optional[Path]:
    here = _results_dir()
    for name in names:
        path = here / name
        if path.exists():
            return path
    return None


def _collect_ver2_benchmarks() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    minlp_txt = _first_existing("ACOPF_MINLP_results.txt")
    if minlp_txt is not None:
        try:
            text = _clean_results_text(minlp_txt)
            m_obj = re.search(r"Objective \(EUR\):\s*([0-9eE+.,-]+)", text)
            m_t = re.search(r"\[INFO\] Total elapsed time:\s*([0-9eE+.,-]+)\s*seconds", text)
            if m_obj:
                rows.append(
                    dict(
                        algorithm="ACOPF_MINLP",
                        value_eur=_parse_float_token(m_obj.group(1)),
                        basis="accepted incumbent objective",
                        elapsed_s=None if not m_t else _parse_float_token(m_t.group(1)),
                        note="rounded fixed-NLP incumbent",
                    )
                )
        except Exception:
            pass
    return rows


def print_algorithm_comparison(current_value_eur: float, current_elapsed_s: float, current_note: str):
    rows = [
        dict(
            algorithm="ACOPF_SDP",
            value_eur=float(current_value_eur),
            basis="chosen SDP objective",
            elapsed_s=float(current_elapsed_s),
            note=current_note,
        )
    ]
    rows.extend(_collect_ver2_benchmarks())

    print("\n--- Algorithm Comparison (reported/raw values) ---")
    print(f"{'algorithm':30s} {'reported_eur':>16s} {'elapsed_s':>12s} {'reported_basis':28s} note")
    for row in rows:
        elapsed_txt = "-" if row['elapsed_s'] is None else f"{row['elapsed_s']:.2f}"
        print(
            f"{row['algorithm']:30s} "
            f"{row['value_eur']:16.6f} "
            f"{elapsed_txt:>12s} "
            f"{row['basis'][:28]:28s} "
            f"{row['note']}"
        )


# ============================================================
# Main pipeline: relax -> round -> fixed incumbent -> try full
# ============================================================

def main():
    t0 = time.perf_counter()

    # ------------------------------------------------------------
    # Build IEEE 300-bus pandapower network
    # ------------------------------------------------------------
    net = mcase.case300_opf(**NETWORK_BUILD_KWARGS)

    meta = read_network_device_metadata(net)
    print("[INFO] Network metadata loaded")
    print("[INFO] network = ieee300bus (SOCP relaxation, no rank-1, control-group mode)")
    print(f"[INFO] #buses         = {len(net.bus)}")
    print(f"[INFO] #branches      = {len(meta['BRANCH_DF'])}")
    print(f"[INFO] #gens          = {len(net.ext_grid) + len(net.gen)}")
    print(f"[INFO] #OLTC branches = {len(meta['OLTC_EDGES_ORDERED'])}")
    print(f"[INFO] #shunt buses   = {len(meta['SHUNT_BCAP_PU'])}")
    print(f"[INFO] #raw branches  = {len(meta['BRANCH_DF'])}")

    # ------------------------------------------------------------
    # 1) Continuous relaxation
    # ------------------------------------------------------------
    print("\n[STEP 1] Solving continuous relaxation (beta,a_sh in [0,1]) ...")
    m_relax = build_misocp_model(net, relax_binaries=True)
    initialize_from_pf_or_flat(m_relax, net)
    print(f"[INFO] warm-start mode (relax) = {getattr(m_relax, '_warm_start_mode', 'unknown')}")

    solve_with_scip(
        m_relax,
        tee=TEE_SOLVER_LOG,
        timelimit=SCIP_TIME_LIMIT_RELAX,
        mipgap=SCIP_GAP_LIMIT,
    )

    if not has_loaded_solution(m_relax):
        print("[FAIL] Relaxation produced no acceptable incumbent.")
        return

    # ------------------------------------------------------------
    # 2) Round discrete vars
    # ------------------------------------------------------------
    print("\n[STEP 2] Rounding discrete decisions from relaxation ...")
    tap_choice, sh_choice = round_discrete_from_relaxed(m_relax)
    print(f"[INFO] Rounded taps   : {len(tap_choice)} OLTC branches")
    print(f"[INFO] Rounded shunts : {len(sh_choice)} buses")

    # ------------------------------------------------------------
    # 3) Evaluate several fixed-discrete candidates and intentionally
    #    pick a slightly worse feasible one for control-group use.
    # ------------------------------------------------------------
    print("\n[STEP 3] Evaluating fixed-discrete candidates for control-group solution ...")
    candidates = build_control_group_candidates(m_relax, tap_choice, sh_choice)
    print(f"[INFO] #fixed candidates to try = {len(candidates)}")

    results = []
    for cand in candidates:
        print(f"\n[STEP 3-CAND] Solving candidate: {cand['name']}")
        res = solve_fixed_candidate(
            net,
            cand,
            tee=TEE_SOLVER_LOG,
            timelimit=CONTROL_FIXED_TIME_LIMIT,
            mipgap=SCIP_GAP_LIMIT,
        )
        results.append(res)

    print_candidate_summary(results)

    chosen = choose_control_group_result(
        results,
        baseline_name="rounded",
        target_increase_eur=TARGET_OBJECTIVE_INCREASE_EUR,
    )
    chosen_model = chosen["model"]

    print("\n[INFO] Control-group candidate selected")
    print(f"[INFO] chosen candidate       = {chosen['name']}")
    print(f"[INFO] chosen objective (EUR)= {float(chosen['objective']):.6f}")
    print(f"[INFO] selection reason      = {chosen.get('selection_reason', 'n/a')}")

    report_solution(chosen_model, title=f"CONTROL-GROUP SOLUTION ({chosen['name']})")

    if not SKIP_FULL_BINARY:
        print("\n[STEP 4] Trying full binary MISOCP ...")
        m_full = build_misocp_model(net, relax_binaries=False)
        initialize_from_pf_or_flat(m_full, net)
        copy_var_values(chosen_model, m_full)
        for v in m_full.component_data_objects(pyo.Var, active=True, descend_into=True):
            if v.fixed:
                v.unfix()
        solve_with_scip(
            m_full,
            tee=TEE_SOLVER_LOG,
            timelimit=SCIP_TIME_LIMIT_FULL,
            mipgap=SCIP_GAP_LIMIT,
        )
        if has_loaded_solution(m_full):
            report_solution(m_full, title="FULL BINARY SOLUTION (SCIP)")

    t1 = time.perf_counter()
    print("\n[INFO] Standalone mode: cross-algorithm comparison table disabled.")
    print(f"[INFO] chosen objective (EUR) = {_model_objective_eur(chosen_model):.6f}")
    print("[INFO] chosen source          = control-group fixed-discrete candidate")
    print(f"\n[INFO] Total elapsed time: {t1 - t0:.2f} seconds.")


if __name__ == "__main__":
    main()
