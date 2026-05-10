# ACOPF_SDP.py
# ------------------------------------------------------------
# Pyomo MISOCP (SDP->SOCP relaxation with 2x2 principal minors)
# ACOPF for modified explicit 41-bus mesh
#
#  - OLTC: binary tap selection + McCormick (x variables per tap)
#  - Switched shunt: binary + McCormick z_i = a_i * v_i, q_sh = bcap * z
#  - Branch flow: linear in W (non-OLTC) / linear in (x vars) (OLTC)
#  - Thermal limits: ||[Pij,Qij]||_2 <= Smax
#
# Solve policy:
#   (1) continuous relaxation
#   (2) round beta/a_sh
#   (3) solve fixed-discrete incumbent
#   (4) optionally try full binary solve
#   (5) fallback to fixed incumbent if full binary gives no incumbent
#
# Network module:
#   ieee39busplus_modified_explicit.py
# ------------------------------------------------------------

import math
import time
from collections import defaultdict
from typing import Dict, Tuple, Any, Optional, List

import pyomo.environ as pyo
from pyomo.opt import SolverStatus, TerminationCondition

import ieee41bus as mcase


# ============================================================
# Solver settings
# ============================================================
SCIP_TIME_LIMIT_FULL = 36000
SCIP_TIME_LIMIT_RELAX = 120
SCIP_TIME_LIMIT_FIXED = 180

SCIP_GAP_LIMIT = 1e-4
TEE_SOLVER_LOG = True

FIX_SLACK_VOLTAGE = True


# ============================================================
# Metadata readers
# ============================================================
def read_network_device_metadata(net):
    if "fixed_oltc_table" not in net:
        raise KeyError("Network metadata 'fixed_oltc_table' not found.")
    if "fixed_shunt_table" not in net:
        raise KeyError("Network metadata 'fixed_shunt_table' not found.")

    oltc_df = net["fixed_oltc_table"]
    shunt_df = net["fixed_shunt_table"]

    oltc_edges_ordered: List[Tuple[int, int]] = []
    oltc_tap_ranges: Dict[Tuple[int, int], Tuple[int, int]] = {}
    oltc_dv_percent: Dict[Tuple[int, int], float] = {}
    oltc_meta: Dict[Tuple[int, int], Dict[str, float]] = {}

    for _, row in oltc_df.iterrows():
        i = int(row["from_bus"])
        j = int(row["to_bus"])
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
        }

    shunt_bcap_pu: Dict[int, float] = {}
    for _, row in shunt_df.iterrows():
        b = int(row["bus"])
        shunt_bcap_pu[b] = float(row["bcap_pu"])

    return {
        "OLTC_EDGES_ORDERED": oltc_edges_ordered,
        "OLTC_TAP_RANGES": oltc_tap_ranges,
        "OLTC_DV_PERCENT": oltc_dv_percent,
        "OLTC_META": oltc_meta,
        "SHUNT_BCAP_PU": shunt_bcap_pu,
    }


# ============================================================
# Helpers
# ============================================================
def edge_key(i, j):
    """Undirected edge key with i<j"""
    return (i, j) if i < j else (j, i)


def build_data_from_pandapower(net):
    """
    Extract per-unit parameters:
      - sn_mva base
      - y = 1/(r+jx), store g=Re(y), b=Im(y)
      - bc assumed 0
      - Smax from max_i_ka
      - generator bounds/cost from poly_cost
    """
    sn = float(net.sn_mva)
    N = [int(i) for i in net.bus.index]

    vL = {i: float(net.bus.at[i, "min_vm_pu"]) ** 2 for i in N}
    vU = {i: float(net.bus.at[i, "max_vm_pu"]) ** 2 for i in N}

    # slack
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

    poly = net.poly_cost.copy() if hasattr(net, "poly_cost") and net.poly_cost is not None else None

    def poly_cost(et: str, element: int):
        if poly is None or poly.empty:
            return 0.0, 0.0, 0.0
        row = poly[(poly["et"] == et) & (poly["element"] == element)]
        if row.empty:
            return 0.0, 0.0, 0.0
        r = row.iloc[0]
        return (
            float(r.get("cp2_eur_per_mw2", 0.0)),
            float(r.get("cp1_eur_per_mw", 0.0)),
            float(r.get("cp0_eur", 0.0)),
        )

    # generators: ext_grid + gen
    gens = []
    for eg in net.ext_grid.index:
        eg = int(eg)
        bus = int(net.ext_grid.at[eg, "bus"])
        pmin = float(net.ext_grid.at[eg, "min_p_mw"]) / sn
        pmax = float(net.ext_grid.at[eg, "max_p_mw"]) / sn
        qmin = float(net.ext_grid.at[eg, "min_q_mvar"]) / sn
        qmax = float(net.ext_grid.at[eg, "max_q_mvar"]) / sn
        c2, c1, c0 = poly_cost("ext_grid", eg)
        gens.append(
            dict(
                type="ext_grid",
                id=eg,
                bus=bus,
                pmin=pmin,
                pmax=pmax,
                qmin=qmin,
                qmax=qmax,
                c2=c2,
                c1=c1,
                c0=c0,
            )
        )

    if hasattr(net, "gen") and len(net.gen.index) > 0:
        for gi in net.gen.index:
            gi = int(gi)
            bus = int(net.gen.at[gi, "bus"])
            pmin = float(net.gen.at[gi, "min_p_mw"]) / sn
            pmax = float(net.gen.at[gi, "max_p_mw"]) / sn
            qmin = float(net.gen.at[gi, "min_q_mvar"]) / sn
            qmax = float(net.gen.at[gi, "max_q_mvar"]) / sn
            c2, c1, c0 = poly_cost("gen", gi)
            gens.append(
                dict(
                    type="gen",
                    id=gi,
                    bus=bus,
                    pmin=pmin,
                    pmax=pmax,
                    qmin=qmin,
                    qmax=qmax,
                    c2=c2,
                    c1=c1,
                    c0=c0,
                )
            )

    # lines -> undirected edges
    E = []
    g = {}
    b = {}
    bc = {}
    Smax = {}
    wbar = {}

    def zbase_ohm(vn_kv):
        return (vn_kv ** 2) / sn

    for lid in net.line.index:
        lid = int(lid)
        fb = int(net.line.at[lid, "from_bus"])
        tb = int(net.line.at[lid, "to_bus"])
        e = edge_key(fb, tb)
        if e in g:
            raise ValueError(f"Parallel line detected for edge {e}. This code assumes no parallels.")

        vn_kv = float(net.bus.at[fb, "vn_kv"])
        zb = zbase_ohm(vn_kv)

        r_ohm = float(net.line.at[lid, "r_ohm_per_km"]) * float(net.line.at[lid, "length_km"])
        x_ohm = float(net.line.at[lid, "x_ohm_per_km"]) * float(net.line.at[lid, "length_km"])
        r_pu = r_ohm / zb
        x_pu = x_ohm / zb

        y = 1.0 / complex(r_pu, x_pu) if (abs(r_pu) + abs(x_pu)) > 1e-12 else (0.0 + 0.0j)
        g[e] = float(y.real)
        b[e] = float(y.imag)
        bc[e] = 0.0

        Imax = float(net.line.at[lid, "max_i_ka"]) if "max_i_ka" in net.line.columns else 1e9
        Smax_mva = math.sqrt(3.0) * vn_kv * Imax
        Smax[e] = Smax_mva / sn

        wbar[e] = math.sqrt(vU[e[0]] * vU[e[1]])
        E.append(e)

    return dict(
        sn=sn,
        N=N,
        E=E,
        vL=vL,
        vU=vU,
        Pd=Pd,
        Qd=Qd,
        g=g,
        b=b,
        bc=bc,
        Smax=Smax,
        wbar=wbar,
        gens=gens,
        slack_bus=slack_bus,
        slack_vm=slack_vm,
    )


# ============================================================
# Model builder
# ============================================================
def build_misocp_model(net, relax_binaries: bool = False):
    data = build_data_from_pandapower(net)
    meta = read_network_device_metadata(net)

    sn = data["sn"]
    N = data["N"]
    E = data["E"]
    Eset = set(E)

    # directed arcs A (both directions)
    A = []
    out = defaultdict(list)
    for (i, j) in E:
        A.append((i, j))
        A.append((j, i))
        out[i].append(j)
        out[j].append(i)

    # OLTC set T is directed exactly as metadata gives
    T = list(meta["OLTC_EDGES_ORDERED"])
    for (i, j) in T:
        if edge_key(i, j) not in Eset:
            raise ValueError(
                f"OLTC edge {(i, j)} not found in net.line edges. "
                f"Check metadata direction/topology consistency."
            )

    taps_by_T = defaultdict(list)
    TAP = []
    alpha = {}
    delta = {}

    for (i, j) in T:
        tmin, tmax = meta["OLTC_TAP_RANGES"][(i, j)]
        dV = float(meta["OLTC_DV_PERCENT"][(i, j)])
        for t in range(int(tmin), int(tmax) + 1):
            tau = 1.0 + (t * dV) / 100.0
            alpha[(i, j, t)] = 1.0 / tau
            delta[(i, j, t)] = 1.0 / (tau ** 2)
            taps_by_T[(i, j)].append(t)
            TAP.append((i, j, t))

    # shunt set
    SHUNT_BCAP_PU = dict(meta["SHUNT_BCAP_PU"])
    C = sorted([int(k) for k in SHUNT_BCAP_PU.keys()])
    Cset = set(C)
    bcap = {i: float(SHUNT_BCAP_PU[i]) for i in C}

    # generators mapping
    gens = data["gens"]
    G = list(range(len(gens)))
    gen_bus = {g: int(gens[g]["bus"]) for g in G}
    G_of_bus = defaultdict(list)
    for g in G:
        G_of_bus[gen_bus[g]].append(g)

    m = pyo.ConcreteModel("MISOCP_SDP_SOCP_relax_explicit_41bus")

    # sets
    m.N = pyo.Set(initialize=N, ordered=True)
    m.E = pyo.Set(initialize=E, dimen=2, ordered=True)      # undirected i<j
    m.A = pyo.Set(initialize=A, dimen=2, ordered=True)      # directed arcs
    m.T = pyo.Set(initialize=T, dimen=2, ordered=True)      # directed OLTC
    m.TAP = pyo.Set(initialize=TAP, dimen=3, ordered=True)  # (i,j,t)
    m.C = pyo.Set(initialize=C, ordered=True)
    m.G = pyo.Set(initialize=G, ordered=True)

    # params
    m.vL = pyo.Param(m.N, initialize=lambda mm, i: float(data["vL"][i]))
    m.vU = pyo.Param(m.N, initialize=lambda mm, i: float(data["vU"][i]))
    m.Pd = pyo.Param(m.N, initialize=lambda mm, i: float(data["Pd"][i]))
    m.Qd = pyo.Param(m.N, initialize=lambda mm, i: float(data["Qd"][i]))

    m.g = pyo.Param(m.E, initialize=lambda mm, i, j: float(data["g"][(i, j)]))
    m.b = pyo.Param(m.E, initialize=lambda mm, i, j: float(data["b"][(i, j)]))
    m.bc = pyo.Param(m.E, initialize=lambda mm, i, j: float(data["bc"][(i, j)]))
    m.Smax = pyo.Param(m.E, initialize=lambda mm, i, j: float(data["Smax"][(i, j)]))
    m.wbar = pyo.Param(m.E, initialize=lambda mm, i, j: float(data["wbar"][(i, j)]))

    m.alpha = pyo.Param(m.TAP, initialize=lambda mm, i, j, t: float(alpha[(i, j, t)]))
    m.delta = pyo.Param(m.TAP, initialize=lambda mm, i, j, t: float(delta[(i, j, t)]))

    m.bcap = pyo.Param(m.C, initialize=lambda mm, i: float(bcap[i]))

    m.pmin = pyo.Param(m.G, initialize=lambda mm, g: float(gens[g]["pmin"]))
    m.pmax = pyo.Param(m.G, initialize=lambda mm, g: float(gens[g]["pmax"]))
    m.qmin = pyo.Param(m.G, initialize=lambda mm, g: float(gens[g]["qmin"]))
    m.qmax = pyo.Param(m.G, initialize=lambda mm, g: float(gens[g]["qmax"]))
    m.c2 = pyo.Param(m.G, initialize=lambda mm, g: float(gens[g]["c2"]))
    m.c1 = pyo.Param(m.G, initialize=lambda mm, g: float(gens[g]["c1"]))
    m.c0 = pyo.Param(m.G, initialize=lambda mm, g: float(gens[g]["c0"]))

    # variables
    m.v = pyo.Var(m.N, bounds=lambda mm, i: (mm.vL[i], mm.vU[i]))
    m.Wre = pyo.Var(m.E, bounds=lambda mm, i, j: (-mm.wbar[i, j], mm.wbar[i, j]))
    m.Wim = pyo.Var(m.E, bounds=lambda mm, i, j: (-mm.wbar[i, j], mm.wbar[i, j]))

    m.Pg = pyo.Var(m.G, bounds=lambda mm, g: (mm.pmin[g], mm.pmax[g]))
    m.Qg = pyo.Var(m.G, bounds=lambda mm, g: (mm.qmin[g], mm.qmax[g]))

    if relax_binaries:
        m.beta = pyo.Var(m.TAP, bounds=(0.0, 1.0))
        m.a_sh = pyo.Var(m.C, bounds=(0.0, 1.0))
    else:
        m.beta = pyo.Var(m.TAP, within=pyo.Binary)
        m.a_sh = pyo.Var(m.C, within=pyo.Binary)

    # McCormick aux vars
    m.x_ii = pyo.Var(m.TAP)
    m.x_jj = pyo.Var(m.TAP)
    m.x_ijR = pyo.Var(m.TAP)
    m.x_ijI = pyo.Var(m.TAP)

    m.z = pyo.Var(m.C)

    # helper: ordered W_ij
    def Wre_ord(i, j):
        e = edge_key(i, j)
        return m.Wre[e]

    def Wim_ord(i, j):
        e = edge_key(i, j)
        return m.Wim[e] if i < j else -m.Wim[e]

    T_undirected = {edge_key(i, j) for (i, j) in T}

    def is_oltc_edge(i, j):
        return edge_key(i, j) in T_undirected

    def oltc_base(i, j):
        ek = edge_key(i, j)
        for (u, v) in T:
            if edge_key(u, v) == ek:
                return (u, v)
        return None

    # slack voltage fixing
    if FIX_SLACK_VOLTAGE:
        sb = int(data["slack_bus"])
        m.v[sb].fix(float(data["slack_vm"]) ** 2)

    # expressions
    m.pG = pyo.Expression(m.N, rule=lambda mm, i: sum(mm.Pg[g] for g in G_of_bus.get(i, [])))
    m.qG = pyo.Expression(m.N, rule=lambda mm, i: sum(mm.Qg[g] for g in G_of_bus.get(i, [])))

    def qsh_expr(mm, i):
        if i not in Cset:
            return 0.0
        return mm.bcap[i] * mm.z[i]
    m.qsh = pyo.Expression(m.N, rule=qsh_expr)

    # Branch flow expressions on directed arcs
    def P_expr(mm, i, j):
        e = edge_key(i, j)
        gij = mm.g[e]
        bij = mm.b[e]

        if not is_oltc_edge(i, j):
            return gij * (mm.v[i] - Wre_ord(i, j)) - bij * Wim_ord(i, j)

        (u, v) = oltc_base(i, j)
        if (i, j) == (u, v):
            s = 0.0
            for t in taps_by_T[(u, v)]:
                a = mm.alpha[u, v, t]
                d = mm.delta[u, v, t]
                s += gij * (d * mm.x_ii[u, v, t] - a * mm.x_ijR[u, v, t]) - bij * (a * mm.x_ijI[u, v, t])
            return s
        else:
            s = 0.0
            for t in taps_by_T[(u, v)]:
                a = mm.alpha[u, v, t]
                s += gij * (mm.x_jj[u, v, t] - a * mm.x_ijR[u, v, t]) + bij * (a * mm.x_ijI[u, v, t])
            return s

    def Q_expr(mm, i, j):
        e = edge_key(i, j)
        gij = mm.g[e]
        bij = mm.b[e]
        bcij = mm.bc[e]

        if not is_oltc_edge(i, j):
            return (-gij) * Wim_ord(i, j) - bij * (mm.v[i] - Wre_ord(i, j)) - 0.5 * bcij * mm.v[i]

        (u, v) = oltc_base(i, j)
        if (i, j) == (u, v):
            s = 0.0
            for t in taps_by_T[(u, v)]:
                a = mm.alpha[u, v, t]
                d = mm.delta[u, v, t]
                s += (-gij) * (a * mm.x_ijI[u, v, t]) - bij * (d * mm.x_ii[u, v, t] - a * mm.x_ijR[u, v, t]) \
                     - 0.5 * bcij * (d * mm.x_ii[u, v, t])
            return s
        else:
            s = 0.0
            for t in taps_by_T[(u, v)]:
                a = mm.alpha[u, v, t]
                s += gij * (a * mm.x_ijI[u, v, t]) - bij * (mm.x_jj[u, v, t] - a * mm.x_ijR[u, v, t]) \
                     - 0.5 * bcij * (mm.x_jj[u, v, t])
            return s

    m.P = pyo.Expression(m.A, rule=P_expr)
    m.Q = pyo.Expression(m.A, rule=Q_expr)

    # constraints
    m.con_soc = pyo.Constraint(
        m.E,
        rule=lambda mm, i, j:
            (2 * mm.Wre[i, j]) ** 2
            + (2 * mm.Wim[i, j]) ** 2
            + (mm.v[i] - mm.v[j]) ** 2
            <= (mm.v[i] + mm.v[j]) ** 2
    )

    m.con_onehot = pyo.Constraint(
        m.T,
        rule=lambda mm, i, j: sum(mm.beta[i, j, t] for t in taps_by_T[(i, j)]) == 1
    )

    m.con_link = pyo.ConstraintList()
    for (i, j) in T:
        e = edge_key(i, j)
        m.con_link.add(sum(m.x_ii[i, j, t] for t in taps_by_T[(i, j)]) == m.v[i])
        m.con_link.add(sum(m.x_jj[i, j, t] for t in taps_by_T[(i, j)]) == m.v[j])
        m.con_link.add(sum(m.x_ijR[i, j, t] for t in taps_by_T[(i, j)]) == m.Wre[e])
        if i < j:
            m.con_link.add(sum(m.x_ijI[i, j, t] for t in taps_by_T[(i, j)]) == m.Wim[e])
        else:
            m.con_link.add(sum(m.x_ijI[i, j, t] for t in taps_by_T[(i, j)]) == -m.Wim[e])

    m.con_mcc = pyo.ConstraintList()
    for (i, j, t) in TAP:
        e = edge_key(i, j)
        wb = float(data["wbar"][e])

        m.con_mcc.add(m.x_ii[i, j, t] >= m.vL[i] * m.beta[i, j, t])
        m.con_mcc.add(m.x_ii[i, j, t] <= m.vU[i] * m.beta[i, j, t])
        m.con_mcc.add(m.x_ii[i, j, t] >= m.v[i] - m.vU[i] * (1 - m.beta[i, j, t]))
        m.con_mcc.add(m.x_ii[i, j, t] <= m.v[i] - m.vL[i] * (1 - m.beta[i, j, t]))

        m.con_mcc.add(m.x_jj[i, j, t] >= m.vL[j] * m.beta[i, j, t])
        m.con_mcc.add(m.x_jj[i, j, t] <= m.vU[j] * m.beta[i, j, t])
        m.con_mcc.add(m.x_jj[i, j, t] >= m.v[j] - m.vU[j] * (1 - m.beta[i, j, t]))
        m.con_mcc.add(m.x_jj[i, j, t] <= m.v[j] - m.vL[j] * (1 - m.beta[i, j, t]))

        m.con_mcc.add(m.x_ijR[i, j, t] >= -wb * m.beta[i, j, t])
        m.con_mcc.add(m.x_ijR[i, j, t] <= wb * m.beta[i, j, t])
        m.con_mcc.add(m.x_ijR[i, j, t] >= m.Wre[e] - wb * (1 - m.beta[i, j, t]))
        m.con_mcc.add(m.x_ijR[i, j, t] <= m.Wre[e] + wb * (1 - m.beta[i, j, t]))

        target_im = m.Wim[e] if i < j else -m.Wim[e]
        m.con_mcc.add(m.x_ijI[i, j, t] >= -wb * m.beta[i, j, t])
        m.con_mcc.add(m.x_ijI[i, j, t] <= wb * m.beta[i, j, t])
        m.con_mcc.add(m.x_ijI[i, j, t] >= target_im - wb * (1 - m.beta[i, j, t]))
        m.con_mcc.add(m.x_ijI[i, j, t] <= target_im + wb * (1 - m.beta[i, j, t]))

    m.con_sh = pyo.ConstraintList()
    for i in C:
        m.con_sh.add(m.z[i] <= m.vU[i] * m.a_sh[i])
        m.con_sh.add(m.z[i] >= m.vL[i] * m.a_sh[i])
        m.con_sh.add(m.z[i] <= m.v[i] - m.vL[i] * (1 - m.a_sh[i]))
        m.con_sh.add(m.z[i] >= m.v[i] - m.vU[i] * (1 - m.a_sh[i]))

    m.con_kclP = pyo.Constraint(
        m.N,
        rule=lambda mm, i: mm.pG[i] - mm.Pd[i] == sum(mm.P[i, j] for j in out[i])
    )
    m.con_kclQ = pyo.Constraint(
        m.N,
        rule=lambda mm, i: mm.qG[i] - mm.Qd[i] + mm.qsh[i] == sum(mm.Q[i, j] for j in out[i])
    )

    m.con_thermal = pyo.Constraint(
        m.A,
        rule=lambda mm, i, j: mm.P[i, j] ** 2 + mm.Q[i, j] ** 2 <= (mm.Smax[edge_key(i, j)]) ** 2
    )

    m.obj = pyo.Objective(
        expr=sum(m.c2[g] * (sn * m.Pg[g]) ** 2 + m.c1[g] * (sn * m.Pg[g]) + m.c0[g] for g in m.G),
        sense=pyo.minimize
    )

    # attach metadata for reporting/rounding
    m._data = data
    m._gens = gens
    m._taps_by_T = taps_by_T
    m._out = out
    m._relax_binaries = relax_binaries
    m._meta = meta
    return m


# ============================================================
# Initialization / rounding / fixing / copy helpers
# ============================================================
def _default_tap_choice(taps):
    return min(taps, key=lambda t: abs(t))


def initialize_flat(m: pyo.ConcreteModel):
    sb = int(m._data["slack_bus"])
    v_slack = float(m._data["slack_vm"]) ** 2

    for i in m.N:
        m.v[i].set_value(v_slack if int(i) == sb else 1.0)

    for (i, j) in m.E:
        m.Wre[i, j].set_value(0.0)
        m.Wim[i, j].set_value(0.0)

    for g in m.G:
        pmin = float(pyo.value(m.pmin[g]))
        pmax = float(pyo.value(m.pmax[g]))
        qmin = float(pyo.value(m.qmin[g]))
        qmax = float(pyo.value(m.qmax[g]))
        pmid = 0.0 if (pmin <= 0.0 <= pmax) else 0.5 * (pmin + pmax)
        qmid = 0.0 if (qmin <= 0.0 <= qmax) else 0.5 * (qmin + qmax)
        m.Pg[g].set_value(pmid)
        m.Qg[g].set_value(qmid)

    for i in m.C:
        m.a_sh[i].set_value(0.0)
        m.z[i].set_value(0.0)

    for (i, j) in m.T:
        taps = list(m._taps_by_T[(i, j)])
        pick = _default_tap_choice(taps)
        for t in taps:
            m.beta[i, j, t].set_value(1.0 if t == pick else 0.0)

        for t in taps:
            if t == pick:
                m.x_ii[i, j, t].set_value(pyo.value(m.v[i]))
                m.x_jj[i, j, t].set_value(pyo.value(m.v[j]))
                m.x_ijR[i, j, t].set_value(0.0)
                m.x_ijI[i, j, t].set_value(0.0)
            else:
                m.x_ii[i, j, t].set_value(0.0)
                m.x_jj[i, j, t].set_value(0.0)
                m.x_ijR[i, j, t].set_value(0.0)
                m.x_ijI[i, j, t].set_value(0.0)


def has_loaded_solution(m: pyo.ConcreteModel) -> bool:
    for v in m.component_data_objects(pyo.Var, active=True, descend_into=True):
        if v.value is not None:
            return True
    return False


def round_discrete_from_relaxed(m_relax: pyo.ConcreteModel) -> Tuple[Dict[Tuple[int, int], int], Dict[int, int]]:
    tap_choice: Dict[Tuple[int, int], int] = {}
    sh_choice: Dict[int, int] = {}

    for (i, j) in m_relax.T:
        best_t, best_v = None, -1e100
        for t in m_relax._taps_by_T[(i, j)]:
            v = float(pyo.value(m_relax.beta[i, j, t]))
            if v > best_v:
                best_v = v
                best_t = int(t)
        if best_t is None:
            best_t = int(_default_tap_choice(list(m_relax._taps_by_T[(i, j)])))
        tap_choice[(int(i), int(j))] = int(best_t)

    for i in m_relax.C:
        v = float(pyo.value(m_relax.a_sh[i]))
        sh_choice[int(i)] = 1 if v >= 0.5 else 0

    return tap_choice, sh_choice


def apply_discrete_fixings(
    m: pyo.ConcreteModel,
    tap_choice: Dict[Tuple[int, int], int],
    sh_choice: Dict[int, int],
    fix: bool = True,
):
    for (i, j) in m.T:
        pick = int(tap_choice[(int(i), int(j))])
        for t in m._taps_by_T[(i, j)]:
            val = 1.0 if int(t) == pick else 0.0
            m.beta[i, j, t].set_value(val)
            if fix:
                m.beta[i, j, t].fix(val)

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
                vsrc = src_comp[idx]
            except Exception:
                continue
            val = vsrc.value
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
) -> Any:
    opt = pyo.SolverFactory("scip", solver_io="nl")
    if opt is None or not opt.available(exception_flag=False):
        opt = pyo.SolverFactory("scip")
    if opt is None or not opt.available(exception_flag=False):
        raise RuntimeError("SCIP solver is not available in Pyomo.")

    opt.options["limits/time"] = float(timelimit)
    opt.options["limits/gap"] = float(mipgap)
    opt.options["display/verblevel"] = 4 if tee else 0

    res = opt.solve(model, tee=tee, load_solutions=True)

    st = res.solver.status
    tc = res.solver.termination_condition
    msg = (getattr(res.solver, "message", "") or "").strip()

    print(f"[INFO] SCIP status={st}, termination={tc}")
    if msg:
        print(f"[INFO] SCIP message: {msg}")

    return res


# ============================================================
# Reporting
# ============================================================
def report_solution(m: pyo.ConcreteModel, title: str = "SOLUTION"):
    if not has_loaded_solution(m):
        print(f"[FAIL] {title}: no variable values loaded -> cannot report.")
        return

    sn = float(m._data["sn"])
    sb = int(m._data["slack_bus"])

    print("\n====================")
    print(title)
    print("====================")
    print(f"Objective (EUR) = {float(pyo.value(m.obj)):.6f}")

    print("\nBus voltages (v=W_ii):")
    for i in m.N:
        vi = float(pyo.value(m.v[i]))
        tag = " [slack]" if int(i) == sb else ""
        print(f"  bus {int(i):2d}: v={vi:.6f}, |V|={math.sqrt(max(vi, 0.0)):.6f}{tag}")

    print("\nGenerators (MW/MVAr):")
    for g in m.G:
        rec = m._gens[int(g)]
        Pg = sn * float(pyo.value(m.Pg[g]))
        Qg = sn * float(pyo.value(m.Qg[g]))
        print(f"  {rec['type']}[{rec['id']}] @ bus {rec['bus']:2d}: P={Pg:.4f} MW, Q={Qg:.4f} MVAr")

    print("\nOLTC taps (chosen beta=1):")
    for (i, j) in m.T:
        chosen = None
        for t in m._taps_by_T[(i, j)]:
            if float(pyo.value(m.beta[i, j, t])) > 0.5:
                chosen = int(t)
                break
        meta = m._meta["OLTC_META"][(int(i), int(j))]
        print(
            f"  ({int(i)},{int(j)}) tap={chosen}   "
            f"range=[{meta['tap_min']},{meta['tap_max']}], dV%={meta['dV_percent']}"
        )

    print("\nShunts (a_sh, q_sh):")
    for i in m.C:
        a = int(round(float(pyo.value(m.a_sh[i]))))
        qpu = float(pyo.value(m.bcap[i])) * float(pyo.value(m.z[i]))
        q_mvar = sn * qpu
        q_rated_mvar = sn * float(m._meta["SHUNT_BCAP_PU"][int(i)])
        print(f"  bus {int(i):2d}: a_sh={a}, q_sh={q_mvar:.3f} MVAr (rated@1pu≈{q_rated_mvar:.3f} MVAr)")


# ============================================================
# Main pipeline: relax -> round -> fixed incumbent -> try full
# ============================================================
def main():
    t0 = time.perf_counter()

    # ------------------------------------------------------------
    # Build modified explicit 41-bus mesh
    # ------------------------------------------------------------
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

    meta = read_network_device_metadata(net)
    print("[INFO] Network metadata loaded")
    print(f"[INFO] #OLTC branches = {len(meta['OLTC_EDGES_ORDERED'])}")
    print(f"[INFO] #shunt buses    = {len(meta['SHUNT_BCAP_PU'])}")

    # ------------------------------------------------------------
    # 1) Continuous relaxation
    # ------------------------------------------------------------
    print("\n[STEP 1] Solving continuous relaxation (beta,a_sh in [0,1]) ...")
    m_relax = build_misocp_model(net, relax_binaries=True)
    initialize_flat(m_relax)

    solve_with_scip(
        m_relax,
        tee=TEE_SOLVER_LOG,
        timelimit=SCIP_TIME_LIMIT_RELAX,
        mipgap=SCIP_GAP_LIMIT,
    )

    if not has_loaded_solution(m_relax):
        print("[FAIL] Relaxation produced no solution.")
        return

    # ------------------------------------------------------------
    # 2) Round discrete vars
    # ------------------------------------------------------------
    print("\n[STEP 2] Rounding discrete decisions from relaxation ...")
    tap_choice, sh_choice = round_discrete_from_relaxed(m_relax)
    print(f"[INFO] Rounded taps: {len(tap_choice)} OLTC branches")
    print(f"[INFO] Rounded shunts: {len(sh_choice)} buses")

    # ------------------------------------------------------------
    # 3) Fixed-discrete solve
    # ------------------------------------------------------------
    print("\n[STEP 3] Solving with DISCRETE variables FIXED to rounded choice ...")
    m_fixed = build_misocp_model(net, relax_binaries=False)
    initialize_flat(m_fixed)
    apply_discrete_fixings(m_fixed, tap_choice, sh_choice, fix=True)

    solve_with_scip(
        m_fixed,
        tee=TEE_SOLVER_LOG,
        timelimit=SCIP_TIME_LIMIT_FIXED,
        mipgap=SCIP_GAP_LIMIT,
    )

    if not has_loaded_solution(m_fixed):
        print("[FAIL] Fixed-discrete problem has no solution.")
        return

    report_solution(m_fixed, title="INCUMBENT (Fixed discrete from relax-and-round)")

    # ------------------------------------------------------------
    # 4) Try full binary solve
    # ------------------------------------------------------------
    print("\n[STEP 4] Trying full binary MISOCP ...")
    m_full = build_misocp_model(net, relax_binaries=False)
    initialize_flat(m_full)
    copy_var_values(m_fixed, m_full)

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
    else:
        print("[WARN] Full binary solve found 0 solutions. Using fixed-discrete incumbent.")

    t1 = time.perf_counter()
    print(f"\n[INFO] Total elapsed time: {t1 - t0:.2f} seconds.")


if __name__ == "__main__":
    main()