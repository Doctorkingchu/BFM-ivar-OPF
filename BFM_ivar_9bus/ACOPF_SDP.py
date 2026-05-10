# ACOPF_SDP.py
# ------------------------------------------------------------
# Pyomo MISOCP (SDP->SOCP relaxation) ACOPF with:
#   - OLTC (binary tap selection) at EXACT locations/spec from ACOPF_MINLP.py
#   - Switched shunts at EXACT buses/spec from ACOPF_MINLP.py
#
# Solver: SCIP (installed in your env)
#
# Run:
#   (kingchu) python ACOPF_SDP.py
# ------------------------------------------------------------

import math
from collections import defaultdict

import pyomo.environ as pyo
import ieee9bus as mcase


# ============================================================
# EXACT device specs/locations copied from ACOPF_MINLP.py
# ============================================================
OLTC_BRANCHES = {
    (3, 4): dict(tap_min=-8, tap_max=8, dV_percent=1.25),  # 17 taps
    (5, 6): dict(tap_min=-6, tap_max=6, dV_percent=1.25),  # 13 taps
    (7, 1): dict(tap_min=-4, tap_max=4, dV_percent=1.25),  # 9 taps
}

SHUNTS = {
    1: dict(q_rated_mvar=10.0, v_rated_pu=1.0),
    4: dict(q_rated_mvar=15.0, v_rated_pu=1.0),
    5: dict(q_rated_mvar=20.0, v_rated_pu=1.0),
    6: dict(q_rated_mvar=10.0, v_rated_pu=1.0),
    8: dict(q_rated_mvar=8.0,  v_rated_pu=1.0),
}

FIX_SLACK_VOLTAGE = True
# ============================================================


def edge_key(i, j):
    """Undirected edge key with i<j"""
    return (i, j) if i < j else (j, i)


def build_data_from_pandapower(net):
    """
    Extract per-unit parameters consistent with your MINLP extraction:
      - sn_mva base
      - line admittance y = 1/(r+jx), store g=Re(y), b=Im(y) (typically b<0 for inductive)
      - bc = 0 in your network
      - Smax from max_i_ka
      - generator bounds/cost from poly_cost
    """
    sn = float(net.sn_mva)
    N = [int(i) for i in net.bus.index]
    nb = len(N)

    # voltage bounds (squared)
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

    # costs
    poly = net.poly_cost.copy() if hasattr(net, "poly_cost") and net.poly_cost is not None else None

    def poly_cost(et: str, element: int):
        if poly is None or poly.empty:
            return 0.0, 0.0, 0.0
        row = poly[(poly["et"] == et) & (poly["element"] == element)]
        if row.empty:
            return 0.0, 0.0, 0.0
        r = row.iloc[0]
        return (float(r.get("cp2_eur_per_mw2", 0.0)),
                float(r.get("cp1_eur_per_mw", 0.0)),
                float(r.get("cp0_eur", 0.0)))

    # generators: ext_grid + gen
    gens = []  # each: bus, pmin_pu, pmax_pu, qmin_pu, qmax_pu, c2,c1,c0 (MW-based)
    for eg in net.ext_grid.index:
        eg = int(eg)
        bus = int(net.ext_grid.at[eg, "bus"])
        pmin = float(net.ext_grid.at[eg, "min_p_mw"]) / sn
        pmax = float(net.ext_grid.at[eg, "max_p_mw"]) / sn
        qmin = float(net.ext_grid.at[eg, "min_q_mvar"]) / sn
        qmax = float(net.ext_grid.at[eg, "max_q_mvar"]) / sn
        c2, c1, c0 = poly_cost("ext_grid", eg)
        gens.append(dict(type="ext_grid", id=eg, bus=bus,
                         pmin=pmin, pmax=pmax, qmin=qmin, qmax=qmax,
                         c2=c2, c1=c1, c0=c0))

    if hasattr(net, "gen") and len(net.gen.index) > 0:
        for gi in net.gen.index:
            gi = int(gi)
            bus = int(net.gen.at[gi, "bus"])
            pmin = float(net.gen.at[gi, "min_p_mw"]) / sn
            pmax = float(net.gen.at[gi, "max_p_mw"]) / sn
            qmin = float(net.gen.at[gi, "min_q_mvar"]) / sn
            qmax = float(net.gen.at[gi, "max_q_mvar"]) / sn
            c2, c1, c0 = poly_cost("gen", gi)
            gens.append(dict(type="gen", id=gi, bus=bus,
                             pmin=pmin, pmax=pmax, qmin=qmin, qmax=qmax,
                             c2=c2, c1=c1, c0=c0))

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

        # per-unit r,x from ohm
        vn_kv = float(net.bus.at[fb, "vn_kv"])
        zb = zbase_ohm(vn_kv)
        r_ohm = float(net.line.at[lid, "r_ohm_per_km"]) * float(net.line.at[lid, "length_km"])
        x_ohm = float(net.line.at[lid, "x_ohm_per_km"]) * float(net.line.at[lid, "length_km"])
        r_pu = r_ohm / zb
        x_pu = x_ohm / zb

        y = 1.0 / complex(r_pu, x_pu) if (abs(r_pu) + abs(x_pu)) > 1e-12 else (0.0 + 0.0j)
        g[e] = float(y.real)
        b[e] = float(y.imag)
        bc[e] = 0.0  # your network uses 0

        Imax = float(net.line.at[lid, "max_i_ka"])
        Smax_mva = math.sqrt(3.0) * vn_kv * Imax
        Smax[e] = Smax_mva / sn

        # wbar = sqrt(vU_i vU_j)
        # (note: vU is squared voltage, so wbar = sqrt(vU_i*vU_j) is consistent with |W_ij| bound)
        # here vU already squared.
        wbar[e] = math.sqrt(vU[e[0]] * vU[e[1]])

        E.append(e)

    return dict(
        sn=sn, N=N, E=E,
        vL=vL, vU=vU,
        Pd=Pd, Qd=Qd,
        g=g, b=b, bc=bc,
        Smax=Smax, wbar=wbar,
        gens=gens,
        slack_bus=slack_bus, slack_vm=slack_vm
    )


def build_misocp_model(net):
    data = build_data_from_pandapower(net)
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

    # OLTC set T is directed as given (tap on i-side)
    T = list(OLTC_BRANCHES.keys())
    for (i, j) in T:
        if edge_key(i, j) not in Eset:
            raise ValueError(f"OLTC edge {(i, j)} not found in net.line (undirected) edges.")

    # full taps (NO stride reduction)
    taps_by_T = defaultdict(list)
    TAP = []
    alpha = {}
    delta = {}
    for (i, j), cfg in OLTC_BRANCHES.items():
        tmin = int(cfg["tap_min"])
        tmax = int(cfg["tap_max"])
        dV = float(cfg["dV_percent"])
        for t in range(tmin, tmax + 1):
            tau = 1.0 + (t * dV) / 100.0
            alpha[(i, j, t)] = 1.0 / tau
            delta[(i, j, t)] = 1.0 / (tau ** 2)
            taps_by_T[(i, j)].append(t)
            TAP.append((i, j, t))

    # shunt set C
    C = sorted(list(SHUNTS.keys()))
    Cset = set(C)
    bcap = {}
    for i in C:
        q_pu = float(SHUNTS[i]["q_rated_mvar"]) / sn
        v_rated_sq = float(SHUNTS[i]["v_rated_pu"]) ** 2
        bcap[i] = q_pu / v_rated_sq if v_rated_sq > 0 else 0.0

    # generators sets/mapping
    gens = data["gens"]
    G = list(range(len(gens)))
    gen_bus = {g: int(gens[g]["bus"]) for g in G}
    G_of_bus = defaultdict(list)
    for g in G:
        G_of_bus[gen_bus[g]].append(g)

    m = pyo.ConcreteModel("MISOCP_SDP_SOCP_relaxation")

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

    # generator bounds/cost (cost is MW-based; Pg is pu)
    m.pmin = pyo.Param(m.G, initialize=lambda mm, g: float(gens[g]["pmin"]))
    m.pmax = pyo.Param(m.G, initialize=lambda mm, g: float(gens[g]["pmax"]))
    m.qmin = pyo.Param(m.G, initialize=lambda mm, g: float(gens[g]["qmin"]))
    m.qmax = pyo.Param(m.G, initialize=lambda mm, g: float(gens[g]["qmax"]))
    m.c2 = pyo.Param(m.G, initialize=lambda mm, g: float(gens[g]["c2"]))
    m.c1 = pyo.Param(m.G, initialize=lambda mm, g: float(gens[g]["c1"]))
    m.c0 = pyo.Param(m.G, initialize=lambda mm, g: float(gens[g]["c0"]))

    # variables
    # v_i = W_ii
    m.v = pyo.Var(m.N, bounds=lambda mm, i: (mm.vL[i], mm.vU[i]))

    # W_ij for undirected (i<j): store Re(W_ij), Im(W_ij) where W_ij corresponds to that i<j ordering
    m.Wre = pyo.Var(m.E, bounds=lambda mm, i, j: (-mm.wbar[i, j], mm.wbar[i, j]))
    m.Wim = pyo.Var(m.E, bounds=lambda mm, i, j: (-mm.wbar[i, j], mm.wbar[i, j]))

    # generator dispatch (pu)
    m.Pg = pyo.Var(m.G, bounds=lambda mm, g: (mm.pmin[g], mm.pmax[g]))
    m.Qg = pyo.Var(m.G, bounds=lambda mm, g: (mm.qmin[g], mm.qmax[g]))

    # OLTC binaries + McCormick auxiliary vars
    m.beta = pyo.Var(m.TAP, within=pyo.Binary)
    m.x_ii = pyo.Var(m.TAP)
    m.x_jj = pyo.Var(m.TAP)
    m.x_ijR = pyo.Var(m.TAP)
    m.x_ijI = pyo.Var(m.TAP)

    # shunt binaries + z_i = a_i * v_i
    m.a_sh = pyo.Var(m.C, within=pyo.Binary)
    m.z = pyo.Var(m.C)

    # ------------------------------------------------------------
    # helper: Re/Im of W_{ij} for an arbitrary ordered pair (i,j)
    #   if i<j: W_{ij} = Wre[i,j] + j Wim[i,j]
    #   if i>j: W_{ij} = Wre[j,i] - j Wim[j,i]  (Hermitian)
    # ------------------------------------------------------------
    def Wre_ord(i, j):
        e = edge_key(i, j)
        return m.Wre[e]

    def Wim_ord(i, j):
        e = edge_key(i, j)
        return m.Wim[e] if i < j else -m.Wim[e]

    # identify if undirected edge is OLTC edge
    T_undirected = {edge_key(i, j) for (i, j) in T}

    def is_oltc_edge(i, j):
        return edge_key(i, j) in T_undirected

    def oltc_base(i, j):
        # returns directed base (u,v) in T matching the undirected pair {i,j}
        ek = edge_key(i, j)
        for (u, v) in T:
            if edge_key(u, v) == ek:
                return (u, v)
        return None

    # ------------------------------------------------------------
    # Slack voltage fixing (same idea as MINLP)
    # ------------------------------------------------------------
    if FIX_SLACK_VOLTAGE:
        sb = int(data["slack_bus"])
        m.v[sb].fix(float(data["slack_vm"]) ** 2)

    # ------------------------------------------------------------
    # Expressions (reduce variable count; taps still full)
    # ------------------------------------------------------------
    m.pG = pyo.Expression(m.N, rule=lambda mm, i: sum(mm.Pg[g] for g in G_of_bus.get(i, [])))
    m.qG = pyo.Expression(m.N, rule=lambda mm, i: sum(mm.Qg[g] for g in G_of_bus.get(i, [])))

    # shunt injection: q_sh = bcap * z (only on C)
    def qsh_expr(mm, i):
        if i not in Cset:
            return 0.0
        return mm.bcap[i] * mm.z[i]
    m.qsh = pyo.Expression(m.N, rule=qsh_expr)

    # ------------------------------------------------------------
    # Branch flow expressions P_ij, Q_ij on directed arcs (i,j) in A
    # ------------------------------------------------------------
    def P_expr(mm, i, j):
        e = edge_key(i, j)
        gij = mm.g[e]
        bij = mm.b[e]
        if not is_oltc_edge(i, j):
            # P_ij = Re((W_ii - W_ij) y*), y* = g - j b
            # -> P = g*(v_i - Re(W_ij)) - b*Im(W_ij)
            return gij * (mm.v[i] - Wre_ord(i, j)) - bij * (Wim_ord(i, j))

        (u, v) = oltc_base(i, j)  # directed base with tap on u-side
        # use same formulas as your given model
        if (i, j) == (u, v):
            # P_uv = sum Re((delta x_ii - alpha(x_ijR + j x_ijI)) y*)
            s = 0.0
            for t in taps_by_T[(u, v)]:
                a = mm.alpha[u, v, t]
                d = mm.delta[u, v, t]
                s += gij * (d*mm.x_ii[u, v, t] - a*mm.x_ijR[u, v, t]) - bij * (a*mm.x_ijI[u, v, t])
            return s
        else:
            # P_vu = sum Re((x_jj - alpha(x_ijR - j x_ijI)) y*)
            s = 0.0
            for t in taps_by_T[(u, v)]:
                a = mm.alpha[u, v, t]
                s += gij * (mm.x_jj[u, v, t] - a*mm.x_ijR[u, v, t]) + bij * (a*mm.x_ijI[u, v, t])
            return s

    def Q_expr(mm, i, j):
        e = edge_key(i, j)
        gij = mm.g[e]
        bij = mm.b[e]
        bcij = mm.bc[e]

        if not is_oltc_edge(i, j):
            # Q_ij = Im((W_ii - W_ij) y*) - (bc/2) W_ii
            # -> Q = -g*Im(W_ij) - b*(v_i - Re(W_ij)) - (bc/2)*v_i
            return (-gij) * (Wim_ord(i, j)) - bij * (mm.v[i] - Wre_ord(i, j)) - 0.5*bcij*mm.v[i]

        (u, v) = oltc_base(i, j)
        if (i, j) == (u, v):
            s = 0.0
            for t in taps_by_T[(u, v)]:
                a = mm.alpha[u, v, t]
                d = mm.delta[u, v, t]
                # Im part: -g*(a x_ijI) - b*(d x_ii - a x_ijR)
                s += (-gij) * (a*mm.x_ijI[u, v, t]) - bij * (d*mm.x_ii[u, v, t] - a*mm.x_ijR[u, v, t]) \
                     - 0.5*bcij*(d*mm.x_ii[u, v, t])
            return s
        else:
            s = 0.0
            for t in taps_by_T[(u, v)]:
                a = mm.alpha[u, v, t]
                # Im part: +g*(a x_ijI) - b*(x_jj - a x_ijR)
                s += (gij) * (a*mm.x_ijI[u, v, t]) - bij * (mm.x_jj[u, v, t] - a*mm.x_ijR[u, v, t]) \
                     - 0.5*bcij*(mm.x_jj[u, v, t])
            return s

    m.P = pyo.Expression(m.A, rule=P_expr)
    m.Q = pyo.Expression(m.A, rule=Q_expr)

    # ------------------------------------------------------------
    # Constraints
    # ------------------------------------------------------------

    # (SOCP relaxation) 2x2 principal minor:
    # || [2Re(Wij), 2Im(Wij), v_i - v_j] ||_2 <= v_i + v_j
    # implemented as convex quadratic inequality:
    m.con_soc = pyo.Constraint(
        m.E,
        rule=lambda mm, i, j: (2*mm.Wre[i, j])**2 + (2*mm.Wim[i, j])**2 + (mm.v[i] - mm.v[j])**2 <= (mm.v[i] + mm.v[j])**2
    )

    # OLTC one-hot: sum_t beta = 1 for each (i,j) in T
    m.con_onehot = pyo.Constraint(m.T, rule=lambda mm, i, j: sum(mm.beta[i, j, t] for t in taps_by_T[(i, j)]) == 1)

    # linking sums:
    # sum x_ii = v_i, sum x_jj = v_j
    # sum x_ijR = Re(W_ij), sum x_ijI = Im(W_ij)  (IMPORTANT: ordered pair (i,j) may have i>j)
    m.con_link = pyo.ConstraintList()
    for (i, j) in T:
        e = edge_key(i, j)
        # x sums
        m.con_link.add(sum(m.x_ii[i, j, t] for t in taps_by_T[(i, j)]) == m.v[i])
        m.con_link.add(sum(m.x_jj[i, j, t] for t in taps_by_T[(i, j)]) == m.v[j])
        m.con_link.add(sum(m.x_ijR[i, j, t] for t in taps_by_T[(i, j)]) == m.Wre[e])
        # Im(W_ij) depends on ordering
        if i < j:
            m.con_link.add(sum(m.x_ijI[i, j, t] for t in taps_by_T[(i, j)]) == m.Wim[e])
        else:
            m.con_link.add(sum(m.x_ijI[i, j, t] for t in taps_by_T[(i, j)]) == -m.Wim[e])

    # McCormick envelopes for x = beta * bounded var
    m.con_mcc = pyo.ConstraintList()
    for (i, j, t) in TAP:
        e = edge_key(i, j)
        wb = float(data["wbar"][e])

        # x_ii = beta * v_i
        m.con_mcc.add(m.x_ii[i, j, t] >= m.vL[i] * m.beta[i, j, t])
        m.con_mcc.add(m.x_ii[i, j, t] <= m.vU[i] * m.beta[i, j, t])
        m.con_mcc.add(m.x_ii[i, j, t] >= m.v[i] - m.vU[i] * (1 - m.beta[i, j, t]))
        m.con_mcc.add(m.x_ii[i, j, t] <= m.v[i] - m.vL[i] * (1 - m.beta[i, j, t]))

        # x_jj = beta * v_j
        m.con_mcc.add(m.x_jj[i, j, t] >= m.vL[j] * m.beta[i, j, t])
        m.con_mcc.add(m.x_jj[i, j, t] <= m.vU[j] * m.beta[i, j, t])
        m.con_mcc.add(m.x_jj[i, j, t] >= m.v[j] - m.vU[j] * (1 - m.beta[i, j, t]))
        m.con_mcc.add(m.x_jj[i, j, t] <= m.v[j] - m.vL[j] * (1 - m.beta[i, j, t]))

        # x_ijR = beta * Re(W_ij) (Re is symmetric)
        m.con_mcc.add(m.x_ijR[i, j, t] >= -wb * m.beta[i, j, t])
        m.con_mcc.add(m.x_ijR[i, j, t] <=  wb * m.beta[i, j, t])
        m.con_mcc.add(m.x_ijR[i, j, t] >= m.Wre[e] - wb * (1 - m.beta[i, j, t]))
        m.con_mcc.add(m.x_ijR[i, j, t] <= m.Wre[e] + wb * (1 - m.beta[i, j, t]))

        # x_ijI = beta * Im(W_ij)  (ordered pair!)
        # target Im(W_ij) = +Wim[e] if i<j else -Wim[e]
        target_im = m.Wim[e] if i < j else -m.Wim[e]
        m.con_mcc.add(m.x_ijI[i, j, t] >= -wb * m.beta[i, j, t])
        m.con_mcc.add(m.x_ijI[i, j, t] <=  wb * m.beta[i, j, t])
        m.con_mcc.add(m.x_ijI[i, j, t] >= target_im - wb * (1 - m.beta[i, j, t]))
        m.con_mcc.add(m.x_ijI[i, j, t] <= target_im + wb * (1 - m.beta[i, j, t]))

    # Shunt: z_i = a_i * v_i (McCormick)
    m.con_sh = pyo.ConstraintList()
    for i in C:
        m.con_sh.add(m.z[i] <= m.vU[i] * m.a_sh[i])
        m.con_sh.add(m.z[i] >= m.vL[i] * m.a_sh[i])
        m.con_sh.add(m.z[i] <= m.v[i] - m.vL[i] * (1 - m.a_sh[i]))
        m.con_sh.add(m.z[i] >= m.v[i] - m.vU[i] * (1 - m.a_sh[i]))

    # KCL:
    # pG - Pd = sum_out P_ij
    # qG - Qd + qsh = sum_out Q_ij
    m.con_kclP = pyo.Constraint(
        m.N,
        rule=lambda mm, i: mm.pG[i] - mm.Pd[i] == sum(mm.P[i, j] for j in out[i])
    )
    m.con_kclQ = pyo.Constraint(
        m.N,
        rule=lambda mm, i: mm.qG[i] - mm.Qd[i] + mm.qsh[i] == sum(mm.Q[i, j] for j in out[i])
    )

    # thermal limits on each directed arc: ||[P,Q]||_2 <= Smax  -> P^2 + Q^2 <= Smax^2
    m.con_thermal = pyo.Constraint(
        m.A,
        rule=lambda mm, i, j: mm.P[i, j]**2 + mm.Q[i, j]**2 <= (mm.Smax[edge_key(i, j)])**2
    )

    # Objective in EUR (same scaling as your MINLP):
    # sum (c2*(sn*Pg)^2 + c1*(sn*Pg) + c0)
    m.obj = pyo.Objective(
        expr=sum(m.c2[g] * (sn*m.Pg[g])**2 + m.c1[g] * (sn*m.Pg[g]) + m.c0[g] for g in m.G),
        sense=pyo.minimize
    )

    # attach for reporting
    m._data = data
    m._gens = gens
    m._taps_by_T = taps_by_T
    return m


def solve_with_scip(model, tee=True, timelimit=None, mipgap=1e-4):
    # Try forcing NL interface first (robust for quadratic/nonlinear forms)
    opt = pyo.SolverFactory("scip", solver_io="nl")
    if opt is None or not opt.available(exception_flag=False):
        opt = pyo.SolverFactory("scip")
    if opt is None or not opt.available(exception_flag=False):
        raise RuntimeError("SCIP solver is not available in Pyomo. But your env shows scip installed.")

    # SCIP options
    # (names are consistent with what you used in ACOPF_MINLP.py)
    if timelimit is not None:
        opt.options["limits/time"] = float(timelimit)
    opt.options["limits/gap"] = float(mipgap)  # relative gap
    # verbosity
    opt.options["display/verblevel"] = 4 if tee else 0

    return opt.solve(model, tee=tee)


def report_solution(m):
    sn = m._data["sn"]
    print("\n====================")
    print("Solved by SCIP (Pyomo MISOCP, full taps kept).")
    print(f"Objective (EUR) = {pyo.value(m.obj):.6f}")
    print("====================\n")

    print("Bus voltages:")
    for i in m.N:
        vi = pyo.value(m.v[i])
        print(f"  bus {i:2d}: v={vi:.6f}, |V|={math.sqrt(max(vi,0.0)):.6f}")

    print("\nGenerators (MW/MVAr):")
    for g in m.G:
        rec = m._gens[int(g)]
        Pg = sn * pyo.value(m.Pg[g])
        Qg = sn * pyo.value(m.Qg[g])
        print(f"  {rec['type']}[{rec['id']}] @ bus {rec['bus']}: P={Pg:.4f} MW, Q={Qg:.4f} MVAr")

    print("\nOLTC taps:")
    for (i, j) in m.T:
        chosen = None
        for t in m._taps_by_T[(i, j)]:
            if pyo.value(m.beta[i, j, t]) > 0.5:
                chosen = t
                break
        cfg = OLTC_BRANCHES[(i, j)]
        print(f"  ({i},{j}) tap={chosen}   range=[{cfg['tap_min']},{cfg['tap_max']}], dV%={cfg['dV_percent']}")

    print("\nShunts:")
    for i in m.C:
        a = int(round(pyo.value(m.a_sh[i])))
        qinj = sn * (pyo.value(m.bcap[i]) * pyo.value(m.z[i]))
        print(f"  bus {i:2d}: a_sh={a}, q_sh={qinj:.3f} MVAr (rated={SHUNTS[i]['q_rated_mvar']} MVAr)")


def main():
    net = mcase.busradial9_opf(slack_vm_pu=1.0, line_max_loading_percent=1e6)
    model = build_misocp_model(net)

    # Solve with SCIP (no license size limit like Gurobi)
    solve_with_scip(model, tee=True, timelimit=300, mipgap=1e-4)

    report_solution(model)


if __name__ == "__main__":
    main()
