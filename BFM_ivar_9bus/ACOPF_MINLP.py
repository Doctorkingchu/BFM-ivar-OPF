# ACOPF_MINLP.py
# ------------------------------------------------------------
# Pyomo BIM AC-OPF / MINLP (OLTC + switched shunt embedded in Ybus)
#
# IMPORTANT:
# - If oltc_branches/shunts are empty, there are NO discrete vars -> NLP.
# - This version ADDS 3 OLTC branches + 5 switched shunts by default,
#   so you WILL see binary vars (beta, a_sh).
#
# How to run:
#   python ACOPF_MINLP.py
# ------------------------------------------------------------

import math
import time
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional, Any

import numpy as np
import pyomo.environ as pyo

import ieee9bus as m  # radial: ieee9bus1, meshed: ieee9bus


# ============================
# User-tunable global settings
# ============================

# --- NLP (IPOPT) stop conditions (fast, "good enough") ---
IPOPT_TOL = 1e-6
IPOPT_MAX_ITER = 2500
IPOPT_MAX_CPU_TIME = 60  # seconds (per solve)

# IPOPT "acceptable" (stop early if decent progress)
IPOPT_ACCEPTABLE_TOL = 1e-4
IPOPT_ACCEPTABLE_ITER = 10
IPOPT_ACCEPTABLE_CONSTR_VIOL = 1e-4

# --- MINLP (SCIP) hard limits to avoid runaway ---
SCIP_TIME_LIMIT = 300         # seconds
SCIP_GAP_LIMIT = 0.01          # 1% gap
SCIP_MEMORY_LIMIT_MB = 4096    # MB
SCIP_NODE_LIMIT = 30000

# --- NLP-based B&B heuristic limits ---
HEURISTIC_BB_MAX_NODES = 40     # keep small for speed
HEURISTIC_NODE_IPOPT_TIME = 20  # seconds per node IPOPT
HEURISTIC_FRAC_TOL = 1e-6

# When True, show solver logs (very verbose)
TEE_SOLVER_LOG = False


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
    v_rated_pu: float  # typically 1.0


@dataclass
class BuildConfig:
    # OLTC branches among existing line branches (from_bus,to_bus)
    oltc_branches: Dict[Tuple[int, int], OLTCBranchConfig]
    # switched shunts at buses
    shunts: Dict[int, ShuntConfig]
    # fix slack V magnitude
    fix_slack_vm: bool = True


# ----------------------------
# Helpers: extract network data
# ----------------------------
def _zbase_ohm(vn_kv: float, sn_mva: float) -> float:
    return (vn_kv ** 2) / sn_mva


def _build_branch_list_from_lines(net) -> List[Tuple[int, int, int]]:
    branches = []
    for e_id in net.line.index:
        fb = int(net.line.at[e_id, "from_bus"])
        tb = int(net.line.at[e_id, "to_bus"])
        branches.append((int(e_id), fb, tb))
    return branches


def extract_per_unit_data(net, cfg: BuildConfig) -> Dict[str, Any]:
    sn = float(net.sn_mva)

    # buses
    buses = [int(i) for i in net.bus.index]
    nbus = len(buses)
    bus_vn_kv = {int(i): float(net.bus.at[i, "vn_kv"]) for i in buses}
    vmin = {int(i): float(net.bus.at[i, "min_vm_pu"]) for i in buses}
    vmax = {int(i): float(net.bus.at[i, "max_vm_pu"]) for i in buses}

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

    # poly_cost (pandapower) for objective
    def _find_poly_cost(et: str, element: int):
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

    # generators = ext_grid + gen
    gen_records = []

    # ext_grid
    for eg in net.ext_grid.index:
        eg = int(eg)
        b = int(net.ext_grid.at[eg, "bus"])
        pmin = float(net.ext_grid.at[eg, "min_p_mw"]) if "min_p_mw" in net.ext_grid.columns else -1e9
        pmax = float(net.ext_grid.at[eg, "max_p_mw"]) if "max_p_mw" in net.ext_grid.columns else 1e9
        qmin = float(net.ext_grid.at[eg, "min_q_mvar"]) if "min_q_mvar" in net.ext_grid.columns else -1e9
        qmax = float(net.ext_grid.at[eg, "max_q_mvar"]) if "max_q_mvar" in net.ext_grid.columns else 1e9
        c2, c1, c0 = _find_poly_cost("ext_grid", eg)
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

    # gen
    if hasattr(net, "gen") and len(net.gen.index) > 0:
        for gi in net.gen.index:
            gi = int(gi)
            b = int(net.gen.at[gi, "bus"])
            pmin = float(net.gen.at[gi, "min_p_mw"])
            pmax = float(net.gen.at[gi, "max_p_mw"])
            qmin = float(net.gen.at[gi, "min_q_mvar"])
            qmax = float(net.gen.at[gi, "max_q_mvar"])
            c2, c1, c0 = _find_poly_cost("gen", gi)
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

    # branches from lines
    branches = _build_branch_list_from_lines(net)
    E = [(fb, tb) for _, fb, tb in branches]
    E_set = set(E)

    g = {}
    b = {}
    bc = {}
    smax_pu = {}

    for e_id, fb, tb in branches:
        r_ohm = float(net.line.at[e_id, "r_ohm_per_km"]) * float(net.line.at[e_id, "length_km"])
        x_ohm = float(net.line.at[e_id, "x_ohm_per_km"]) * float(net.line.at[e_id, "length_km"])

        zbase = _zbase_ohm(bus_vn_kv[fb], sn)
        r_pu = r_ohm / zbase
        x_pu = x_ohm / zbase

        y = 1.0 / complex(r_pu, x_pu) if (r_pu != 0.0 or x_pu != 0.0) else complex(0.0, 0.0)
        g[(fb, tb)] = float(y.real)
        b[(fb, tb)] = float(y.imag)
        bc[(fb, tb)] = 0.0

        Imax = float(net.line.at[e_id, "max_i_ka"]) if "max_i_ka" in net.line.columns else 1e9
        Vkv = bus_vn_kv[fb]
        Smax_mva = math.sqrt(3.0) * Vkv * Imax
        smax_pu[(fb, tb)] = Smax_mva / sn

    # OLTC sets from cfg
    T = []
    K = {}
    alpha_tap = {}
    delta_tap = {}

    for (u, v), tcfg in cfg.oltc_branches.items():
        # IMPORTANT: we only allow OLTC on an existing directed edge in net.line
        if (u, v) in E_set:
            ij = (u, v)
        elif (v, u) in E_set:
            ij = (v, u)
        else:
            raise ValueError(f"OLTC branch {(u, v)} not found in net.line directed edges.")
        T.append(ij)
        taps = list(range(int(tcfg.tap_min), int(tcfg.tap_max) + 1))
        K[ij] = taps
        for tap in taps:
            tau = 1.0 + (tap * float(tcfg.dV_percent)) / 100.0
            alpha_tap[(ij, tap)] = 1.0 / tau
            delta_tap[(ij, tap)] = 1.0 / (tau * tau)

    # shunt set C
    C = sorted([int(i) for i in cfg.shunts.keys()])
    bcap_pu = {}
    for i in C:
        scfg = cfg.shunts[i]
        q_pu = float(scfg.q_rated_mvar) / sn
        bcap_pu[i] = q_pu / float(scfg.v_rated_pu)

    # base Y0 excluding OLTC branches
    Y0 = np.zeros((nbus, nbus), dtype=complex)
    bus_to_pos = {buses[k]: k for k in range(nbus)}
    T_set = set(T)

    for (i, j) in E:
        if (i, j) in T_set:
            continue
        yij = complex(g[(i, j)], b[(i, j)])
        pi = bus_to_pos[i]
        pj = bus_to_pos[j]
        Y0[pi, pj] += -yij
        Y0[pj, pi] += -yij
        Y0[pi, pi] += yij + 1j * (bc[(i, j)] / 2.0)
        Y0[pj, pj] += yij + 1j * (bc[(i, j)] / 2.0)

    return {
        "sn_mva": sn,
        "buses": buses,
        "bus_to_pos": bus_to_pos,
        "slack_bus": slack_bus,
        "slack_vm_pu": slack_vm_pu,
        "vmin": vmin,
        "vmax": vmax,
        "Pd_pu": Pd_pu,
        "Qd_pu": Qd_pu,
        "gen_records": gen_records,
        "E": E,
        "g": g,
        "b": b,
        "bc": bc,
        "smax_pu": smax_pu,
        "T": T,
        "K": K,
        "alpha_tap": alpha_tap,
        "delta_tap": delta_tap,
        "C": C,
        "bcap_pu": bcap_pu,
        "G0": Y0.real,
        "B0": Y0.imag,
        "fix_slack_vm": cfg.fix_slack_vm,
    }


# ----------------------------
# Pyomo model builder
# ----------------------------
def build_pyomo_model(data: Dict[str, Any], relax_binaries: bool = False) -> pyo.ConcreteModel:
    sn = data["sn_mva"]
    buses = data["buses"]
    bus_to_pos = data["bus_to_pos"]
    slack_bus = data["slack_bus"]
    slack_vm_pu = data["slack_vm_pu"]

    E = data["E"]
    g = data["g"]
    b = data["b"]
    bc = data["bc"]
    smax_pu = data["smax_pu"]

    T = data["T"]
    K = data["K"]
    alpha_tap = data["alpha_tap"]
    delta_tap = data["delta_tap"]

    C = data["C"]
    bcap_pu = data["bcap_pu"]

    G0 = data["G0"]
    B0 = data["B0"]

    Pd = data["Pd_pu"]
    Qd = data["Qd_pu"]
    vmin = data["vmin"]
    vmax = data["vmax"]

    gen_records = data["gen_records"]
    Gset = list(range(len(gen_records)))

    model = pyo.ConcreteModel(name="BIM_MINLP_OLTC_SHUNT_YBUS")

    # Sets
    model.N = pyo.Set(initialize=buses, ordered=True)
    model.G = pyo.Set(initialize=Gset, ordered=True)
    model.E = pyo.Set(initialize=E, dimen=2, ordered=True)
    model.T = pyo.Set(initialize=T, dimen=2, ordered=True)
    model.C = pyo.Set(initialize=C, ordered=True)

    # Parameters: base admittance
    model.G0 = pyo.Param(
        model.N, model.N,
        initialize=lambda m, i, k: float(G0[bus_to_pos[i], bus_to_pos[k]]),
        mutable=False
    )
    model.B0 = pyo.Param(
        model.N, model.N,
        initialize=lambda m, i, k: float(B0[bus_to_pos[i], bus_to_pos[k]]),
        mutable=False
    )

    # Load parameters
    model.Pd = pyo.Param(model.N, initialize=lambda m, i: float(Pd[i]), mutable=False)
    model.Qd = pyo.Param(model.N, initialize=lambda m, i: float(Qd[i]), mutable=False)

    # Voltage bounds (V in pu)
    model.Vmin = pyo.Param(model.N, initialize=lambda m, i: float(vmin[i]), mutable=False)
    model.Vmax = pyo.Param(model.N, initialize=lambda m, i: float(vmax[i]), mutable=False)

    # Generator params
    model.gen_bus = pyo.Param(model.G, initialize=lambda m, gg: int(gen_records[gg]["bus"]), within=pyo.Any)
    model.Pgmin = pyo.Param(model.G, initialize=lambda m, gg: float(gen_records[gg]["pmin_pu"]))
    model.Pgmax = pyo.Param(model.G, initialize=lambda m, gg: float(gen_records[gg]["pmax_pu"]))
    model.Qgmin = pyo.Param(model.G, initialize=lambda m, gg: float(gen_records[gg]["qmin_pu"]))
    model.Qgmax = pyo.Param(model.G, initialize=lambda m, gg: float(gen_records[gg]["qmax_pu"]))
    model.c2 = pyo.Param(model.G, initialize=lambda m, gg: float(gen_records[gg]["c2"]))
    model.c1 = pyo.Param(model.G, initialize=lambda m, gg: float(gen_records[gg]["c1"]))
    model.c0 = pyo.Param(model.G, initialize=lambda m, gg: float(gen_records[gg]["c0"]))

    # Branch params
    model.g = pyo.Param(model.E, initialize=lambda m, i, j: float(g[(i, j)]))
    model.b = pyo.Param(model.E, initialize=lambda m, i, j: float(b[(i, j)]))
    model.bc = pyo.Param(model.E, initialize=lambda m, i, j: float(bc[(i, j)]))
    model.Smax = pyo.Param(model.E, initialize=lambda m, i, j: float(smax_pu[(i, j)]))

    # Shunt bcap
    model.bcap = pyo.Param(model.C, initialize=lambda m, i: float(bcap_pu[i]), default=0.0)

    # OLTC tap index set (i,j,tap)
    beta_index = []
    for (i, j) in T:
        for tap in K[(i, j)]:
            beta_index.append((i, j, int(tap)))
    model.BETA_INDEX = pyo.Set(initialize=beta_index, dimen=3, ordered=True)

    model.alpha_tap = pyo.Param(
        model.BETA_INDEX,
        initialize=lambda m, i, j, tap: float(alpha_tap[((i, j), int(tap))]),
        mutable=False
    )
    model.delta_tap = pyo.Param(
        model.BETA_INDEX,
        initialize=lambda m, i, j, tap: float(delta_tap[((i, j), int(tap))]),
        mutable=False
    )

    # Variables
    model.Pg = pyo.Var(model.G, bounds=lambda m, gg: (m.Pgmin[gg], m.Pgmax[gg]))
    model.Qg = pyo.Var(model.G, bounds=lambda m, gg: (m.Qgmin[gg], m.Qgmax[gg]))

    model.V = pyo.Var(model.N, bounds=lambda m, i: (m.Vmin[i], m.Vmax[i]))
    model.theta = pyo.Var(model.N, bounds=(-math.pi, math.pi))

    model.v = pyo.Var(model.N, bounds=lambda m, i: (m.Vmin[i] ** 2, m.Vmax[i] ** 2))

    model.Pinj = pyo.Var(model.N)
    model.Qinj = pyo.Var(model.N)

    model.Pij = pyo.Var(model.E)
    model.Qij = pyo.Var(model.E)

    if relax_binaries:
        model.beta = pyo.Var(model.BETA_INDEX, bounds=(0.0, 1.0))
        model.a_sh = pyo.Var(model.C, bounds=(0.0, 1.0))
    else:
        model.beta = pyo.Var(model.BETA_INDEX, within=pyo.Binary)
        model.a_sh = pyo.Var(model.C, within=pyo.Binary)

    model.alpha = pyo.Var(model.T)
    model.delta = pyo.Var(model.T)

    # Constraints
    model.v_def = pyo.Constraint(model.N, rule=lambda m, i: m.v[i] == m.V[i] ** 2)

    model.Pinj_def = pyo.Constraint(
        model.N,
        rule=lambda m, i: m.Pinj[i] == sum(m.Pg[gg] for gg in m.G if int(m.gen_bus[gg]) == int(i)) - m.Pd[i]
    )
    model.Qinj_def = pyo.Constraint(
        model.N,
        rule=lambda m, i: m.Qinj[i] == sum(m.Qg[gg] for gg in m.G if int(m.gen_bus[gg]) == int(i)) - m.Qd[i]
    )

    # OLTC selection: one-hot + alpha/delta linear combination
    def onehot_rule(m, i, j):
        taps = K[(i, j)]
        return sum(m.beta[i, j, int(t)] for t in taps) == 1
    model.onehot = pyo.Constraint(model.T, rule=onehot_rule)

    def alpha_sel_rule(m, i, j):
        taps = K[(i, j)]
        return m.alpha[i, j] == sum(m.alpha_tap[i, j, int(t)] * m.beta[i, j, int(t)] for t in taps)
    model.alpha_sel = pyo.Constraint(model.T, rule=alpha_sel_rule)

    def delta_sel_rule(m, i, j):
        taps = K[(i, j)]
        return m.delta[i, j] == sum(m.delta_tap[i, j, int(t)] * m.beta[i, j, int(t)] for t in taps)
    model.delta_sel = pyo.Constraint(model.T, rule=delta_sel_rule)

    # Slack constraints
    model.slack_angle = pyo.Constraint(expr=model.theta[slack_bus] == 0.0)
    if data["fix_slack_vm"]:
        model.slack_vm = pyo.Constraint(expr=model.V[slack_bus] == float(slack_vm_pu))

    # Ybus expressions
    T_set = set(T)
    C_set = set(C)

    def Gexpr(m, i, k):
        expr = m.G0[i, k]
        for (p, q) in T:
            gij = m.g[p, q]
            if (i == p and k == p):
                expr += gij * m.delta[p, q]
            elif (i == q and k == q):
                expr += gij
            elif (i == p and k == q):
                expr += -gij * m.alpha[p, q]
            elif (i == q and k == p):
                expr += -gij * m.alpha[p, q]
        return expr

    def Bexpr(m, i, k):
        expr = m.B0[i, k]
        for (p, q) in T:
            bij = m.b[p, q]
            bcij = m.bc[p, q]
            if (i == p and k == p):
                expr += (bij + 0.5 * bcij) * m.delta[p, q]
            elif (i == q and k == q):
                expr += (bij + 0.5 * bcij)
            elif (i == p and k == q):
                expr += -bij * m.alpha[p, q]
            elif (i == q and k == p):
                expr += -bij * m.alpha[p, q]
        if i == k and i in C_set:
            expr += -m.a_sh[i] * m.bcap[i]
        return expr

    model.Gik = pyo.Expression(model.N, model.N, rule=Gexpr)
    model.Bik = pyo.Expression(model.N, model.N, rule=Bexpr)

    # BIM power balance
    def BIM_P_rule(m, i):
        return m.Pinj[i] == m.V[i] * sum(
            m.V[k] * (m.Gik[i, k] * pyo.cos(m.theta[i] - m.theta[k]) +
                      m.Bik[i, k] * pyo.sin(m.theta[i] - m.theta[k]))
            for k in m.N
        )
    model.BIM_P = pyo.Constraint(model.N, rule=BIM_P_rule)

    def BIM_Q_rule(m, i):
        return m.Qinj[i] == m.V[i] * sum(
            m.V[k] * (m.Gik[i, k] * pyo.sin(m.theta[i] - m.theta[k]) -
                      m.Bik[i, k] * pyo.cos(m.theta[i] - m.theta[k]))
            for k in m.N
        )
    model.BIM_Q = pyo.Constraint(model.N, rule=BIM_Q_rule)

    # Branch flows for thermal limits
    def alphaE(m, i, j):
        return m.alpha[i, j] if (i, j) in T_set else 1.0

    def deltaE(m, i, j):
        return m.delta[i, j] if (i, j) in T_set else 1.0

    model.alphaE = pyo.Expression(model.E, rule=lambda m, i, j: alphaE(m, i, j))
    model.deltaE = pyo.Expression(model.E, rule=lambda m, i, j: deltaE(m, i, j))

    def Pij_rule(m, i, j):
        gij = m.g[i, j]
        bij = m.b[i, j]
        return m.Pij[i, j] == gij * m.deltaE[i, j] * m.v[i] - m.V[i] * m.V[j] * (
            gij * m.alphaE[i, j] * pyo.cos(m.theta[i] - m.theta[j]) +
            bij * m.alphaE[i, j] * pyo.sin(m.theta[i] - m.theta[j])
        )
    model.Pij_def = pyo.Constraint(model.E, rule=Pij_rule)

    def Qij_rule(m, i, j):
        gij = m.g[i, j]
        bij = m.b[i, j]
        bcij = m.bc[i, j]
        return m.Qij[i, j] == -(bij * m.deltaE[i, j] + 0.5 * bcij * m.deltaE[i, j]) * m.v[i] - m.V[i] * m.V[j] * (
            gij * m.alphaE[i, j] * pyo.sin(m.theta[i] - m.theta[j]) -
            bij * m.alphaE[i, j] * pyo.cos(m.theta[i] - m.theta[j])
        )
    model.Qij_def = pyo.Constraint(model.E, rule=Qij_rule)

    model.thermal = pyo.Constraint(
        model.E,
        rule=lambda m, i, j: m.Pij[i, j] ** 2 + m.Qij[i, j] ** 2 <= (m.Smax[i, j] ** 2)
    )

    # Objective (Pg in pu -> MW)
    model.obj = pyo.Objective(
        rule=lambda m: sum(
            m.c2[gg] * (sn * m.Pg[gg]) ** 2 + m.c1[gg] * (sn * m.Pg[gg]) + m.c0[gg]
            for gg in m.G
        ),
        sense=pyo.minimize
    )

    return model


# ----------------------------
# Solver utilities
# ----------------------------
def _count_discrete_vars(model: pyo.ConcreteModel) -> Tuple[int, int]:
    from pyomo.environ import Var
    nbin, nint = 0, 0
    for v in model.component_data_objects(Var, descend_into=True):
        if v.is_binary():
            nbin += 1
        elif v.is_integer():
            nint += 1
    return nbin, nint


def _solve_nlp(model: pyo.ConcreteModel,
               tee: bool = False,
               tol: float = IPOPT_TOL,
               max_iter: int = IPOPT_MAX_ITER,
               max_cpu_time: int = IPOPT_MAX_CPU_TIME) -> Optional[float]:
    for name in ["ipopt", "cyipopt", "appsi_ipopt"]:
        solver = pyo.SolverFactory(name)
        if solver is None or not solver.available(exception_flag=False):
            continue

        try:
            solver.options["tol"] = tol
            solver.options["max_iter"] = int(max_iter)
            solver.options["max_cpu_time"] = float(max_cpu_time)

            solver.options["acceptable_tol"] = float(IPOPT_ACCEPTABLE_TOL)
            solver.options["acceptable_iter"] = int(IPOPT_ACCEPTABLE_ITER)
            solver.options["acceptable_constr_viol_tol"] = float(IPOPT_ACCEPTABLE_CONSTR_VIOL)

            solver.options["print_level"] = 5 if tee else 0
        except Exception:
            pass

        res = solver.solve(model, tee=tee)
        tc = res.solver.termination_condition

        if tc in [
            pyo.TerminationCondition.optimal,
            pyo.TerminationCondition.locallyOptimal,
            pyo.TerminationCondition.feasible,
            pyo.TerminationCondition.maxIterations,
            pyo.TerminationCondition.maxTimeLimit,
        ]:
            return pyo.value(model.obj)

        return None

    raise RuntimeError("No NLP solver available (ipopt/cyipopt/appsi_ipopt).")


def _try_solve_with_scip_minlp(model: pyo.ConcreteModel,
                              time_limit_sec: int = SCIP_TIME_LIMIT,
                              gap_limit: float = SCIP_GAP_LIMIT,
                              memory_limit_mb: int = SCIP_MEMORY_LIMIT_MB,
                              node_limit: int = SCIP_NODE_LIMIT,
                              tee: bool = False) -> bool:
    solver = pyo.SolverFactory("scip")
    if solver is None or not solver.available(exception_flag=False):
        return False

    try:
        solver.options["limits/time"] = float(time_limit_sec)
        solver.options["limits/gap"] = float(gap_limit)
        solver.options["limits/memory"] = float(memory_limit_mb)  # MB
        solver.options["limits/nodes"] = int(node_limit)
        solver.options["display/verblevel"] = 4 if tee else 0
    except Exception:
        pass

    try:
        res = solver.solve(model, tee=tee)
        tc = res.solver.termination_condition
        if tc in [
            pyo.TerminationCondition.optimal,
            pyo.TerminationCondition.locallyOptimal,
            pyo.TerminationCondition.feasible,
            pyo.TerminationCondition.maxTimeLimit,
        ]:
            return True
    except Exception:
        return False

    return False


# ----------------------------
# NLP-based Branch-and-Bound (heuristic)
# ----------------------------
def solve_with_nlp_branch_and_bound(data: Dict[str, Any],
                                   max_nodes: int = HEURISTIC_BB_MAX_NODES,
                                   frac_tol: float = HEURISTIC_FRAC_TOL,
                                   tee: bool = False) -> Dict[str, Any]:
    best = {"obj": float("inf"), "sol": None}
    node_count = 0

    # branch variables: beta (i,j,tap) + a_sh(i)
    beta_keys = list(data["alpha_tap"].keys())  # ((i,j),tap)
    beta_triplets = [(ij[0], ij[1], int(tap)) for (ij, tap) in beta_keys]
    shunt_keys = list(data["C"])

    branch_vars = [("beta", key) for key in beta_triplets] + [("a_sh", i) for i in shunt_keys]

    def _get_val(model, kind, key):
        if kind == "beta":
            i, j, tap = key
            return pyo.value(model.beta[i, j, tap])
        else:
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
        best_frac = 0.0
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

        model = build_pyomo_model(data, relax_binaries=True)
        for kind, key, val in fixings:
            _fix(model, kind, key, val)

        obj = _solve_nlp(
            model,
            tee=tee,
            tol=IPOPT_TOL,
            max_iter=min(IPOPT_MAX_ITER, 1500),
            max_cpu_time=HEURISTIC_NODE_IPOPT_TIME
        )
        if obj is None:
            return

        if obj >= best["obj"] - 1e-9:
            return

        # check integrality
        all_int = True
        for kind, key in branch_vars:
            v = _get_val(model, kind, key)
            if not _is_integral(v):
                all_int = False
                break

        if all_int:
            best["obj"] = obj
            best["sol"] = model
            return

        pick = _pick_branch(model)
        if pick is None:
            best["obj"] = obj
            best["sol"] = model
            return

        kind, key, v = pick
        order = [1, 0] if v >= 0.5 else [0, 1]
        for val in order:
            dfs(fixings + [(kind, key, val)])

    dfs([])
    return {"best_obj": best["obj"], "best_model": best["sol"], "nodes": node_count}


# ----------------------------
# Output
# ----------------------------
def _print_solution(model: pyo.ConcreteModel, data: Dict[str, Any]):
    sn = data["sn_mva"]
    buses = data["buses"]
    gen_records = data["gen_records"]
    slack_bus = data["slack_bus"]

    print("\n--- Bus Voltages (pu) ---")
    for i in buses:
        th_rad = pyo.value(model.theta[i])
        th_deg = th_rad * 180.0 / math.pi
        print(f"Bus {i}: V={pyo.value(model.V[i]):.6f}, theta(deg)={th_deg:+.6f}"
              + ("  [slack]" if i == slack_bus else ""))

    print("\n--- Generator Dispatch (MW / Mvar) ---")
    for gg in model.G:
        rec = gen_records[int(gg)]
        Pg_mw = sn * pyo.value(model.Pg[gg])
        Qg_mvar = sn * pyo.value(model.Qg[gg])
        print(f"{rec['type']}[{rec['id']}] @ bus {rec['bus']}: P={Pg_mw:.4f} MW, Q={Qg_mvar:.4f} Mvar")

    if len(list(model.T)) > 0:
        print("\n--- OLTC taps (beta one-hot) ---")
        for (i, j) in model.T:
            best = None
            for (ii, jj, tap) in model.BETA_INDEX:
                if ii == i and jj == j:
                    v = pyo.value(model.beta[ii, jj, tap])
                    if best is None or v > best[1]:
                        best = (tap, v)
            if best is not None:
                print(f"OLTC ({i},{j}): tap={best[0]} (beta={best[1]:.6f}), "
                      f"alpha={pyo.value(model.alpha[i, j]):.6f}, delta={pyo.value(model.delta[i, j]):.6f}")

    if len(list(model.C)) > 0:
        print("\n--- Switched shunt status (a_sh) ---")
        for i in model.C:
            print(f"Shunt @ bus {i}: a_sh={pyo.value(model.a_sh[i]):.6f}, bcap_pu={pyo.value(model.bcap[i]):.6f}")

    print("\n--- Thermal check (|S_ij| in pu) ---")
    for (i, j) in model.E:
        P = pyo.value(model.Pij[i, j])
        Q = pyo.value(model.Qij[i, j])
        Smag = math.sqrt(max(P * P + Q * Q, 0.0))
        print(f"({i}->{j}): |S|={Smag:.6f} pu  <=  Smax={pyo.value(model.Smax[i, j]):.6f} pu")


# ----------------------------
# Main
# ----------------------------
def main():
    # Build pandapower net
    t0 = time.perf_counter()
    try:
        net = m.busradial9_opf(slack_vm_pu=1.0, line_max_loading_percent=1e6)

        # ============================================================
        # ADD DISCRETE DEVICES HERE (so bin vars exist for sure)
        # ============================================================
        # 1) OLTC 3개: 기존 선로 위에 "탭이 i-side에 있는 변압기"로 해석하여 적용
        #    - 반드시 net.line에 존재하는 (from_bus,to_bus) 또는 (to_bus,from_bus) 여야 함.
        #    - IEEE9 radial 구성에서 존재하는 선로: (0,3),(3,4),(4,5),(2,5),(5,6),(6,7),(7,1),(8,3)
        #
        # 2) Switched shunt 5개: 버스에 바이너리 on/off capacitor
        #    - q_rated_mvar는 + (capacitive)로 가정 -> B_ii에 (-a*bcap)로 들어가므로
        #      Q 방정식에서 "전압 상승(무효 주입)" 효과가 나도록 동작
        #      (네 수식의 부호 정의를 그대로 따름)
        #
        # NOTE: 너무 큰 q_rated를 주면 수치적으로 난해/비현실적일 수 있으니 5~25 Mvar 정도로 둠.
        oltc_branches: Dict[Tuple[int, int], OLTCBranchConfig] = {
            (3, 4): OLTCBranchConfig(tap_min=-8, tap_max=8, dV_percent=1.25),  # 17 taps
            (5, 6): OLTCBranchConfig(tap_min=-6, tap_max=6, dV_percent=1.25),  # 13 taps
            (7, 1): OLTCBranchConfig(tap_min=-4, tap_max=4, dV_percent=1.25),  # 9 taps
        }

        shunts: Dict[int, ShuntConfig] = {
            1: ShuntConfig(q_rated_mvar=10.0, v_rated_pu=1.0),
            4: ShuntConfig(q_rated_mvar=15.0, v_rated_pu=1.0),
            5: ShuntConfig(q_rated_mvar=20.0, v_rated_pu=1.0),
            6: ShuntConfig(q_rated_mvar=10.0, v_rated_pu=1.0),
            8: ShuntConfig(q_rated_mvar=8.0,  v_rated_pu=1.0),
        }

        # build config
        cfg = BuildConfig(oltc_branches=oltc_branches, shunts=shunts, fix_slack_vm=True)
        data = extract_per_unit_data(net, cfg)

        # Build MINLP model
        model = build_pyomo_model(data, relax_binaries=False)

        # Detect discrete vars
        nbin, nint = _count_discrete_vars(model)
        print(f"[INFO] discrete vars detected: bin={nbin}, int={nint}")
        print(f"[INFO] #OLTC branches={len(data['T'])}, #SwitchedShunts={len(data['C'])}, #BetaIndex={len(list(model.BETA_INDEX))}")

        # If no discrete vars (should NOT happen now)
        if (nbin + nint) == 0:
            print("[WARN] No discrete vars detected (unexpected). Solving NLP with IPOPT.")
            obj = _solve_nlp(model, tee=TEE_SOLVER_LOG)
            if obj is None:
                print("[FAIL] IPOPT could not find a feasible solution.")
                return
            print("\n[SOLVED] by IPOPT (NLP).")
            print("Objective (EUR):", obj)
            _print_solution(model, data)
            return

        # Try SCIP MINLP
        print("[INFO] Discrete vars present -> trying SCIP with strong limits.")
        ok = _try_solve_with_scip_minlp(
            model,
            time_limit_sec=SCIP_TIME_LIMIT,
            gap_limit=SCIP_GAP_LIMIT,
            memory_limit_mb=SCIP_MEMORY_LIMIT_MB,
            node_limit=SCIP_NODE_LIMIT,
            tee=TEE_SOLVER_LOG
        )
        if ok:
            print("\n[SOLVED] by SCIP (MINLP).")
            print("Objective (EUR):", pyo.value(model.obj))
            _print_solution(model, data)
            return

        # Fallback: heuristic NLP-B&B
        print("\n[INFO] SCIP not available/failed/limited. Running NLP-B&B heuristic (early stop).")
        result = solve_with_nlp_branch_and_bound(data, max_nodes=HEURISTIC_BB_MAX_NODES, tee=False)
        best_model = result["best_model"]
        if best_model is None:
            print("[FAIL] No feasible solution found by NLP-B&B heuristic.")
            return

        print("\n[SOLVED] by NLP-B&B heuristic.")
        print("Nodes explored:", result["nodes"])
        print("Best objective (EUR):", result["best_obj"])
        _print_solution(best_model, data)
    
    finally:
        t1 = time.perf_counter()
        print(f"\n[INFO] Total elapsed time: {t1 - t0:.2f} seconds.")


if __name__ == "__main__":
    main()
