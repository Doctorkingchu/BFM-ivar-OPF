# DCOPF.py
# ------------------------------------------------------------
# Pyomo MIQP DC-OPF with OLTC tap selection (one-hot) via Big-M
# - Aligns notation with your ACOPF code style
# - Shunt capacitor excluded (as requested)
# - Uses pandapower net builder (ieee9bus / ieee9bus1) for data
# - Solves with SCIP (recommended) and ACCEPTS "gap limit reached"
#   even if Pyomo termination_condition is reported as "other".
#
# How to run:
#   python DCOPF.py
#
# Requirements:
#   pip install pyomo numpy pandapower
#   (and SCIP installed + accessible by Pyomo "scip" solver)
# ------------------------------------------------------------

import math
import time
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional, Any

import numpy as np
import pyomo.environ as pyo
from pyomo.opt import SolverStatus, TerminationCondition

import ieee9bus as m  # radial: ieee9bus1, meshed: ieee9bus


# ============================
# User-tunable global settings
# ============================

DEFAULT_SOLVER = "scip"  # MIQP solver

# SCIP limits
SCIP_TIME_LIMIT = 180          # seconds
SCIP_GAP_LIMIT = 1e-4          # relative MIP gap, e.g., 1e-4 = 0.01%
SCIP_MEMORY_LIMIT_MB = 4096    # MB
SCIP_NODE_LIMIT = 300000

# Big-M / bounds safety
THETA_MAX = math.pi            # bound for theta vars [-pi, pi]
TEE_SOLVER_LOG = True          # show solver logs


# ----------------------------
# Configuration containers
# ----------------------------
@dataclass
class OLTCBranchConfig:
    tap_min: int
    tap_max: int
    dV_percent: float  # e.g., 1.25 means 1.25% per tap step


@dataclass
class BuildConfig:
    oltc_branches: Dict[Tuple[int, int], OLTCBranchConfig]
    fix_slack_angle: bool = True


# ----------------------------
# Helpers: branch list + pu
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


def extract_dc_miqp_data(net, cfg: BuildConfig) -> Dict[str, Any]:
    """
    Build DCOPF data with OLTC taps.

    DC model assumptions:
      - |V| ~= 1 pu, ignore Q, ignore losses
      - line flow: f_ij = b_ij * (theta_i - theta_j)
        where b_ij is series susceptance of the *branch model*
      - thermal/flow limit: |f_ij| <= Fmax_ij

    OLTC modeling (DC approximation):
      - For OLTC branch (i,j), tap changes effective series susceptance.
      - We use a simple scaling consistent with your ACOPF "alpha=1/tau":
            b_ij^{tap} = b_ij * alpha_ij^{tap}
        where tau = 1 + tap*dV%/100, alpha=1/tau
      - Then f_ij^{tap} = b_ij^{tap} (theta_i - theta_j) when beta=1.
    """
    sn = float(net.sn_mva)

    # buses
    buses = [int(i) for i in net.bus.index]
    nbus = len(buses)

    # slack (assume first ext_grid)
    if len(net.ext_grid.index) < 1:
        raise ValueError("pandapower net must have an ext_grid (slack).")
    eg0 = int(net.ext_grid.index[0])
    slack_bus = int(net.ext_grid.at[eg0, "bus"])

    # loads aggregated per bus (MW -> pu)
    Pd = {i: 0.0 for i in buses}
    if hasattr(net, "load") and len(net.load.index) > 0:
        for li in net.load.index:
            b = int(net.load.at[li, "bus"])
            Pd[b] += float(net.load.at[li, "p_mw"])
    Pd_pu = {i: Pd[i] / sn for i in buses}

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

    # generators = ext_grid + gen (active power only)
    gen_records = []

    # ext_grid
    for eg in net.ext_grid.index:
        eg = int(eg)
        b = int(net.ext_grid.at[eg, "bus"])
        pmin = float(net.ext_grid.at[eg, "min_p_mw"]) if "min_p_mw" in net.ext_grid.columns else -1e9
        pmax = float(net.ext_grid.at[eg, "max_p_mw"]) if "max_p_mw" in net.ext_grid.columns else 1e9
        c2, c1, c0 = _find_poly_cost("ext_grid", eg)
        gen_records.append({
            "type": "ext_grid",
            "id": eg,
            "bus": b,
            "pmin_pu": pmin / sn,
            "pmax_pu": pmax / sn,
            "c2": c2, "c1": c1, "c0": c0,
        })

    # gen
    if hasattr(net, "gen") and len(net.gen.index) > 0:
        for gi in net.gen.index:
            gi = int(gi)
            b = int(net.gen.at[gi, "bus"])
            pmin = float(net.gen.at[gi, "min_p_mw"])
            pmax = float(net.gen.at[gi, "max_p_mw"])
            c2, c1, c0 = _find_poly_cost("gen", gi)
            gen_records.append({
                "type": "gen",
                "id": gi,
                "bus": b,
                "pmin_pu": pmin / sn,
                "pmax_pu": pmax / sn,
                "c2": c2, "c1": c1, "c0": c0,
            })

    # branches from lines
    branches = _build_branch_list_from_lines(net)
    E = [(fb, tb) for _, fb, tb in branches]
    E_set = set(E)

    # per-unit series susceptance b_ij = imag(1/(r+jx))  (NOTE: negative for inductive)
    # DC uses -1/x in many texts; here we keep b_ij = imag(y) and use f=b*(theta_i-theta_j)
    # It works consistently as long as sign is consistent.
    bus_vn_kv = {int(i): float(net.bus.at[i, "vn_kv"]) for i in buses}

    b = {}
    fmax_pu = {}

    for e_id, fb, tb in branches:
        r_ohm = float(net.line.at[e_id, "r_ohm_per_km"]) * float(net.line.at[e_id, "length_km"])
        x_ohm = float(net.line.at[e_id, "x_ohm_per_km"]) * float(net.line.at[e_id, "length_km"])
        zbase = _zbase_ohm(bus_vn_kv[fb], sn)
        r_pu = r_ohm / zbase
        x_pu = x_ohm / zbase

        if (r_pu == 0.0 and x_pu == 0.0):
            y = complex(0.0, 0.0)
        else:
            y = 1.0 / complex(r_pu, x_pu)

        b[(fb, tb)] = float(y.imag)  # susceptance

        # line max_i_ka -> Smax -> Pmax approx (DC uses active flow)
        Imax = float(net.line.at[e_id, "max_i_ka"]) if "max_i_ka" in net.line.columns else 1e9
        Vkv = bus_vn_kv[fb]
        Smax_mva = math.sqrt(3.0) * Vkv * Imax
        fmax_pu[(fb, tb)] = (Smax_mva / sn)

    # OLTC sets from cfg (must exist on a directed edge in net.line)
    T = []
    K = {}
    alpha_tap = {}   # alpha=1/tau
    b_tap = {}       # b^{tap} for DC flow
    for (u, v), tcfg in cfg.oltc_branches.items():
        if (u, v) in E_set:
            ij = (u, v)
        elif (v, u) in E_set:
            ij = (v, u)
        else:
            raise ValueError(f"OLTC branch {(u, v)} not found in net.line directed edges.")
        T.append(ij)

        taps = list(range(int(tcfg.tap_min), int(tcfg.tap_max) + 1))
        K[ij] = taps

        base_b = b[ij]
        for tap in taps:
            tau = 1.0 + (tap * float(tcfg.dV_percent)) / 100.0
            alpha = 1.0 / tau
            alpha_tap[(ij, int(tap))] = alpha
            # DC approx: susceptance scales with alpha (simple, stable)
            b_tap[(ij, int(tap))] = base_b * alpha

    # Directed arc set A for nodal balance
    # Use both directions for each physical line so we can write sum_out f_ij = injection.
    A = []
    for (i, j) in E:
        A.append((i, j))
        A.append((j, i))

    # For reverse arcs, use same magnitude but consistent sign:
    # If original has b(i,j), define b(j,i)=b(i,j) (flow eq uses theta_j-theta_i, so it flips naturally).
    for (i, j) in E:
        if (j, i) not in b:
            b[(j, i)] = b[(i, j)]
        if (j, i) not in fmax_pu:
            fmax_pu[(j, i)] = fmax_pu[(i, j)]

    # For OLTC reverse arc: if OLTC is on (i,j) directed, we will model OLTC on that directed pair only.
    # If you want OLTC to apply regardless of direction, place it in cfg on the directed edge you prefer.

    # Big-M suggestion:
    # |b_tap|*|theta_i-theta_j| <= |b_tap|*(2*THETA_MAX) -> use max over taps/lines
    bmax = 0.0
    for (i, j) in E:
        bmax = max(bmax, abs(b[(i, j)]))
    for key, val in b_tap.items():
        bmax = max(bmax, abs(val))
    Mf = float(bmax * (2.0 * THETA_MAX) + 1e-6)  # safe margin

    return {
        "sn_mva": sn,
        "buses": buses,
        "slack_bus": slack_bus,
        "Pd_pu": Pd_pu,
        "gen_records": gen_records,

        "E": E,          # directed as in net.line
        "A": A,          # both directions

        "b": b,
        "Fmax": fmax_pu,

        "T": T,
        "K": K,
        "b_tap": b_tap,

        "Mf": Mf,
        "theta_max": THETA_MAX,
        "fix_slack_angle": cfg.fix_slack_angle,
    }


# ----------------------------
# Pyomo model builder (MIQP)
# ----------------------------
def build_miqp_dcopf_model(data: Dict[str, Any]) -> pyo.ConcreteModel:
    sn = data["sn_mva"]
    buses = data["buses"]
    slack_bus = data["slack_bus"]
    Pd = data["Pd_pu"]

    gen_records = data["gen_records"]
    Gset = list(range(len(gen_records)))

    A = data["A"]
    b = data["b"]
    Fmax = data["Fmax"]

    T = data["T"]
    K = data["K"]
    b_tap = data["b_tap"]

    Mf = data["Mf"]
    theta_max = data["theta_max"]

    model = pyo.ConcreteModel(name="DC_MIQP_OLTC")

    # Sets
    model.N = pyo.Set(initialize=buses, ordered=True)
    model.G = pyo.Set(initialize=Gset, ordered=True)
    model.A = pyo.Set(initialize=A, dimen=2, ordered=True)     # directed arcs
    model.T = pyo.Set(initialize=T, dimen=2, ordered=True)     # OLTC directed arcs

    # Parameters: loads
    model.Pd = pyo.Param(model.N, initialize=lambda m, i: float(Pd[i]), mutable=False)

    # Generator params
    model.gen_bus = pyo.Param(model.G, initialize=lambda m, gg: int(gen_records[gg]["bus"]), within=pyo.Any)
    model.Pgmin = pyo.Param(model.G, initialize=lambda m, gg: float(gen_records[gg]["pmin_pu"]))
    model.Pgmax = pyo.Param(model.G, initialize=lambda m, gg: float(gen_records[gg]["pmax_pu"]))
    model.c2 = pyo.Param(model.G, initialize=lambda m, gg: float(gen_records[gg]["c2"]))
    model.c1 = pyo.Param(model.G, initialize=lambda m, gg: float(gen_records[gg]["c1"]))
    model.c0 = pyo.Param(model.G, initialize=lambda m, gg: float(gen_records[gg]["c0"]))

    # Line params (for non-OLTC arcs too)
    model.b = pyo.Param(model.A, initialize=lambda m, i, j: float(b[(i, j)]))
    model.Fmax = pyo.Param(model.A, initialize=lambda m, i, j: float(Fmax[(i, j)]))

    # OLTC tap index set (i,j,tap)
    beta_index = []
    for (i, j) in T:
        for tap in K[(i, j)]:
            beta_index.append((i, j, int(tap)))
    model.BETA_INDEX = pyo.Set(initialize=beta_index, dimen=3, ordered=True)

    model.b_tap = pyo.Param(
        model.BETA_INDEX,
        initialize=lambda m, i, j, tap: float(b_tap[((i, j), int(tap))]),
        mutable=False
    )

    # Variables
    model.Pg = pyo.Var(model.G, bounds=lambda m, gg: (m.Pgmin[gg], m.Pgmax[gg]))
    model.theta = pyo.Var(model.N, bounds=(-theta_max, theta_max))

    # Active flow on arcs
    model.f = pyo.Var(model.A)

    # OLTC tap selection
    model.beta = pyo.Var(model.BETA_INDEX, within=pyo.Binary)
    model.f_tap = pyo.Var(model.BETA_INDEX)  # f_{ij}^{tap}

    # Slack angle
    if data["fix_slack_angle"]:
        model.slack_angle = pyo.Constraint(expr=model.theta[slack_bus] == 0.0)

    # Power balance:
    # injection(i) = sum_{(i,j) in A} f_{ij}
    # injection(i) = sum_g at i Pg - Pd_i
    def p_balance_rule(m, i):
        inj = sum(m.Pg[gg] for gg in m.G if int(m.gen_bus[gg]) == int(i)) - m.Pd[i]
        return inj == sum(m.f[i, j] for (ii, j) in m.A if ii == i)
    model.p_balance = pyo.Constraint(model.N, rule=p_balance_rule)

    # Identify OLTC arcs set for quick membership
    T_set = set(T)

    # DC flow for NON-OLTC arcs:
    # f_ij = b_ij (theta_i - theta_j)
    def dc_flow_non_oltc_rule(m, i, j):
        if (i, j) in T_set:
            return pyo.Constraint.Skip
        return m.f[i, j] == m.b[i, j] * (m.theta[i] - m.theta[j])
    model.dc_flow_non_oltc = pyo.Constraint(model.A, rule=dc_flow_non_oltc_rule)

    # OLTC one-hot
    def onehot_rule(m, i, j):
        taps = K[(i, j)]
        return sum(m.beta[i, j, int(t)] for t in taps) == 1
    model.onehot = pyo.Constraint(model.T, rule=onehot_rule)

    # Aggregate f_ij equals sum of tap flows
    def agg_flow_rule(m, i, j):
        taps = K[(i, j)]
        return m.f[i, j] == sum(m.f_tap[i, j, int(t)] for t in taps)
    model.agg_flow = pyo.Constraint(model.T, rule=agg_flow_rule)

    # Big-M linking:
    # if beta=1 => f_tap = b_tap*(theta_i - theta_j)
    # Implement as two inequalities (no ranged inequality with variable bounds)
    def tap_link_ub_rule(m, i, j, tap):
        return m.f_tap[i, j, tap] - m.b_tap[i, j, tap] * (m.theta[i] - m.theta[j]) <= Mf * (1 - m.beta[i, j, tap])

    def tap_link_lb_rule(m, i, j, tap):
        return m.f_tap[i, j, tap] - m.b_tap[i, j, tap] * (m.theta[i] - m.theta[j]) >= -Mf * (1 - m.beta[i, j, tap])

    model.tap_link_ub = pyo.Constraint(model.BETA_INDEX, rule=tap_link_ub_rule)
    model.tap_link_lb = pyo.Constraint(model.BETA_INDEX, rule=tap_link_lb_rule)

    # Tighten: when beta=0 => f_tap=0 by bounds
    def tap_flow_ub_rule(m, i, j, tap):
        return m.f_tap[i, j, tap] <= m.Fmax[i, j] * m.beta[i, j, tap]

    def tap_flow_lb_rule(m, i, j, tap):
        return m.f_tap[i, j, tap] >= -m.Fmax[i, j] * m.beta[i, j, tap]

    model.tap_flow_ub = pyo.Constraint(model.BETA_INDEX, rule=tap_flow_ub_rule)
    model.tap_flow_lb = pyo.Constraint(model.BETA_INDEX, rule=tap_flow_lb_rule)

    # Line flow limits for all arcs
    def flow_ub_rule(m, i, j):
        return m.f[i, j] <= m.Fmax[i, j]

    def flow_lb_rule(m, i, j):
        return m.f[i, j] >= -m.Fmax[i, j]

    model.flow_ub = pyo.Constraint(model.A, rule=flow_ub_rule)
    model.flow_lb = pyo.Constraint(model.A, rule=flow_lb_rule)

    # Objective (Pg in pu -> MW)
    # This is quadratic in Pg -> MIQP
    model.obj = pyo.Objective(
        rule=lambda m: sum(
            m.c2[gg] * (sn * m.Pg[gg]) ** 2 + m.c1[gg] * (sn * m.Pg[gg]) + m.c0[gg]
            for gg in m.G
        ),
        sense=pyo.minimize
    )

    return model


# ----------------------------
# Solver wrapper (accept gap)
# ----------------------------
def solve_miqp(model: pyo.ConcreteModel,
               solver_name: str = "scip",
               timelimit: Optional[float] = None,
               mipgap: Optional[float] = None,
               memlimit_mb: Optional[int] = None,
               nodelimit: Optional[int] = None,
               tee: bool = True):
    opt = pyo.SolverFactory(solver_name)
    if opt is None or not opt.available(exception_flag=False):
        raise RuntimeError(f"Solver '{solver_name}' not available in Pyomo.")

    sname = solver_name.lower()

    if sname == "scip":
        if timelimit is not None:
            opt.options["limits/time"] = float(timelimit)
        if nodelimit is not None:
            opt.options["limits/nodes"] = int(nodelimit)
        if memlimit_mb is not None:
            opt.options["limits/memory"] = float(memlimit_mb)
        if mipgap is not None:
            opt.options["limits/gap"] = float(mipgap)
        # optional: quieter
        # opt.options["display/verblevel"] = 4 if tee else 0

    res = opt.solve(model, tee=tee, load_solutions=True)

    tc = res.solver.termination_condition
    st = res.solver.status
    msg = (getattr(res.solver, "message", "") or "")
    msg_low = msg.lower()

    # detect whether variable values got loaded
    has_values = any(
        (v.value is not None)
        for v in model.component_data_objects(pyo.Var, active=True, descend_into=True)
    )

    # Accept "other" if it looks like "gap limit reached" or values exist
    acceptable_tc = {
        TerminationCondition.optimal,
        TerminationCondition.locallyOptimal,
        TerminationCondition.feasible,
        TerminationCondition.maxTimeLimit,
        TerminationCondition.maxIterations,
        TerminationCondition.other,
    }

    gapish = ("gap" in msg_low) or ("gap limit" in msg_low) or ("gap limit reached" in msg_low)

    ok_status = (st in {SolverStatus.ok, SolverStatus.warning})
    ok_tc = (tc in acceptable_tc)

    print(f"[INFO] solver.status = {st}")
    print(f"[INFO] termination_condition = {tc}")
    if msg.strip():
        print(f"[INFO] solver.message = {msg.strip()}")

    if ok_status and ok_tc and has_values:
        if tc == TerminationCondition.other and gapish:
            print("[INFO] Accepted solution: solver stopped due to gap/other condition, but a feasible incumbent is loaded.")
        elif tc == TerminationCondition.other:
            print("[WARN] termination_condition=other, but variable values are loaded. Proceeding with loaded solution.")
        return res

    raise RuntimeError(f"Solve failed or no solution loaded. status={st}, termination={tc}, message={msg}")


# ----------------------------
# Output
# ----------------------------
def _print_solution(model: pyo.ConcreteModel, data: Dict[str, Any]):
    sn = data["sn_mva"]
    buses = data["buses"]
    gen_records = data["gen_records"]
    slack_bus = data["slack_bus"]

    print("\n====================")
    print(" DC-OPF (MIQP) RESULT")
    print("====================")
    print("Objective (EUR):", float(pyo.value(model.obj)))

    print("\n--- Bus Angles (deg) ---")
    for i in buses:
        th_rad = pyo.value(model.theta[i])
        th_deg = th_rad * 180.0 / math.pi
        tag = " [slack]" if i == slack_bus else ""
        print(f"Bus {i}: theta={th_deg:+.6f}{tag}")

    print("\n--- Generator Dispatch (MW) ---")
    for gg in model.G:
        rec = gen_records[int(gg)]
        Pg_mw = sn * pyo.value(model.Pg[gg])
        print(f"{rec['type']}[{rec['id']}] @ bus {rec['bus']}: P={Pg_mw:.6f} MW")

    # OLTC tap results
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
                tap, beta_val = best
                # report implied b_tap
                b_eff = pyo.value(model.b_tap[i, j, tap])
                print(f"OLTC ({i},{j}): tap={tap} (beta={beta_val:.6f}), b_eff={b_eff:.6f}")

    # Flow check (only print physical directed edges E and their reverse)
    print("\n--- Line Flows f_ij (pu, MW approx) ---")
    printed = set()
    for (i, j) in data["E"]:
        if (i, j) in printed:
            continue
        printed.add((i, j))
        printed.add((j, i))
        fij = pyo.value(model.f[i, j])
        fji = pyo.value(model.f[j, i])
        print(f"Edge {i}<->{j}:  f({i}->{j})={fij:+.6f} pu ({sn*fij:+.3f} MW), "
              f"f({j}->{i})={fji:+.6f} pu ({sn*fji:+.3f} MW)")

    # Feasibility residuals (power balance)
    print("\n--- Nodal balance residuals (pu) ---")
    for i in buses:
        inj = sum(pyo.value(model.Pg[gg]) for gg in model.G if int(pyo.value(model.gen_bus[gg])) == int(i)) - float(data["Pd_pu"][i])
        out = sum(pyo.value(model.f[i, j]) for (ii, j) in data["A"] if ii == i)
        res = inj - out
        print(f"Bus {i}: inj - sum_out = {res:+.3e}")


# ----------------------------
# Main
# ----------------------------
def main():
    t0 = time.perf_counter()
    try:
        # Build pandapower net (same as your ACOPF code)
        net = m.busradial9_opf(slack_vm_pu=1.0, line_max_loading_percent=1e6)

        # ============================================================
        # OLTC spec: SAME as your ACOPF_MINLP
        # (Shunt capacitors excluded for DCOPF MIQP)
        # ============================================================
        oltc_branches: Dict[Tuple[int, int], OLTCBranchConfig] = {
            (3, 4): OLTCBranchConfig(tap_min=-8, tap_max=8, dV_percent=1.25),  # 17 taps
            (5, 6): OLTCBranchConfig(tap_min=-6, tap_max=6, dV_percent=1.25),  # 13 taps
            (7, 1): OLTCBranchConfig(tap_min=-4, tap_max=4, dV_percent=1.25),  # 9 taps
        }

        cfg = BuildConfig(oltc_branches=oltc_branches, fix_slack_angle=True)
        data = extract_dc_miqp_data(net, cfg)

        print("[INFO] DC-MIQP data summary")
        print(f"  #buses={len(data['buses'])}, #gens={len(data['gen_records'])}, #lines={len(data['E'])}")
        print(f"  #OLTC lines={len(data['T'])}, #beta-vars={sum(len(data['K'][ij]) for ij in data['T'])}")
        print(f"  Mf={data['Mf']:.6f}, theta_max={data['theta_max']:.6f} rad")

        # Build model
        model = build_miqp_dcopf_model(data)

        print(f"[INFO] Solving with solver='{DEFAULT_SOLVER}' (MIQP)...")
        _ = solve_miqp(
            model,
            solver_name=DEFAULT_SOLVER,
            timelimit=SCIP_TIME_LIMIT,
            mipgap=SCIP_GAP_LIMIT,
            memlimit_mb=SCIP_MEMORY_LIMIT_MB,
            nodelimit=SCIP_NODE_LIMIT,
            tee=TEE_SOLVER_LOG
        )

        # Print results
        _print_solution(model, data)

    finally:
        t1 = time.perf_counter()
        print(f"\n[INFO] Total elapsed time: {t1 - t0:.2f} seconds.")


if __name__ == "__main__":
    main()
