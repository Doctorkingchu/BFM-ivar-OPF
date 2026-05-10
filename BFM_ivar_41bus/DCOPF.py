# DCOPF.py
# ------------------------------------------------------------
# Pyomo MIQP DC-OPF for modified explicit 41-bus IEEE 39-plus system
# with OLTC tap selection (one-hot) via Big-M
#
# - Shunt capacitor EXCLUDED
# - Objective quadratic in Pg (MW-based poly cost) -> MIQP
# - Constraints linear + binary -> MIQP
# - Solver: SCIP (recommended)
#
# IMPORTANT modeling choice for OLTC:
#   - OLTC is defined on the ordered directed arc (i,j).
#   - We enforce OLTC DC flow ONLY on that direction:
#         f_ij = sum_t f_ij^t, and if beta=1 then f_ij^t = b_ij^t (theta_i-theta_j)
#   - Reverse-direction flow is enforced by physics:
#         f_ji = - f_ij
#
# Network module:
#   ieee41bus.py
# ------------------------------------------------------------

import math
import time
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional, Any

import pyomo.environ as pyo
from pyomo.opt import SolverStatus, TerminationCondition

import ieee41bus as mcase


# ============================
# User-tunable global settings
# ============================

DEFAULT_SOLVER = "scip"

# SCIP limits
SCIP_TIME_LIMIT = 36000
SCIP_GAP_LIMIT = 1e-4          # <-- requested change
SCIP_MEMORY_LIMIT_MB = 8192
SCIP_NODE_LIMIT = 500000

TEE_SOLVER_LOG = True

# Angle bounds
THETA_MAX = 1.0  # rad

# Optional subset of OLTC edges to activate.
# If None -> use all OLTC candidates from network metadata.
ACTIVE_OLTC_EDGES: Optional[List[Tuple[int, int]]] = None

# Network build settings
NETWORK_BUILD_KWARGS = dict(
    slack_vm_pu=1.0,
    line_max_loading_percent=1e6,
    stress_q_over_p=0.95,
    stress_load_mw_each=300.0,
    r_loss_scale=3.0,
    max_i_ka_base=2.0,
    max_i_ka_stress=1.0,
    line_smax_pu_overrides=None,
)


# ----------------------------
# Config containers
# ----------------------------
@dataclass
class OLTCBranchConfig:
    tap_min: int
    tap_max: int
    dV_percent: float


@dataclass
class BuildConfig:
    oltc_branches: Dict[Tuple[int, int], OLTCBranchConfig]
    fix_slack_angle: bool = True


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


def _build_branch_list_from_lines(net) -> List[Tuple[int, int, int]]:
    branches = []
    for e_id in net.line.index:
        fb = int(net.line.at[e_id, "from_bus"])
        tb = int(net.line.at[e_id, "to_bus"])
        branches.append((int(e_id), fb, tb))
    return branches


def _default_tap_choice(taps):
    return min(taps, key=lambda t: abs(t))


def read_network_oltc_metadata(
    net,
    active_oltc_edges: Optional[List[Tuple[int, int]]] = None
) -> Dict[Tuple[int, int], OLTCBranchConfig]:
    """
    Read OLTC metadata from net["fixed_oltc_table"] and align directions
    with actual directed arcs in net.line.
    """
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
            raise ValueError(
                f"OLTC metadata edge ({i0},{j0}) not found in either direction in net.line."
            )

        oltc_branches[(i, j)] = OLTCBranchConfig(
            tap_min=int(row["tap_min"]),
            tap_max=int(row["tap_max"]),
            dV_percent=float(row["dV_percent"]),
        )

    if len(oltc_branches) == 0:
        raise ValueError("No active OLTC branches found after metadata filtering.")

    return oltc_branches


# ----------------------------
# Data extraction
# ----------------------------
def extract_dc_miqp_data(net, cfg: BuildConfig) -> Dict[str, Any]:
    """
    Extract MIQP DCOPF data from pandapower net.

    DC assumptions:
      - |V| ≈ 1 pu, ignore Q, ignore losses
      - flow: f_ij = b_ij (theta_i - theta_j)
      - |f_ij| <= Fmax_ij

    OLTC DC approximation:
      - tau(t) = 1 + tap*dV%/100
      - alpha(t) = 1/tau(t)
      - b_ij^tap = b_ij * alpha(t)
    """
    sn = float(net.sn_mva)

    buses = [int(i) for i in net.bus.index]

    # slack bus
    if len(net.ext_grid.index) < 1:
        raise ValueError("pandapower net must have an ext_grid (slack).")
    eg0 = int(net.ext_grid.index[0])
    slack_bus = int(net.ext_grid.at[eg0, "bus"])

    # loads (MW -> pu)
    Pd_mw = {i: 0.0 for i in buses}
    if hasattr(net, "load") and len(net.load.index) > 0:
        for li in net.load.index:
            b = int(net.load.at[li, "bus"])
            Pd_mw[b] += float(net.load.at[li, "p_mw"])
    Pd_pu = {i: Pd_mw[i] / sn for i in buses}

    # generators: ext_grid + gen (active only)
    gen_records = []

    for eg in net.ext_grid.index:
        eg = int(eg)
        b = int(net.ext_grid.at[eg, "bus"])
        pmin = float(net.ext_grid.at[eg, "min_p_mw"]) if "min_p_mw" in net.ext_grid.columns else -1e9
        pmax = float(net.ext_grid.at[eg, "max_p_mw"]) if "max_p_mw" in net.ext_grid.columns else 1e9
        c2, c1, c0 = _find_poly_cost(net, "ext_grid", eg)
        gen_records.append({
            "type": "ext_grid",
            "id": eg,
            "bus": b,
            "pmin_pu": pmin / sn,
            "pmax_pu": pmax / sn,
            "c2": c2, "c1": c1, "c0": c0,
        })

    if hasattr(net, "gen") and len(net.gen.index) > 0:
        for gi in net.gen.index:
            gi = int(gi)
            b = int(net.gen.at[gi, "bus"])
            pmin = float(net.gen.at[gi, "min_p_mw"])
            pmax = float(net.gen.at[gi, "max_p_mw"])
            c2, c1, c0 = _find_poly_cost(net, "gen", gi)
            gen_records.append({
                "type": "gen",
                "id": gi,
                "bus": b,
                "pmin_pu": pmin / sn,
                "pmax_pu": pmax / sn,
                "c2": c2, "c1": c1, "c0": c0,
            })

    # lines -> physical edges
    branches = _build_branch_list_from_lines(net)

    # build undirected edge list U, and directed arc set A (both directions)
    U = []
    A = []

    # per-unit b_ij and Fmax_ij on directed arcs
    bus_vn_kv = {int(i): float(net.bus.at[i, "vn_kv"]) for i in buses}

    b_dir = {}
    Fmax_dir = {}

    EPS = 1e-12
    seen_undir = set()

    for e_id, fb, tb in branches:
        vn_kv = bus_vn_kv[fb]
        zbase = _zbase_ohm(vn_kv, sn)
        r_ohm = float(net.line.at[e_id, "r_ohm_per_km"]) * float(net.line.at[e_id, "length_km"])
        x_ohm = float(net.line.at[e_id, "x_ohm_per_km"]) * float(net.line.at[e_id, "length_km"])
        r_pu = r_ohm / zbase
        x_pu = x_ohm / zbase

        # DC susceptance
        if abs(x_pu) > EPS:
            bval = -1.0 / x_pu
        else:
            y = 1.0 / complex(r_pu, x_pu) if (abs(r_pu) + abs(x_pu)) > EPS else (0.0 + 0.0j)
            bval = float(y.imag)

        Imax = float(net.line.at[e_id, "max_i_ka"]) if "max_i_ka" in net.line.columns else 1e9
        Smax_mva = math.sqrt(3.0) * vn_kv * Imax
        Fmax_pu = Smax_mva / sn

        b_dir[(fb, tb)] = float(bval)
        b_dir[(tb, fb)] = float(bval)
        Fmax_dir[(fb, tb)] = float(Fmax_pu)
        Fmax_dir[(tb, fb)] = float(Fmax_pu)

        A.append((fb, tb))
        A.append((tb, fb))

        ek = (min(fb, tb), max(fb, tb))
        if ek not in seen_undir:
            seen_undir.add(ek)
            U.append(ek)

    A = list(dict.fromkeys(A))
    U = list(dict.fromkeys(U))

    # OLTC set T
    T = []
    K = {}
    b_tap = {}
    Mftap = {}
    theta_diff_bound = {}

    for (u, v), tcfg in cfg.oltc_branches.items():
        if (u, v) not in b_dir:
            raise ValueError(f"OLTC branch {(u, v)} not found in directed DC arc set.")
        ij = (u, v)
        T.append(ij)

        taps = list(range(int(tcfg.tap_min), int(tcfg.tap_max) + 1))
        K[ij] = taps

        base_b = float(b_dir[ij])
        abs_b_list = []
        for tap in taps:
            tau = 1.0 + (tap * float(tcfg.dV_percent)) / 100.0
            alpha = 1.0 / tau
            bt = base_b * alpha
            b_tap[(ij, int(tap))] = float(bt)
            abs_b_list.append(abs(bt))

        min_abs_b = max(min(abs_b_list), 1e-9)
        Fmax_ij = float(Fmax_dir[ij])
        theta_bound = min(2.0 * THETA_MAX, Fmax_ij / min_abs_b)
        theta_diff_bound[ij] = float(theta_bound)

        for tap in taps:
            bt = abs(float(b_tap[(ij, int(tap))]))
            Mftap[(ij, int(tap))] = float(bt * theta_bound + 1e-6)

    return {
        "sn_mva": sn,
        "buses": buses,
        "slack_bus": slack_bus,
        "Pd_pu": Pd_pu,
        "gen_records": gen_records,
        "U": U,
        "A": A,
        "b": b_dir,
        "Fmax": Fmax_dir,
        "T": T,
        "K": K,
        "b_tap": b_tap,
        "Mftap": Mftap,
        "theta_diff_bound": theta_diff_bound,
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
    A = data["A"]
    b = data["b"]
    Fmax = data["Fmax"]

    T = data["T"]
    K = data["K"]
    b_tap = data["b_tap"]
    Mftap = data["Mftap"]
    theta_diff_bound = data["theta_diff_bound"]

    gen_records = data["gen_records"]
    Gset = list(range(len(gen_records)))

    model = pyo.ConcreteModel(name="DC_MIQP_modified_explicit_41BUS_OLTC")

    # Sets
    model.N = pyo.Set(initialize=buses, ordered=True)
    model.G = pyo.Set(initialize=Gset, ordered=True)
    model.A = pyo.Set(initialize=A, dimen=2, ordered=True)
    model.T = pyo.Set(initialize=T, dimen=2, ordered=True)

    # Params
    model.Pd = pyo.Param(model.N, initialize=lambda m, i: float(Pd[i]), mutable=False)

    model.gen_bus = pyo.Param(model.G, initialize=lambda m, gg: int(gen_records[int(gg)]["bus"]), within=pyo.Any)
    model.Pgmin = pyo.Param(model.G, initialize=lambda m, gg: float(gen_records[int(gg)]["pmin_pu"]))
    model.Pgmax = pyo.Param(model.G, initialize=lambda m, gg: float(gen_records[int(gg)]["pmax_pu"]))
    model.c2 = pyo.Param(model.G, initialize=lambda m, gg: float(gen_records[int(gg)]["c2"]))
    model.c1 = pyo.Param(model.G, initialize=lambda m, gg: float(gen_records[int(gg)]["c1"]))
    model.c0 = pyo.Param(model.G, initialize=lambda m, gg: float(gen_records[int(gg)]["c0"]))

    model.b = pyo.Param(model.A, initialize=lambda m, i, j: float(b[(i, j)]))
    model.Fmax = pyo.Param(model.A, initialize=lambda m, i, j: float(Fmax[(i, j)]))

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
    model.Mf = pyo.Param(
        model.BETA_INDEX,
        initialize=lambda m, i, j, tap: float(Mftap[((i, j), int(tap))]),
        mutable=False
    )

    # Variables
    model.Pg = pyo.Var(model.G, bounds=lambda m, gg: (m.Pgmin[gg], m.Pgmax[gg]))
    model.theta = pyo.Var(model.N, bounds=(-data["theta_max"], data["theta_max"]))
    model.f = pyo.Var(model.A)

    model.beta = pyo.Var(model.BETA_INDEX, within=pyo.Binary)
    model.f_tap = pyo.Var(model.BETA_INDEX)

    if data["fix_slack_angle"]:
        model.slack_angle = pyo.Constraint(expr=model.theta[slack_bus] == 0.0)

    out_neighbors = {i: [] for i in buses}
    for (i, j) in A:
        out_neighbors[i].append(j)

    def p_balance_rule(m, i):
        inj = sum(m.Pg[gg] for gg in m.G if int(m.gen_bus[gg]) == int(i)) - m.Pd[i]
        return inj == sum(m.f[i, j] for j in out_neighbors[int(i)])
    model.p_balance = pyo.Constraint(model.N, rule=p_balance_rule)

    T_set = set(T)
    T_undir = {(min(i, j), max(i, j)) for (i, j) in T_set}

    def is_oltc_arc(i, j) -> bool:
        return (min(i, j), max(i, j)) in T_undir

    def dc_flow_non_oltc_rule(m, i, j):
        if is_oltc_arc(i, j):
            return pyo.Constraint.Skip
        return m.f[i, j] == m.b[i, j] * (m.theta[i] - m.theta[j])
    model.dc_flow_non_oltc = pyo.Constraint(model.A, rule=dc_flow_non_oltc_rule)

    model.onehot = pyo.ConstraintList()
    model.agg_flow = pyo.ConstraintList()
    model.tap_link_ub = pyo.ConstraintList()
    model.tap_link_lb = pyo.ConstraintList()
    model.tap_flow_ub = pyo.ConstraintList()
    model.tap_flow_lb = pyo.ConstraintList()
    model.reverse_flow = pyo.ConstraintList()

    for (i, j) in T:
        taps = K[(i, j)]

        model.onehot.add(sum(model.beta[i, j, int(t)] for t in taps) == 1)
        model.agg_flow.add(model.f[i, j] == sum(model.f_tap[i, j, int(t)] for t in taps))

        if (j, i) in model.A:
            model.reverse_flow.add(model.f[j, i] == -model.f[i, j])

        for t in taps:
            t = int(t)
            Mf = model.Mf[i, j, t]

            model.tap_link_ub.add(
                model.f_tap[i, j, t] - model.b_tap[i, j, t] * (model.theta[i] - model.theta[j])
                <= Mf * (1 - model.beta[i, j, t])
            )
            model.tap_link_lb.add(
                model.f_tap[i, j, t] - model.b_tap[i, j, t] * (model.theta[i] - model.theta[j])
                >= -Mf * (1 - model.beta[i, j, t])
            )

            model.tap_flow_ub.add(model.f_tap[i, j, t] <= model.Fmax[i, j] * model.beta[i, j, t])
            model.tap_flow_lb.add(model.f_tap[i, j, t] >= -model.Fmax[i, j] * model.beta[i, j, t])

        thb = float(theta_diff_bound[(i, j)])
        model.add_component(
            f"theta_diff_ub_{i}_{j}",
            pyo.Constraint(expr=(model.theta[i] - model.theta[j] <= thb))
        )
        model.add_component(
            f"theta_diff_lb_{i}_{j}",
            pyo.Constraint(expr=(model.theta[i] - model.theta[j] >= -thb))
        )

    def flow_ub_rule(m, i, j):
        return m.f[i, j] <= m.Fmax[i, j]

    def flow_lb_rule(m, i, j):
        return m.f[i, j] >= -m.Fmax[i, j]

    model.flow_ub = pyo.Constraint(model.A, rule=flow_ub_rule)
    model.flow_lb = pyo.Constraint(model.A, rule=flow_lb_rule)

    model.obj = pyo.Objective(
        expr=sum(
            model.c2[g] * (sn * model.Pg[g]) ** 2
            + model.c1[g] * (sn * model.Pg[g])
            + model.c0[g]
            for g in model.G
        ),
        sense=pyo.minimize
    )

    return model


# ----------------------------
# Initialization
# ----------------------------
def initialize_flat(m: pyo.ConcreteModel, data: Dict[str, Any]):
    sb = int(data["slack_bus"])

    for i in m.N:
        m.theta[i].set_value(0.0 if int(i) == sb else 0.0)

    for g in m.G:
        pmin = float(pyo.value(m.Pgmin[g]))
        pmax = float(pyo.value(m.Pgmax[g]))
        pmid = 0.0 if (pmin <= 0.0 <= pmax) else 0.5 * (pmin + pmax)
        m.Pg[g].set_value(pmid)

    for (i, j) in m.A:
        m.f[i, j].set_value(0.0)

    taps_by_T = data["K"]
    for (i, j) in m.T:
        pick = _default_tap_choice(list(taps_by_T[(i, j)]))
        for t in taps_by_T[(i, j)]:
            val = 1.0 if int(t) == int(pick) else 0.0
            m.beta[i, j, t].set_value(val)
            m.f_tap[i, j, t].set_value(0.0)


# ----------------------------
# Solver wrapper
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

    if solver_name.lower() == "scip":
        if timelimit is not None:
            opt.options["limits/time"] = float(timelimit)
        if nodelimit is not None:
            opt.options["limits/nodes"] = int(nodelimit)
        if memlimit_mb is not None:
            opt.options["limits/memory"] = float(memlimit_mb)
        if mipgap is not None:
            opt.options["limits/gap"] = float(mipgap)
        opt.options["display/verblevel"] = 4 if tee else 0

    res = opt.solve(model, tee=tee, load_solutions=True)

    tc = res.solver.termination_condition
    st = res.solver.status
    msg = (getattr(res.solver, "message", "") or "")

    print(f"[INFO] solver.status = {st}")
    print(f"[INFO] termination_condition = {tc}")
    if msg.strip():
        print(f"[INFO] solver.message = {msg.strip()}")

    has_values = any(
        (v.value is not None)
        for v in model.component_data_objects(pyo.Var, active=True, descend_into=True)
    )

    # Treat incumbent-loaded limit stops as acceptable.
    bad_tc = {
        TerminationCondition.infeasible,
        TerminationCondition.unbounded,
        TerminationCondition.infeasibleOrUnbounded,
        TerminationCondition.invalidProblem,
        TerminationCondition.noSolution,
        TerminationCondition.solverFailure,
        TerminationCondition.internalSolverError,
        TerminationCondition.error,
    }

    if st in {SolverStatus.ok, SolverStatus.warning} and has_values and tc not in bad_tc:
        return res

    raise RuntimeError(
        f"Solve failed or no solution loaded. status={st}, termination={tc}, msg={msg}"
    )


# ----------------------------
# Reporting
# ----------------------------
def report_solution(model, data: Dict[str, Any]):
    sn = data["sn_mva"]
    buses = data["buses"]
    gen_records = data["gen_records"]
    slack_bus = data["slack_bus"]

    print("\n====================")
    print(" DCOPF (MIQP) RESULT")
    print("====================")
    print(f"Objective (EUR): {float(pyo.value(model.obj)):.6f}")

    print("\n--- Bus Angles (deg) ---")
    for i in buses:
        th = float(pyo.value(model.theta[i]))
        deg = th * 180.0 / math.pi
        tag = " [slack]" if i == slack_bus else ""
        print(f"Bus {i:2d}: theta={deg:+.6f}{tag}")

    print("\n--- Generator Dispatch (MW) ---")
    for gg in model.G:
        rec = gen_records[int(gg)]
        Pg_mw = sn * float(pyo.value(model.Pg[gg]))
        print(f"{rec['type']}[{rec['id']}] @ bus {rec['bus']:2d}: P={Pg_mw:.6f} MW")

    if len(list(model.T)) > 0:
        print("\n--- OLTC Selected Taps ---")
        for (i, j) in model.T:
            best_t = None
            for t in data["K"][(i, j)]:
                if float(pyo.value(model.beta[i, j, t])) > 0.5:
                    best_t = int(t)
                    break
            if best_t is None:
                best_t = max(data["K"][(i, j)], key=lambda t: float(pyo.value(model.beta[i, j, t])))
            b_eff = float(pyo.value(model.b_tap[i, j, best_t]))
            print(f"OLTC ({i:2d},{j:2d}): tap={best_t:>3d}, b_eff={b_eff:.6f}")

    out_neighbors = {i: [] for i in buses}
    for (i, j) in data["A"]:
        out_neighbors[i].append(j)

    print("\n--- Nodal balance residuals (pu) ---")
    for i in buses:
        inj = sum(
            float(pyo.value(model.Pg[gg]))
            for gg in model.G
            if int(pyo.value(model.gen_bus[gg])) == int(i)
        ) - float(data["Pd_pu"][i])
        out = sum(float(pyo.value(model.f[i, j])) for j in out_neighbors[i])
        res = inj - out
        print(f"Bus {i:2d}: {res:+.3e}")

    print("\n--- Line flows (showing i<j direction) ---")
    for (i, j) in data["U"]:
        fij = float(pyo.value(model.f[i, j])) if (i, j) in model.A else float("nan")
        fmax = float(data["Fmax"][(i, j)]) if (i, j) in data["Fmax"] else float("nan")
        print(f"Edge {i:2d}-{j:2d}: f={fij:+.6f} pu ({sn*fij:+.2f} MW), Fmax={fmax:.6f} pu")


# ----------------------------
# Main
# ----------------------------
def main():
    t0 = time.perf_counter()

    # 1) Build modified 41-bus network
    net = mcase.busmeshed39_opf(**NETWORK_BUILD_KWARGS)

    # 2) Read OLTC metadata from network
    oltc_branches = read_network_oltc_metadata(net, active_oltc_edges=ACTIVE_OLTC_EDGES)
    cfg = BuildConfig(oltc_branches=oltc_branches, fix_slack_angle=True)

    # 3) Extract data and build model
    data = extract_dc_miqp_data(net, cfg)

    n_beta = sum(len(data["K"][ij]) for ij in data["T"])
    print("[INFO] DC-MIQP data summary")
    print(f"  #buses={len(data['buses'])}, #gens={len(data['gen_records'])}, #undirected_lines={len(data['U'])}, #arcs={len(data['A'])}")
    print(f"  #OLTC lines={len(data['T'])}, #beta-vars={n_beta}")
    print(f"  THETA_MAX={THETA_MAX:.4f} rad, solver={DEFAULT_SOLVER}")
    print(f"  SCIP_GAP_LIMIT={SCIP_GAP_LIMIT}, SCIP_NODE_LIMIT={SCIP_NODE_LIMIT}")

    model = build_miqp_dcopf_model(data)
    initialize_flat(model, data)

    # 4) Solve
    print("\n[INFO] Solving MIQP DCOPF with SCIP...")
    _ = solve_miqp(
        model,
        solver_name=DEFAULT_SOLVER,
        timelimit=SCIP_TIME_LIMIT,
        mipgap=SCIP_GAP_LIMIT,
        memlimit_mb=SCIP_MEMORY_LIMIT_MB,
        nodelimit=SCIP_NODE_LIMIT,
        tee=TEE_SOLVER_LOG
    )

    # 5) Report
    report_solution(model, data)

    t1 = time.perf_counter()
    print(f"\n[INFO] Total elapsed time: {t1 - t0:.2f} seconds.")


if __name__ == "__main__":
    main()