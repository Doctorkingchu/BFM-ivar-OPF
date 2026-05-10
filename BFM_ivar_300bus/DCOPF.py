# DCOPF.py
# ------------------------------------------------------------
# Standalone Pyomo MIQP DC-OPF on the ver3 IEEE 300-bus system
# with OLTC tap selection (one-hot) via Big-M.
#
# This file intentionally depends only on:
#   - ieee300bus.py   (network builder / metadata)
#   - Pyomo + SCIP
#
# It does not import helper logic from BFMag / BFMit / SDP files.
# ------------------------------------------------------------

import math
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import pyomo.environ as pyo
from pyomo.opt import SolverStatus, TerminationCondition

import ieee300bus as mcase


# ============================
# User-tunable global settings
# ============================

DEFAULT_SOLVER = "scip"

SCIP_TIME_LIMIT = 36000
SCIP_GAP_LIMIT = 1e-4
SCIP_MEMORY_LIMIT_MB = 8192
SCIP_NODE_LIMIT = 500000

TEE_SOLVER_LOG = True

THETA_MAX = 1.0  # rad

# If None, use all OLTC candidates from network metadata.
ACTIVE_OLTC_EDGES: Optional[List[Tuple[int, int]]] = None

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


# ----------------------------
# Config containers
# ----------------------------
@dataclass
class OLTCBranchConfig:
    tap_min: int
    tap_max: int
    dV_percent: float


@dataclass
class BranchTableConfig:
    oltc_branches: Dict[Tuple[int, int], OLTCBranchConfig]
    fix_slack_angle: bool = True


# ----------------------------
# Helpers
# ----------------------------
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


def _merge_parallel_series_equivalent(rows: List[dict]) -> Tuple[float, float, float]:
    """
    Merge parallel directed rows from branch_params_pu_table into one equivalent branch.
    Uses admittance summation for z_eq and sums synthetic ratings.
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


def build_branch_table_cfg_from_net_metadata(
    net,
    active_oltc_edges: Optional[List[Tuple[int, int]]] = None,
) -> BranchTableConfig:
    if "fixed_oltc_table" not in net:
        raise KeyError("Network metadata 'fixed_oltc_table' not found.")

    requested = None
    if active_oltc_edges is not None:
        requested = {tuple(map(int, edge)) for edge in active_oltc_edges}

    oltc_branches: Dict[Tuple[int, int], OLTCBranchConfig] = {}
    oltc_df = net["fixed_oltc_table"]

    for _, row in oltc_df.iterrows():
        i = int(row["from_bus_pp"])
        j = int(row["to_bus_pp"])
        if requested is not None and (i, j) not in requested and (j, i) not in requested:
            continue
        oltc_branches[(i, j)] = OLTCBranchConfig(
            tap_min=int(row["tap_min"]),
            tap_max=int(row["tap_max"]),
            dV_percent=float(row["dV_percent"]),
        )

    if requested is not None and len(oltc_branches) == 0:
        raise ValueError("No active OLTC branches found after metadata filtering.")

    return BranchTableConfig(oltc_branches=oltc_branches, fix_slack_angle=True)


# ----------------------------
# Data extraction
# ----------------------------
def extract_dc_miqp_data(
    net,
    active_oltc_edges: Optional[List[Tuple[int, int]]] = None,
) -> Dict[str, Any]:
    """
    Extract MIQP DCOPF data directly from ieee300bus metadata.

    DC assumptions:
      - ignore reactive power and losses
      - flow on non-OLTC branches: f_ij = b_ij * (theta_i - theta_j)
      - |f_ij| <= Fmax_ij

    OLTC DC approximation:
      - tau(t) = 1 + tap*dV%/100
      - alpha(t) = 1/tau(t)
      - b_ij^tap = b_ij * alpha(t)
    """
    cfg = build_branch_table_cfg_from_net_metadata(net, active_oltc_edges=active_oltc_edges)

    sn = float(net.sn_mva)
    buses = [int(i) for i in net.bus.index]

    if len(net.ext_grid.index) < 1:
        raise ValueError("pandapower net must have ext_grid.")
    eg0 = int(net.ext_grid.index[0])
    slack_bus = int(net.ext_grid.at[eg0, "bus"])

    Pd = {i: 0.0 for i in buses}
    if hasattr(net, "load") and len(net.load.index) > 0:
        for li in net.load.index:
            b = int(net.load.at[li, "bus"])
            Pd[b] += float(net.load.at[li, "p_mw"])
    Pd_pu = {i: Pd[i] / sn for i in buses}

    gen_records = []
    for eg in net.ext_grid.index:
        eg = int(eg)
        b = int(net.ext_grid.at[eg, "bus"])
        pmin = float(net.ext_grid.at[eg, "min_p_mw"]) if "min_p_mw" in net.ext_grid.columns else -1e9
        pmax = float(net.ext_grid.at[eg, "max_p_mw"]) if "max_p_mw" in net.ext_grid.columns else 1e9
        c2, c1, c0 = _find_poly_cost(net, "ext_grid", eg)
        gen_records.append(
            {
                "type": "ext_grid",
                "id": eg,
                "bus": b,
                "pmin_pu": pmin / sn,
                "pmax_pu": pmax / sn,
                "c2": c2,
                "c1": c1,
                "c0": c0,
            }
        )

    if hasattr(net, "gen") and len(net.gen.index) > 0:
        for gi in net.gen.index:
            gi = int(gi)
            b = int(net.gen.at[gi, "bus"])
            pmin = float(net.gen.at[gi, "min_p_mw"])
            pmax = float(net.gen.at[gi, "max_p_mw"])
            c2, c1, c0 = _find_poly_cost(net, "gen", gi)
            gen_records.append(
                {
                    "type": "gen",
                    "id": gi,
                    "bus": b,
                    "pmin_pu": pmin / sn,
                    "pmax_pu": pmax / sn,
                    "c2": c2,
                    "c1": c1,
                    "c0": c0,
                }
            )

    if "branch_params_pu_table" not in net or net["branch_params_pu_table"] is None or net["branch_params_pu_table"].empty:
        raise KeyError("Network metadata 'branch_params_pu_table' not found or empty.")

    brdf = net["branch_params_pu_table"].copy()
    oltc_dir_set = set(cfg.oltc_branches.keys())

    grouped: Dict[Tuple[int, int], List[dict]] = {}
    for _, row in brdf.iterrows():
        fb0 = int(row["from_bus_pp"])
        tb0 = int(row["to_bus_pp"])
        if (fb0, tb0) in oltc_dir_set:
            fb, tb = fb0, tb0
        elif (tb0, fb0) in oltc_dir_set:
            fb, tb = tb0, fb0
        else:
            fb, tb = fb0, tb0
        grouped.setdefault((fb, tb), []).append(row)

    E: List[Tuple[int, int]] = []
    x: Dict[Tuple[int, int], float] = {}
    r: Dict[Tuple[int, int], float] = {}
    Smax: Dict[Tuple[int, int], float] = {}

    for key, rows in grouped.items():
        E.append(key)
        req_pu, xeq_pu, smax_mva = _merge_parallel_series_equivalent(rows)
        r[key] = float(req_pu)
        x[key] = float(xeq_pu)
        Smax[key] = float(smax_mva) / sn

    U: List[Tuple[int, int]] = []
    A: List[Tuple[int, int]] = []
    b_dir: Dict[Tuple[int, int], float] = {}
    Fmax_dir: Dict[Tuple[int, int], float] = {}
    seen_undir = set()
    eps = 1e-12

    for (i, j) in E:
        ek = (min(int(i), int(j)), max(int(i), int(j)))
        if ek not in seen_undir:
            seen_undir.add(ek)
            U.append(ek)

        x_pu = float(x[(i, j)])
        r_pu = float(r[(i, j)])
        if abs(x_pu) > eps:
            bval = -1.0 / x_pu
        else:
            y = 1.0 / complex(r_pu, x_pu) if (abs(r_pu) + abs(x_pu)) > eps else (0.0 + 0.0j)
            bval = float(y.imag)

        fmax_pu = float(Smax[(i, j)])
        for arc in [(int(i), int(j)), (int(j), int(i))]:
            if arc not in b_dir:
                A.append(arc)
            b_dir[arc] = float(bval)
            Fmax_dir[arc] = float(fmax_pu)

    T = list(cfg.oltc_branches.keys())
    K: Dict[Tuple[int, int], List[int]] = {}
    b_tap: Dict[Tuple[Tuple[int, int], int], float] = {}
    Mftap: Dict[Tuple[Tuple[int, int], int], float] = {}
    theta_diff_bound: Dict[Tuple[int, int], float] = {}

    for (u, v) in T:
        if (u, v) not in b_dir:
            raise ValueError(f"OLTC branch {(u, v)} not found in directed DC arc set.")

        tcfg = cfg.oltc_branches[(u, v)]
        taps = list(range(int(tcfg.tap_min), int(tcfg.tap_max) + 1))
        K[(u, v)] = taps

        base_b = float(b_dir[(u, v)])
        abs_b_list = []
        for tap in taps:
            tau = 1.0 + (tap * float(tcfg.dV_percent)) / 100.0
            alpha = 1.0 / tau
            bt = base_b * alpha
            b_tap[((u, v), int(tap))] = float(bt)
            abs_b_list.append(abs(bt))

        min_abs_b = max(min(abs_b_list), 1e-9)
        fmax_uv = float(Fmax_dir[(u, v)])
        theta_bound = min(2.0 * THETA_MAX, fmax_uv / min_abs_b)
        theta_diff_bound[(u, v)] = float(theta_bound)

        for tap in taps:
            bt = abs(float(b_tap[((u, v), int(tap))]))
            Mftap[((u, v), int(tap))] = float(bt * theta_bound + 1e-6)

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

    model = pyo.ConcreteModel(name="DC_MIQP_ver3_300bus_standalone")

    model.N = pyo.Set(initialize=buses, ordered=True)
    model.G = pyo.Set(initialize=Gset, ordered=True)
    model.A = pyo.Set(initialize=A, dimen=2, ordered=True)
    model.T = pyo.Set(initialize=T, dimen=2, ordered=True)

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
        mutable=False,
    )
    model.Mf = pyo.Param(
        model.BETA_INDEX,
        initialize=lambda m, i, j, tap: float(Mftap[((i, j), int(tap))]),
        mutable=False,
    )

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
            mf = model.Mf[i, j, t]
            model.tap_link_ub.add(
                model.f_tap[i, j, t] - model.b_tap[i, j, t] * (model.theta[i] - model.theta[j])
                <= mf * (1 - model.beta[i, j, t])
            )
            model.tap_link_lb.add(
                model.f_tap[i, j, t] - model.b_tap[i, j, t] * (model.theta[i] - model.theta[j])
                >= -mf * (1 - model.beta[i, j, t])
            )
            model.tap_flow_ub.add(model.f_tap[i, j, t] <= model.Fmax[i, j] * model.beta[i, j, t])
            model.tap_flow_lb.add(model.f_tap[i, j, t] >= -model.Fmax[i, j] * model.beta[i, j, t])

        thb = float(theta_diff_bound[(i, j)])
        model.add_component(
            f"theta_diff_ub_{i}_{j}",
            pyo.Constraint(expr=(model.theta[i] - model.theta[j] <= thb)),
        )
        model.add_component(
            f"theta_diff_lb_{i}_{j}",
            pyo.Constraint(expr=(model.theta[i] - model.theta[j] >= -thb)),
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
        sense=pyo.minimize,
    )

    return model


# ----------------------------
# Initialization
# ----------------------------
def initialize_flat(model: pyo.ConcreteModel, data: Dict[str, Any]) -> None:
    slack_bus = int(data["slack_bus"])

    for i in model.N:
        model.theta[i].set_value(0.0 if int(i) == slack_bus else 0.0)

    for g in model.G:
        pmin = float(pyo.value(model.Pgmin[g]))
        pmax = float(pyo.value(model.Pgmax[g]))
        pmid = 0.0 if (pmin <= 0.0 <= pmax) else 0.5 * (pmin + pmax)
        model.Pg[g].set_value(pmid)

    for (i, j) in model.A:
        model.f[i, j].set_value(0.0)

    for (i, j) in model.T:
        pick = _default_tap_choice(list(data["K"][(i, j)]))
        for t in data["K"][(i, j)]:
            val = 1.0 if int(t) == int(pick) else 0.0
            model.beta[i, j, t].set_value(val)
            model.f_tap[i, j, t].set_value(0.0)


# ----------------------------
# Solver wrapper
# ----------------------------
def solve_miqp(
    model: pyo.ConcreteModel,
    solver_name: str = "scip",
    timelimit: Optional[float] = None,
    mipgap: Optional[float] = None,
    memlimit_mb: Optional[int] = None,
    nodelimit: Optional[int] = None,
    tee: bool = True,
):
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

    raise RuntimeError(f"Solve failed or no solution loaded. status={st}, termination={tc}, msg={msg}")


# ----------------------------
# Reporting
# ----------------------------
def report_solution(model: pyo.ConcreteModel, data: Dict[str, Any]) -> None:
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
        pg_mw = sn * float(pyo.value(model.Pg[gg]))
        print(f"{rec['type']}[{rec['id']}] @ bus {rec['bus']:2d}: P={pg_mw:.6f} MW")

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
def main() -> None:
    t0 = time.perf_counter()

    net = mcase.case300_opf(**NETWORK_BUILD_KWARGS)
    data = extract_dc_miqp_data(net, active_oltc_edges=ACTIVE_OLTC_EDGES)

    n_beta = sum(len(data["K"][ij]) for ij in data["T"])
    scenario_name = None
    if "build_profile" in net and isinstance(net["build_profile"], dict):
        scenario_name = net["build_profile"].get("scenario_name", None)

    print("[INFO] Standalone DC-MIQP data summary")
    if scenario_name:
        print(f"  scenario={scenario_name}")
    print(f"  #buses={len(data['buses'])}, #gens={len(data['gen_records'])}, #undirected_branches={len(data['U'])}, #arcs={len(data['A'])}")
    print(f"  #OLTC branches={len(data['T'])}, #beta-vars={n_beta}")
    print(f"  THETA_MAX={THETA_MAX:.4f} rad, solver={DEFAULT_SOLVER}")
    print(f"  SCIP_GAP_LIMIT={SCIP_GAP_LIMIT}, SCIP_NODE_LIMIT={SCIP_NODE_LIMIT}")

    model = build_miqp_dcopf_model(data)
    initialize_flat(model, data)

    print("\n[INFO] Solving MIQP DCOPF with SCIP...")
    _ = solve_miqp(
        model,
        solver_name=DEFAULT_SOLVER,
        timelimit=SCIP_TIME_LIMIT,
        mipgap=SCIP_GAP_LIMIT,
        memlimit_mb=SCIP_MEMORY_LIMIT_MB,
        nodelimit=SCIP_NODE_LIMIT,
        tee=TEE_SOLVER_LOG,
    )

    report_solution(model, data)

    t1 = time.perf_counter()
    print(f"\n[INFO] Total elapsed time: {t1 - t0:.2f} seconds.")


if __name__ == "__main__":
    main()
