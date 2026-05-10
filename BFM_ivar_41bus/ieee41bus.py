# ieee41bus.py
# ------------------------------------------------------------
# IEEE 39-bus plus (fixed 41-bus version, buses 0..40)
#
# Non-exact mesh experiment profile
# ---------------------------------
# This version is tuned so that:
#   - the network remains a stressed but meaningful 41-bus mesh,
#   - ACOPF can still be feasible in many cases,
#   - OPF-cr / MISOCP can become noticeably inaccurate on mesh cycles.
#
# Key design ideas
# ----------------
# 1) Strong load pocket around buses 39 and 40
# 2) Multiple short cycles near the pocket
# 3) Moderate bottlenecks at pocket-entry / cycle-coupling branches
# 4) Recommended asymmetric OLTC profile (metadata only)
# 5) Recommended partial shunt profile (metadata only)
#
# Notes
# -----
# - Transformer rows are still stored in net.line for Pyomo extractor compatibility.
# - OLTC / shunt are exposed only as metadata for downstream OPF / MINLP / verifier code.
# - They are NOT automatically activated as fixed devices in pandapower.
# ------------------------------------------------------------

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pandapower as pp


# ============================================================
# Device metadata
# ============================================================

@dataclass(frozen=True)
class OLTCBranchConfig:
    tap_min: int = -4
    tap_max: int = 4
    dV_percent: float = 1.25


# Candidate OLTC branches
FIXED_OLTC_EDGES_ORDERED: List[Tuple[int, int]] = [
    (11, 12),
    (12, 13),
    (19, 33),
    (22, 35),
    (0, 1),
    (0, 36),
    (1, 2),
    (1, 24),
    (7, 39),
    (8, 39),
    (12, 40),
    (13, 40),
]

FIXED_OLTC_CONFIGS_UNDIR: Dict[Tuple[int, int], OLTCBranchConfig] = {
    tuple(sorted(edge)): OLTCBranchConfig()
    for edge in FIXED_OLTC_EDGES_ORDERED
}

# Shunt candidate buses
FIXED_SHUNT_BUSES: List[int] = [2, 3, 7, 10, 13, 15, 18, 19, 23, 25, 28, 38, 39]

# Moderate shunt support for non-exact mesh experiments
FIXED_SHUNT_BCAP_PU: Dict[int, float] = {
    2: 0.10,
    3: 0.08,
    7: 0.10,
    10: 0.08,
    13: 0.15,
    15: 0.10,
    18: 0.15,
    19: 0.18,
    23: 0.10,
    25: 0.12,
    28: 0.18,
    38: 0.08,
    39: 0.15,
}

# Recommended asymmetric OLTC taps for downstream OPF / verifier codes
# Metadata only: not automatically applied in pandapower
RECOMMENDED_NONEXACT_OLTC_TAPS: Dict[Tuple[int, int], int] = {
    (11, 12): -3,
    (12, 13):  0,
    (19, 33):  1,
    (22, 35):  1,
    (0,  1):  -4,
    (0, 36):  -3,
    (1,  2):   1,
    (1, 24):   0,
    (7, 39):  -4,
    (8, 39):   2,   # asymmetric
    (12, 40):  3,   # asymmetric
    (13, 40): -3,   # asymmetric
}

# Recommended shunt statuses for downstream OPF / verifier codes
# Metadata only: not automatically applied in pandapower
RECOMMENDED_NONEXACT_SHUNT_STATUS: Dict[int, int] = {
    2: 1,
    3: 1,
    7: 0,
    10: 0,
    13: 1,
    15: 0,
    18: 1,
    19: 1,
    23: 0,
    25: 0,
    28: 1,
    38: 0,
    39: 1,
}


# ============================================================
# Optional per-line Smax overrides (pu, on sn_mva base)
# These are moderate bottlenecks: enough to expose mesh inaccuracy,
# but not so tight that ACOPF necessarily becomes infeasible.
# ============================================================
LINE_SMAX_PU_OVERRIDES = {
    # trunk / upstream
    (0, 1):   3.0,
    (0, 36):  2.8,
    (1, 24):  2.4,

    # pocket-entry bottlenecks around buses 39/40
    (7, 39):  2.4,
    (8, 39):  2.3,
    (12, 40): 1.7,
    (13, 40): 1.7,
    (39, 40): 1.6,
    (34, 40): 3.0,

    # cycle-coupling bottlenecks
    (3, 12):  1.7,
    (8, 12):  1.7,
}


# ============================================================
# Topology classification
# ============================================================
TRANSFORMER_ORIGIN_LINES = {
    (11, 29), (11, 12), (5, 30), (9, 31), (18, 32), (19, 33),
    (21, 34), (22, 35), (24, 36), (1, 29), (28, 37), (18, 19),
}

# Stress / shortcut edges for non-exact mesh profile
STRESS_LINES = {
    # pocket-entry edges
    (7, 39), (8, 39), (12, 40), (13, 40), (39, 40), (34, 40),

    # cycle-forming shortcut edges
    (3, 7), (4, 8), (8, 12), (3, 12), (31, 4),
}


# ============================================================
# Helpers
# ============================================================
def _edge_in_set(fb: int, tb: int, edge_set) -> bool:
    return (fb, tb) in edge_set or (tb, fb) in edge_set


def _smaxpu_to_imax_ka(smax_pu: float, sn_mva: float, vn_kv: float) -> float:
    """Convert Smax_pu (on sn_mva base) to max_i_ka using S = sqrt(3)*V*I."""
    smax_mva = float(smax_pu) * float(sn_mva)
    return smax_mva / (math.sqrt(3.0) * float(vn_kv))


def _apply_circle_qcapability_curve_to_gens(
    net,
    smax_gen_mva_by_bus,
    n_points=6,
    curve_style="straightLineYValues",
):
    """
    Build a q_capability_curve_table for circular capability:
        P^2 + Q^2 <= Smax^2
    """
    try:
        from pandapower.control import util as ppctrl_util
    except Exception:
        return False

    creator = None
    if hasattr(ppctrl_util, "create_q_capability_curve_characteristics_object"):
        creator = ppctrl_util.create_q_capability_curve_characteristics_object
    elif hasattr(ppctrl_util, "create_q_capability_characteristics_object"):
        creator = ppctrl_util.create_q_capability_characteristics_object

    if creator is None:
        return False

    if "q_capability_curve_table" not in net:
        net["q_capability_curve_table"] = pd.DataFrame(
            columns=["id_q_capability_curve", "p_mw", "q_min_mvar", "q_max_mvar"]
        )

    qtab = net.q_capability_curve_table
    next_id = 0 if qtab.empty else int(qtab["id_q_capability_curve"].max()) + 1

    for col, default in [
        ("reactive_capability_curve", False),
        ("id_q_capability_curve_characteristic", np.nan),
        ("curve_style", None),
    ]:
        if col not in net.gen.columns:
            net.gen[col] = default

    for gi in net.gen.index:
        bus = int(net.gen.at[gi, "bus"])
        Smax = smax_gen_mva_by_bus.get(bus, None)
        if Smax is None:
            continue

        Pmax = float(net.gen.at[gi, "max_p_mw"])
        if Smax < Pmax:
            raise ValueError(f"Gen@bus{bus}: Smax({Smax}) < Pmax({Pmax}) -> infeasible rating")

        p_points = np.linspace(0.0, Pmax, int(n_points))
        rows = []
        for p in p_points:
            qcap = math.sqrt(max(Smax**2 - p**2, 0.0))
            rows.append(
                {
                    "id_q_capability_curve": next_id,
                    "p_mw": float(p),
                    "q_min_mvar": float(-qcap),
                    "q_max_mvar": float(qcap),
                }
            )

        qtab = pd.concat([qtab, pd.DataFrame(rows)], ignore_index=True)

        net.gen.at[gi, "reactive_capability_curve"] = True
        net.gen.at[gi, "id_q_capability_curve_characteristic"] = int(next_id)
        net.gen.at[gi, "curve_style"] = curve_style

        next_id += 1

    net["q_capability_curve_table"] = qtab
    creator(net)

    if hasattr(ppctrl_util, "q_capability_curve_table_diagnostic"):
        ppctrl_util.q_capability_curve_table_diagnostic(net)

    return True


def _get_changed_topology_line_params() -> List[Tuple[int, int, float, float]]:
    """
    Explicit branch parameters (pu on system base).

    Returns
    -------
    list of (from_bus, to_bus, r_pu, x_pu)
    """
    return [
        # ====================================================
        # Base lines
        # ====================================================
        (0, 1,   0.0035, 0.0411),
        (0, 36,  0.0010, 0.0250),
        (1, 2,   0.0013, 0.0151),
        (1, 24,  0.0070, 0.0086),
        (2, 3,   0.0013, 0.0213),
        (2, 17,  0.0011, 0.0133),
        (3, 4,   0.0008, 0.0128),
        (3, 13,  0.0008, 0.0129),
        (4, 5,   0.0002, 0.0026),
        (4, 7,   0.0008, 0.0112),
        (5, 6,   0.0006, 0.0092),
        (5, 10,  0.0007, 0.0082),
        (6, 7,   0.0004, 0.0046),
        (7, 8,   0.0023, 0.0363),
        (8, 38,  0.0010, 0.0250),
        (9, 10,  0.0004, 0.0043),
        (9, 12,  0.0004, 0.0043),
        (12, 13, 0.0009, 0.0101),
        (13, 14, 0.0018, 0.0217),
        (14, 15, 0.0009, 0.0094),
        (15, 16, 0.0007, 0.0089),
        (15, 18, 0.0016, 0.0195),
        (18, 20, 0.0008, 0.0135),
        (15, 23, 0.0003, 0.0059),
        (16, 17, 0.0007, 0.0082),
        (16, 26, 0.0013, 0.0173),
        (20, 21, 0.0008, 0.0140),
        (21, 22, 0.0006, 0.0096),
        (22, 23, 0.0022, 0.0350),
        (24, 25, 0.0032, 0.0323),
        (25, 26, 0.0014, 0.0147),
        (25, 27, 0.0043, 0.0474),
        (25, 28, 0.0057, 0.0625),
        (27, 28, 0.0014, 0.0151),

        # ====================================================
        # Transformer-origin branches (kept as lines)
        # ====================================================
        (11, 29, 0.0016, 0.0435),
        (11, 12, 0.0016, 0.0435),
        (5, 30,  0.0008, 0.0220),
        (9, 31,  0.0007, 0.0190),
        (18, 32, 0.0007, 0.0142),
        (19, 33, 0.0009, 0.0180),
        (21, 34, 0.0006, 0.0143),
        (22, 35, 0.0005, 0.0272),
        (24, 36, 0.0006, 0.0232),
        (1, 29,  0.0007, 0.0181),
        (28, 37, 0.0008, 0.0156),
        (18, 19, 0.0007, 0.0138),

        # ====================================================
        # Stress / shortcut branches for non-exact mesh profile
        # ====================================================
        (7, 39,  0.0011, 0.0118),
        (8, 39,  0.0010, 0.0109),
        (12, 40, 0.0012, 0.0126),
        (13, 40, 0.0011, 0.0119),
        (39, 40, 0.0010, 0.0108),
        (34, 40, 0.0010, 0.0134),

        # cycle-forming shortcuts
        (3, 7,   0.0013, 0.0160),
        (4, 8,   0.0012, 0.0154),
        (8, 12,  0.0013, 0.0169),
        (3, 12,  0.0014, 0.0176),
        (31, 4,  0.0015, 0.0190),
    ]


def get_network_metadata() -> Dict[str, object]:
    return {
        "fixed_oltc_edges_ordered": list(FIXED_OLTC_EDGES_ORDERED),
        "fixed_oltc_configs_undir": dict(FIXED_OLTC_CONFIGS_UNDIR),
        "fixed_shunt_buses": list(FIXED_SHUNT_BUSES),
        "fixed_shunt_bcap_pu": dict(FIXED_SHUNT_BCAP_PU),
        "transformer_origin_lines": set(TRANSFORMER_ORIGIN_LINES),
        "stress_lines": set(STRESS_LINES),
        "recommended_nonexact_oltc_taps": dict(RECOMMENDED_NONEXACT_OLTC_TAPS),
        "recommended_nonexact_shunt_status": dict(RECOMMENDED_NONEXACT_SHUNT_STATUS),
    }


# ============================================================
# Main builder
# ============================================================
def busmeshed39_opf(
    slack_vm_pu: float = 1.0,
    line_max_loading_percent: float = 120.0,
    stress_q_over_p: float = 0.95,
    stress_load_mw_each: float = 300.0,
    r_loss_scale: float = 3.0,
    max_i_ka_base: float = 2.0,
    max_i_ka_stress: float = 1.0,
    line_smax_pu_overrides: Optional[Dict[Tuple[int, int], float]] = None,
):
    """
    Fixed 41-bus IEEE 39-plus network with non-exact mesh experiment profile.

    Notes
    -----
    - Internal bus indices: 0..40
    - Buses 39 and 40 are included directly.
    - r_used = r_loss_scale * r_base
    - x_used = x_base
    """
    if r_loss_scale <= 0:
        raise ValueError("r_loss_scale must be positive.")
    if max_i_ka_base <= 0 or max_i_ka_stress <= 0:
        raise ValueError("max_i_ka_base/max_i_ka_stress must be positive.")

    overrides = LINE_SMAX_PU_OVERRIDES if line_smax_pu_overrides is None else dict(line_smax_pu_overrides)

    def _get_override_smax_pu(fb: int, tb: int):
        if (fb, tb) in overrides:
            return overrides[(fb, tb)]
        if (tb, fb) in overrides:
            return overrides[(tb, fb)]
        return None

    def _add_line_pu(net, buses, fb, tb, r_pu, x_pu, Z_base, name, max_i_ka):
        return pp.create_line_from_parameters(
            net,
            from_bus=buses[fb],
            to_bus=buses[tb],
            length_km=1.0,
            r_ohm_per_km=float(r_pu) * Z_base,
            x_ohm_per_km=float(x_pu) * Z_base,
            c_nf_per_km=0.0,
            max_i_ka=float(max_i_ka),
            r0_ohm_per_km=float(r_pu) * Z_base,
            x0_ohm_per_km=float(x_pu) * Z_base,
            c0_nf_per_km=0.0,
            name=name,
        )

    # --------------------------------------------------------
    # Base network
    # --------------------------------------------------------
    net = pp.create_empty_network(sn_mva=100.0)
    sn_mva = float(net.sn_mva)
    vn_kv = 345.0

    # Fixed 41 buses: 0..40
    buses = [pp.create_bus(net, vn_kv=vn_kv, name=f"Bus {b}") for b in range(41)]
    slack_bus = buses[0]

    net.bus["min_vm_pu"] = 0.95
    net.bus["max_vm_pu"] = 1.05

    # Slack
    eg = pp.create_ext_grid(net, bus=slack_bus, vm_pu=slack_vm_pu, va_degree=0.0, name="Slack")
    if "controllable" not in net.ext_grid.columns:
        net.ext_grid["controllable"] = True
    net.ext_grid.at[eg, "controllable"] = True

    net.ext_grid.loc[eg, ["min_p_mw", "max_p_mw"]] = [0.0, 3000.0]
    net.ext_grid.loc[eg, ["min_q_mvar", "max_q_mvar"]] = [-3000.0, 3000.0]
    net.ext_grid.loc[eg, ["min_vm_pu", "max_vm_pu"]] = [0.95, 1.05]

    pp.create_poly_cost(
        net,
        element=eg,
        et="ext_grid",
        cp2_eur_per_mw2=0.008,
        cp1_eur_per_mw=12.0,
        cp0_eur=0.0,
    )

    # --------------------------------------------------------
    # Generators
    # remote-cheap / local-expensive profile
    # --------------------------------------------------------
    pv_vm = {
        29: 1.0475,
        30: 0.9820,
        31: 0.9831,
        32: 0.9972,
        33: 1.0123,
        34: 1.0493,
        35: 1.0635,
        36: 1.0278,
        37: 1.0265,
        38: 1.0300,
    }

    gen_data = [
        # bus,   p0,             pmin, pmax,       qmin,              qmax,              c2,     c1,   c0
        (1,   0.30 * 500.0,      0.0,  500.0, -0.60 * 500.0,   0.60 * 500.0,   0.020,   9.0,  0.0),
        (5,   0.30 * 450.0,      0.0,  450.0, -0.60 * 450.0,   0.60 * 450.0,   0.018,   9.5,  0.0),

        # remote / comparatively cheaper
        (29,  0.30 * 350.0,      0.0,  350.0, -0.60 * 350.0,   0.60 * 350.0,   0.026,  10.0, 0.0),
        (30,  0.30 * 400.0,      0.0,  400.0, -0.60 * 400.0,   0.60 * 400.0,   0.024,  10.4, 0.0),
        (31,  0.30 * 900.0,      0.0,  900.0, -0.60 * 900.0,   0.60 * 900.0,   0.008,  10.2, 0.0),
        (32,  0.30 * 850.0,      0.0,  850.0, -0.60 * 850.0,   0.60 * 850.0,   0.009,  10.4, 0.0),
        (33,  0.30 * 700.0,      0.0,  700.0, -0.60 * 700.0,   0.60 * 700.0,   0.012,  11.2, 0.0),
        (37,  0.30 * 1100.0,     0.0, 1100.0, -0.60 * 1100.0,  0.60 * 1100.0,  0.007,  11.5, 0.0),

        # local / pocket-near / more expensive
        (34,  0.30 * 900.0,      0.0,  900.0, -0.60 * 900.0,   0.60 * 900.0,   0.016,  13.8, 0.0),
        (35,  0.30 * 750.0,      0.0,  750.0, -0.60 * 750.0,   0.60 * 750.0,   0.016,  13.5, 0.0),
        (36,  0.30 * 700.0,      0.0,  700.0, -0.60 * 700.0,   0.60 * 700.0,   0.017,  13.2, 0.0),
        (38,  0.30 * 1300.0,     0.0, 1300.0, -0.60 * 1300.0,  0.60 * 1300.0,  0.014,  15.0, 0.0),
    ]

    for bus_id, p0, pmin, pmax, qmin, qmax, c2, c1, c0 in gen_data:
        vm_set = float(pv_vm.get(bus_id, 1.0))
        g = pp.create_gen(
            net,
            bus=buses[bus_id],
            p_mw=p0,
            vm_pu=vm_set,
            name=f"Gen@{bus_id}",
            controllable=True,
        )
        net.gen.loc[g, ["min_p_mw", "max_p_mw"]] = [pmin, pmax]
        net.gen.loc[g, ["min_q_mvar", "max_q_mvar"]] = [qmin, qmax]
        net.gen.loc[g, ["min_vm_pu", "max_vm_pu"]] = [0.95, 1.05]
        pp.create_poly_cost(
            net,
            element=g,
            et="gen",
            cp2_eur_per_mw2=c2,
            cp1_eur_per_mw=c1,
            cp0_eur=c0,
        )

    # --------------------------------------------------------
    # Loads
    # --------------------------------------------------------
    load_scale = 0.55
    q_over_p = 0.30

    base_load_p_pdf = {
        3: 322.0,
        4: 500.0,
        7: 233.8,
        8: 522.0,
        12: 7.5,
        15: 320.0,
        16: 329.0,
        18: 158.0,
        20: 628.0,
        21: 274.0,
        23: 247.5,
        24: 308.6,
        25: 224.0,
        26: 139.0,
        27: 281.0,
        28: 206.0,
        29: 283.5,
        31: 9.2,
        39: 1104.0,
    }

    extra_loads_pdf = [
        (5, 60.0),
        (9, 45.0),
        (10, 55.0),
        (11, 40.0),
        (13, 35.0),
        (14, 30.0),
        (17, 25.0),
        (19, 50.0),
        (22, 45.0),
    ]

    load_data = []
    for k, p in base_load_p_pdf.items():
        bus_id = int(k) - 1
        p_scaled = float(load_scale) * float(p)
        q_scaled = float(q_over_p) * p_scaled
        load_data.append((bus_id, p_scaled, q_scaled))

    for k, p in extra_loads_pdf:
        bus_id = int(k) - 1
        p_scaled = float(load_scale) * float(p)
        q_scaled = float(q_over_p) * p_scaled
        load_data.append((bus_id, p_scaled, q_scaled))

    # Fixed stress loads on buses 39, 40
    pL = float(stress_load_mw_each)
    qL = float(stress_q_over_p) * pL
    load_data.append((39, pL, qL))
    load_data.append((40, pL, qL))

    for bus_id, p, q in load_data:
        pp.create_load(net, bus=buses[bus_id], p_mw=p, q_mvar=q, name=f"Load@{bus_id}")

    # --------------------------------------------------------
    # Lines
    # --------------------------------------------------------
    Z_base = (vn_kv ** 2) / net.sn_mva
    line_params = _get_changed_topology_line_params()

    for fb, tb, r_pu_base, x_pu_base in line_params:
        is_stress = _edge_in_set(fb, tb, STRESS_LINES)
        Imax = max_i_ka_stress if is_stress else max_i_ka_base

        smax_pu_override = _get_override_smax_pu(fb, tb)
        if smax_pu_override is not None:
            Imax = _smaxpu_to_imax_ka(float(smax_pu_override), sn_mva=sn_mva, vn_kv=vn_kv)

        _add_line_pu(
            net,
            buses,
            fb,
            tb,
            r_pu=float(r_pu_base) * float(r_loss_scale),
            x_pu=float(x_pu_base),
            Z_base=Z_base,
            name=f"Line {fb}-{tb}",
            max_i_ka=Imax,
        )

    # pandapower OPF line loading limit (%)
    net.line["max_loading_percent"] = float(line_max_loading_percent)

    # explicit parameter table
    net["line_params_pu_table"] = pd.DataFrame(
        [
            {
                "from_bus": fb,
                "to_bus": tb,
                "r_pu_base": r_pu_base,
                "x_pu_base": x_pu_base,
                "r_pu_used": float(r_pu_base) * float(r_loss_scale),
                "x_pu_used": float(x_pu_base),
                "is_transformer_origin": _edge_in_set(fb, tb, TRANSFORMER_ORIGIN_LINES),
                "is_stress_line": _edge_in_set(fb, tb, STRESS_LINES),
            }
            for (fb, tb, r_pu_base, x_pu_base) in line_params
        ]
    )

    # --------------------------------------------------------
    # Store metadata for downstream OPF / MINLP code
    # --------------------------------------------------------
    net["fixed_oltc_edges_ordered"] = list(FIXED_OLTC_EDGES_ORDERED)
    net["fixed_oltc_table"] = pd.DataFrame(
        [
            {
                "from_bus": i,
                "to_bus": j,
                "tap_min": FIXED_OLTC_CONFIGS_UNDIR[tuple(sorted((i, j)))].tap_min,
                "tap_max": FIXED_OLTC_CONFIGS_UNDIR[tuple(sorted((i, j)))].tap_max,
                "dV_percent": FIXED_OLTC_CONFIGS_UNDIR[tuple(sorted((i, j)))].dV_percent,
            }
            for (i, j) in FIXED_OLTC_EDGES_ORDERED
        ]
    )

    net["fixed_shunt_buses"] = list(FIXED_SHUNT_BUSES)
    net["fixed_shunt_table"] = pd.DataFrame(
        [{"bus": b, "bcap_pu": FIXED_SHUNT_BCAP_PU[b]} for b in FIXED_SHUNT_BUSES]
    )

    net["transformer_origin_lines"] = list(sorted(TRANSFORMER_ORIGIN_LINES))
    net["stress_lines"] = list(sorted(STRESS_LINES))

    # Recommended non-exact mesh discrete profiles
    net["recommended_nonexact_oltc_taps"] = pd.DataFrame(
        [{"from_bus": i, "to_bus": j, "tap": tap}
         for (i, j), tap in RECOMMENDED_NONEXACT_OLTC_TAPS.items()]
    )
    net["recommended_nonexact_shunt_status"] = pd.DataFrame(
        [{"bus": b, "status": int(RECOMMENDED_NONEXACT_SHUNT_STATUS[b])}
         for b in sorted(RECOMMENDED_NONEXACT_SHUNT_STATUS.keys())]
    )

    # --------------------------------------------------------
    # Generator P-Q capability
    # --------------------------------------------------------
    smax_gen_mva = {}
    for bus_id, _, _, pmax, *_ in gen_data:
        smax_gen_mva[int(bus_id)] = 1.20 * float(pmax)

    ok = _apply_circle_qcapability_curve_to_gens(
        net,
        smax_gen_mva,
        n_points=7,
        curve_style="straightLineYValues",
    )

    if not ok:
        for gi in net.gen.index:
            bus = int(net.gen.at[gi, "bus"])
            Smax = smax_gen_mva.get(bus, None)
            if Smax is None:
                continue
            Pmax = float(net.gen.at[gi, "max_p_mw"])
            qcap = math.sqrt(max(Smax**2 - Pmax**2, 0.0))
            net.gen.at[gi, "min_q_mvar"] = max(float(net.gen.at[gi, "min_q_mvar"]), -qcap)
            net.gen.at[gi, "max_q_mvar"] = min(float(net.gen.at[gi, "max_q_mvar"]), qcap)

    return net


if __name__ == "__main__":
    net = busmeshed39_opf()

    print(net)
    print("\n[INFO] #buses =", len(net.bus))
    print("[INFO] #lines =", len(net.line))
    print("[INFO] #gens  =", len(net.gen))
    print("[INFO] #loads =", len(net.load))

    print("\n[INFO] OLTC candidate branches:")
    print(net["fixed_oltc_table"])

    print("\n[INFO] Shunt candidate buses:")
    print(net["fixed_shunt_table"])

    print("\n[INFO] Recommended non-exact OLTC taps:")
    print(net["recommended_nonexact_oltc_taps"])

    print("\n[INFO] Recommended non-exact shunt status:")
    print(net["recommended_nonexact_shunt_status"])

    print("\n[INFO] Explicit line parameter table:")
    print(net["line_params_pu_table"][[
        "from_bus", "to_bus",
        "r_pu_base", "x_pu_base",
        "r_pu_used", "x_pu_used",
        "is_transformer_origin", "is_stress_line"
    ]])