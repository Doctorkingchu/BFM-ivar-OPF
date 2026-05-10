"""Generate a Python warm-start snapshot for BFM_ivar_mit.py.

Parses DCOPF_results.txt for per-edge P (pu) and per-bus theta (deg). Q is
set to 0 (DCOPF assumes lossless), vm to 1.0 (DCOPF assumes flat voltage).
This is a much better seed than the uniform (alpha*smax)^2 / vmean^2
heuristic because P is spatially correct -- iter 1 sees realistic loss only
on edges that actually carry flow, instead of phantom losses spread
uniformly across all 3566 edges that saturate KCL slack.

Run once after DCOPF.py completes; BFM_ivar_mit picks the snapshot up
automatically on its next run (auto-invoked from BFM_ivar_mit.main() when
bfmag_warmstart_snapshot.npz is missing).

Usage:
    cd BFM_ivar_3012bus
    python -B make_warmstart_snapshot.py                 # DCOPF source
    python -B make_warmstart_snapshot.py --dcopf-results my_dcopf.txt
"""
from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np


SNAPSHOT_FILENAME = "bfmag_warmstart_snapshot.npz"
DEFAULT_DCOPF_RESULTS = "DCOPF_results.txt"

_BUS_ANGLE_RE = re.compile(r"^Bus\s+(\d+):\s+theta=([+\-]?\d+\.\d+)")
_LINE_FLOW_RE = re.compile(
    r"^Edge\s+(\d+)-(\d+):\s+f=([+\-]?\d+\.\d+)\s+pu"
)


def parse_dcopf_results(
    path: Path,
) -> Tuple[Dict[int, float], Dict[Tuple[int, int], float]]:
    """Parse a DCOPF result text file produced by DCOPF.py.

    Returns:
        bus_theta_deg: bus_idx -> theta in degrees
        edge_p_pu:     (i, j) -> P_pu in DCOPF's i<j convention with sign
                       (positive = power flows i to j).
    """
    if not path.is_file():
        raise FileNotFoundError(f"DCOPF result file not found: {path}")

    bus_theta_deg: Dict[int, float] = {}
    edge_p_pu: Dict[Tuple[int, int], float] = {}

    section = None
    with path.open(encoding="utf-8", errors="replace") as fh:
        for line in fh:
            stripped = line.rstrip()
            if stripped.startswith("--- Bus Angles"):
                section = "angles"
                continue
            if stripped.startswith("--- Line flows"):
                section = "flows"
                continue
            if stripped.startswith("--- Generator Dispatch") or \
               stripped.startswith("--- OLTC") or \
               stripped.startswith("--- Nodal balance") or \
               stripped.startswith("--- Bus Voltages"):
                section = None
                continue

            if section == "angles":
                m = _BUS_ANGLE_RE.match(stripped.lstrip())
                if m:
                    bus_theta_deg[int(m.group(1))] = float(m.group(2))
            elif section == "flows":
                m = _LINE_FLOW_RE.match(stripped.lstrip())
                if m:
                    i = int(m.group(1))
                    j = int(m.group(2))
                    f_pu = float(m.group(3))
                    edge_p_pu[(i, j)] = f_pu

    if not bus_theta_deg:
        raise ValueError(
            f"No bus angle entries parsed from {path}. Expected lines like "
            f"'Bus  0: theta=+0.316941'."
        )
    if not edge_p_pu:
        raise ValueError(
            f"No edge flow entries parsed from {path}. Expected lines like "
            f"'Edge 8-10: f=-2.415722 pu'."
        )

    return bus_theta_deg, edge_p_pu


def _build_net_and_bfm_edges():
    """Build the same net BFMag uses, return (net, E, sn_mva).

    Mirrors NETWORK_BUILD_KWARGS from BFM_ivar_mit.py so the
    warm-start snapshot is keyed on the same BFM edge set.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import ieee3012bus as mcase
    from BFM_ivar_mit import (
        NETWORK_BUILD_KWARGS,
        build_cfg_from_net_metadata,
        extract_data_fullmesh_branch_table,
    )

    net = mcase.case3012_opf(**NETWORK_BUILD_KWARGS)
    cfg = build_cfg_from_net_metadata(net)
    data = extract_data_fullmesh_branch_table(net, cfg)
    return net, data


def make_dcopf_snapshot(
    dcopf_path: Path,
    out_path: Path,
) -> None:
    """Build a warm-start snapshot from DCOPF results."""
    print(f"[1/3] Parsing DCOPF results from {dcopf_path.name}...")
    bus_theta_deg, edge_p_pu = parse_dcopf_results(dcopf_path)
    print(f"      Parsed {len(bus_theta_deg)} bus angles, {len(edge_p_pu)} edge flows.")

    print("[2/3] Building net + BFM edge set (same kwargs as BFMag driver)...")
    net, data = _build_net_and_bfm_edges()
    E = data["E"]
    sn_mva = float(data["sn_mva"])
    buses = data["buses"]
    print(f"      net has {len(buses)} buses, BFM has {len(E)} directed edges, sn_mva={sn_mva}.")

    # Map DCOPF (min,max) bus pairs -> P (signed in min->max direction).
    # DCOPF reports edges in i<j canonical order; sign carries direction.
    dcopf_lookup: Dict[frozenset, float] = {}
    for (i, j), p in edge_p_pu.items():
        if i == j:
            continue
        key = frozenset((i, j))
        # If the same {i,j} pair appears more than once (shouldn't but
        # defensive), keep the first since they should agree.
        if key in dcopf_lookup:
            continue
        # Store in (min,max) direction.
        if i <= j:
            dcopf_lookup[key] = (i, j, p)
        else:
            dcopf_lookup[key] = (j, i, -p)

    print("[3/3] Matching BFM edges to DCOPF flows...")
    edge_from = np.zeros(len(E), dtype=np.int32)
    edge_to = np.zeros(len(E), dtype=np.int32)
    P_pu_arr = np.zeros(len(E), dtype=np.float64)
    Q_pu_arr = np.zeros(len(E), dtype=np.float64)
    matched = 0
    unmatched = 0
    for idx, (i, j) in enumerate(E):
        edge_from[idx] = int(i)
        edge_to[idx] = int(j)
        key = frozenset((int(i), int(j)))
        rec = dcopf_lookup.get(key)
        if rec is None:
            unmatched += 1
            continue
        a, b, p_in_a_to_b = rec
        # Convert DCOPF's a->b direction to BFM's i->j direction.
        if (i, j) == (a, b):
            P_pu_arr[idx] = float(p_in_a_to_b)
        elif (i, j) == (b, a):
            P_pu_arr[idx] = -float(p_in_a_to_b)
        else:
            unmatched += 1
            continue
        matched += 1

    print(f"      matched {matched}/{len(E)} edges, unmatched {unmatched}.")
    if matched == 0:
        raise RuntimeError(
            "Failed to match any BFM edges to DCOPF flows. Are the two using "
            "the same bus index convention? (Both should use pp 0-indexed.)"
        )

    bus_idx_arr = np.array(sorted(buses), dtype=np.int32)
    theta_deg_arr = np.zeros(len(bus_idx_arr), dtype=np.float64)
    vm_pu_arr = np.ones(len(bus_idx_arr), dtype=np.float64)
    bus_unmatched = 0
    for k, b in enumerate(bus_idx_arr):
        t = bus_theta_deg.get(int(b))
        if t is None:
            bus_unmatched += 1
            continue
        theta_deg_arr[k] = float(t)

    if bus_unmatched > 0:
        print(f"      WARN: {bus_unmatched} buses had no DCOPF angle (set to 0.0).")

    # Statistics for sanity check.
    p_abs = np.abs(P_pu_arr)
    print(
        f"      P_pu stats: max={p_abs.max():.3f}, mean={p_abs.mean():.3f}, "
        f"#edges with |P|>0.01: {int((p_abs > 0.01).sum())}/{len(E)}"
    )
    nonzero_p = (p_abs > 1e-9).sum()
    if nonzero_p < 0.3 * len(E):
        print(
            f"      WARN: only {nonzero_p}/{len(E)} edges have nonzero P. "
            f"Snapshot may be sparse — check that DCOPF solved a proper case."
        )

    # ell_pf = (P^2 + Q^2) / vm_fr^2.  DCOPF gives Q≈0, vm≈1, so ell_pf ≈ P^2.
    ell_pf_arr = (P_pu_arr ** 2 + Q_pu_arr ** 2) / 1.0  # vm^2 = 1.0

    np.savez(
        out_path,
        source="dcopf",
        bus_idx=bus_idx_arr,
        theta_deg=theta_deg_arr,
        vm_pu=vm_pu_arr,
        edge_from=edge_from,
        edge_to=edge_to,
        P_pu=P_pu_arr,
        Q_pu=Q_pu_arr,
        ell_pf=ell_pf_arr,
        sn_mva=np.float64(sn_mva),
    )
    print(f"\n[OK] Snapshot written to {out_path.name}")
    print(
        f"     {len(bus_idx_arr)} buses, {len(E)} edges, "
        f"max ell_pf = {float(ell_pf_arr.max()):.3f} pu^2, "
        f"mean ell_pf = {float(ell_pf_arr.mean()):.4f} pu^2"
    )
    p_loss_mw = sn_mva * sum(
        float(data["r"][(int(edge_from[k]), int(edge_to[k]))]) * float(ell_pf_arr[k])
        for k in range(len(E))
    )
    q_loss_mvar = sn_mva * sum(
        float(data["x"][(int(edge_from[k]), int(edge_to[k]))]) * float(ell_pf_arr[k])
        for k in range(len(E))
    )
    print(
        f"     implied iter-1 P-loss ~ {p_loss_mw:.1f} MW, "
        f"Q-loss ~ {q_loss_mvar:.1f} MVAr "
        f"(heuristic warm-start was 652.1 MW / 3520.0 MVAr uniform)."
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dcopf-results", default=DEFAULT_DCOPF_RESULTS,
        help="DCOPF result text file (default: %(default)s)"
    )
    parser.add_argument(
        "--out", default=SNAPSHOT_FILENAME,
        help="Output .npz path (default: %(default)s)"
    )
    args = parser.parse_args()

    out_path = Path(args.out).resolve()
    dcopf_path = Path(args.dcopf_results).resolve()
    make_dcopf_snapshot(dcopf_path, out_path)


if __name__ == "__main__":
    main()
