# BFM-ivar — IEEE 300-bus

Paper-supplemental code for the 300-bus row of the optimality / time
table and the 300-bus Pareto figure.  The IEEE 300-bus system (411 lines,
112 independent loops) is the scalability test in the paper: the largest
case on which every Table II method is run end-to-end.

## Files

| File                 | Role                                                         | Paper method                          |
| ---                  | ---                                                          | ---                                   |
| `ieee300bus.py`      | Network builder (`case300_opf(...)`)                         | (not a method)                        |
| `ACOPF_MINLP.py`     | Full BIM AC-OPF (non-convex, SCIP + IPOPT warm-start)        | ACOPF (MINLP)                         |
| `ACOPF_SDP.py`       | SDP-to-SOCP relaxation of ACOPF                              | ACOPF (SDP)                           |
| `DCOPF.py`           | Linearized AC PF, MIQCP                                      | DCOPF                                 |
| `BFM_MINLP.py`       | Branch-flow MINLP reference                                  | BFM (MINLP)                           |
| `BFM_MISOCP.py`      | SOC relaxation of BFM                                        | BFM (MISOCP)                          |
| `BFM_it.py`          | Iterative BFM-ar without angle recovery                      | BFM-it                                |
| `BFM_ivar_no_mit.py` | Iterative BFM with angle recovery, **no** mitigations        | BFM-ivar (no mitigation)              |
| `BFM_ivar_mit.py`    | Iterative BFM with angle recovery **+** Sec. 4.2 mitigations | **Proposed (BFM-ivar + mitigation)**  |

Each method file imports `ieee300bus as m` (or `as mcase`).

**Sibling-file interactions:**

- `ACOPF_SDP.py` reads `ACOPF_MINLP_results.txt` if it is present in the
  folder (produced by a prior `ACOPF_MINLP.py` run); this fills the
  "Algorithm Comparison" table at the end of the SDP run with the MINLP
  baseline.  If the file is absent the comparison table is just the SDP
  row alone — nothing fails.
- `BFM_ivar_mit.py` has an optional lazy `import BFM_MISOCP as socp_mod`
  (the [C13] warm-start feature is off by default).

## Dependencies

```
pip install pyomo pandapower numpy pandas
```

External solvers:

- **SCIP** — required.
- **IPOPT** — optional; used as a continuous warm-start by `ACOPF_MINLP.py`
  and `BFM_MINLP.py`.
- **Gurobi** — optional; preferred-first by `BFM_ivar_mit.py` /
  `BFM_ivar_no_mit.py` if importable, else SCIP fallback.

## How to run

For the cleanest table reproduction, run `ACOPF_MINLP.py` first so its
`ACOPF_MINLP_results.txt` is on disk for the SDP file to pick up; the
rest are order-independent.

```bash
cd BFM_ivar_300bus
python -B ACOPF_MINLP.py        # writes ACOPF_MINLP_results.txt
python -B ACOPF_SDP.py
python -B DCOPF.py
python -B BFM_MINLP.py
python -B BFM_MISOCP.py
python -B BFM_it.py
python -B BFM_ivar_no_mit.py
python -B BFM_ivar_mit.py       # the proposed method
```

`ACOPF_MINLP.py` and `BFM_ivar_no_mit.py` are the long ones at this
scale — the former times out at the configured 10-hour limit; the
latter takes ~4 hours and diverges (this divergence IS the paper's
ablation result).

## Reproducing the paper's 300-bus results

Expected outputs (paper Table for the 300-bus system, in EUR/h):

| Method                       | Reported `f_gen^eval` | Time (s)        | Status      |
| ---                          | ---                   | ---             | ---         |
| `ACOPF_MINLP.py`             | 827,960.22            | > 36,000        | Timeout     |
| `ACOPF_SDP.py`               | 809,921.04            | 526.41          | Optimal     |
| `DCOPF.py`                   | 812,066.25            | 3.96            | Optimal     |
| `BFM_MINLP.py`               | 828,841.88            | 30.49           | Suboptimal  |
| `BFM_MISOCP.py`              | 833,266.60            | 5.63            | Optimal     |
| `BFM_it.py`                  | 826,506.70            | 112.02          | Converged   |
| `BFM_ivar_no_mit.py`         | 1,085,441.34          | 14,086.63       | Diverged    |
| **`BFM_ivar_mit.py`**        | **808,412.67**        | **1,481.28**    | Converged   |

The proposed method attains the lowest cost (808,412.67) in 1,481 s —
0.45 % below DCOPF, 2.36 % below ACOPF (MINLP) (timed out), and far
below `BFM_ivar_no_mit` (diverged at 1,085,441 after ~ 4 h).  ACOPF (SDP),
the closest competitor in cost, is + 0.19 % higher without angle
recovery.

Wall-clock times are indicative — they depend on the SCIP build and
machine.

## Notes

- This folder is derived from the experiment folder
  `branchflowmodel_300bus`.  Of the six `ver{2,3,4,5,6}_ieee300bus.py`
  builder variants kept there, only `ver5_ieee300bus.py` is used by the
  paper-cited solvers; it has been renamed to `ieee300bus.py` here, and
  the other builder versions are not shipped.
- `BFM_ivar_no_mit.py` is the paper's ablation baseline (BFM-ivar
  without any of the Sec. 4.2 mitigations).  Its divergence on the
  300-bus case is one of the paper's narrative anchors — the
  mitigation features are not optional polish, they are required for
  convergence at this scale.
- The "Algorithm Comparison" table that `ACOPF_SDP.py` prints at the end
  of its run is a developer-side helper; on first run it shows only the
  SDP row, on later runs it picks up the MINLP baseline from
  `ACOPF_MINLP_results.txt` if present.
