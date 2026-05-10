# BFM-ivar — IEEE 41-bus mesh

Paper-supplemental code for the 41-bus row of the optimality / time table
and the 41-bus Pareto figure.  The 41-bus system is a modified IEEE
39-bus (the "39-plus" stressed mesh used in the paper) with 11 independent
loops, additional stress lines, and a load pocket — the case that most
clearly exposes BFM-relaxation limitations and the mitigation gain.

## Files

| File                 | Role                                                         | Paper method                          |
| ---                  | ---                                                          | ---                                   |
| `ieee41bus.py`       | Network builder (`busmeshed39_opf(...)`)                     | (not a method)                        |
| `ACOPF_MINLP.py`     | Full BIM AC-OPF (non-convex, SCIP + IPOPT warm-start)        | ACOPF (MINLP)                         |
| `ACOPF_SDP.py`       | SDP-to-SOCP relaxation of ACOPF                              | ACOPF (SDP)                           |
| `DCOPF.py`           | Linearized AC PF, MIQCP                                      | DCOPF                                 |
| `BFM_MINLP.py`       | Branch-flow MINLP reference                                  | BFM (MINLP)                           |
| `BFM_MISOCP.py`      | SOC relaxation of BFM                                        | BFM (MISOCP)                          |
| `BFM_it.py`          | Iterative BFM-ar without angle recovery (Jo *et al.*)        | BFM-it                                |
| `BFM_ivar_no_mit.py` | Iterative BFM with angle recovery, **no** mitigations        | BFM-ivar (no mitigation)              |
| `BFM_ivar_mit.py`    | Iterative BFM with angle recovery **+** Sec. 4.2 mitigations | **Proposed (BFM-ivar + mitigation)**  |

Each method file imports `ieee41bus as m` (or `as mcase`).
`BFM_ivar_mit.py` also has an optional lazy `import BFM_MISOCP as
socp_mod` (the [C13] feature is off by default; enabling it requires the
sibling `BFM_MISOCP.py`).

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

Each method is independent:

```bash
cd BFM_ivar_41bus
python -B ACOPF_MINLP.py
python -B ACOPF_SDP.py
python -B DCOPF.py
python -B BFM_MINLP.py
python -B BFM_MISOCP.py
python -B BFM_it.py
python -B BFM_ivar_no_mit.py
python -B BFM_ivar_mit.py        # the proposed method
```

`ACOPF_MINLP.py`, `ACOPF_SDP.py`, and `BFM_MINLP.py` will hit the
10-hour (36,000 s) solver time limit on this case — that timeout
behavior is itself the paper's reported result for these methods.  Set
your own shorter limit by editing `SCIP_TIME_LIMIT` near the top of each
file if a faster smoke test is needed.

## Reproducing the paper's 41-bus results

Expected outputs (paper Table for the 41-bus system, in EUR/h):

| Method                       | Reported `f_gen^eval` | Time (s)      | Status     |
| ---                          | ---                   | ---           | ---        |
| `ACOPF_MINLP.py`             | 66,636.72             | > 36,000      | Timeout    |
| `ACOPF_SDP.py`               | 66,306.46             | > 36,000      | Timeout    |
| `DCOPF.py`                   | 67,348.27             | 884.61        | Optimal    |
| `BFM_MINLP.py`               | 66,250.73             | > 36,000      | Timeout    |
| `BFM_MISOCP.py`              | 66,405.73             | 36.04         | Optimal    |
| `BFM_it.py`                  | 67,427.17             | 53.17         | Converged  |
| `BFM_ivar_no_mit.py`         | 94,105.90             | 229.53        | Diverged   |
| **`BFM_ivar_mit.py`**        | **66,277.43**         | **1,451.55**  | Converged  |

ACOPF (MINLP/SDP) and BFM (MINLP) all hit the 10-hour timeout,
confirming exact solutions are infeasible at this scale.  The proposed
method attains 66,277.43 (+ 0.04 % vs. BFM (MINLP), which times out
without angle recovery; − 1.59 % vs. DCOPF), dominating the time–cost
plane among angle-recovering methods.

`BFM_ivar_no_mit.py` diverges to 94,105.90 — this is the paper's
ablation row demonstrating that the Sec. 4.2 mitigation features are
load-bearing on meshed networks.

Wall-clock times are indicative — they depend on the SCIP build and
machine.

## Notes

- This folder originated from the experiment folder
  `branchflowmodel_41bus_relaxation`; the `_relaxation` suffix is dropped
  here for the upload package.
- `BFM_ivar_mit.py` (source: `ver2_BFMag_final.py`) is the paper's
  proposed method on this network; the sibling `ver3_BFMag_final.py`
  that exists in the source experiment folder is a newer un-paper'd
  variant and is intentionally not included.
- The network builder `ieee41bus.busmeshed39_opf(...)` is the
  "39-plus" stressed-mesh variant of IEEE 39-bus, not the standard
  IEEE 39-bus case.
