# BFM-ivar — IEEE 9-bus

Paper-supplemental code for the 9-bus row of the optimality / time table
and the 9-bus Pareto figure.  The 9-bus system is the simplest setting: a
single loop, a small variable count, and the cleanest accuracy comparison
between the eight benchmark methods.

## Files

| File                  | Role                                                         | Paper method                          |
| ---                   | ---                                                          | ---                                   |
| `ieee9bus.py`         | Network builder (`busmeshed9_opf(...)`)                      | (not a method)                        |
| `ACOPF_MINLP.py`      | Full BIM AC-OPF (non-convex, SCIP + IPOPT warm-start)        | ACOPF (MINLP)                         |
| `ACOPF_SDP.py`        | SDP-to-SOCP relaxation of ACOPF (2×2 principal minors)       | ACOPF (SDP)                           |
| `DCOPF.py`            | Linearized AC PF, MIQP                                       | DCOPF                                 |
| `BFM_MINLP.py`        | Branch-flow MINLP reference (keeps `P²+Q²=δ·v·ell`)          | BFM (MINLP)                           |
| `BFM_MISOCP.py`       | SOC relaxation of BFM                                        | BFM (MISOCP)                          |
| `BFM_it.py`           | Iterative BFM-ar without angle recovery (Jo *et al.*)        | BFM-it                                |
| `BFM_ivar_no_mit.py`  | Iterative BFM with angle recovery, **no** mitigations        | BFM-ivar (no mitigation)              |
| `BFM_ivar_mit.py`     | Iterative BFM with angle recovery **+** Sec. 4.2 mitigations | **Proposed (BFM-ivar + mitigation)**  |

Each method file imports `ieee9bus as m` (or `as mcase`); they do not
import each other.

## Dependencies

```
pip install pyomo pandapower numpy pandas
```

External solvers:

- **SCIP** — required for every method file.
- **IPOPT** — optional; used as a continuous warm-start inside
  `ACOPF_MINLP.py` and `BFM_MINLP.py`.
- **Gurobi** — optional; some files prefer it if importable, otherwise
  silently fall back to SCIP.

## How to run

Run each method directly:

```bash
cd BFM_ivar_9bus
python -B ACOPF_MINLP.py
python -B ACOPF_SDP.py
python -B DCOPF.py
python -B BFM_MINLP.py
python -B BFM_MISOCP.py
python -B BFM_it.py
python -B BFM_ivar_no_mit.py
python -B BFM_ivar_mit.py        # the proposed method
```

The eight methods are independent — there is no required order.  Each
prints its gen-cost, wall-clock time, OLTC tap selection, and switched-
shunt schedule to stdout.

## Reproducing the paper's 9-bus results

Expected outputs (paper Table for the 9-bus system, in EUR/h):

| Method                       | Reported `f_gen^eval` | Time (s) | Angle rec. |
| ---                          | ---                   | ---      | ---        |
| `ACOPF_MINLP.py`             | 5019.43               | 300.72   | ✓          |
| `ACOPF_SDP.py`               | 5015.89               | 13.00    | ✗          |
| `DCOPF.py`                   | 5220.36               | 1.51     | ✓          |
| `BFM_MINLP.py`               | 5022.99               | 4.53     | ✗          |
| `BFM_MISOCP.py`              | 5031.10               | 0.26     | ✗          |
| `BFM_it.py`                  | 5050.91               | 0.50     | ✗          |
| `BFM_ivar_no_mit.py`         | 5052.44               | 1.16     | ✓          |
| **`BFM_ivar_mit.py`**        | **5019.53**           | **6.11** | ✓          |

The proposed method matches `ACOPF_MINLP` within 0.002% with angle
recovery; DCOPF deviates by ≈ 4% and ACOPF (SDP), though slightly lower,
lacks angle recovery.

Wall-clock times are indicative — they depend on the SCIP build and
machine.

## Notes

- The bare `ieee9bus2`-named builder of the source experiment folder has
  been renamed to `ieee9bus.py` here.
- The 9-bus case is small enough that every method file finishes in
  seconds; the most time-expensive is `ACOPF_MINLP.py` (the only
  non-convex MINLP that does not collapse to LP under round-and-fix).
