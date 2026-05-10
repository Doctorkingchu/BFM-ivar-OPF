# Supplemental Code

Python (and MATLAB on the 3012-bus case) code that produced the numerical
results of:

> **A Branch Flow Model for Optimal Power Flow in Meshed Power Networks
> via Iterative Voltage-Angle Reconstruction**
> Yeonouk Chu *et al.*, submitted to IEEE Transactions on Smart Grid.

The paper proposes **BFM-ivar**, an iterative branch-flow OPF method that
reconstructs voltage angles inside every outer iteration and adds the four
mitigation features of Sec. 4.2 (trust-region proximal regularization,
loss surrogate, hybrid `ell` estimator, scheduled slack contraction).
Numerical experiments are run on four test networks.

## Layout

One folder per test network. Each folder is self-contained — run scripts
from inside their own folder.

| Folder | Test network | # benchmark methods | Reproduces |
| --- | --- | --- | --- |
| [`BFM_ivar_9bus/`](BFM_ivar_9bus/README.md)      | IEEE 9-bus                              | 8 (full Table II) | 9-bus opt/time table, 9-bus Pareto figure |
| [`BFM_ivar_41bus/`](BFM_ivar_41bus/README.md)    | IEEE 41-bus mesh (39-plus, modified)    | 8 (full Table II) | 41-bus opt/time table, 41-bus Pareto figure |
| [`BFM_ivar_300bus/`](BFM_ivar_300bus/README.md)  | IEEE 300-bus                            | 8 (full Table II) | 300-bus opt/time table, 300-bus Pareto figure |
| [`BFM_ivar_3012bus/`](BFM_ivar_3012bus/README.md)| Polish 3012-bus (`case3012wp`)          | 3 (MATPOWER MIPS, DCOPF, Proposed) | 3012-bus opt/time table, 3012-bus Pareto figure |

The 3012-bus case carries only three methods because ACOPF (MINLP),
BFM (MINLP), and MISOCP-class methods do not converge in Pyomo+SCIP at
this scale; the ACOPF reference is therefore computed in MATLAB via
MATPOWER MIPS.

## Paper Table II method → file mapping

For the 9-bus, 41-bus, and 300-bus folders, the eight Python files map
one-to-one to the benchmark methods of the paper's Table II:

| File                       | Paper method                          |
| ---                        | ---                                   |
| `ieee<N>bus.py`            | (network builder; not a method)       |
| `ACOPF_MINLP.py`           | ACOPF (MINLP)                         |
| `ACOPF_SDP.py`             | ACOPF (SDP)                           |
| `DCOPF.py`                 | DCOPF                                 |
| `BFM_MINLP.py`             | BFM (MINLP)                           |
| `BFM_MISOCP.py`            | BFM (MISOCP)                          |
| `BFM_it.py`                | BFM-it                                |
| `BFM_ivar_no_mit.py`       | BFM-ivar (no mitigation)              |
| `BFM_ivar_mit.py`          | **Proposed (BFM-ivar + mitigation)**  |

The 3012-bus folder uses `ACOPF.m` (MATLAB MATPOWER MIPS) instead of
`ACOPF_MINLP.py / ACOPF_SDP.py`, drops the BFM-class and BFM-it files
(non-convergent at this scale), and merges `BFM_ivar_no_mit.py` into
`BFM_ivar_mit.py` so the proposed file is self-contained.

## Common dependencies

All Python solvers run in a single environment with:

- Python ≥ 3.10
- `pyomo`, `pandapower`, `numpy`, `pandas`
- **SCIP** solver — invoked from Pyomo via `SolverFactory("scip", solver_io="nl")`
  (AMPL NL interface).  A native-interface fallback `SolverFactory("scip")`
  is also tried.
- `networkx` (optional; used by `BFM_ivar_mit.py` to build the T4 cycle
  basis; falls back to no-op if not installed)

Optional / per-file:

- **Gurobi** — some files probe for Gurobi first and gracefully fall back
  to SCIP if it is not on `PATH`. No license required to use the files as
  shipped.
- **IPOPT** — used as a continuous-NLP warm-start by `ACOPF_MINLP.py` and
  `BFM_MINLP.py`.  Without IPOPT these files still run but skip the
  warm-start step.

The **3012-bus** folder additionally requires:

- **MATLAB R2020a or later** with **MATPOWER ≥ 7.0** on the MATLAB path,
  for `ACOPF.m`.  See <https://matpower.org/download> and run
  `install_matpower` once before launching `ACOPF.m`.

## Quick start

```bash
cd BFM_ivar_9bus
python -B ACOPF_MINLP.py     # one of the eight (three on 3012-bus) methods
```

Each method file writes a `<method>_results.txt` log next to itself with
the gen-cost, OLTC tap selection, switched-shunt schedule, and per-bus PF
diagnostics.

See the per-folder `README.md` for the exact workflow that reproduces
each table's numbers.

## Conventions

- **Run from inside the folder.**  Each method file does
  `import ieee<N>bus as ...` by bare module name; running from a parent
  folder will not find the builder.
- **Windows + Dropbox host.**  Invoke Python with `python -B` to suppress
  `__pycache__/` directories that the host's sync agent would otherwise
  re-upload on every run.
- **No package manager files** (no `requirements.txt` / `pyproject.toml`).
  The environment is assumed pre-installed.  An indicative `pip install`
  list is given in each per-folder README.
- **No automated test harness.**  Verification is per-script (each writes
  its results log; cross-script comparisons are done by reading the logs).
