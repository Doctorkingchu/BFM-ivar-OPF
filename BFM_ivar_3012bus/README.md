# BFM-ivar — Polish 3012-bus (`case3012wp`)

Paper-supplemental code for the 3012-bus row of the optimality / time
table and the 3012-bus Pareto figure.  The Polish 3012-bus winter
2007-08 evening-peak case (3,012 buses, 3,572 lines, 506 generators) is
the paper's largest stress test.

This folder differs structurally from the other three:

- Only **three methods** appear in the paper's 3012-bus table — MATPOWER
  MIPS, DCOPF, and the proposed BFM-ivar with mitigation.  ACOPF
  (MINLP), BFM (MINLP), and MISOCP-class methods do not converge in
  Pyomo+SCIP at this scale, so they are not shipped.
- The ACOPF reference is computed in **MATLAB** via MATPOWER MIPS
  (`ACOPF.m`), not Python.
- `BFM_ivar_mit.py` is **self-contained**: the BFMag base and the
  Sec. 4.2 mitigation wrappers are merged into a single file (no
  separate `BFM_ivar_no_mit.py` here).
- A helper script `make_warmstart_snapshot.py` is included — it produces
  the iter-1 warm-start that the proposed method needs to converge at
  3012-bus scale (the built-in heuristic fallback diverges on this
  case).

## Files

| File                          | Role                                                              | Paper method                           |
| ---                           | ---                                                               | ---                                    |
| `case3012wp.m`                | MATPOWER case data, Polish winter 2007-08 evening peak            | (case data; not a method)              |
| `ACOPF.m`                     | MATPOWER MIPS AC-OPF reference (run from MATLAB)                  | MATPOWER MIPS                          |
| `ieee3012bus.py`              | Python network builder (`case3012_opf(...)`)                      | (not a method)                         |
| `DCOPF.py`                    | Linearized AC PF, MIQCP                                           | DCOPF                                  |
| `BFM_ivar_mit.py`             | Iterative BFM with angle recovery **+** Sec. 4.2 mitigations      | **Proposed (BFM-ivar + mitigation)**   |
| `make_warmstart_snapshot.py`  | DCOPF → `bfmag_warmstart_snapshot.npz` parser; helper, not a method | (helper)                             |

`case3012wp.m` keeps its canonical MATPOWER filename — the function
inside is `function mpc = case3012wp`, and renaming the file would break
the `mpc = case3012wp;` call inside `ACOPF.m`.

## Dependencies

Python side:

```
pip install pyomo pandapower numpy pandas networkx
```

External solvers:

- **SCIP** — required for `DCOPF.py` and `BFM_ivar_mit.py`.
- **Gurobi** — optional; `BFM_ivar_mit.py` prefers it if importable,
  else SCIP fallback.

MATLAB side (only for `ACOPF.m`):

- **MATLAB R2020a or later**
- **MATPOWER 7.0 or later** — see <https://matpower.org/download>.  Run
  `install_matpower` once before launching `ACOPF.m`.  This folder's
  `case3012wp.m` shadows MATPOWER's bundled copy so local edits win over
  the library copy.

## How to run — reproducing the paper's 3012-bus row

The three methods are run independently, but `BFM_ivar_mit.py` has a
runtime dependency on the DCOPF result file, so DCOPF must be run first.

### 1. MATPOWER MIPS reference (≈ 6 s of solver time)

In MATLAB:

```matlab
cd path/to/BFM_ivar_3012bus
ACOPF
```

Produces `ACOPF_results.txt` (the run log) and `ACOPF_snapshot.mat`
(the converged dispatch as a `struct`).  Expected reported objective:
**2,473,364.14 EUR/h** in ≈ 6.39 s.

### 2. DCOPF (≈ 13,000 s of solver time)

```bash
cd BFM_ivar_3012bus
python -B DCOPF.py
```

Produces `DCOPF_results.txt`.  Expected reported objective:
**2,405,278.94 EUR/h** in ≈ 13,416.36 s.

### 3. BFM-ivar with mitigation — the proposed method (≈ 5,000 s)

```bash
python -B BFM_ivar_mit.py
```

This file does the following at startup:

1. Auto-checks for `bfmag_warmstart_snapshot.npz`.  If missing,
2. Auto-checks for `DCOPF_results.txt` (from step 2 above).  If found,
3. Calls `make_warmstart_snapshot.make_dcopf_snapshot()` to generate
   `bfmag_warmstart_snapshot.npz` from the DCOPF results.
4. Runs the BFM-ivar iteration with mitigation features (T4 cycle-KVL
   penalty, W3 adaptive hybrid weight, H1 top-K KCL hard buses,
   distributed slack, capacity release, scheduled slack contraction).
5. Verifies the slack-interior property (Sec. 3.2.3 / Prop. 2) at the
   end of `main()`.

Produces `BFM_ivar_mit_results.txt` (the run log).  Expected reported
objective: **2,297,978.72 EUR/h** in ≈ 5,023.64 s.

### Why the warm-start step is required

The (alpha · S_max)² / v_mean² heuristic warm-start that BFM-ivar falls
back to spreads ~ 650 MW active and ~ 3.5 GVAr reactive of fictitious
losses *uniformly* across all 3,566 BFM edges.  At 3012-bus scale this
saturates the per-bus KCL slack budget on a few stiff buses and triggers
an iter-2 REJECT cascade, so the method effectively will not converge
without a better seed.  `pandapower.runpp` on `case3012wp` fails with
"Voltage controlling elements at the same bus have different setpoints"
and is not a viable fallback.

The DCOPF result, parsed by `make_warmstart_snapshot.py` into a
`.npz` with per-edge P (pu) and per-bus θ (deg), gives a spatially
correct seed where iter 1 sees realistic loss only on the edges that
actually carry flow.  This is the path the paper's reported result is
produced on.

## Reproducing the paper's 3012-bus table

| Method                  | Reported `f_gen^eval` (EUR/h) | Time (s)     |
| ---                     | ---                           | ---          |
| `ACOPF.m`               | 2,473,364.14                  | 6.39         |
| `DCOPF.py`              | 2,405,278.94                  | 13,416.36    |
| **`BFM_ivar_mit.py`**   | **2,297,978.72**              | **5,023.64** |

The proposed method delivers a 7.09 % cost reduction vs. MATPOWER MIPS
(which holds OLTC taps and switched shunts fixed), and a 4.46 % cost
reduction vs. DCOPF — by exploiting the 24 OLTC and 9 switched-shunt
degrees of freedom that MATPOWER ACOPF treats as fixed.

Wall-clock times are indicative — they depend on the SCIP and MATLAB
builds and the machine.

## Notes

- Why is there no `BFM_ivar_no_mit.py` here when the other three folders
  have it?  The original two-file mit / no-mit split (where the mit
  wrapper imported the no-mit module as `base` and monkey-patched it)
  has been collapsed into a single self-contained `BFM_ivar_mit.py`
  on this network.  The merged file is ~ 4,700 lines and applies every
  override at the constant definition site — no runtime monkey-patching.
- H1 (top-K KCL hard-equality buses) reads its source from the
  *previous* run of `BFM_ivar_mit.py`.  The first run finds no prior log
  and the feature is a no-op (the algorithm still converges); the
  second run picks up the previous run's top-K KCL slack-absorbing
  buses and applies the strict-equality constraint to them.
- `pandapower.runpp` is known to fail on `case3012wp` with "Voltage
  controlling elements at the same bus have different setpoints" — the
  warm-start cascade in `BFM_ivar_mit.py` already accounts for this by
  preferring the DCOPF-seeded `.npz` over a `runpp`-derived seed.
