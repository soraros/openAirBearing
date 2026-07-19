# AGENTS.md

Externally pressurized air bearing analysis library (porous-media restrictors).
Two generations of the package coexist:

- **v1** (`openairbearing/*.py`, `openairbearing/app/`) — upstream's current
  FEM-based implementation (skfem; `bearings.py`, `solution_analytic.py`,
  `solution_fem.py`, `fem_utils.py`, `mesh.py`). Fixes only, do not extend.
- **v2** (`openairbearing/v2/`) — our data-oriented rewrite. All new work goes here.

## Branches

- `v2` — main line: upstream main (0.2.x, FEM rewrite) + the v2 package.
  No panel code.
- `gui` — Panel web UI (`openairbearing/panel/`). Frozen at 0.1.x; left as-is.
- `fix/v1-bugs` — v1 (0.1.x) bug-fix series for the upstream PR. Frozen:
  upstream's rewrite removed the code those fixes target, so they apply to
  the 0.1.x line only.
- `backup/dev-2026-07-18` — pre-restructure snapshot. Safe to delete.
- `main` — tracks upstream (Aalto-Arotor/openAirBearing).

## Tooling (uv, Python ≥3.13)

```bash
uv sync                      # install/sync the venv (.venv)
uv run pytest                # or: .venv/bin/python -m pytest tests/
uv run ruff check .          # lint (repo-wide must stay green)
uv run ruff format           # format; v2 scope must stay clean
```

- Dependencies in `[project]`, dev tools in `[dependency-groups]`
  (`[tool.uv] default-groups = ["dev"]`). `uv.lock` is gitignored.
- **numba is pinned `>=0.62.1,<0.63.0`** (Intel macOS / llvmlite compat) and
  **numpy `<2.4`** (numba 0.62.x compat). Do not bump these without checking numba.
- **jax is deliberately not a dependency**: nothing imports it directly, and
  jax 0.6.2+ ships no Intel macOS wheels. `skfem.autodiff.NonlinearForm`
  (which needs jax) is imported lazily in `solution_fem.py` so the package
  imports everywhere; jax-dependent FEM tests are `importorskip`-guarded
  (they skip on Intel macs).
- Build backend: hatchling. Package ships `py.typed`.

## v2 architecture (absim_fvm style)

Layers: `primitives` (Gas/AIR/P_ATM, type aliases, validation) → `geometry`
(pad specs, `BoundarySpec`, `SurfaceError`, `GridSpec`) → `operating`
(`PorousRestrictor`, `OperatingState`) → `problem.py` (`BearingSpec` →
`build_problem` → immutable `BearingProblem`) → `solvers` (analytic, numba FDM
kernels, `solve_bearing`, frozen `Solution`) → `postprocess/plots.py`.
`catalog.py` holds the v1-default spec factories.

Conventions:

- Frozen dataclasses, pure functions, immutable arrays; `(nh, nx[, ny])` layout.
- Pads own their grid semantics (axes, boundaries, film, weights) via `PadBase`
  defaults + per-pad `EDGES` row; the compiler just orchestrates.
- Everything typed. v2 has its own `openairbearing/v2/ruff.toml` adding ANN rules
  on top of root config (2-space indent, line-length 100, preview,
  I/UP/NPY/RUF). Legacy v1 paths are grandfathered via per-file-ignores — do not
  add new paths there.
- Numba kernels: `@njit(cache=True)` (+ `parallel=True` for per-sample loops),
  plain-array signatures.
- Validate at construction/API boundary; assume invariants inside the lib.
- No sig-dig rounding, no hidden mutable state, no per-type dispatch chains.

## Testing

- `tests/` — upstream's suite (FEM-based; jax-gated tests skip on Intel macs)
  + `tests/v2/` (v2 suite: pinned values, BC/physics checks, analytic↔numeric
  cross-validation, headless example runs).
- Pinned v2 values use rtol 1e-8/1e-6 — regenerate only deliberately.
- Examples in `examples/v2/` expose `build_figures()` for headless testing;
  `.show()` only under `main()`.
- The old `tests/v2/test_cross_v1.py` is retired: upstream's rewrite removed
  the 0.1.x v1 API it compared against.

## v1 (0.1.x) quirks that shaped v2's catalog (historical, pre-rewrite)

- The 0.1.x v1 computed κ for linear/rect/journal with the base-class
  `psc=0.6e6+pa` (init ordering bug); v2's catalog reproduces those outputs by
  making the calibration pressure explicit data. 0.1.x also rounded κ to 3
  significant digits; v2 does not.
- 0.1.x rect 2-D used grid spacings inconsistent with its nodes; v2 uses
  consistent spacings and trapezoidal weights (ΣdA = area exactly).
- 0.1.x 2-D polar was broken and journal's θ grid endpoint-inclusive; v2
  implements both properly.
- Flow sign convention: `qc` carries its own sign; `qs = qa - qc` is total outflow.
- Upstream's rewrite still contains some of these bugs in new clothes
  (e.g. rect `dx=xa/(nx+1)` in `bearings.py`, `blocked` without
  `block_in`/`block_A`) — follow-up PR material; check current code first.

## Housekeeping

- `notebooks/` is untracked local work — never commit it without being asked.
- CI lint workflow uses black (legacy); the enforced local gate is ruff.

## Commit style

Uniform message shape, all lowercase (identifiers stay verbatim):

- subject: one imperative line, no prefix, no period
  (`fix the rectangular grid spacing`, `add PadBase with shared pad defaults`)
- body: one or two concise sentences — what was wrong, what changed
- outline: for multi-part changes, one dash per part
- for correctness bugs: a tiny repro example between body and outline

Example:

```
fix the repr crash for non-journal bearings

c and e were field(init=False) without defaults, assigned only by
JournalBearing:

  >>> repr(CircularBearing())
  AttributeError: 'CircularBearing' object has no attribute 'c'

give them default=None like theta/clearance.
```

Keep commits small and single-goal; fixups belong squashed into their
parent, never as standalone commits.


