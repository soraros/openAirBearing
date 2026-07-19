# AGENTS.md

Externally pressurized air bearing analysis library (porous-media restrictors).
Two generations of the package coexist:

- **v1** (`openairbearing/*.py`, `openairbearing/app/`) — legacy mutable-dataclass
  design plus the Dash web app. Kept working, do not extend; fixes only.
- **v2** (`openairbearing/v2/`) — the data-oriented rewrite. All new work goes here.

## Branches

- `v2` — main line: v1 lib + v2. No panel code.
- `gui` — Panel web UI (`openairbearing/panel/`, panel entry point, VS Code tasks).
  Branched off before the v2 work; v1-only. Panel deps exist only there.
- `backup/dev-2026-07-18` — pre-restructure snapshot (dev incl. panel commit). Safe to
  delete once the split is confirmed good.
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

- `tests/` — v1 suite (pinned v1 regression values) + `tests/v2/` (v2 suite:
  pinned values, BC/physics checks, analytic↔numeric cross-validation,
  v1↔v2 cross-checks in `test_cross_v1.py`, headless example runs).
- 262 tests must stay green. Pinned v2 values use rtol 1e-8/1e-6 — regenerate
  only deliberately.
- Examples in `examples/v2/` expose `build_figures()` for headless testing;
  `.show()` only under `main()`.

## v1 quirks that v2 deliberately handles differently (don't "fix" tests over these)

- v1 computes κ for linear/rect/journal with the base-class `psc=0.6e6+pa`
  (post-init ordering bug); v2's catalog reproduces v1 outputs by making the
  calibration pressure explicit data. 1-D v1↔v2 agreement is ~5e-4 (v1 rounds κ
  to 3 significant digits).
- v1's rect 2-D grid spacing (`dx=lx/(nx+1)`) is inconsistent with its nodes and
  its dA under-integrates the area; v2 uses consistent spacings and trapezoidal
  weights (ΣdA = area exactly), so 2-D absolute values differ from v1.
- v1's 2-D polar path is broken (no-op `factors` line); v2 implements it properly.
- Journal: v1's endpoint-inclusive θ grid causes artifacts; v2 uses an
  endpoint-free periodic grid.
- Flow sign convention: `qc` carries its own sign; `qs = qa - qc` is total outflow.

## Housekeeping

- `notebooks/` is untracked local work — never commit it without being asked.
- CI lint workflow uses black (legacy); the enforced local gate is ruff.
- Commit style: imperative subject; v2 work so far is small focused commits.
