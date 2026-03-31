---
description: "Pytest conventions for this repo. Covers assertions, fixtures, naming, and conftest.py resources."
applyTo: "tests/**/*.py"
---
# Test Writing Guidelines

Run tests via `runTests` tool, not terminal.

## Assertions

| Pattern | When |
|---------|------|
| `np.testing.assert_allclose(a, b, rtol=...)` | Array comparisons (shows failed elements) |
| `np.testing.assert_array_equal(a, b)` | Exact integer/boolean arrays |
| `assert cond, f"msg {val}"` | Scalars, bounds, booleans |

Never `assert np.allclose(...)` — no diagnostics on failure.

## Organization

| Structure | When |
|-----------|------|
| Function | Default — independent tests |
| Class | Shared class-scoped fixture, or white-box tests needing documented rationale |

## Fixtures

| Scope | When |
|-------|------|
| `function` | Cheap setup, isolation needed |
| `module` | Expensive setup (mesh), read-only — **default for this repo** |
| `class` | Expensive setup modified per-test |

## conftest.py

Shared resources live in `conftest.py` — don't duplicate in test files:
- **Constants**: `P_SUPPLY`, `GAP`, `R_PAD`, `BCS`, tolerances
- **Mesh fixtures**: `unit_square`, `three_cell_strip`
- **Analytical solutions**: `analytical_psi`, `analytical_pressure`, `analytical_load` (re-exported from `absim_fvm.analytical`), `l2_error`

Every tolerance constant must have a rationale comment: `L2_REL_TOL = 0.02  # refined mesh, 2nd-order scheme`

## Naming & Docs

- Name: `test_<what>_<behavior>` — `test_solve_pressure_bounded`
- Docstring: *why* it matters, not *what* it does

## Parametrization

Use for: linearity tests, edge cases, multiple presets with identical assertions.

Avoid when test logic differs per case — write separate functions.

## White-box Tests

Testing private attrs (`solver._ilu`) requires documented justification in class docstring — explains why black-box alternative is insufficient.
