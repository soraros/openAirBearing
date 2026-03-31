---
description: "Pytest conventions. Assertions, fixtures, naming, conftest.py."
applyTo: "tests/**/*.py"
---
# Test Guidelines

Run tests via `runTests` tool, not terminal.

## Assertions

| Pattern | When |
| --- | --- |
| `np.testing.assert_allclose(a, b, rtol=...)` | Array comparisons (shows failed elements) |
| `np.testing.assert_array_equal(a, b)` | Exact integer/boolean arrays |
| `assert cond, f"msg {val}"` | Scalars, bounds, booleans |

Never `assert np.allclose(...)` — no diagnostics on failure.

## Organization

Default: standalone functions. Use classes only for shared class-scoped fixtures or white-box tests with documented rationale.

## Fixtures

| Scope | When |
| --- | --- |
| `function` | Cheap setup, isolation needed |
| `module` | Expensive setup (solver), read-only |

## conftest.py (planned)

Shared resources belong in `conftest.py`:
- Tolerance constants with rationale: `RTOL_NUMERIC = 0.05  # FDM vs analytic, coarse grid`
- Default bearing fixtures: `CircularBearing()`, `AnnularBearing()`, etc.
- Analytical reference solutions for cross-validation

## Naming

`test_<what>_<behavior>` — `test_pressure_bounded_by_supply`

Docstring: *why* it matters, not *what* it does.

## Parametrize

Use for: multiple bearing types with identical assertions, grid refinement sweeps. Avoid when test logic differs per case.
