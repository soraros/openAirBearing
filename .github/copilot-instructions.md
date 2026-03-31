# openAirBearing

Air bearing analysis library. Analytic + numeric (FDM) solvers for externally pressurized porous bearings: circular, annular, rectangular, journal. Compressible Reynolds equation.

**Module map**: `bearings.py` (`BaseBearing` → `CircularBearing`, `AnnularBearing`, …), `solvers.py` (`solve_bearing`, analytic + 1D/2D numeric), `utils.py` (derived quantities), `plots.py` (Plotly figures), `config.py` (`DEMO_MODE`). Current web UI: `app/` (Dash).

**Entry point**: `pip install -e .` then `openairbearing` (Dash app) or `python examples/simple_circular_bearing.py`

| Instruction file | Scope | Auto-loads on |
| --- | --- | --- |
| [`tests.instructions.md`](instructions/tests.instructions.md) | Pytest conventions | `tests/**/*.py` |
| [`writing.instructions.md`](instructions/writing.instructions.md) | Instruction file standards | `**/*instructions*.md` |

## Workflow

`pip install -e . && pytest`

Never commit without explicit user approval.

| Task | Tool | Instead of |
| --- | --- | --- |
| Run tests | `runTests` | `pytest` CLI |
| Errors | `get_errors` | linter CLI |
| Symbols | `vscode_listCodeUsages` | grep |

## Code Style

2-space indent, `ruff` for formatting. Type hints on everything.

Short domain names (`ha`, `pa`, `mu`, `kappa`). Name for what it *is*: `load_capacity` not `compute_load_capacity`. One-line docstrings; expand only for non-obvious math/semantics.

**Design** — dataclass bearings, pure functions over methods:
1. **Data-oriented** — parameters in `@dataclass`, solver logic in standalone functions (`solve_bearing(b, soltype)`).
2. **Vectorized** — `numpy` broadcasting; no performance-critical paths.
3. **Units** — SI internally (Pa, m, kg/m³). Convert at I/O boundaries only.

## Domain

$$\nabla \cdot (h^3 \nabla \psi) = 0, \quad \psi = p^2$$

Feeding parameter: `beta = 12 * mu * kappa / hp` — couples porous layer permeability to air gap.

## Roadmap

Library:
- [ ] Refactor `BaseBearing` (reduce field sprawl, separate geometry from solver config)
- [ ] Add `conftest.py` with shared fixtures and analytical references
- [ ] Improve test coverage (2D numeric, journal bearing)

UI:
- [ ] Add Panel UI alongside Dash (see [`ui-framework-decision.md`](../docs/ui-framework-decision.md))
- [ ] Keep Dash app functional for comparison against Panel implementation
