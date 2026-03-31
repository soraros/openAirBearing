# absim_fvm

2D FVM simulation of thrust air bearings. Compressible Reynolds equation on unstructured triangular meshes.

**Module map**: `mesh.py` (Mesh), `film.py` (FVM solver), `solver.py` (IterativeSolver), `restrictor.py` (Restrictor + Orifice), `bearing.py` (Inlet, Bearing), `result.py` (BearingResult, solve_coupling), `geometry.py` (Pad, PadCircle, PadRect, PadAnnular, Hole, Groove protocols + OccResult), `geom.py` (gmsh → Mesh), `vis.py` (plotting). Shared: `constants.py` (Gas + AIR), `typing.py`.

**Entry point**: `uv run python examples/single.py`

| Instruction file | Scope | Auto-loads on |
|------------------|-------|---------------|
| [`fvm.instructions.md`](instructions/fvm.instructions.md) | Solver correctness constraints | `film.py`, `solver.py`, `mesh.py`, their tests |
| [`tests.instructions.md`](instructions/tests.instructions.md) | Pytest conventions, fixtures, conftest.py | `tests/**/*.py` |
| [`writing.instructions.md`](instructions/writing.instructions.md) | Instruction file writing standards | `*instructions*.md` |

**Design docs**: [`fvm_design.md`](../docs/fvm_design.md), [`api_design.md`](../docs/api_design.md), [`orifice_model.md`](../docs/orifice_model.md).

## Workflow

`uv sync && uv run ruff format . && uv run ruff check --fix . && uv run pytest`

Always `uv run <cmd>`. Never activate venv manually.

Never commit without explicit user approval. Stage only when a logical unit is complete.

Prefer structured tools over CLI:

| Task | Tool | Instead of |
|------|------|------------|
| Run Python | `pylanceRunCodeSnippet` | `python -c`, terminal |
| Run tests | `runTests` | `pytest` CLI |
| Errors | `get_errors`, `pylanceFileSyntaxErrors` | linter CLI |
| Symbols | `vscode_listCodeUsages` | grep |
| Refactor | `pylanceInvokeRefactoring` | manual multi-file edits |
| Git ops | gitkraken tools | git CLI |
| Notebooks | `run_notebook_cell` | jupyter CLI |

## Code Style

2-space indent, 100-char lines, `ruff` for formatting + import sorting.

Short names for domain quantities (`h`, `psi`, `mu`). Name for what it *is*: `mass_flow` not `compute_mass_flow`. Type hints everywhere, including HOFs.

Comments only when code can't express it: encoding conventions (`# 0=interior`), semantic nuance (`# size at curve, NOT at distance=0`), non-obvious "why" (`# gmsh auto-assigns IDs from 1`). Skip obvious docstrings; one-line default; expand only for subtle math/semantics.

**Design** — numba-compatible hot paths drive these choices:
1. **Data-oriented** — arrays in `@dataclass`/`NamedTuple`, pure `f(arrays) → arrays`. No deep inheritance.
2. **Immutable** — `@cached_property` for derived, new objects over mutation.
3. **Vectorized** — `np.where`, broadcasting, fancy indexing. Vectorize before reaching for loops.
4. **`@njit`** for hot paths — `parallel=True` + `prange`. Clarity over clever tricks.

Meters/pascals internally, mm/bar for display. Convert at I/O boundaries only.

## Domain

$$\nabla \cdot (h^3 \nabla \psi) = 0, \quad \psi = p^2$$

Cell-centered FVM, two-point flux. Mass flow: $\dot{m} = \frac{\rho_0}{24\mu p_0} \int h^3 \nabla\psi \cdot \hat{n} \, dL$

## Roadmap

- [ ] Numba-ify `film.py` assembly
- [ ] SVG → gmsh geometry pipeline
- [ ] Rectangular pads with grooves
- [ ] Transient solver (dynamic stiffness/damping)
