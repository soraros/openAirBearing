# UI Framework Decision

## Requirements

Staged interactive form for air bearing simulation: geometry → mesh → solve → plot. Many parameters (complex bearings with grooves, multiple zones). Geometry preview should be live-reactive; solver execution explicitly triggered.

## Options

**Dash** (current openAirBearing UI). Callbacks wired by string IDs; ~20 inputs feed a single monolithic callback returning ~11 outputs. Inline style dicts mixed into layout code. No natural stage separation — every parameter change reruns the full solver.

**Streamlit**. Script-reruns-on-every-interaction. Clean for flat parameter panels, but modifying a mesh parameter re-triggers geometry definition by default. `st.session_state` and `st.fragment` work around this — patching over the execution model, not working with it.

**Panel** (HoloViz/Anaconda). Persistent-state, event-driven. Each stage is a `param.Parameterized` class; `@param.depends(...)` declares which outputs react to which inputs. Built-in `Pipeline` for multi-stage workflows. Same code runs in Jupyter and as standalone server. Smaller community than Streamlit; actively maintained.

## Decision: Panel

- **Staged workflow** — `param.Parameterized` classes into `pn.Tabs` or `Pipeline`. Each stage owns its parameters and outputs.
- **Selective reactivity** — `@param.depends("groove_depth")` re-renders only the geometry preview; mesh stage untouched.
- **Explicit solver trigger** — heavy computation behind a button, not fired on every slider move.
- **Notebook + server** — iterate in Jupyter, deploy with `panel serve`.

The UI remains a thin shell over `solve_bearing()` and the existing Plotly figure functions. The Dash app stays functional for comparison.

## Implementation sketch

```python
class BearingGeometry(param.Parameterized):
    bearing_type = param.Selector(default="circular",
        objects=["circular", "annular", "rectangular", "grooved"])
    outer_radius = param.Number(default=18.5, bounds=(1, 100), step=0.5, label="Outer radius (mm)")
    porous_thickness = param.Number(default=4.5, bounds=(0.1, 20), step=0.1, label="Porous layer thickness (mm)")

    @param.depends("bearing_type", "outer_radius", "porous_thickness")
    def shape_preview(self):
        b = self._to_bearing()
        return pn.pane.Plotly(plot_xz_shape(b))

class MeshConfig(param.Parameterized):
    nx = param.Integer(default=30, bounds=(10, 500), label="Radial points")
    ny = param.Integer(default=1, bounds=(1, 500), label="Circumferential points")
    nh = param.Integer(default=20, bounds=(3, 100), label="Height points")
    ha_min = param.Number(default=1.0, bounds=(0.1, 50), step=0.5, label="Min air gap (μm)")
    ha_max = param.Number(default=20.0, bounds=(1, 100), step=0.5, label="Max air gap (μm)")

class SolverStage(param.Parameterized):
    solver = param.ListSelector(default=["analytic"],
        objects=["analytic", "numeric", "numeric2d"])

    def __init__(self, geometry, mesh, **kwargs):
        super().__init__(**kwargs)
        self.geometry = geometry
        self.mesh = mesh

    def solve(self, event=None):
        bearing = self.geometry.to_bearing(self.mesh)
        self.results = [solve_bearing(bearing, s) for s in self.solver]

# App assembly
geo = BearingGeometry()
mesh = MeshConfig()
solver = SolverStage(geometry=geo, mesh=mesh)

app = pn.Tabs(
    ("Geometry", pn.Row(pn.Param(geo, expand=True, width=350), geo.shape_preview)),
    ("Mesh",     pn.Row(pn.Param(mesh, expand=True, width=350), mesh.preview)),
    ("Results",  solver.results_panel()),
)
app.servable()
```

## Comparison

| Aspect | Dash (current) | Panel (planned) |
| --- | --- | --- |
| Reactivity | All-or-nothing via callbacks | Per-parameter via `@param.depends` |
| Stages | Single page, all inputs fire together | `pn.Tabs` / `Pipeline`, independent |
| Widgets | Manual `dcc.Input` + string IDs | Auto from `param.Parameterized` |
| Styling | Inline Python dicts | CSS classes + stylesheet |
| Notebook | No | Same code runs in Jupyter |
| Heavy computation | Blocks on every change | Explicit trigger via button |
