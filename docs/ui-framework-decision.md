# UI Framework Decision

## Context

We're building an interactive web interface for an air bearing FVM simulator. The design is compositional — bearings can have complex geometry (grooves, multiple zones), which means more parameters, staged workflows, and heavier solver calls than a simple parameter-sweep tool.

The interface should feel like an interactive form where the designer works through stages: define geometry → generate mesh → run solver → inspect plots. Some elements should be "live" — e.g. a bearing profile diagram that reshapes as you drag a groove-depth slider — while others (solver execution) are explicitly triggered.

We looked at the existing openAirBearing project (Plotly Dash) as a reference point for what we want to improve upon.

## Options considered

**Plotly Dash** is what openAirBearing uses. The app is split into `app.py` (Dash instance + HTML template), `layouts.py` (widget tree with inline styles), and `callbacks.py` (four callbacks wired by string IDs). It works, but the developer experience is mediocre: every input and output is connected by hand-managed string IDs, inline style dicts are mixed into layout code, and there's no natural way to represent staged workflows. Every parameter change fires through a single monolithic callback that reruns the entire solver. The framework is actively maintained but hasn't changed its core model in years.

**Streamlit** is the most popular Python UI framework for scientific apps. Its script-reruns-on-every-interaction model is elegant for simple tools but fights you once you need independent stages. Modifying a mesh parameter shouldn't re-trigger geometry definition, but Streamlit's default behaviour does exactly that. `st.session_state` and `st.fragment` can work around this, but you're patching over the execution model rather than working with it. Great for prototyping, less great for our staged, multi-concern workflow.

**Panel** (HoloViz/Anaconda) uses a persistent-state, event-driven model built on `param.Parameterized` classes. Each stage is a class whose parameters automatically generate widgets; `@param.depends(...)` decorators declare exactly which outputs react to which inputs. It has a built-in `Pipeline` widget for multi-stage workflows with forward/back navigation and state passing. It works identically in Jupyter notebooks and as a standalone server. The community is smaller than Streamlit's, but it's actively maintained and is the standard choice in scientific Python when you need more than a flat parameter panel.

## Decision: Panel

Panel's execution model matches our requirements directly:

- **Staged workflow** maps onto `param.Parameterized` classes composed into `pn.Tabs` or `pn.pipeline.Pipeline`. Each stage owns its parameters and its preview/output. No session-state hacks.
- **Selective reactivity** — `@param.depends("groove_depth", "groove_count")` means only the geometry preview re-renders when groove parameters change; the mesh stage is untouched. This is fundamental to how Panel works, not an opt-in escape hatch.
- **Heavy solver calls** can be triggered by an explicit button (`pn.widgets.Button`) rather than firing on every slider move. Panel doesn't force auto-execution.
- **Notebook + server duality** — during development we can iterate in Jupyter; the same code serves as a standalone app with `panel serve app.py`.
- **Plotting flexibility** — Panel wraps Plotly, Matplotlib, and Bokeh interchangeably via `pn.pane.Plotly`, `pn.pane.Matplotlib`, etc.

## What we'd do differently from openAirBearing

The existing Dash app is a useful reference for *what* to display (bearing shape, pressure distribution, load capacity, stiffness, flow rates) and for the "thin UI over library" pattern (the web layer just converts units and calls `solve_bearing()`). We keep that pattern. The structural changes:

### 1. Replace string-ID callback wiring with `param` declarations

openAirBearing's Dash app manually connects ~20 input IDs to a single callback with ~11 outputs. In Panel, parameters are declared on the class and the framework generates widgets and dependency tracking automatically:

```python
class BearingGeometry(param.Parameterized):
    bearing_type = param.Selector(default="circular",
        objects=["circular", "annular", "rectangular", "grooved"])
    outer_radius = param.Number(default=18.5, bounds=(1, 100), step=0.5, label="Outer radius (mm)")
    porous_thickness = param.Number(default=4.5, bounds=(0.1, 20), step=0.1, label="Porous layer thickness (mm)")
    # ... more params

    @param.depends("bearing_type", "outer_radius", "porous_thickness")
    def shape_preview(self):
        """Returns a live-updating figure of the bearing cross-section."""
        b = self._to_bearing()  # convert to library's CircularBearing etc.
        return pn.pane.Plotly(plot_xz_shape(b))
```

No string IDs to keep in sync. Adding a parameter means adding one line to the class.

### 2. Split the monolithic callback into stages

The Dash app's `update_bearing` callback takes every parameter and returns every figure. In Panel, each concern is a separate class with its own reactive outputs:

```python
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
        # update output panes
```

Changing `nx` doesn't rebuild the geometry preview. Changing `bearing_type` doesn't rerun the solver until you explicitly ask.

### 3. Move styles out of Python

The Dash app has a 60-line `STYLES` dict and per-element inline style dicts scattered through `layouts.py`. Panel components accept CSS classes via the `css_classes` parameter, and Panel supports external stylesheets via `pn.config.raw_css` or a `static/` directory. Keep Python code about structure, keep CSS about appearance.

### 4. Use `pn.Tabs` for the staged layout

```python
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

`pn.Param(geo)` auto-generates a widget panel from the `param.Parameterized` class — sliders for numbers, dropdowns for selectors, etc. — with no manual widget construction.

### 5. Keep the library layer untouched

The bearing dataclasses, `solve_bearing()`, and the Plotly figure functions (`plot_load_capacity`, `plot_stiffness`, etc.) are called from the Panel stages exactly as the Dash app calls them. The UI is a shell; the library is the product.

## Summary

| Aspect | Dash (current) | Panel (planned) |
| --- | --- | --- |
| Reactivity | All-or-nothing via callbacks | Per-parameter via `@param.depends` |
| Stage management | None (single page, all inputs fire together) | `pn.Tabs` / `Pipeline`, each stage independent |
| Widget generation | Manual `dcc.Input` + string IDs | Auto from `param.Parameterized` |
| Styling | Inline Python dicts | CSS classes + external stylesheet |
| Notebook support | No | Same code runs in Jupyter |
| Heavy computation | Blocks on every change | Explicit trigger via button |
