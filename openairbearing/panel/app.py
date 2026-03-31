import panel as pn

from openairbearing.panel.bearings import VISIBILITY
from openairbearing.panel.model import BearingApp

pn.extension("plotly", sizing_mode="stretch_width")

RAW_CSS = """
.bk-root .card-title {
  font-size: 1.1em;
  font-weight: 600;
}
body {
  max-width: 1400px;
  margin: 0 auto;
  padding: 0 20px;
  font-family: Arial, Helvetica, sans-serif;
}
"""


def _conditional_row(app: BearingApp, field: str, widget) -> pn.Row:
  """Wrap a widget in a Row whose visibility tracks bearing_type."""
  row = pn.Row(widget, sizing_mode="stretch_width")

  def _update_vis(*events):
    case = app.bearing_type
    row.visible = VISIBILITY.get(case, {}).get(field, False)

  app.param.watch(_update_vis, ["bearing_type"])
  _update_vis()
  return row


def create_app() -> pn.Column:
  app = BearingApp(name="Open Air Bearing")

  # --- Card 1: Configuration ---
  w_case = pn.widgets.Select.from_param(app.param.bearing_type, width=200)
  w_solver = pn.widgets.CheckBoxGroup.from_param(app.param.solver)

  config_card = pn.Card(
    pn.Row(
      pn.Column("**Case**", w_case),
      pn.Column("**Solvers**", w_solver),
    ),
    title="Configuration",
    collapsible=True,
  )

  # --- Card 2: Bearing Parameters + Shape Preview ---
  # Geometry
  w_hp = pn.widgets.FloatInput.from_param(app.param.hp, width=120)
  w_xa = pn.widgets.FloatInput.from_param(app.param.xa, width=120)
  w_xc = pn.widgets.FloatInput.from_param(app.param.xc, width=120)
  w_ya = pn.widgets.FloatInput.from_param(app.param.ya, width=120)

  xc_row = _conditional_row(app, "xc", w_xc)
  ya_row = _conditional_row(app, "ya", w_ya)

  geom_col = pn.Column(
    "### Geometry",
    w_hp,
    w_xa,
    xc_row,
    ya_row,
    sizing_mode="stretch_width",
  )

  # Permeability
  w_kappa = pn.widgets.FloatInput.from_param(
    app.param.kappa,
    step=1e-16,
    format="%.4e",
    width=150,
  )
  w_qsc = pn.widgets.FloatInput.from_param(app.param.Qsc, width=120)

  perm_col = pn.Column(
    "### Permeability",
    w_kappa,
    w_qsc,
    sizing_mode="stretch_width",
  )

  # Fluid
  w_pa = pn.widgets.FloatInput.from_param(app.param.pa, width=120)
  w_ps = pn.widgets.FloatInput.from_param(app.param.ps, width=120)
  w_pc = pn.widgets.FloatInput.from_param(app.param.pc, width=120)
  w_rho = pn.widgets.FloatInput.from_param(app.param.rho, width=120)
  w_mu = pn.widgets.FloatInput.from_param(
    app.param.mu,
    step=1e-7,
    format="%.2e",
    width=150,
  )

  pc_row = _conditional_row(app, "pc", w_pc)

  fluid_col = pn.Column(
    "### Fluid & Pressure",
    w_pa,
    w_ps,
    pc_row,
    w_rho,
    w_mu,
    sizing_mode="stretch_width",
  )

  # Left: params stacked
  params_col = pn.Column(
    geom_col,
    perm_col,
    fluid_col,
    sizing_mode="stretch_width",
    max_width=380,
  )

  # Right: live shape preview
  shape_pane = pn.panel(app.shape_preview, sizing_mode="stretch_width")

  params_card = pn.Card(
    pn.Row(params_col, shape_pane, sizing_mode="stretch_width"),
    title="Bearing Parameters",
    collapsible=True,
  )

  # --- Card 3: Mesh & Solver ---
  w_ha_min = pn.widgets.FloatInput.from_param(app.param.ha_min, width=120)
  w_ha_max = pn.widgets.FloatInput.from_param(app.param.ha_max, width=120)
  w_nh = pn.widgets.IntInput.from_param(app.param.nh, width=100)
  w_nx = pn.widgets.IntInput.from_param(app.param.nx, width=100)
  w_ny = pn.widgets.IntInput.from_param(app.param.ny, width=100)

  ny_row = _conditional_row(app, "ny", w_ny)

  w_error_type = pn.widgets.Select.from_param(app.param.error_type, width=150)
  w_error = pn.widgets.FloatInput.from_param(app.param.error, width=120)
  w_psi = pn.widgets.FloatInput.from_param(app.param.Psi, width=120)

  solve_btn = pn.widgets.Button(
    name="Solve",
    button_type="primary",
    width=200,
    height=40,
  )
  solve_btn.on_click(app.solve)

  mesh_card = pn.Card(
    pn.Row(
      pn.Column(
        "### Air Gap Sweep",
        w_ha_min,
        w_ha_max,
        w_nh,
        "### Grid",
        w_nx,
        ny_row,
        sizing_mode="stretch_width",
      ),
      pn.Column(
        "### Error Model",
        w_error_type,
        w_error,
        w_psi,
        sizing_mode="stretch_width",
      ),
      sizing_mode="stretch_width",
    ),
    pn.Row(solve_btn, align="center"),
    title="Mesh & Solver",
    collapsible=True,
  )

  # --- Card 4: Results ---
  results_pane = pn.panel(app.result_plots, sizing_mode="stretch_width")
  results_card = pn.Card(results_pane, title="Results", collapsible=True)

  # --- Assemble ---
  header = pn.pane.Markdown("# absim", styles={"text-align": "center"})

  return pn.Column(
    header,
    config_card,
    params_card,
    mesh_card,
    results_card,
    sizing_mode="stretch_width",
  )


def main():
  pn.config.raw_css.append(RAW_CSS)  # type: ignore[union-attr]
  app = create_app()
  app.servable()
  pn.serve({"/": app}, port=5007, show=False, title="absim")


if __name__ == "__main__":
  main()
