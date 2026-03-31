from dash import dcc, html

from openairbearing.config import DEMO_MODE
from openairbearing.plots import empty_figure, plot_bearing_shape, plot_key_results

STYLES = {
  "input": {
    "width": "100px",
    "borderRadius": "4px",
    "border": "1px solid #ddd",
    "padding": "4px",
  },
  "input_container": {
    "display": "grid",
    "grid-template-columns": "200px 100px 20px",
    "marginBottom": "20px",
    "gap": "20px",
    "align-items": "center",
  },
  "toggle_container": {
    "display": "none",
    "grid-template-columns": "200px 100px 20px",
    "marginTop": "20px",
    "gap": "20px",
    "align-items": "center",
  },
  "reset_button": {
    "padding": "2px 6px",
    "fontSize": "14px",
    "backgroundColor": "#f8f9fa",
    "border": "1px solid #ddd",
    "borderRadius": "4px",
    "cursor": "pointer",
    "height": "25px",
    "width": "30px",
  },
  "header_container": {
    "display": "flex",
    "justifyContent": "space-between",
    "alignItems": "center",
    "marginBottom": "20px",
  },
  "reset_all_button": {
    "padding": "5px 10px",
    "fontSize": "14px",
    "backgroundColor": "#f8f9fa",
    "border": "1px solid #ddd",
    "borderRadius": "4px",
    "cursor": "pointer",
    "height": "30px",
  },
  "input_column": {
    "width": "370px",
    "minWidth": "370px",  # Prevent inputs from getting too narrow
    "display": "inline-block",
    "vertical-align": "top",
    "padding": "20px",
    "border": "1px solid black",
    "borderRadius": "8px",
    "flex": "0 1 auto",  # Don't grow, allow shrink, auto basis
  },
  "plot_box": {
    "width": "calc(100% - 40px)",  # Dynamic width based on container
    "display": "inline-block",
    "vertical-align": "top",
    "justifyContent": "space-between",
    "padding": "20px",
    "border": "1px solid black",
    "borderRadius": "8px",
    "flex": "1 1 auto",  # Allow grow and shrink
  },
  "plot_column": {
    "width": "calc(100% - 550px)",  # Dynamic width based on container
    "minWidth": "600px",  # Minimum width for plots
    "display": "inline-block",
    "vertical-align": "top",
    "padding": "0px",
    "flex": "1 1 auto",  # Allow grow and shrink
  },
  "container": {
    "display": "flex",
    "alignItems": "flex-start",
    "justifyContent": "space-between",
    "width": "100%",
    "flexWrap": "wrap",  # Allow wrapping on smaller screens
    "gap": "20px",  # Space between columns
  },
}


def create_layout(default_bearing, bearing, results):
  """Create the main app layout.

  Args:
      bearing: Bearing instance
      results: List of calculation results

  Returns:
      html.Div: Main application layout
  """
  return html.Div([
    html.Div(
      [
        html.Img(
          src="/assets/favicon.ico",
          style={
            "height": "40px",
            "margin": "10px 5px",
            "verticalAlign": "middle",
          },
        ),
        html.H1(
          "Open Air Bearing",
          style={
            "textAlign": "center",
            "display": "inline-block",
            "margin": "10px 0",
          },
        ),
      ],
      style={
        "display": "flex",
        "alignItems": "center",
        "justifyContent": "center",
      },
    ),
    html.Div(
      [
        create_input_layout(default_bearing),
        create_results_layout(bearing, results),
      ],
      style=STYLES["container"],
    ),
  ])


def create_input_layout(default_bearing):
  return html.Div(
    [
      html.Div(
        [
          html.H3("Setup", style={"margin": "0"}),
          html.Button(
            "Reset All",
            id="reset-all",
            title="Reset all values to default",
            style=STYLES["reset_all_button"],
          ),
        ],
        style=STYLES["header_container"],
      ),
      # html.H4("Setup"),
      html.Div(
        [
          html.Label("Simulated Case"),
          dcc.Dropdown(
            id="case-select",
            options=[
              {"label": "Circular thrust", "value": "circular"},
              {"label": "Annular thrust", "value": "annular"},
              {"label": "Infinitely long", "value": "infinite"},
              {"label": "Rectangular", "value": "rectangular"},
              {"label": "Journal", "value": "journal", "disabled": True},
            ],
            value="circular",
            style={"width": "150px"},
          ),
          html.Label(""),
          html.Label("Solution selection"),
          dcc.Checklist(
            id="solver-select",
            options=[
              {
                "label": "Analytic",
                "value": "analytic",
                "disabled": False,
              },
              {
                "label": "Numeric",
                "value": "numeric",
                "disabled": False,
              },
              {
                "label": "Numeric 2d",
                "value": "numeric2d",
                "disabled": False,
              },
            ],
            value=["analytic"],
            style={"color": "black", "width": "150px"},
          ),
        ],
        style=STYLES["input_container"],
      ),
      # Geometry inputs
      html.H4("Bearing parameters"),
      html.Div(
        [
          html.Label("Porous Layer Thickness (mm)"),
          dcc.Input(
            id="hp-input",
            type="number",
            min=0.01,
            value=default_bearing.hp * 1e3,  # Convert m to mm
            style=STYLES["input"],
          ),
          html.Button(
            "↺",
            id="hp-reset",
            title="Reset to default",
            style=STYLES["reset_button"],
          ),
          html.Label("Outer radius / length (mm)"),
          dcc.Input(
            id="xa-input",
            type="number",
            min=0.01,
            value=default_bearing.xa * 1e3,  # Convert m to mm
            style=STYLES["input"],
          ),
          html.Button(
            "↺",
            id="xa-reset",
            title="Reset to default",
            style=STYLES["reset_button"],
          ),
        ],
        style=STYLES["input_container"],
      ),
      html.Div(
        [
          html.Label("Inner radius (mm)"),
          dcc.Input(
            id="xc-input",
            type="number",
            min=0.01,
            value=default_bearing.xc * 1e3,  # Convert m to mm
            style=STYLES["input"],
          ),
          html.Button(
            "↺",
            id="xc-reset",
            title="Reset to default",
            style=STYLES["reset_button"],
          ),
        ],
        id="xc-container",
        style=STYLES["toggle_container"],
      ),
      html.Div(
        [
          html.Label("Length (mm)"),
          dcc.Input(
            id="ya-input",
            type="number",
            min=0.01,
            value=default_bearing.ya * 1e3,  # Convert m to mm
            style=STYLES["input"],
          ),
          html.Button(
            "↺",
            id="ya-reset",
            title="Reset to default",
            style=STYLES["reset_button"],
          ),
        ],
        id="ya-container",
        style=STYLES["toggle_container"],
      ),
      html.Div(
        [
          html.Label("Permeability (m^2)"),
          dcc.Input(
            id="kappa-input",
            type="number",
            value=default_bearing.kappa,
            min=0,
            step=1e-16,
            inputMode="numeric",
            style=STYLES["input"],
          ),
          html.Button(
            "↺",
            id="kappa-reset",
            title="Reset to default",
            style=STYLES["reset_button"],
          ),
          html.Label("Free flow (l/min)"),
          dcc.Input(
            id="Qsc-input",
            type="number",
            value=default_bearing.Qsc,
            min=0.1,
            step=0.1,
            inputMode="numeric",
            style=STYLES["input"],
          ),
          html.Button(
            "↺",
            id="Qsc-reset",
            title="Reset to default",
            style=STYLES["reset_button"],
          ),
        ],
        style=STYLES["input_container"],
      ),
      # Geometry inputs
      html.H4("Numerical model specific:"),
      html.Div(
        [
          html.Label("Geometrical error type"),
          dcc.Dropdown(
            id="error-select",
            options=[
              {
                "label": "Linear",
                "value": "linear",
              },
              {
                "label": "Quadratic",
                "value": "quadratic",
              },
              {
                "label": "Tilt x",
                "value": "tiltx",
                "disabled": True,
              },
              {
                "label": "Tilt y",
                "value": "tilty",
                "disabled": True,
              },
            ],
            value="linear",
            style={"width": "150px"},
          ),
          html.Label(""),
          html.Label("Geometry error (μm)"),
          dcc.Input(
            id="error-input",
            type="number",
            step=0.5,
            value=default_bearing.error * 1e6,
            style=STYLES["input"],
          ),
          html.Button(
            "↺",
            id="error-reset",
            title="Reset to default",
            style=STYLES["reset_button"],
          ),
          html.Label("Slip coefficeint Φ"),
          dcc.Input(
            id="psi-input",
            type="number",
            min=0,
            step=0.01,
            value=default_bearing.Psi * 1e6,
            style=STYLES["input"],
          ),
          html.Button(
            "↺",
            id="psi-reset",
            title="Reset to default",
            style=STYLES["reset_button"],
          ),
        ],
        style=STYLES["input_container"],
      ),
      html.H4("Load parameters"),
      html.Div(
        [
          html.Label("Ambient Pressure (MPa)"),
          dcc.Input(
            id="pa-input",
            type="number",
            value=default_bearing.pa * 1e-6,  # Convert Pa to MPa
            min=1e-6,
            step=0.1,
            inputMode="numeric",
            style=STYLES["input"],
          ),
          html.Button(
            "↺",
            id="pa-reset",
            title="Reset to default",
            style=STYLES["reset_button"],
          ),
          html.Label("Supply Pressure (MPa)"),
          dcc.Input(
            id="ps-input",
            type="number",
            value=default_bearing.ps * 1e-6,  # Convert Pa to MPa
            min=0.1,
            step=0.1,
            inputMode="numeric",
            style=STYLES["input"],
          ),
          html.Button(
            "↺",
            id="ps-reset",
            title="Reset to default",
            style=STYLES["reset_button"],
          ),
        ],
        style=STYLES["input_container"],
      ),
      html.Div(
        [
          html.Label("Chamber Pressure (MPa)"),
          dcc.Input(
            id="pc-input",
            type="number",
            value=default_bearing.pc * 1e-6,  # Convert Pa to MPa
            min=0,
            step=0.01,
            inputMode="numeric",
            style=STYLES["input"],
          ),
          html.Button(
            "↺",
            id="pc-reset",
            title="Reset to default",
            style=STYLES["reset_button"],
          ),
        ],
        id="pc-container",
        style=STYLES["toggle_container"],
      ),
      html.H4("Fluid properties"),
      html.Div(
        [
          html.Label("Air Density (kg/m³)"),
          dcc.Input(
            id="rho-input",
            type="number",
            value=default_bearing.rho,
            style=STYLES["input"],
          ),
          html.Button(
            "↺",
            id="rho-reset",
            title="Reset to default",
            style=STYLES["reset_button"],
          ),
          html.Label("Dynamic Viscosity (Pa·s)"),
          dcc.Input(
            id="mu-input",
            type="number",
            value=default_bearing.mu,
            style=STYLES["input"],
          ),
          html.Button(
            "↺",
            id="mu-reset",
            title="Reset to default",
            style=STYLES["reset_button"],
          ),
        ],
        style=STYLES["input_container"],
      ),
      html.H4("Model parameters"),
      html.Div(
        [
          html.Label("Minimum Height (μm)"),
          dcc.Input(
            id="ha-min-input",
            type="number",
            value=default_bearing.ha_min * 1e6,  # Convert m to μm
            min=0,
            step=0.5,
            style=STYLES["input"],
          ),
          html.Button(
            "↺",
            id="ha-min-reset",
            title="Reset to default",
            style=STYLES["reset_button"],
          ),
          html.Label("Maximum Height (μm)"),
          dcc.Input(
            id="ha-max-input",
            type="number",
            value=default_bearing.ha_max * 1e6,  # Convert m to μm
            style=STYLES["input"],
          ),
          html.Button(
            "↺",
            id="ha-max-reset",
            title="Reset to default",
            style=STYLES["reset_button"],
          ),
          html.Label("Number of height points"),
          dcc.Input(
            id="nh-input",
            type="number",
            min=3,
            max=100 if DEMO_MODE else None,
            step=1,
            value=default_bearing.nh,
            style=STYLES["input"],
          ),
          html.Button(
            "↺",
            id="nh-reset",
            title="Reset to default",
            style=STYLES["reset_button"],
          ),
          html.Label("Number of x direction points"),
          dcc.Input(
            id="nx-input",
            type="number",
            value=default_bearing.nx,
            min=3,
            max=100 if DEMO_MODE else None,
            step=1,
            style=STYLES["input"],
          ),
          html.Button(
            "↺",
            id="nx-reset",
            title="Reset to default",
            style=STYLES["reset_button"],
          ),
        ],
        style=STYLES["input_container"],
      ),
      html.Div(
        [
          html.Label("Number of y direction points"),
          dcc.Input(
            id="ny-input",
            type="number",
            value=default_bearing.ny,
            min=3,
            max=100 if DEMO_MODE else None,
            step=1,
            style=STYLES["input"],
          ),
          html.Button(
            "↺",
            id="ny-reset",
            title="Reset to default",
            style=STYLES["reset_button"],
          ),
        ],
        id="ny-container",
        style=STYLES["toggle_container"],
      ),
    ],
    style=STYLES["input_column"],
  )


def create_results_layout(bearing, results):
  """Create the results section layout.

  Args:
      bearing: Bearing instance
      results: List of calculation results

  Returns:
      html.Div: Results layout with plots arranged in a grid.
  """
  # Get the list of plots from plot_key_results
  shape_figures = plot_bearing_shape(bearing)
  plot_figures = plot_key_results(bearing, results)

  # Pad the list to ensure it has a multiple of 3 elements
  while len(shape_figures) % 3 != 0:
    shape_figures.append(empty_figure())
  while len(plot_figures) % 3 != 0:
    plot_figures.append(empty_figure())
  # Create rows of plots (3 plots per row)
  shape_rows = []
  for i in range(0, len(shape_figures), 3):
    row = html.Div(
      [
        html.Div(
          dcc.Graph(
            id=f"shape-plot-{j}",
            figure=shape_figures[j],
            config={"displayModeBar": False},
            style={"height": "400px"},
          ),
          style={
            "width": "33%",
            "margin": "0px",
            "padding": "0px",
          },
        )
        for j in range(i, min(i + 3, len(shape_figures)))  # Add up to 3 plots per row
      ],
      style={
        "display": "flex",
        "justifyContent": "space-between",
      },  # Flexbox for row layout
    )
    shape_rows.append(row)
  result_rows = []
  for i in range(0, len(plot_figures), 3):
    row = html.Div(
      [
        html.Div(
          dcc.Graph(
            id=f"result-plot-{j}",
            figure=plot_figures[j],
            config={"displayModeBar": False},
            style={"height": "400px"},
          ),
          style={
            "width": "calc(33% - 20px)",
            "margin": "10px",
            "padding": "0px",
          },
        )
        for j in range(i, min(i + 3, len(plot_figures)))  # Add up to 3 plots per row
      ],
      style={
        "display": "flex",
        "justifyContent": "space-between",
      },  # Flexbox for row layout
    )
    result_rows.append(row)

  # Combine all rows into a single column
  return html.Div(
    [
      html.Div(
        [
          html.H3(
            "Bearing shape",
            style={"margin": "0 0 20px 0", "textAlign": "left"},
          ),
          *shape_rows,
        ],
        style=STYLES["plot_box"],
      ),
      html.Div(style={"height": "20px"}),
      html.Div(
        [
          html.H3("Results", style={"margin": "0 0 20px 0", "textAlign": "left"}),
          *result_rows,
        ],
        style=STYLES["plot_box"],
      ),
    ],
    style=STYLES["plot_column"],  # Style for the overall results section
  )
