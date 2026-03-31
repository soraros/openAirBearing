from dash import Input, Output
import dash


from openairbearing.plots import plot_bearing_shape, plot_key_results, empty_figure
from openairbearing.bearings import (
    CircularBearing,
    AnnularBearing,
    InfiniteLinearBearing,
    RectangularBearing,
    JournalBearing,
)
from openairbearing.solvers import solve_bearing
from openairbearing.utils import get_kappa, get_Qsc, get_beta


def get_bearing(case):
    """Return default bearing instance based on case."""
    match case:
        case "circular":
            return CircularBearing
        case "annular":
            return AnnularBearing
        case "infinite":
            return InfiniteLinearBearing
        case "rectangular":
            return RectangularBearing
        case "journal":
            return JournalBearing
        case _:
            raise TypeError("no default bearing defined")


def register_callbacks(app):
    """Register all callbacks for the application."""

    # Add callback to handle parameter updates
    @app.callback(
        [
            *[Output(f"shape-plot-{i}", "figure") for i in range(3)],
            *[Output(f"result-plot-{i}", "figure") for i in range(6)],
            Output("kappa-input", "value", allow_duplicate=True),
            Output("Qsc-input", "value", allow_duplicate=True),
        ],
        [
            Input("case-select", "value"),
            Input("solver-select", "value"),
            Input("pa-input", "value"),
            Input("ps-input", "value"),
            Input("pc-input", "value"),
            Input("rho-input", "value"),
            Input("mu-input", "value"),
            Input("hp-input", "value"),
            Input("xa-input", "value"),
            Input("xc-input", "value"),
            Input("nx-input", "value"),
            Input("ya-input", "value"),
            Input("ny-input", "value"),
            Input("ha-min-input", "value"),
            Input("ha-max-input", "value"),
            Input("nh-input", "value"),
            Input("kappa-input", "value"),
            Input("Qsc-input", "value"),
            Input("error-select", "value"),
            Input("error-input", "value"),
            Input("psi-input", "value"),
        ],
        prevent_initial_call="initial_duplicate",
    )
    def update_bearing(
        case,
        solvers,
        pa_mpa,
        ps_mpa,
        pc_mpa,
        rho,
        mu,
        hp_mm,
        xa_mm,
        xc_mm,
        nx,
        ya_mm,
        ny,
        ha_min_um,
        ha_max_um,
        nh,
        kappa,
        Qsc,
        error_type,
        error_um,
        Psi,
    ):
        """Update bearing parameters and recalculate results.

        Args:
            case: Selected bearing case
            solvers: List of selected solvers
            pa_mpa: Ambient pressure in MPa
            ...

        Returns:
            tuple: Updated figures and values
        """
        ctx = dash.callback_context
        if not ctx.triggered:
            raise dash.exceptions.PreventUpdate

        input_id = ctx.triggered[0]["prop_id"].split(".")[0]

        try:
            # Get default bearing for selected case
            bearing_class = get_bearing(case)

            # Create new bearing instance with appropriate class
            b = bearing_class(
                pa=pa_mpa * 1e6,
                ps=ps_mpa * 1e6,
                pc=pc_mpa * 1e6,
                rho=rho,
                mu=mu,
                hp=hp_mm * 1e-3,
                xa=xa_mm * 1e-3,
                xc=xc_mm * 1e-3,
                ya=ya_mm * 1e-3,
                nx=int(nx),
                ny=int(ny),
                ha_min=ha_min_um * 1e-6,
                ha_max=ha_max_um * 1e-6,
                nh=int(nh),
                error_type=error_type,
                error=error_um * 1e-6,
                Psi=Psi,
            )

            # Update kappa or Qsc based on which input changed
            if input_id == "kappa-input" and kappa is not None:
                b.kappa = kappa
                b.Qsc = get_Qsc(b)
                new_kappa = kappa
                new_Qsc = b.Qsc
            elif input_id == "Qsc-input" and Qsc is not None:
                b.Qsc = Qsc
                b.kappa = get_kappa(b)
                new_kappa = b.kappa
                new_Qsc = Qsc
            else:
                # For other inputs, maintain current values
                b.kappa = kappa if kappa is not None else get_kappa(b)
                b.Qsc = Qsc if Qsc is not None else get_Qsc(b)
                new_kappa = b.kappa
                new_Qsc = b.Qsc

            b.beta = get_beta(b)

            # Calculate results for each selected solver
            results = []

            if "analytic" in solvers:
                results.append(solve_bearing(b, soltype="analytic"))
            if "numeric" in solvers:
                results.append(solve_bearing(b, soltype="numeric"))
            if "numeric2d" in solvers:
                results.append(solve_bearing(b, soltype="numeric2d"))

            shape_figures = plot_bearing_shape(b)
            plot_figures = plot_key_results(b, results)

            return (
                *shape_figures + [empty_figure()] * (3 - len(shape_figures)),
                *plot_figures + [empty_figure()] * (6 - len(plot_figures)),
                new_kappa,
                new_Qsc,
            )

        except Exception as e:
            print(f"Error: {e}")
            return (
                *[empty_figure() for _ in range(3)],
                *[empty_figure() for _ in range(6)],
                dash.no_update,
                dash.no_update,
            )

    @app.callback(
        Output("pc-container", "style"),
        Output("xc-container", "style"),
        Output("ya-container", "style"),
        Output("ny-container", "style"),
        Input("case-select", "value"),
    )
    def toggle_containers(case):
        """Show/hide input containers based on selected case."""
        base_style = {
            "grid-template-columns": "200px 100px 20px",
            "gap": "20px",
            "marginTop": "20px",
            "marginBottom": "20px",
            "align-items": "center",
        }
        match case:
            case "annular":
                return (
                    {**base_style, "display": "grid"},
                    {**base_style, "display": "grid"},
                    {**base_style, "display": "none"},
                    {**base_style, "display": "none"},
                )
            case "infinite":
                return (
                    {**base_style, "display": "grid"},
                    {**base_style, "display": "none"},
                    {**base_style, "display": "none"},
                    {**base_style, "display": "none"},
                )
            case "rectangular":
                return (
                    {**base_style, "display": "grid"},
                    {**base_style, "display": "none"},
                    {**base_style, "display": "grid"},
                    {**base_style, "display": "grid"},
                )
            case "journal":
                return (
                    {**base_style, "display": "grid"},
                    {**base_style, "display": "none"},
                    {**base_style, "display": "grid"},
                    {**base_style, "display": "grid"},
                )
            case _:
                return (
                    {**base_style, "display": "none"},
                    {**base_style, "display": "none"},
                    {**base_style, "display": "none"},
                    {**base_style, "display": "none"},
                )

    @app.callback(
        [Output("solver-select", "options"), Output("solver-select", "value")],
        Input("case-select", "value"),
    )
    def update_solver_options(case):
        """Update available solvers based on the selected case."""

        case_solvers = {
            "circular": [
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
                    "disabled": True,
                },
            ],
            "annular": [
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
                    "disabled": True,
                },
            ],
            "infinite": [
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
                    "disabled": True,
                },
            ],
            "rectangular": [
                {
                    "label": "Analytic",
                    "value": "analytic",
                    "disabled": True,
                },
                {
                    "label": "Numeric",
                    "value": "numeric",
                    "disabled": True,
                },
                {
                    "label": "Numeric 2d",
                    "value": "numeric2d",
                    "disabled": False,
                },
            ],
            "journal": [
                {
                    "label": "Analytic",
                    "value": "analytic",
                    "disabled": True,
                },
                {
                    "label": "Numeric",
                    "value": "numeric",
                    "disabled": True,
                },
                {
                    "label": "Numeric 2d",
                    "value": "numeric2d",
                    "disabled": False,
                },
            ],
        }
        # Default solver selections for each case
        case_defaults = {
            "circular": ["analytic"],
            "annular": ["analytic"],
            "infinite": ["analytic"],
            "rectangular": ["numeric2d"],
            "journal": ["numeric2d"],
        }
        # Get the solvers for the selected case
        solvers = case_solvers.get(case, [])
        defaults = case_defaults.get(case, [])

        return solvers, defaults

    @app.callback(
        [
            Output("rho-input", "value"),
            Output("mu-input", "value"),
            Output("hp-input", "value"),
            Output("xa-input", "value"),
            Output("xc-input", "value"),
            Output("ya-input", "value"),
            Output("kappa-input", "value", allow_duplicate=True),
            Output("Qsc-input", "value", allow_duplicate=True),
            Output("pa-input", "value"),
            Output("pc-input", "value"),
            Output("ps-input", "value"),
            Output("ha-min-input", "value"),
            Output("ha-max-input", "value"),
            Output("nx-input", "value"),
            Output("ny-input", "value"),
            Output("nh-input", "value"),
            Output("error-input", "value"),
            Output("psi-input", "value"),
        ],
        [
            Input("reset-all", "n_clicks"),
            Input("rho-reset", "n_clicks"),
            Input("mu-reset", "n_clicks"),
            Input("hp-reset", "n_clicks"),
            Input("xa-reset", "n_clicks"),
            Input("xc-reset", "n_clicks"),
            Input("ya-reset", "n_clicks"),
            Input("kappa-reset", "n_clicks"),
            Input("Qsc-reset", "n_clicks"),
            Input("pa-reset", "n_clicks"),
            Input("pc-reset", "n_clicks"),
            Input("ps-reset", "n_clicks"),
            Input("ha-min-reset", "n_clicks"),
            Input("ha-max-reset", "n_clicks"),
            Input("nx-reset", "n_clicks"),
            Input("ny-reset", "n_clicks"),
            Input("nh-reset", "n_clicks"),
            Input("error-reset", "n_clicks"),
            Input("psi-reset", "n_clicks"),
            Input("case-select", "value"),
        ],
        prevent_initial_call=True,
    )
    def reset_values(reset_all, *args):
        ctx = dash.callback_context
        if not ctx.triggered:
            raise dash.exceptions.PreventUpdate

        button_id = ctx.triggered[0]["prop_id"].split(".")[0]

        # Fetch defaults based on current case selection
        case = args[-1]
        b = get_bearing(case)()

        # Current values dictionary with proper unit conversions
        current_values = {
            "rho": b.rho,
            "mu": b.mu,
            "hp": b.hp * 1e3,  # Convert to mm
            "xa": b.xa * 1e3,  # Convert to mm
            "xc": b.xc * 1e3,  # Convert to mm
            "ya": b.ya * 1e3,  # Convert to mm
            "kappa": b.kappa,
            "Qsc": b.Qsc,
            "pa": b.pa * 1e-6,  # Convert to MPa
            "pc": b.pc * 1e-6,  # Convert to MPa
            "ps": b.ps * 1e-6,  # Convert to MPa
            "ha_min": b.ha_min * 1e6,  # Convert to μm
            "ha_max": b.ha_max * 1e6,  # Convert to μm
            "nx": b.nx,
            "ny": b.ny,
            "nh": b.nh,
            "error": b.error * 1e6,
            "psi": b.Psi,
        }

        # Reset all values when Reset All clicked OR when case changes
        if button_id in ["reset-all", "case-select"]:
            return list(current_values.values())

        # For individual reset buttons
        param = button_id.replace("-reset", "")
        return [
            current_values[p] if p == param else dash.no_update
            for p in [
                "rho",
                "mu",
                "hp",
                "xa",
                "xc",
                "ya",
                "kappa",
                "Qsc",
                "pa",
                "pc",
                "ps",
                "ha_min",
                "ha_max",
                "nx",
                "ny",
                "nh",
                "error",
                "psi",
            ]
        ]
