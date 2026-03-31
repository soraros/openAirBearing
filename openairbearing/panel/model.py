from typing import cast

import panel as pn
import param

from openairbearing.config import DEMO_MODE
from openairbearing.panel.bearings import (
  BEARING_CLASSES,
  SOLVER_AVAILABILITY,
  SOLVER_DEFAULTS,
)
from openairbearing.plots import plot_bearing_shape, plot_key_results
from openairbearing.solvers import solve_bearing
from openairbearing.utils import get_beta, get_kappa, get_Qsc

NX_MAX = 100 if DEMO_MODE else 500
NY_MAX = 100 if DEMO_MODE else 500
NH_MAX = 100 if DEMO_MODE else 100


class BearingApp(param.Parameterized):
  """All bearing parameters + solve workflow in one class."""

  # --- Configuration ---
  bearing_type: str = cast(
    str,
    param.Selector(
      default="circular",
      objects=["circular", "annular", "infinite", "rectangular"],
      label="Simulated case",
    ),
  )
  solver: list[str] = cast(
    list[str],
    param.ListSelector(
      default=["analytic"],
      objects=["analytic", "numeric"],
      label="Solution selection",
    ),
  )

  # --- Geometry (display units: mm) ---
  hp: float = cast(
    float,
    param.Number(
      default=4.5, bounds=(0.01, None), step=0.1, label="Porous layer thickness (mm)"
    ),
  )
  xa: float = cast(
    float,
    param.Number(
      default=18.5, bounds=(0.01, None), step=0.5, label="Outer radius / length (mm)"
    ),
  )
  xc: float = cast(
    float,
    param.Number(
      default=0.001, bounds=(0.0, None), step=0.5, label="Inner radius (mm)"
    ),
  )
  ya: float = cast(
    float, param.Number(default=0.0, bounds=(0.0, None), step=0.5, label="Length (mm)")
  )

  # --- Permeability ---
  kappa: float | None = cast(
    float | None,
    param.Number(default=None, label="Permeability (m²)"),  # type: ignore[arg-type]
  )
  Qsc: float = cast(
    float,
    param.Number(default=2.8, bounds=(0.01, None), step=0.1, label="Free flow (L/min)"),
  )

  # --- Fluid (display units: MPa for pressures) ---
  pa: float = cast(
    float,
    param.Number(
      default=0.101325, bounds=(1e-6, None), step=0.01, label="Ambient pressure (MPa)"
    ),
  )
  ps: float = cast(
    float,
    param.Number(
      default=0.701325, bounds=(0.01, None), step=0.1, label="Supply pressure (MPa)"
    ),
  )
  pc: float = cast(
    float,
    param.Number(
      default=0.101325, bounds=(0.0, None), step=0.01, label="Chamber pressure (MPa)"
    ),
  )
  rho: float = cast(
    float, param.Number(default=1.293, bounds=(0.01, None), label="Air density (kg/m³)")
  )
  mu: float = cast(
    float,
    param.Number(
      default=1.85e-5, bounds=(1e-8, None), label="Dynamic viscosity (Pa·s)"
    ),
  )

  # --- Mesh ---
  nx: int = cast(
    int, param.Integer(default=30, bounds=(3, NX_MAX), label="Radial / x points")
  )
  ny: int = cast(
    int,
    param.Integer(default=1, bounds=(1, NY_MAX), label="Circumferential / y points"),
  )
  nh: int = cast(
    int, param.Integer(default=20, bounds=(3, NH_MAX), label="Height points")
  )
  ha_min: float = cast(
    float,
    param.Number(default=1.0, bounds=(0.01, None), step=0.5, label="Min air gap (μm)"),
  )
  ha_max: float = cast(
    float,
    param.Number(default=20.0, bounds=(0.1, None), step=0.5, label="Max air gap (μm)"),
  )

  # --- Error model ---
  error_type: str = cast(
    str,
    param.Selector(
      default="linear",
      objects=["none", "linear", "quadratic"],
      label="Geometrical error type",
    ),
  )
  error: float = cast(
    float, param.Number(default=0.0, step=0.5, label="Geometry error (μm)")
  )
  Psi: float = cast(
    float,
    param.Number(
      default=0.0, bounds=(0.0, None), step=0.01, label="Slip coefficient Φ"
    ),
  )

  # --- Internal trigger (bumped after solve) ---
  _result_version: int = cast(int, param.Integer(default=0))

  def __init__(self, **kwargs):
    self._results: list = []
    self._result_bearing = None
    self._syncing: bool = False
    super().__init__(**kwargs)
    # Set defaults from the default bearing
    self._apply_bearing_defaults("circular")
    # Watch kappa / Qsc for bidirectional sync
    self.param.watch(self._on_kappa_changed, ["kappa"])
    self.param.watch(self._on_qsc_changed, ["Qsc"])
    # Initial solve so app isn't blank
    self._do_solve()

  # ------------------------------------------------------------------
  # Bearing construction
  # ------------------------------------------------------------------

  def _to_bearing(self):
    """Convert display-unit params → SI and instantiate the bearing dataclass."""
    cls = BEARING_CLASSES[self.bearing_type]
    return cls(
      pa=self.pa * 1e6,
      ps=self.ps * 1e6,
      pc=self.pc * 1e6,
      rho=self.rho,
      mu=self.mu,
      hp=self.hp * 1e-3,
      xa=self.xa * 1e-3,
      xc=self.xc * 1e-3,
      ya=self.ya * 1e-3,
      nx=self.nx,
      ny=self.ny,
      ha_min=self.ha_min * 1e-6,
      ha_max=self.ha_max * 1e-6,
      nh=self.nh,
      error_type=self.error_type,
      error=self.error * 1e-6,
      Psi=self.Psi,
    )

  # ------------------------------------------------------------------
  # Bearing-type change → reset defaults, update solver options
  # ------------------------------------------------------------------

  @param.depends("bearing_type", watch=True)
  def _on_bearing_type_changed(self):
    self._apply_bearing_defaults(self.bearing_type)

  def _apply_bearing_defaults(self, case: str):
    """Reset parameter values to the per-type defaults."""
    cls = BEARING_CLASSES[case]
    b = cls()

    with param.parameterized.batch_call_watchers(self):
      self.hp = b.hp * 1e3
      self.xa = b.xa * 1e3
      self.xc = b.xc * 1e3
      self.ya = b.ya * 1e3
      self.pa = b.pa * 1e-6
      self.ps = b.ps * 1e-6
      self.pc = b.pc * 1e-6
      self.rho = b.rho
      self.mu = b.mu
      self.nx = b.nx
      self.ny = b.ny
      self.nh = b.nh
      self.ha_min = b.ha_min * 1e6
      self.ha_max = b.ha_max * 1e6
      self.error_type = "linear"
      self.error = b.error * 1e6
      self.Psi = b.Psi
      self.Qsc = b.Qsc

      # kappa is derived during __post_init__, so read it
      self._syncing = True
      self.kappa = b.kappa
      self._syncing = False

      # Update solver list for this bearing type
      available = SOLVER_AVAILABILITY.get(case, ["analytic"])
      self.param.solver.objects = available
      self.solver = SOLVER_DEFAULTS.get(case, available[:1])

  # ------------------------------------------------------------------
  # Bidirectional kappa ↔ Qsc
  # ------------------------------------------------------------------

  def _on_kappa_changed(self, event):
    if self._syncing or event.new is None:
      return
    self._syncing = True
    try:
      b = self._to_bearing()
      b.kappa = event.new
      self.Qsc = get_Qsc(b)
    finally:
      self._syncing = False

  def _on_qsc_changed(self, event):
    if self._syncing or event.new is None:
      return
    self._syncing = True
    try:
      b = self._to_bearing()
      b.Qsc = event.new
      self.kappa = get_kappa(b)
    finally:
      self._syncing = False

  # ------------------------------------------------------------------
  # Shape preview (live-reactive)
  # ------------------------------------------------------------------

  @param.depends(
    "bearing_type",
    "xa",
    "xc",
    "ya",
    "hp",
    "ha_min",
    "ha_max",
    "kappa",
    "error_type",
    "error",
  )
  def shape_preview(self):
    try:
      b = self._to_bearing()
      if self.kappa is not None:
        b.kappa = self.kappa
      b.beta = get_beta(b)
      figs = plot_bearing_shape(b)
      plots = [
        pn.pane.Plotly(f, sizing_mode="stretch_width", min_height=350) for f in figs
      ]
      return pn.Row(*plots, sizing_mode="stretch_width")
    except Exception as exc:
      return pn.pane.Alert(f"Shape preview error: {exc}", alert_type="warning")

  # ------------------------------------------------------------------
  # Solve
  # ------------------------------------------------------------------

  def _do_solve(self):
    """Run selected solvers and store results."""
    try:
      b = self._to_bearing()
      if self.kappa is not None:
        b.kappa = self.kappa
      b.Qsc = self.Qsc
      b.beta = get_beta(b)

      results = []
      for s in self.solver:
        results.append(solve_bearing(b, soltype=s))

      self._results = results
      self._result_bearing = b
      self._result_version += 1
    except Exception as exc:
      self._results = []
      self._result_bearing = None
      self._result_version += 1
      print(f"Solve error: {exc}")

  def solve(self, event=None):
    """Button callback."""
    self._do_solve()

  # ------------------------------------------------------------------
  # Result plots (react to _result_version)
  # ------------------------------------------------------------------

  @param.depends("_result_version")
  def result_plots(self):
    if not self._results or self._result_bearing is None:
      return pn.pane.Markdown("*Click **Solve** to compute results.*")
    try:
      figs = plot_key_results(self._result_bearing, self._results)
      rows = []
      for i in range(0, len(figs), 3):
        row_figs = figs[i : i + 3]
        row = pn.Row(
          *[
            pn.pane.Plotly(f, sizing_mode="stretch_width", min_height=350)
            for f in row_figs
          ],
          sizing_mode="stretch_width",
        )
        rows.append(row)
      return pn.Column(*rows, sizing_mode="stretch_width")
    except Exception as exc:
      return pn.pane.Alert(f"Plot error: {exc}", alert_type="danger")
