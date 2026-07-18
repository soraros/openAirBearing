"""Plotly figures for solutions and pad shapes.

All functions are pure: solutions carry their compiled problem, so every
figure is built from explicit data. Each plot accepts one solution or a
sequence (overlaid curves, colored per solution method).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import plotly.graph_objects as go

from openairbearing.v2.geometry.pads import (
  AnnularPad,
  CircularPad,
  JournalPad,
  LinearPad,
  RectangularPad,
)
from openairbearing.v2.primitives.types import SampleScalar
from openairbearing.v2.problem import BearingProblem, BearingSpec, build_problem
from openairbearing.v2.solvers.solution import Solution

__all__ = [
  "AXIS_STYLE",
  "FIG_LAYOUT",
  "PLOT_FONT",
  "SOLVER_COLORS",
  "plot_ambient_flow_rate",
  "plot_chamber_flow_rate",
  "plot_key_results",
  "plot_load_capacity",
  "plot_pad_xy",
  "plot_pad_xz",
  "plot_pressure_distribution",
  "plot_stiffness",
  "plot_supply_flow_rate",
]

PLOT_FONT = {"family": "Arial", "size": 12}

SOLVER_COLORS = {
  "analytic": "blue",
  "numeric": "red",
  "numeric2d": "red",
}

AXIS_STYLE = {
  "title_font": PLOT_FONT,
  "tickfont": PLOT_FONT,
  "showline": True,
  "showgrid": False,
  "linecolor": "black",
  "ticks": "inside",
  "mirror": True,
}
FIG_LAYOUT = {
  "font": PLOT_FONT,
  "plot_bgcolor": "white",
  "paper_bgcolor": "white",
  "legend": {"orientation": "h", "yanchor": "bottom", "y": 1.1, "xanchor": "center", "x": 0.5},
  "showlegend": True,
  "margin": {"l": 10, "r": 10, "t": 50, "b": 10},
}


# ── Helpers ───────────────────────────────────────────────────────────────────


def _as_list(solutions: Solution | Sequence[Solution]) -> list[Solution]:
  if isinstance(solutions, Solution):
    return [solutions]
  out = list(solutions)
  if not out:
    raise ValueError("at least one solution is required")
  return out


def _color(solution: Solution) -> str:
  return SOLVER_COLORS.get(solution.method, "purple")


def _sample_label(problem: BearingProblem) -> str:
  return "e (μm)" if problem.journal else "h (μm)"


def _x_label(problem: BearingProblem) -> str:
  if problem.journal:
    return "θ (rad)"
  return "r (mm)" if problem.polar else "x (mm)"


def _samples_um(solution: Solution) -> SampleScalar:
  return solution.samples * 1e6


def _marker_sizes(solution: Solution) -> list[int]:
  return [8 if i == solution.peak_index else 0 for i in range(solution.problem.n_samples)]


def _curve(solution: Solution, x: SampleScalar, y: SampleScalar) -> go.Scatter:
  color = _color(solution)
  return go.Scatter(
    x=x,
    y=y,
    name=solution.method,
    mode="lines+markers",
    marker={"color": color, "size": _marker_sizes(solution), "symbol": "circle"},
    line={"color": color},
  )


# ── Performance curves ────────────────────────────────────────────────────────


def plot_load_capacity(solutions: Solution | Sequence[Solution]) -> go.Figure:
  """Load capacity over the sample sweep."""
  fig = go.Figure()
  for sol in _as_list(solutions):
    fig.add_trace(_curve(sol, _samples_um(sol), sol.load))
  problem = _as_list(solutions)[0].problem
  fig.update_xaxes(
    title_text=_sample_label(problem), range=[0, problem.samples[-1] * 1e6], **AXIS_STYLE
  )
  fig.update_yaxes(title_text="w (N)", **AXIS_STYLE)
  fig.update_layout(title="Load Capacity", **FIG_LAYOUT)
  return fig


def plot_stiffness(solutions: Solution | Sequence[Solution]) -> go.Figure:
  """Static stiffness over the sample sweep."""
  fig = go.Figure()
  for sol in _as_list(solutions):
    fig.add_trace(_curve(sol, _samples_um(sol), sol.stiffness * 1e-6))
  problem = _as_list(solutions)[0].problem
  fig.update_xaxes(
    title_text=_sample_label(problem), range=[0, problem.samples[-1] * 1e6], **AXIS_STYLE
  )
  fig.update_yaxes(title_text="k (N/μm)", **AXIS_STYLE)
  fig.update_layout(title="Static stiffness", **FIG_LAYOUT)
  return fig


def plot_supply_flow_rate(solutions: Solution | Sequence[Solution]) -> go.Figure:
  """Supply flow rate over the sample sweep."""
  return _flow_figure(solutions, "q_supply", "q<sub>s</sub> (L/min)", "Supply flow")


def plot_ambient_flow_rate(solutions: Solution | Sequence[Solution]) -> go.Figure:
  """Ambient-edge flow rate over the sample sweep."""
  return _flow_figure(solutions, "q_ambient", "q<sub>a</sub> (L/min)", "Ambient flow")


def plot_chamber_flow_rate(solutions: Solution | Sequence[Solution]) -> go.Figure:
  """Chamber-edge flow rate over the sample sweep."""
  return _flow_figure(solutions, "q_chamber", "q<sub>c</sub> (L/min)", "Chamber flow")


def _flow_figure(
  solutions: Solution | Sequence[Solution], attr: str, y_label: str, title: str
) -> go.Figure:
  fig = go.Figure()
  for sol in _as_list(solutions):
    fig.add_trace(_curve(sol, _samples_um(sol), getattr(sol, attr)))
  problem = _as_list(solutions)[0].problem
  fig.update_xaxes(
    title_text=_sample_label(problem), range=[0, problem.samples[-1] * 1e6], **AXIS_STYLE
  )
  fig.update_yaxes(title_text=y_label, **AXIS_STYLE)
  fig.update_layout(title=title, **FIG_LAYOUT)
  return fig


def plot_key_results(solutions: Solution | Sequence[Solution]) -> tuple[go.Figure, ...]:
  """The standard figure set: load, stiffness, pressure, and flows."""
  sols = _as_list(solutions)
  figs = [
    plot_load_capacity(sols),
    plot_stiffness(sols),
    plot_pressure_distribution(sols),
    plot_supply_flow_rate(sols),
    plot_ambient_flow_rate(sols),
  ]
  if sols[0].problem.pad.SEAL:
    figs.append(plot_chamber_flow_rate(sols))
  return tuple(figs)


# ── Pressure distribution ─────────────────────────────────────────────────────


def plot_pressure_distribution(
  solutions: Solution | Sequence[Solution], *, slider: bool = True
) -> go.Figure:
  """Pressure distribution: curves at key gaps (1-D) or a contour stack (2-D)."""
  sols = _as_list(solutions)
  fig = go.Figure()
  for sol in sols:
    if sol.p.ndim == 2:
      _pressure_curves_1d(fig, sol)
    else:
      _pressure_contour_2d(fig, sol, slider=slider)
  fig.update_layout(title="Pressure distribution", **FIG_LAYOUT)
  return fig


def _pressure_curves_1d(fig: go.Figure, sol: Solution) -> None:
  problem = sol.problem
  color = _color(sol)
  pa = problem.state.p_ambient
  x = problem.x * 1e3
  if problem.journal:
    x = problem.x
  fig.add_trace(
    go.Scatter(
      x=[None],
      y=[None],
      mode="lines",
      line={"color": color},
      name=sol.method,
      showlegend=True,
    )
  )
  indices = [0, sol.peak_index, problem.n_samples - 1]
  label_at = np.round(np.linspace(problem.x.size, 0, 5)[1:-1]).astype(int)
  for k, t_loc in zip(indices, label_at, strict=True):
    sample = problem.samples[k]
    pressures = (sol.p[k] - pa) * 1e-6
    fig.add_trace(
      go.Scatter(
        x=x,
        y=pressures,
        mode="lines+text",
        textposition="top center",
        text=[f"{sample * 1e6:.2f} μm" if i == t_loc else None for i in range(x.size)],
        textfont={"color": color},
        name=f"{sol.method} {sample * 1e6:.1f} μm",
        line={"color": color},
        showlegend=False,
      )
    )
  fig.update_xaxes(title_text=_x_label(problem), **AXIS_STYLE)
  fig.update_yaxes(title_text="p (MPa)", **AXIS_STYLE)


def _pressure_contour_2d(fig: go.Figure, sol: Solution, *, slider: bool) -> None:
  problem = sol.problem
  pa = problem.state.p_ambient
  k0 = sol.peak_index
  pressures = (sol.p - pa) * 1e-6
  x = problem.x if problem.journal else problem.x * 1e3
  y = problem.y * 1e3
  fig.add_trace(
    go.Contour(
      z=pressures[k0].T,
      x=x,
      y=y,
      colorscale="Viridis",
      zmin=0,
      zmax=float(np.max(pressures)),
      contours={
        "coloring": "heatmap",
        "showlabels": True,
        "labelfont": {"size": 10, "color": "white"},
      },
      colorbar={"title": "p (MPa)", "thickness": 15},
      name=sol.method,
    )
  )
  fig.update_xaxes(title_text=_x_label(problem), **AXIS_STYLE)
  fig.update_yaxes(title_text="y (mm)", **AXIS_STYLE)
  if slider:
    steps = [
      {
        "method": "update",
        "args": [{"z": [pressures[k].T]}, {"title": f"Pressure distribution (h = {s * 1e6:.1f})"}],
        "label": f"{s * 1e6:.1f} μm",
      }
      for k, s in enumerate(problem.samples)
    ]
    fig.update_layout(
      sliders=[
        {
          "active": k0,
          "currentvalue": {"prefix": "Film height: ", "font": {"size": 14}},
          "pad": {"t": 50},
          "steps": steps,
        }
      ]
    )


# ── Pad shape ─────────────────────────────────────────────────────────────────


def plot_pad_xy(spec: BearingSpec | BearingProblem) -> go.Figure:
  """Pad outline in the XY plane."""
  problem = build_problem(spec) if isinstance(spec, BearingSpec) else spec
  pad = problem.pad
  fig = go.Figure()
  theta = np.linspace(0.0, 2.0 * np.pi, 100)
  if isinstance(pad, (CircularPad, AnnularPad)):
    fig.add_trace(
      go.Scatter(
        x=pad.r * np.cos(theta) * 1e3,
        y=pad.r * np.sin(theta) * 1e3,
        fill="toself",
        fillcolor="lightgrey",
        line={"color": "black"},
        name="Shape",
        showlegend=True,
      )
    )
    if isinstance(pad, AnnularPad):
      fig.add_trace(
        go.Scatter(
          x=pad.r_inner * np.cos(theta) * 1e3,
          y=pad.r_inner * np.sin(theta) * 1e3,
          fill="toself",
          fillcolor="white",
          line={"color": "black"},
          name="Shape",
          showlegend=False,
        )
      )
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
  elif isinstance(pad, RectangularPad):
    fig.add_trace(
      go.Scatter(
        x=np.array([-1, -1, 1, 1, -1]) * pad.lx * 0.5e3,
        y=np.array([-1, 1, 1, -1, -1]) * pad.ly * 0.5e3,
        fill="toself",
        fillcolor="lightgrey",
        mode="lines",
        line={"color": "black"},
        name="Shape",
        showlegend=False,
      )
    )
    fig.update_xaxes(range=np.array([-0.6, 0.6]) * pad.lx * 1e3, scaleanchor="y", scaleratio=1)
    fig.update_yaxes(range=np.array([-0.6, 0.6]) * pad.ly * 1e3)
  elif isinstance(pad, LinearPad):
    fig.add_trace(
      go.Scatter(
        x=np.array([1, 1]) * pad.length * 1e3,
        y=np.array([0, 1000]),
        mode="lines",
        line={"color": "black"},
        name="Shape",
        showlegend=False,
      )
    )
    fig.add_trace(
      go.Scatter(
        x=np.array([0, 0]),
        y=np.array([0, 1000]),
        fill="tonextx",
        fillcolor="lightgrey",
        mode="lines",
        line={"color": "black"},
        name="Shape",
        showlegend=False,
      )
    )
    fig.update_xaxes(range=np.array([-0.5, 1.5]) * pad.length * 1e3)
    fig.update_yaxes(range=[0, 1000])
  elif isinstance(pad, JournalPad):
    fig.add_trace(
      go.Scatter(
        x=np.array([-1, -1, 1, 1, -1]) * np.pi,
        y=np.array([-1, 1, 1, -1, -1]) * pad.length * 0.5e3,
        fill="toself",
        fillcolor="lightgrey",
        mode="lines",
        line={"color": "black"},
        name="Shape",
        showlegend=False,
      )
    )
    fig.update_xaxes(title_text="θ (rad)")
  fig.update_xaxes(
    title_text="x (mm)" if not isinstance(pad, JournalPad) else "θ (rad)", **AXIS_STYLE
  )
  fig.update_yaxes(title_text="y (mm)", **AXIS_STYLE)
  fig.update_layout(title="XY profile", **FIG_LAYOUT)
  return fig


def plot_pad_xz(spec: BearingSpec | BearingProblem) -> go.Figure:
  """Manufacturing-error profile: section curve (1-D) or contour (2-D)."""
  problem = build_problem(spec) if isinstance(spec, BearingSpec) else spec
  geom = problem.geom
  fig = go.Figure()
  if geom.ndim == 1:
    x = np.concatenate(([-problem.x[-1]], -np.flip(problem.x), problem.x, [problem.x[-1]])) * 1e3
    y = np.concatenate(([100.0], np.flip(geom), geom, [100.0])) * 1e6
    fig.add_trace(
      go.Scatter(
        x=x,
        y=y,
        fill="toself",
        fillcolor="lightgrey",
        mode="lines",
        line={"color": "black"},
        name="Bearing",
        showlegend=True,
      )
    )
    fig.update_yaxes(range=[-0.5, 1.0 + float(np.max(geom)) * 1e6], title_text="Shape (μm)")
  else:
    x = problem.x if problem.journal else problem.x * 1e3
    fig.add_trace(
      go.Contour(
        z=geom.T * 1e6,
        x=x,
        y=problem.y * 1e3,
        colorscale="Viridis",
        zmin=0,
        zmax=float(np.max(geom)) * 1e6 if np.max(geom) > 0 else 1.0,
        contours={
          "coloring": "heatmap",
          "showlabels": True,
          "labelfont": {"size": 10, "color": "white"},
        },
        colorbar={"title": "(μm)", "thickness": 15},
        name="Profile",
      )
    )
    fig.update_yaxes(title_text="y (mm)")
  fig.update_xaxes(title_text=_x_label(problem), **AXIS_STYLE)
  fig.update_layout(title="XZ profile", **FIG_LAYOUT)
  return fig
