"""Run every v2 example's headless figure builder."""

import importlib.util
from pathlib import Path

import plotly.graph_objects as go
import pytest

EXAMPLES_DIR = Path(__file__).resolve().parents[2] / "examples" / "v2"


def _load_example(path: Path):
  spec = importlib.util.spec_from_file_location(f"example_v2_{path.stem}", path)
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


@pytest.mark.parametrize(
  "path",
  sorted(EXAMPLES_DIR.glob("*.py")),
  ids=lambda p: p.stem,
)
def test_example_build_figures(path):
  """Each example builds its figures without rendering."""
  module = _load_example(path)
  figs = module.build_figures()
  assert len(figs) > 0
  for fig in figs:
    assert isinstance(fig, go.Figure)
    assert len(fig.data) > 0
