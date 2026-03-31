import pytest

from openairbearing.app.callbacks import get_bearing
from openairbearing.bearings import (
  AnnularBearing,
  CircularBearing,
  InfiniteLinearBearing,
  RectangularBearing,
)


def test_get_bearing():
  assert get_bearing("circular") == CircularBearing
  assert get_bearing("annular") == AnnularBearing
  assert get_bearing("infinite") == InfiniteLinearBearing
  assert get_bearing("rectangular") == RectangularBearing
  with pytest.raises(TypeError, match="no default bearing defined"):
    get_bearing("unknown")


# psutil multiprocess selenium "dash[testing]"
# def test_update_bearing(dash_duo):
#     app = import_app("openairbearing.app.app")  # Replace with the actual path to your Dash app
#     dash_duo.start_server(app)

#     # Simulate user selecting a bearing case
#     dash_duo.select_dcc_dropdown("#case-select", "circular")

#     # Simulate user input for parameters
#     dash_duo.find_element("#pa-input").send_keys("0.1")  # Ambient pressure in MPa
#     dash_duo.find_element("#ps-input").send_keys("0.6")  # Supply pressure in MPa

#     # Wait for the callback to update the outputs
#     dash_duo.wait_for_text_to_equal("#kappa-input", "calculated_value", timeout=5)
#     dash_duo.wait_for_text_to_equal("#Qsc-input", "calculated_value", timeout=5)

#     # Verify the output figures
#     assert dash_duo.find_element("#bearing-plots").is_displayed()
#     assert dash_duo.find_element("#bearing-shape").is_displayed()
