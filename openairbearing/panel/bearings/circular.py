from openairbearing.bearings import CircularBearing

from ._base import BearingTypeConfig

CONFIG = BearingTypeConfig(
  key="circular",
  cls=CircularBearing,
  solvers=["analytic", "numeric"],
  default_solvers=["analytic"],
  visibility={"pc": False, "xc": False, "ya": False, "ny": False},
)
