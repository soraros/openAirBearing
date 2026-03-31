from openairbearing.bearings import RectangularBearing

from ._base import BearingTypeConfig

CONFIG = BearingTypeConfig(
  key="rectangular",
  cls=RectangularBearing,
  solvers=["numeric2d"],
  default_solvers=["numeric2d"],
  visibility={"pc": True, "xc": False, "ya": True, "ny": True},
)
