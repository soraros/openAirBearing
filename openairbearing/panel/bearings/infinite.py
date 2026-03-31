from openairbearing.bearings import InfiniteLinearBearing

from ._base import BearingTypeConfig

CONFIG = BearingTypeConfig(
  key="infinite",
  cls=InfiniteLinearBearing,
  solvers=["analytic", "numeric"],
  default_solvers=["analytic"],
  visibility={"pc": True, "xc": False, "ya": False, "ny": False},
)
