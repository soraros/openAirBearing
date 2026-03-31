from openairbearing.bearings import AnnularBearing

from ._base import BearingTypeConfig

CONFIG = BearingTypeConfig(
  key="annular",
  cls=AnnularBearing,
  solvers=["analytic", "numeric"],
  default_solvers=["analytic"],
  visibility={"pc": True, "xc": True, "ya": False, "ny": False},
)
