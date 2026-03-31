from openairbearing.bearings import JournalBearing

from ._base import BearingTypeConfig

CONFIG = BearingTypeConfig(
  key="journal",
  cls=JournalBearing,
  solvers=["numeric2d"],
  default_solvers=["numeric2d"],
  visibility={"pc": True, "xc": False, "ya": True, "ny": True},
)
