from dataclasses import dataclass, field


@dataclass(frozen=True)
class BearingTypeConfig:
  """Per-bearing-type metadata for the Panel UI."""

  key: str
  cls: type
  solvers: list[str]
  default_solvers: list[str]
  visibility: dict[str, bool] = field(default_factory=lambda: {
    "pc": False, "xc": False, "ya": False, "ny": False,
  })
