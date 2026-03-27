from abc import abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import ClassVar, Final, Protocol

import numpy as np
from numpy.typing import NDArray

from .utils import get_area, get_beta, get_geom, get_kappa

type F64 = NDArray[np.float64]


@dataclass
class BaseBearing:
    """Base class for all bearing types."""

    pa: Final[float] = 101325.0  # atmospheric pressure
    pc: float = pa
    ps: float = 0.6e6 + pa

    rho: float = 1.293
    mu: float = 1.85e-5

    hp: float = 4.5e-3

    ha_min: float = 1e-6
    ha_max: float = 20e-6

    xa: float = 37 / 2 * 1e-3
    xc: float = 0
    ya: float = 0
    nh: int = 20
    nx: int = 30  # number of points in x direction
    ny: int = 1  # number of points in y direction, it's 1 for 1D bearings

    Psi: float = 0

    error_type: str = "none"
    error: float = 0e-6

    blocked: bool = False
    block_x: float = 25.2e-3 / 2
    block_w: float = 1e-3

    Qsc: float = 3  # L/min  this is the standard flow rate
    psc: float = 0.6e6 + pa  # standard supply pressure

    # clearance and eccentricity for journal bearings
    c: float = field(init=False)  # clearance
    e: F64 = field(init=False)  # eccentricity array
    theta: F64 | None = field(init=False, default=None)
    clearance: F64 | None = field(init=False, default=None)

    x: F64 = field(init=False)
    dx: F64 = field(init=False)
    y: F64 = field(init=False)
    dy: F64 = field(init=False)

    ha: F64 = field(init=False)
    A: float = field(init=False)
    kappa: float = field(init=False)
    beta: float = field(init=False)
    geom: F64 = field(init=False)

    case: str = "base"
    type: str = "bearing"
    csys: str = "cartesian"

    def __post_init__(self):
        self.ha = np.linspace(self.ha_min, self.ha_max, self.nh)
        self.x = np.linspace(self.xc, self.xa, self.nx)
        self.y = np.linspace(0, self.ya, self.ny)
        self.dx = np.gradient(self.x)
        self.dy = np.array(1) if self.ny == 1 else np.gradient(self.y)
        self.A = get_area(self)
        self.geom = get_geom(self)
        self.kappa = get_kappa(self)
        self.beta = get_beta(self)


class BearingType(Enum):
    BEARING = "bearing"
    SEAL = "seal"


class CoordinateSystem(Enum):
    CARTESIAN = "cartesian"
    POLAR = "polar"


class BearingT(Protocol):
    type: ClassVar[BearingType]
    coord_system: ClassVar[CoordinateSystem]

    @property
    @abstractmethod
    def area(self) -> float: ...


type Bearing = (
    CircularBearing
    | AnnularBearing
    | InfiniteLinearBearing
    | RectangularBearing
    | JournalBearing
)

type Bearing_ = Circular | Annular


def get_area_(bearing: Bearing) -> float:
    b = bearing
    match bearing:
        case CircularBearing():
            return 0

    return None

    match b.case:
        case "circular":
            A = np.pi * b.xa**2
        case "annular":
            A = np.pi * (b.xa**2 - b.xc**2)
        case "infinite":
            A = b.xa
        case "rectangular":
            A = b.xa * b.ya
        case "journal":
            A = 2 * np.pi * b.xa * b.ya
        case _:
            raise ValueError(f"Unknown case: {b.case}")

    return A


@dataclass
class Circular(BearingT):
    xa: float
    type = BearingType.BEARING
    coord_system = CoordinateSystem.POLAR

    @property
    def area(self) -> float:
        return np.pi * self.xa**2

    @property
    def geometry(self) -> float:
        return self.area

    def beta(self) -> float:
        return 0.0


@dataclass
class Annular:
    xa: float
    xc: float

    @property
    def area(self) -> float:
        return np.pi * (self.xa**2 - self.xc**2)


@dataclass
class InfiniteLinear:
    xa: float

    @property
    def area(self) -> float:
        return self.xa


@dataclass
class Rectangular:
    xa: float
    ya: float

    @property
    def area(self) -> float:
        return self.xa * self.ya


@dataclass
class Journal:
    xa: float
    ya: float

    @property
    def area(self) -> float:
        return 2 * np.pi * self.xa * self.ya


@dataclass
class CircularBearing(BaseBearing):
    """Base class for circular thrust bearing"""

    case: str = "circular"
    type: str = "bearing"
    csys: str = "polar"  # csys is short for coordinate system
    # ny takes the default value of 1 from BaseBearing, meaning 1D formulation

    xc: float = 1e-6
    xa: float = 37e-3 / 2
    Qsc: float = 2.8  # L/min

    def __post_init__(self):
        super().__post_init__()
        self.psc = 0.6e6 + self.pa


@dataclass
class AnnularBearing(BaseBearing):
    """Base class for annular bearing"""

    case: str = "annular"
    type: str = "seal"
    csys: str = "polar"
    # ny takes the default value of 1 from BaseBearing, meaning 1D formulation

    xa: float = 58e-3 / 2
    xc: float = 25e-3 / 2

    Qsc: float = 3  # L/min

    def __post_init__(self):
        super().__post_init__()
        self.psc = 0.6e6 + self.pa


@dataclass
class InfiniteLinearBearing(BaseBearing):
    """Base class for Infinitely long linear bearing bearing"""

    case: str = "infinite"
    type: str = "seal"
    csys: str = "cartesian"

    ps: float = 0.41e6
    xa: float = 40e-3

    Qsc: float = 37  # L/min

    def __post_init__(self):
        super().__post_init__()
        self.psc = 0.41e6 + self.pa


@dataclass
class RectangularBearing(BaseBearing):
    """Base class for rectangular thrust bearing"""

    case: str = "rectangular"
    type: str = "bearing"
    csys: str = "cartesian"

    xa: float = 80e-3
    ya: float = 40e-3
    nx: int = 40
    ny: int = 20

    ps: float = 0.41e6

    Qsc: float = 2.94  # L/min

    def __post_init__(self):
        super().__post_init__()
        self.psc = 0.41e6 + self.pa
        self.x = np.linspace(-self.xa / 2, self.xa / 2, self.nx)
        self.y = np.linspace(-self.ya / 2, self.ya / 2, self.ny)
        self.dx = self.xa / (self.nx + 1)
        self.dy = self.ya / (self.ny + 1)
        self.geom = get_geom(self)  # calculate after x y


@dataclass
class JournalBearing(BaseBearing):
    """Base class for journal bearing"""

    case: str = "journal"
    type: str = "seal"
    csys: str = "cartesian"

    xa: float = 50.02e-3 / 2  # radius, the minimum radius of the bearing
    ya: float = 89e-3  # length
    nx: int = 80  # circumferential points
    ny: int = 50  # axial points
    hp: float = 3e-3

    c = 40e-6  # clearance (journal radius - shaft radius)

    ps: float = 0.41e6

    Qsc: float = 15  # L/min

    def __post_init__(self):
        super().__post_init__()
        self.psc = 0.41e6 + self.pa
        self.theta = np.linspace(-np.pi, np.pi, self.nx)
        self.x = self.theta
        self.y = np.linspace(-self.ya / 2, self.ya / 2, self.ny)
        self.dx = 2 * np.pi / (self.nx + 1)
        self.dy = self.ya / (self.ny)

        self.ha_min = 0.01e-6
        self.ha_max = self.c / 2 - 1e-6
        e_min = self.ha_min
        e_max = self.ha_max
        self.e = np.linspace(e_min, e_max, self.nh)
        self.clearance = self.xa - np.sqrt(
            (self.xa - self.c / 2) ** 2
            + self.e[None, None, :] ** 2
            + 2
            * self.e[None, None, :]
            * (self.xa - self.c / 2)
            * np.cos(self.theta[:, None, None])
        )
        # import matplotlib.pyplot as plt
        # plt.plot(self.theta, np.squeeze(self.clearance) * 1e6)
        # plt.plot(self.theta, np.cos(self.theta))
        # plt.show()
        # print(self.dx*self.nx)
        self.geom = get_geom(self)  # calculate after x y
