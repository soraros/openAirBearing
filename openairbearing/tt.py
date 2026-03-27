from abc import ABC, ABCMeta, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Final, Protocol, overload


class P(metaclass=ABCMeta):
    @abstractmethod
    def f(self): ...


class Q(ABC):
    @abstractmethod
    def f(self): ...


class C(P):
    # ...
    def f(self) -> int: ...
