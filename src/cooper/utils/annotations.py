
from collections.abc import Sequence
from enum import Enum, auto
from typing import TypeVar, Union

T = TypeVar("T")
OneOrSequence = Union[T, Sequence[T]]


class ConstraintType(Enum):

    EQUALITY = auto()
    INEQUALITY = auto()
