
from __future__ import annotations

import abc
from typing import TYPE_CHECKING

import torch

from cooper.penalty_coefficients.penalty_coefficients import DensePenaltyCoefficient, IndexedPenaltyCoefficient
from cooper.utils import ConstraintType

if TYPE_CHECKING:
    from cooper.constraints import Constraint, ConstraintState


class PenaltyCoefficientUpdater(abc.ABC):

    def step(self, observed_constraints: dict[Constraint, ConstraintState]) -> None:
        pass

    @abc.abstractmethod
    def update_penalty_coefficient_(self, constraint: Constraint, constraint_state: ConstraintState) -> None:
        """Update the penalty coefficient of a constraint.

        Args:
            constraint: The constraint for which the penalty coefficient is updated.
            constraint_state: The constraint state of the constraint.
        """


class FeasibilityDrivenPenaltyCoefficientUpdater(PenaltyCoefficientUpdater, abc.ABC):
    def __init__(self, violation_tolerance: float, has_restart: bool) -> None:
        if violation_tolerance < 0.0:
            raise ValueError("Violation tolerance must be non-negative.")
        self.violation_tolerance = violation_tolerance
        self.has_restart = has_restart

    def update_penalty_coefficient_(self, constraint: Constraint, constraint_state: ConstraintState) -> None:
        pass

    @abc.abstractmethod
    def _compute_updated_penalties(
        self, current_penalty_value: torch.Tensor, should_increase_penalty: torch.Tensor
    ) -> torch.Tensor:
        """Compute updated penalty values based on violation status."""


class MultiplicativePenaltyCoefficientUpdater(FeasibilityDrivenPenaltyCoefficientUpdater):

    def __init__(
        self, growth_factor: float = 1.01, violation_tolerance: float = 1e-4, has_restart: bool = True
    ) -> None:
        super().__init__(violation_tolerance, has_restart)
        self.growth_factor = growth_factor

    def _compute_updated_penalties(
        self, current_penalty_value: torch.Tensor, should_increase_penalty: torch.Tensor
    ) -> torch.Tensor:
        pass


class AdditivePenaltyCoefficientUpdater(FeasibilityDrivenPenaltyCoefficientUpdater):

    def __init__(self, increment: float = 1.0, violation_tolerance: float = 1e-4, has_restart: bool = True) -> None:
        super().__init__(violation_tolerance, has_restart)
        self.increment = increment

    def _compute_updated_penalties(
        self, current_penalty_value: torch.Tensor, should_increase_penalty: torch.Tensor
    ) -> torch.Tensor:
        pass
