
import math
from typing import Optional, Union

import torch

from cooper.multipliers import Multiplier
from cooper.penalty_coefficients import PenaltyCoefficient
from cooper.utils import ConstraintType


def evaluate_constraint_factor(
    module: Union[Multiplier, PenaltyCoefficient],
    constraint_features: Optional[torch.Tensor],
    expand_shape: tuple[int, ...],
) -> torch.Tensor:
    pass


def compute_primal_weighted_violation(
    constraint_factor_value: torch.Tensor, violation: torch.Tensor
) -> Optional[torch.Tensor]:
    pass


def compute_dual_weighted_violation(multiplier_value: torch.Tensor, violation: torch.Tensor) -> torch.Tensor:
    pass


def compute_quadratic_penalty(
    penalty_coefficient_value: torch.Tensor, violation: torch.Tensor, constraint_type: ConstraintType
) -> Optional[torch.Tensor]:
    pass


def compute_primal_quadratic_augmented_contribution(
    multiplier_value: torch.Tensor,
    penalty_coefficient_value: torch.Tensor,
    violation: torch.Tensor,
    constraint_type: ConstraintType,
) -> Optional[torch.Tensor]:
    pass
