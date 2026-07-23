
from typing import Literal, Optional

from cooper.constraints.constraint_state import ConstraintState
from cooper.formulations import ContributionStore, Formulation, Lagrangian
from cooper.multipliers import Multiplier
from cooper.penalty_coefficients import PenaltyCoefficient
from cooper.utils import ConstraintType


class Constraint:

    def __init__(
        self,
        constraint_type: ConstraintType,
        formulation_type: type[Formulation] = Lagrangian,
        multiplier: Optional[Multiplier] = None,
        penalty_coefficient: Optional[PenaltyCoefficient] = None,
    ) -> None:
        self._name = None

        self.constraint_type = constraint_type
        self.formulation_type = formulation_type
        self.formulation = formulation_type(constraint_type=self.constraint_type)

        self.multiplier = multiplier
        self.formulation.sanity_check_multiplier(multiplier)
        if self.multiplier is not None:
            self.multiplier.set_constraint_type(constraint_type)

        self.penalty_coefficient = penalty_coefficient
        self.formulation.sanity_check_penalty_coefficient(penalty_coefficient)

    @property
    def name(self) -> str:
        pass

    @name.setter
    def name(self, name: str) -> None:
        pass

    def compute_contribution_to_lagrangian(
        self, constraint_state: ConstraintState, primal_or_dual: Literal["primal", "dual"]
    ) -> Optional[ContributionStore]:
        pass

    def __repr__(self) -> str:
        repr_ = f"constraint_type={self.constraint_type}, formulation={self.formulation}"
        if self.multiplier is not None:
            repr_ += f", multiplier={self.multiplier}"
        if self.penalty_coefficient is not None:
            repr_ += f", penalty_coefficient={self.penalty_coefficient}"
        return f"Constraint({repr_})"
