
import abc
from typing import Any, Literal, NamedTuple, Optional

import torch

import cooper.formulations.utils as formulation_utils
from cooper.constraints.constraint_state import ConstraintState
from cooper.multipliers import Multiplier
from cooper.penalty_coefficients import PenaltyCoefficient
from cooper.utils import ConstraintType


class ContributionStore(NamedTuple):
    lagrangian_contribution: torch.Tensor
    multiplier_value: Optional[torch.Tensor] = None
    penalty_coefficient_value: Optional[torch.Tensor] = None


class Formulation(abc.ABC):

    expects_multiplier: bool
    expects_penalty_coefficient: bool

    def __init__(self, constraint_type: ConstraintType) -> None:
        if constraint_type not in {ConstraintType.EQUALITY, ConstraintType.INEQUALITY}:
            raise ValueError(f"{type(self).__name__} requires either an equality or inequality constraint.")
        self.constraint_type = constraint_type

    def __repr__(self) -> str:
        return f"{type(self).__name__}(constraint_type={self.constraint_type})"

    def sanity_check_multiplier(self, multiplier: Optional[Multiplier]) -> None:
        pass

    def sanity_check_penalty_coefficient(self, penalty_coefficient: Optional[PenaltyCoefficient]) -> None:
        pass

    def _prepare_kwargs_for_lagrangian_contribution(
        self,
        constraint_state: ConstraintState,
        multiplier: Optional[Multiplier],
        penalty_coefficient: Optional[PenaltyCoefficient],
        primal_or_dual: Literal["primal", "dual"],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        pass

    @abc.abstractmethod
    def compute_contribution_to_primal_lagrangian(self, *args: Any, **kwargs: Any) -> Optional[ContributionStore]:
        """Computes the contribution of a given constraint violation to the *primal*
        Lagrangian.

        Returns ``None`` if the constraint does not contribute to the primal update
        (i.e., when ``ConstraintState.contributes_to_primal_update=False``).
        """
        raise NotImplementedError

    @abc.abstractmethod
    def compute_contribution_to_dual_lagrangian(self, *args: Any, **kwargs: Any) -> Optional[ContributionStore]:
        """Computes the contribution of a given constraint violation to the *dual*
        Lagrangian.

        Returns ``None`` if the constraint does not contribute to the dual update
        (i.e., when ``ConstraintState.contributes_to_dual_update=False``).
        """
        raise NotImplementedError


class Lagrangian(Formulation):

    expects_multiplier = True
    expects_penalty_coefficient = False

    def compute_contribution_to_primal_lagrangian(
        self, constraint_state: ConstraintState, multiplier: Multiplier
    ) -> Optional[ContributionStore]:
        pass

    def compute_contribution_to_dual_lagrangian(
        self, constraint_state: ConstraintState, multiplier: Multiplier
    ) -> Optional[ContributionStore]:
        pass


class QuadraticPenalty(Formulation):

    expects_multiplier = False
    expects_penalty_coefficient = True

    def compute_contribution_to_primal_lagrangian(
        self, constraint_state: ConstraintState, penalty_coefficient: PenaltyCoefficient
    ) -> Optional[ContributionStore]:
        pass

    def compute_contribution_to_dual_lagrangian(  # noqa: PLR6301
        self,
        constraint_state: ConstraintState,  # noqa: ARG002
        penalty_coefficient: PenaltyCoefficient,  # noqa: ARG002
    ) -> None:
        pass


class AugmentedLagrangian(Formulation):

    expects_multiplier = True
    expects_penalty_coefficient = True

    def compute_contribution_to_primal_lagrangian(
        self, constraint_state: ConstraintState, multiplier: Multiplier, penalty_coefficient: PenaltyCoefficient
    ) -> Optional[ContributionStore]:
        pass

    def compute_contribution_to_dual_lagrangian(
        self, constraint_state: ConstraintState, multiplier: Multiplier, penalty_coefficient: PenaltyCoefficient
    ) -> Optional[ContributionStore]:
        pass
