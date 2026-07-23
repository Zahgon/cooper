
import abc
from collections import OrderedDict
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, Literal, Optional

import torch
from typing_extensions import Self

from cooper.constraints import Constraint, ConstraintState
from cooper.multipliers import Multiplier
from cooper.penalty_coefficients import PenaltyCoefficient

__all__ = [
    "CMPState",
    "ConstrainedMinimizationProblem",
    "LagrangianStore",
]


@dataclass
class LagrangianStore:

    lagrangian: Optional[torch.Tensor] = None
    multiplier_values: dict[Constraint, torch.Tensor] = field(default_factory=dict)
    penalty_coefficient_values: dict[Constraint, torch.Tensor] = field(default_factory=dict)

    def backward(self) -> None:
        pass

    def observed_multiplier_values(self) -> Iterator[torch.Tensor]:
        pass

    def observed_penalty_coefficient_values(self) -> Iterator[torch.Tensor]:
        pass


@dataclass
class CMPState:

    loss: Optional[torch.Tensor] = None
    observed_constraints: dict[Constraint, ConstraintState] = field(default_factory=dict)
    misc: Optional[dict] = None

    def _compute_primal_or_dual_lagrangian(self, primal_or_dual: Literal["primal", "dual"]) -> LagrangianStore:
        pass

    def compute_primal_lagrangian(self) -> LagrangianStore:
        pass

    def compute_dual_lagrangian(self) -> LagrangianStore:
        pass

    def named_observed_violations(self) -> Iterator[tuple[str, torch.Tensor]]:
        pass

    def named_observed_strict_violations(self) -> Iterator[tuple[str, torch.Tensor]]:
        pass

    def named_observed_constraint_features(self) -> Iterator[tuple[str, torch.Tensor]]:
        pass

    def named_observed_strict_constraint_features(self) -> Iterator[tuple[str, torch.Tensor]]:
        pass


class ConstrainedMinimizationProblem(abc.ABC):

    def __init__(self) -> None:
        self._constraints = OrderedDict()

    def _register_constraint(self, name: str, constraint: Constraint) -> None:
        pass

    def constraints(self) -> Iterator[Constraint]:
        """Return an iterator over the registered constraints of the CMP."""
        yield from self._constraints.values()

    def named_constraints(self) -> Iterator[tuple[str, Constraint]]:
        pass

    def multipliers(self) -> Iterator[Multiplier]:
        pass

    def named_multipliers(self) -> Iterator[tuple[str, Multiplier]]:
        pass

    def penalty_coefficients(self) -> Iterator[PenaltyCoefficient]:
        """Returns an iterator over the penalty coefficients associated with the
        registered constraints of the CMP. Constraints without penalty coefficients
        are skipped.
        """
        for constraint in self.constraints():
            if constraint.penalty_coefficient is not None:
                yield constraint.penalty_coefficient

    def named_penalty_coefficients(self) -> Iterator[tuple[str, PenaltyCoefficient]]:
        pass

    def dual_parameters(self) -> Iterator[torch.nn.Parameter]:
        pass

    def to(self, *args: Any, **kwargs: Any) -> Self:
        pass

    def state_dict(self) -> dict:
        pass

    def load_state_dict(self, state_dict: dict) -> None:
        pass

    def __setattr__(self, name: str, value: Any) -> None:
        if isinstance(value, Constraint):
            self._register_constraint(name, value)
        else:
            super().__setattr__(name, value)

    def __getattr__(self, name: str) -> Any:
        if name in self._constraints:
            return self._constraints[name]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def __delattr__(self, name: str) -> None:
        if name in self._constraints:
            del self._constraints[name]
        else:
            super().__delattr__(name)

    def __repr__(self) -> str:
        repr_str = f"{type(self).__name__}"
        if len(self._constraints) < 5:  # noqa: PLR2004
            repr_str += "\n\t(constraints=[\n"
            for i, (name, constraint) in enumerate(self.named_constraints()):
                suffix = ",\n" if i < len(self._constraints) - 1 else "\n"
                repr_str += f"\t\t{name}: {constraint}{suffix}"
            repr_str += "\t\t]\n\t)"
        return repr_str

    @abc.abstractmethod
    def compute_cmp_state(self, *args: Any, **kwargs: Any) -> CMPState:
        """Computes the state of the CMP based on the current value of the primal
        parameters.

        The signature of this function may be adjusted to accommodate situations
        that require a model, (mini-batched) inputs/targets, or other arguments to be
        passed.

        .. note::
            When it is prohibitively expensive to compute the loss or constraints
            exactly, the :py:class:`CMPState` may contain **stochastic estimates**. This
            is often the case when mini-batches are used to approximate the loss and
            constraints.

            Just as in the unconstrained case, these approximations can lead to a
            compromise in the stability of the optimization process.

        """

    @staticmethod
    def sanity_check_cmp_state(cmp_state: CMPState) -> None:
        pass

    def compute_violations(self, *args: Any, **kwargs: Any) -> CMPState:
        """Computes the violation of the CMP constraints based on the current value of the
        primal parameters. This function returns a :py:class:`~.CMPState` instance
        containing the observed constraint values. Note that the returned
        :py:class:`~.CMPState` may have ``loss=None``, as the loss value is not
        necessarily computed when only evaluating the constraints.

        The function signature may be adjusted to accommodate situations that require a
        model, (mini-batched) inputs/targets, or other arguments.

        In some cases, the computation of constraints may be independent of loss
        evaluation. In such situations,
        :py:meth:`CMP.compute_violations<.ConstrainedMinimizationProblem.compute_violations>` can be called
        as part of the execution of
        :py:meth:`CMP.compute_cmp_state<.ConstrainedMinimizationProblem.compute_cmp_state>`.
        """
        raise NotImplementedError
