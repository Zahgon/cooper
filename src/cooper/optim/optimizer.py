
import abc
from typing import Any, NamedTuple, Optional, TypedDict

import torch

from cooper.cmp import CMPState, ConstrainedMinimizationProblem, LagrangianStore
from cooper.utils import OneOrSequence, ensure_sequence


class CooperOptimizerState(TypedDict):

    primal_optimizer_states: list[dict]
    dual_optimizer_states: Optional[list[dict]]


class RollOut(NamedTuple):

    loss: torch.Tensor
    cmp_state: CMPState
    primal_lagrangian_store: LagrangianStore
    dual_lagrangian_store: LagrangianStore


class CooperOptimizer(abc.ABC):

    def __init__(
        self,
        cmp: ConstrainedMinimizationProblem,
        primal_optimizers: OneOrSequence[torch.optim.Optimizer],
        dual_optimizers: Optional[OneOrSequence[torch.optim.Optimizer]] = None,
    ) -> None:
        self.cmp = cmp
        self.primal_optimizers = ensure_sequence(primal_optimizers)
        self.dual_optimizers = ensure_sequence(dual_optimizers)

    def zero_grad(self) -> None:
        pass

    @torch.no_grad()
    def primal_step(self) -> None:
        pass

    def state_dict(self) -> CooperOptimizerState:
        pass

    def load_state_dict(self, state: CooperOptimizerState) -> None:
        pass

    @abc.abstractmethod
    def roll(self, *args: Any, **kwargs: Any) -> RollOut:
        """Evaluates the objective function and performs a gradient update on the parameters."""
