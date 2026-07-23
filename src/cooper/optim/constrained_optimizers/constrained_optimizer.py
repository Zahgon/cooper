

import abc
from typing import Any

import torch

from cooper.cmp import ConstrainedMinimizationProblem
from cooper.optim.optimizer import CooperOptimizer, RollOut
from cooper.utils import OneOrSequence


class ConstrainedOptimizer(CooperOptimizer, abc.ABC):

    def __init__(
        self,
        cmp: ConstrainedMinimizationProblem,
        primal_optimizers: OneOrSequence[torch.optim.Optimizer],
        dual_optimizers: OneOrSequence[torch.optim.Optimizer],
    ) -> None:
        super().__init__(cmp=cmp, primal_optimizers=primal_optimizers, dual_optimizers=dual_optimizers)
        self.base_sanity_checks()
        self.custom_sanity_checks()

    def base_sanity_checks(self) -> None:
        pass

    def custom_sanity_checks(self) -> None:
        """Performs custom sanity checks on the initialization of ``ConstrainedOptimizer``."""

    @torch.no_grad()
    def dual_step(self) -> None:
        pass

    @abc.abstractmethod
    def roll(self, *args: Any, **kwargs: Any) -> RollOut:
        """Performs a full update step on the primal and dual variables."""
