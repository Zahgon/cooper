

from typing import Optional

import torch

from cooper.optim.constrained_optimizers.constrained_optimizer import ConstrainedOptimizer
from cooper.optim.optimizer import RollOut


class ExtrapolationConstrainedOptimizer(ConstrainedOptimizer):

    def custom_sanity_checks(self) -> None:
        pass

    @torch.no_grad()
    def primal_extrapolation_step(self) -> None:
        pass

    @torch.no_grad()
    def dual_extrapolation_step(self) -> None:
        pass

    def roll(self, compute_cmp_state_kwargs: Optional[dict] = None) -> RollOut:
        pass
