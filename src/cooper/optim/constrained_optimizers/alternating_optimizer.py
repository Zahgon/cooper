

from typing import Optional

import torch

from cooper.optim.constrained_optimizers.constrained_optimizer import ConstrainedOptimizer
from cooper.optim.optimizer import RollOut


class AlternatingPrimalDualOptimizer(ConstrainedOptimizer):

    def roll(
        self, compute_cmp_state_kwargs: Optional[dict] = None, compute_violations_kwargs: Optional[dict] = None
    ) -> RollOut:
        pass


class AlternatingDualPrimalOptimizer(ConstrainedOptimizer):

    def roll(self, compute_cmp_state_kwargs: Optional[dict] = None) -> RollOut:
        pass
