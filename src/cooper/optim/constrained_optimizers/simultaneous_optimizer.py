

from typing import Optional

from cooper.optim.constrained_optimizers.constrained_optimizer import ConstrainedOptimizer
from cooper.optim.optimizer import RollOut


class SimultaneousOptimizer(ConstrainedOptimizer):

    def roll(self, compute_cmp_state_kwargs: Optional[dict] = None) -> RollOut:
        pass
