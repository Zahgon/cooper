

from typing import Optional

from cooper.cmp import LagrangianStore
from cooper.optim.optimizer import CooperOptimizer, RollOut


class UnconstrainedOptimizer(CooperOptimizer):

    def roll(self, compute_cmp_state_kwargs: Optional[dict] = None) -> RollOut:
        pass
