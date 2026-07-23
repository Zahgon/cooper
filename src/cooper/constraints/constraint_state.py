
from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class ConstraintState:

    violation: torch.Tensor
    constraint_features: Optional[torch.Tensor] = None
    strict_violation: Optional[torch.Tensor] = None
    strict_constraint_features: Optional[torch.Tensor] = None
    contributes_to_primal_update: bool = True
    contributes_to_dual_update: bool = True

    def __post_init__(self) -> None:
        """Checks that the constraint state is well-formed.

        Raises:
            ValueError: If `strict_constraint_features` are provided, but `strict_violation` is not.
        """
        if self.strict_constraint_features is not None and self.strict_violation is None:
            raise ValueError("`strict_violation` must be provided if `strict_constraint_features` is provided.")

    def extract_violations(self, do_unsqueeze: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
        pass

    def extract_constraint_features(self) -> tuple[torch.Tensor, torch.Tensor]:
        pass
