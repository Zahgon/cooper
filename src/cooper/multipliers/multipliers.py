

import abc
from typing import Any, Optional

import torch

from cooper.utils import ConstraintType


class Multiplier(torch.nn.Module, abc.ABC):
    expects_constraint_features: bool
    constraint_type: ConstraintType

    @abc.abstractmethod
    def forward(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Return the current value of the multiplier."""

    @abc.abstractmethod
    def post_step_(self) -> None:
        """Post-step function for multipliers. This function is called after each step of
        the dual optimizer, and allows for additional post-processing of the implicit
        multiplier module or its parameters.
        """

    def sanity_check(self) -> None:
        """Perform sanity checks on the multiplier. This method is called after setting
        the constraint type and ensures consistency between the multiplier and the
        constraint type. For example, multipliers for inequality constraints must be
        non-negative.
        """

    def set_constraint_type(self, constraint_type: ConstraintType) -> None:
        pass


class ExplicitMultiplier(Multiplier):

    def __init__(
        self,
        num_constraints: Optional[int] = None,
        init: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()

        self.weight = self.initialize_weight(num_constraints=num_constraints, init=init, device=device, dtype=dtype)

    @staticmethod
    def initialize_weight(
        num_constraints: Optional[int],
        init: Optional[torch.Tensor],
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        pass

    @property
    def device(self) -> torch.device:
        pass

    def sanity_check(self) -> None:
        pass

    @torch.no_grad()
    def post_step_(self) -> None:
        pass

    def __repr__(self) -> str:
        return f"{type(self).__name__}(num_constraints={self.weight.shape[0]})"


class DenseMultiplier(ExplicitMultiplier):

    expects_constraint_features = False

    def forward(self) -> torch.Tensor:
        pass


class IndexedMultiplier(ExplicitMultiplier):

    expects_constraint_features = True

    def __init__(
        self,
        num_constraints: Optional[int] = None,
        init: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__(num_constraints, init, device, dtype)
        if self.weight.dim() == 1:
            self.weight.data = self.weight.data.unsqueeze(-1)

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        pass


class ImplicitMultiplier(Multiplier):

    @abc.abstractmethod
    def forward(self) -> torch.Tensor:
        pass

    @abc.abstractmethod
    def post_step_(self) -> None:
        """This method is called after each step of the dual optimizer and allows for
        additional post-processing of the implicit multiplier module or its parameters.
        """
