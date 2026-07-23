
import abc
from typing import Any, Optional

import torch
from typing_extensions import Self


class PenaltyCoefficient(abc.ABC):

    expects_constraint_features: bool
    _value: Optional[torch.Tensor] = None

    def __init__(self, init: torch.Tensor) -> None:
        if init.dim() > 1:
            raise ValueError("init must either be a scalar or a 1D tensor of shape `(num_constraints,)`.")
        self.init = init.clone()
        self.value = init

    @property
    def value(self) -> torch.Tensor:
        pass

    @value.setter
    def value(self, value: torch.Tensor) -> None:
        pass

    def to(self, *args: Any, **kwargs: Any) -> Self:
        pass

    def state_dict(self) -> dict:
        pass

    def load_state_dict(self, state_dict: dict) -> None:
        pass

    def sanity_check(self) -> None:
        pass

    def __repr__(self) -> str:
        if self.value.numel() <= 10:  # noqa: PLR2004
            return f"{type(self).__name__}({self.value})"
        return f"{type(self).__name__}(shape={self.value.shape})"

    @abc.abstractmethod
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Return the current value of the penalty coefficient."""


class DensePenaltyCoefficient(PenaltyCoefficient):

    expects_constraint_features = False

    @torch.no_grad()
    def __call__(self) -> torch.Tensor:
        """Return the current value of the penalty coefficient."""
        return self.value.clone()


class IndexedPenaltyCoefficient(PenaltyCoefficient):

    expects_constraint_features = True

    @torch.no_grad()
    def __call__(self, indices: torch.Tensor) -> torch.Tensor:
        """Return the current value of the penalty coefficient at the provided indices.

        Args:
            indices: Tensor of indices for which to return the penalty coefficient.

        Raises:
            ValueError: If ``indices`` is not of type ``torch.long``.
        """
        if indices.dtype != torch.long:
            raise ValueError("Indices must be of type torch.long.")

        if self.value.dim() == 0:
            return self.value.clone()

        coefficient_values = torch.nn.functional.embedding(indices, self.value.unsqueeze(1), sparse=False)

        return torch.flatten(coefficient_values)
