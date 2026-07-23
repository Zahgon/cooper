

import math
from collections.abc import Callable, Iterable
from typing import NoReturn, Optional

import torch











class ExtragradientOptimizer(torch.optim.Optimizer):

    def __init__(self, params: Iterable, defaults: dict) -> None:
        super().__init__(params, defaults)
        self.params_copy: list[torch.nn.Parameter] = []

    def update(self, p: torch.Tensor, group: dict) -> NoReturn:
        raise NotImplementedError

    def extrapolation(self) -> None:
        pass

    def step(self, closure: Optional[Callable] = None) -> Optional[torch.Tensor]:
        pass


class ExtraSGD(ExtragradientOptimizer):

    def __init__(
        self,
        params: Iterable,
        lr: float = 1e-3,
        momentum: float = 0,
        dampening: float = 0,
        weight_decay: float = 0,
        nesterov: bool = False,
        maximize: bool = False,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = {
            "lr": lr,
            "momentum": momentum,
            "dampening": dampening,
            "weight_decay": weight_decay,
            "nesterov": nesterov,
            "maximize": maximize,
        }
        if nesterov and (momentum == 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super().__init__(params, defaults)

    def __setstate__(self, state: dict) -> None:
        super(torch.optim.SGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault("nesterov", False)

    def update(self, p: torch.Tensor, group: dict) -> Optional[torch.Tensor]:
        pass


class ExtraAdam(ExtragradientOptimizer):

    def __init__(
        self,
        params: Iterable,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0,
        amsgrad: bool = False,
        maximize: bool = False,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
            "amsgrad": amsgrad,
            "maximize": maximize,
        }
        super().__init__(params, defaults)

    def __setstate__(self, state: dict) -> None:
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("amsgrad", False)

    def update(self, p: torch.Tensor, group: dict) -> Optional[torch.Tensor]:
        pass
