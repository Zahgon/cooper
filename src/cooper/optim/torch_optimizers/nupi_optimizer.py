

import warnings
from collections.abc import Callable, Iterable
from enum import Enum
from typing import Optional

import torch


class nuPIInitType(Enum):

    ZEROS = 0
    SGD = 1


class nuPI(torch.optim.Optimizer):
    def __init__(
        self,
        params: Iterable[torch.Tensor],
        lr: float,
        weight_decay: Optional[float] = 0.0,
        Kp: Optional[torch.Tensor] = 0.0,
        Ki: Optional[torch.Tensor] = 1.0,
        ema_nu: float = 0.0,
        init_type: nuPIInitType = nuPIInitType.SGD,
        maximize: bool = False,
    ) -> None:
        r"""Implements the ``nuPI`` controller as a PyTorch optimizer.

        Controllers are designed to guide a system toward a desired state by adjusting a
        control variable. This is achieved by measuring the error, which is the
        difference between the desired and current states, and using this error to
        modify the control variable, thereby influencing the system.

        For this controller, the error signal is derived from the gradient of a loss
        function :math:`L` being optimized with respect to a parameter
        :math:`\vtheta`. Here, :math:`\vtheta` acts as the control variable, while the
        **gradient** of :math:`L` serves as the error signal, defined as
        :math:`\ve_t = \nabla L_t(\vtheta_t)`. The control objective of setting
        :math:`\nabla L_t(\vtheta_t) = 0` corresponds to finding a stationary point
        of the loss function, thereby minimizing (or maximizing) it.

        .. note::
            When applied to the Lagrange multipliers of a constrained minimization
            problem, the control state :math:`\nabla L_t(\vtheta_t)` corresponds to the
            gradient of the Lagrangian function with respect to the multipliers (e.g.,
            :math:`\nabla_{\vlambda} \Lag(\vx, \vlambda) = \vg(\vx)` for
            inequality-constrained problems). Setting this gradient to (less than or
            equal to) zero corresponds to finding a point that satisfies the
            constraints.

        The ``nuPI`` controller updates parameters as follows:

        .. math::
            \vxi_t &= \nu \vxi_{t-1} + (1 - \nu) \ve_t, \\
            \vtheta_1 &= \vtheta_0 - \eta (K_P \vxi_0 + K_I \ve_0), \\
            \vtheta_{t+1} &= \vtheta_t - \eta (K_I \ve_t + K_P (\vxi_t - \vxi_{t-1}))

        Here, :math:`\vxi_t` is a smoothed version of the error signal (:math:`\ve_t`),
        using an exponential moving average (EMA) with coefficient :math:`\nu`.
        :math:`K_P` and :math:`K_I` are the proportional and integral gains,
        respectively, while the learning rate :math:`\eta` is kept separate
        to allow comparison with other optimizers.

        Weight decay is applied based only on the error signal :math:`\ve_t`, following
        a similar approach to PyTorch's AdamW optimizer.

        When ``maximize=False``, the parameter update is multiplied by :math:`-1` before
        being applied.

        **Initialization Schemes**:
        The initialization of the ``nuPI`` controller requires specifying the initial
        smoothed error signal, :math:`\vxi_{-1}`, which impacts the first parameter
        update. Two initialization schemes are available:

        - ``nuPIInitType.ZEROS``: Initializes :math:`\vxi_{-1} = \vzero`. The first update rule becomes:

            .. math::
                \vtheta_1 = \vtheta_0 - \eta (K_P \ve_0 + K_I \ve_0) = \vtheta_0 - \eta (K_P + K_I) \ve_0.

        - ``nuPIInitType.SGD``: Initializes :math:`\vxi_{-1} = \ve_0`, producing a first step identical to SGD:

            .. math::
                \vxi_0 &= \ve_0, \\
                \vtheta_1 &= \vtheta_0 - \eta (K_P \ve_0 + K_I \ve_0) = \vtheta_0 - \eta K_I \ve_0.

        .. note::
            nuPI(:math:`\eta`, :math:`K_P=0`, :math:`K_I=1`, :math:`\nu=0`) corresponds
            to SGD with learning rate :math:`\eta`.

            nuPI(:math:`\eta`, :math:`K_P=1`, :math:`K_I=1`, :math:`\nu=0`) corresponds
            to the optimistic gradient method :cite:p:`popov1980modification`.

        Args:
            params: iterable of parameters to optimize, or dicts defining parameter groups.
            lr: learning rate.
            weight_decay: weight decay (L2 penalty). Defaults to 0.
            Kp: proportional gain. Defaults to 0.
            Ki: integral gain. Defaults to 1.
            ema_nu: EMA coefficient for the smoothed error signal. Defaults to 0,
                meaning no smoothing is applied.
            init_type: initialization scheme for :math:`\vxi_{-1}`. Defaults to
                ``nuPIInitType.SGD``, which matches the first step of SGD.
            maximize: whether to maximize the objective with respect to the parameters
                instead of minimizing. Defaults to ``False``.

        Raises:
            ValueError: If the learning rate, or weight decay is negative.
            ValueError: If the EMA coefficient is not in the range :math:`(-1, 1)`.
            ValueError: If the initialization type is invalid.
            NotImplementedError: If multiple parameter groups are used with non-scalar
                proportional and integral gains.

        Warnings:
            If a negative proportional or integral gain is used.
            If both proportional and integral gains are zero.
            If the EMA coefficient is negative.
        """
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if not -1 < ema_nu < 1.0:
            raise ValueError(f"Invalid nu value: {ema_nu}")

        if init_type not in {nuPIInitType.ZEROS, nuPIInitType.SGD}:
            raise ValueError(f"Invalid init_type: {init_type}")

        if not isinstance(Kp, torch.Tensor):
            Kp = torch.tensor(Kp)
        if not isinstance(Ki, torch.Tensor):
            Ki = torch.tensor(Ki)

        if torch.any(Kp < 0.0):
            warnings.warn(f"Using a negative Kp coefficient: {Kp}")
        if torch.any(Ki < 0.0):
            warnings.warn(f"Using a negative Ki coefficient: {Kp}")
        if torch.all(Kp == 0.0) and torch.all(Ki == 0.0):
            warnings.warn("All PI coefficients are zero")
        if ema_nu < 0:
            warnings.warn("nuPI optimizer instantiated with negative EMA coefficient")

        defaults = {
            "lr": lr,
            "weight_decay": weight_decay,
            "Kp": Kp,
            "Ki": Ki,
            "ema_nu": ema_nu,
            "maximize": maximize,
            "init_type": init_type,
        }

        super().__init__(params, defaults)

        if len(self.param_groups) > 1 and Kp.shape != torch.Size([1]):
            raise NotImplementedError("When using multiple parameter groups, Kp and Ki must be scalars")

    @staticmethod
    def disambiguate_update_function(is_grad_sparse: bool, init_type: nuPIInitType) -> Callable:
        pass

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None) -> Optional[float]:
        pass

    def load_state_dict(self, state_dict: dict) -> None:
        pass


def _nupi_zero_init(
    param: torch.Tensor,
    state: dict,
    lr: float,
    weight_decay: float,
    Kp: torch.Tensor,
    Ki: torch.Tensor,
    ema_nu: float,
    maximize: bool,
) -> None:
    pass


def _sparse_nupi_zero_init(
    param: torch.Tensor,
    state: dict,
    lr: float,
    weight_decay: float,
    Kp: float,
    Ki: float,
    ema_nu: float,
    maximize: bool,
) -> None:
    pass


def _nupi_sgd_init(
    param: torch.Tensor,
    state: dict,
    lr: float,
    weight_decay: float,
    Kp: torch.Tensor,
    Ki: torch.Tensor,
    ema_nu: float,
    maximize: bool,
) -> None:
    pass


def _sparse_nupi_sgd_init(
    param: torch.Tensor,
    state: dict,
    lr: float,
    weight_decay: float,
    Kp: float,
    Ki: float,
    ema_nu: float,
    maximize: bool,
) -> None:
    pass
