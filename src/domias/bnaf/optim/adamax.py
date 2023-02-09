# stdlib
from typing import Any, Callable, Optional, Tuple

# third party
import torch


class Adamax(torch.optim.Optimizer):
    def __init__(
        self,
        params: Any,
        lr: float = 2e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0,
        polyak: float = 0,
    ) -> None:
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if not 0.0 <= polyak <= 1.0:
            raise ValueError("Invalid polyak decay term: {}".format(polyak))

        defaults = dict(
            lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, polyak=polyak
        )
        super(Adamax, self).__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None) -> Optional[float]:

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("Adamax does not support sparse gradients")
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p.data)
                    state["exp_inf"] = torch.zeros_like(p.data)
                    # Exponential moving average of param
                    state["exp_avg_param"] = torch.zeros_like(p.data)

                exp_avg, exp_inf = state["exp_avg"], state["exp_inf"]
                beta1, beta2 = group["betas"]
                eps = group["eps"]

                state["step"] += 1

                if group["weight_decay"] != 0:
                    grad = grad.add(group["weight_decay"], p.data)

                # Update biased first moment estimate.
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                # Update the exponentially weighted infinity norm.
                norm_buf = torch.cat(
                    [
                        exp_inf.mul_(beta2).unsqueeze(0),
                        grad.abs().add_(eps).unsqueeze_(0),
                    ],
                    0,
                )
                torch.max(
                    norm_buf, 0, keepdim=False, out=(exp_inf, exp_inf.new().long())
                )

                bias_correction = 1 - beta1 ** state["step"]
                clr = group["lr"] / bias_correction

                p.data.addcdiv_(-clr, exp_avg, exp_inf)

                polyak = self.defaults["polyak"]
                state["exp_avg_param"] = (
                    polyak * state["exp_avg_param"] + (1 - polyak) * p.data
                )

        return loss

    def swap(self) -> None:
        """
        Swapping the running average of params and the current params for saving parameters using polyak averaging
        """
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                new = p.data
                p.data = state["exp_avg_param"]
                state["exp_avg_param"] = new

    def substitute(self) -> None:
        for group in self.param_groups:
            for p in group["params"]:
                p.data = self.state[p]["exp_avg_param"]
