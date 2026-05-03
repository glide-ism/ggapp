from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from typing import Any

import cupy as cp

import torch
from torch import Tensor
from torch.optim import Optimizer

class GGaPPWhiten(torch.autograd.Function):

    @staticmethod
    def forward(ctx,model,u):
        ctx.model = model
        u_ = cp.array(u.data)
        z = model.whiten(u_)
        return torch.tensor(z)

    @staticmethod
    def backward(ctx,gz):
        model = ctx.model
        gz_ = cp.array(gz.data)
        gu = model.whiten(gz_)
        return None, torch.tensor(gu)

class GGaPPMap(torch.autograd.Function):

    @staticmethod
    def forward(ctx,model,z):
        ctx.model = model
        z_ = cp.array(z.data)
        u = model.forward(z_)
        return torch.tensor(u)

    @staticmethod
    def backward(ctx,gu):
        model = ctx.model
        gu_ = cp.array(gu.data)
        gz = model.forward(gu_)
        return None, torch.tensor(gz)

GradTransform = Callable[[Tensor, Tensor, dict[str, Any], dict[str, Any]], Tensor | None]
# Signature: fn(param, grad, state, group) -> transformed_grad
# If the function mutates grad in-place, it may return None.

class PSGD(Optimizer):
    """SGD with optional user-specified gradient transforms.

    The transform is applied *after* optional weight decay is added to the
    gradient and *before* momentum / Nesterov are applied.

    This makes it suitable for preconditioned-gradient methods such as:
        g <- P g

    Parameters
    ----------
    params:
        Standard PyTorch optimizer parameter iterable or param-group iterable.
    lr:
        Learning rate.
    momentum:
        Momentum factor.
    dampening:
        Dampening for momentum.
    weight_decay:
        L2 penalty added directly to the gradient.
    nesterov:
        Whether to use Nesterov momentum.
    maximize:
        Maximize instead of minimize.
    grad_transform:
        Optional default transform for all parameters in a group.
    per_param_grad_transform:
        Optional mapping from parameter object -> transform function.
        This overrides the group-level transform for that parameter.

    Notes
    -----
    - Sparse gradients are not supported in this sketch.
    - If you want the transform to act on the *raw* objective gradient rather
      than the gradient + weight decay, set `weight_decay=0` here and include
      any regularization explicitly in your loss.
    """

    def __init__(
        self,
        params: Iterable[Tensor] | Iterable[dict[str, Any]],
        lr: float = 1e-3,
        momentum: float = 0.0,
        dampening: float = 0.0,
        weight_decay: float = 0.0,
        nesterov: bool = False,
        maximize: bool = False,
        grad_transform: GradTransform | None = None,
        per_param_grad_transform: Mapping[Tensor, GradTransform] | None = None,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if nesterov and (momentum <= 0.0 or dampening != 0.0):
            raise ValueError("Nesterov momentum requires momentum > 0 and dampening == 0")

        defaults = dict(
            lr=lr,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=nesterov,
            maximize=maximize,
            grad_transform=grad_transform,
        )
        super().__init__(params, defaults)

        self._per_param_grad_transform: dict[int, GradTransform] = {}
        if per_param_grad_transform is not None:
            self._per_param_grad_transform = {
                id(param): fn for param, fn in per_param_grad_transform.items()
            }

    def _get_transform(
        self,
        param: Tensor,
        group: dict[str, Any],
    ) -> GradTransform | None:
        fn = self._per_param_grad_transform.get(id(param))
        if fn is not None:
            return fn
        return group.get("grad_transform", None)

    @torch.no_grad()
    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            dampening = group["dampening"]
            weight_decay = group["weight_decay"]
            nesterov = group["nesterov"]
            maximize = group["maximize"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("PSGD does not support sparse gradients")

                # Standard sign convention
                if maximize:
                    grad = -grad

                # Standard SGD-style weight decay
                if weight_decay != 0.0:
                    grad = grad.add(p, alpha=weight_decay)

                # User-defined transform: g <- T(g)
                transform = self._get_transform(p, group)
                state = self.state[p]
                if transform is not None:
                    transformed = transform(p, grad, state, group)
                    if transformed is not None:
                        grad = transformed

                # Momentum / Nesterov
                if momentum != 0.0:
                    buf = state.get("momentum_buffer")
                    if buf is None:
                        buf = torch.clone(grad).detach()
                        state["momentum_buffer"] = buf
                    else:
                        buf.mul_(momentum).add_(grad, alpha=1.0 - dampening)

                    if nesterov:
                        grad = grad.add(buf, alpha=momentum)
                    else:
                        grad = buf

                # Parameter update
                p.add_(grad, alpha=-lr)

        return loss

