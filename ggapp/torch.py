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

TensorTransform = Callable[[Tensor, Tensor, dict[str, Any], dict[str, Any]], Tensor | None]
# Signature: fn(param, x, state, group) -> y
# `x` is the tensor to transform.
# May return None if it mutates x in-place and wants caller to keep using x.

class PreconditionedAdam(Optimizer):
    r"""Adam in latent/whitened coordinates, updates applied in physical coordinates.

    For each parameter tensor m, this optimizer assumes a local change of variables

        z = L (m - mu)

    but it does not need mu explicitly. It only needs:

      grad_to_latent:   g_m -> g_z = L^{-T} g_m
      latent_to_param:  dz  -> dm  = L^{-1} dz

    Adam moments are accumulated in latent coordinates on g_z. The resulting
    adaptive latent step dz is then mapped back to physical space via latent_to_param.

    Notes
    -----
    - For self-adjoint priors, you may often use the same solver/kernel family for
      both transforms, but the API keeps them separate for generality.
    - Sparse gradients are not supported in this sketch.
    - If you need hard constraints in physical space, add a post-step projection.
    """

    def __init__(
        self,
        params: Iterable[Tensor] | Iterable[dict[str, Any]],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        amsgrad: bool = False,
        maximize: bool = False,
        # group-level defaults
        grad_to_latent: TensorTransform | None = None,
        latent_to_param: TensorTransform | None = None,
        # per-parameter overrides
        per_param_grad_to_latent: Mapping[Tensor, TensorTransform] | None = None,
        per_param_latent_to_param: Mapping[Tensor, TensorTransform] | None = None,
        # optional physical-space projection after update
        post_step_proj: TensorTransform | None = None,
        per_param_post_step_proj: Mapping[Tensor, TensorTransform] | None = None,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        beta1, beta2 = betas
        if not (0.0 <= beta1 < 1.0):
            raise ValueError(f"Invalid beta1 value: {beta1}")
        if not (0.0 <= beta2 < 1.0):
            raise ValueError(f"Invalid beta2 value: {beta2}")

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
            maximize=maximize,
            grad_to_latent=grad_to_latent,
            latent_to_param=latent_to_param,
            post_step_proj=post_step_proj,
        )
        super().__init__(params, defaults)

        self._per_param_grad_to_latent = (
            {id(p): fn for p, fn in per_param_grad_to_latent.items()}
            if per_param_grad_to_latent is not None else {}
        )
        self._per_param_latent_to_param = (
            {id(p): fn for p, fn in per_param_latent_to_param.items()}
            if per_param_latent_to_param is not None else {}
        )
        self._per_param_post_step_proj = (
            {id(p): fn for p, fn in per_param_post_step_proj.items()}
            if per_param_post_step_proj is not None else {}
        )

    def _get_transform(
        self,
        param: Tensor,
        group: dict[str, Any],
        key: str,
        overrides: dict[int, TensorTransform],
    ) -> TensorTransform | None:
        fn = overrides.get(id(param))
        if fn is not None:
            return fn
        return group.get(key, None)

    @torch.no_grad()
    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            amsgrad = group["amsgrad"]
            maximize = group["maximize"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                g_m = p.grad
                if g_m.is_sparse:
                    raise RuntimeError("PAdam does not support sparse gradients")

                # Standard sign convention
                if maximize:
                    g_m = -g_m

                # Usually best left at 0.0 for inverse problems; included for completeness.
                if weight_decay != 0.0:
                    g_m = g_m.add(p, alpha=weight_decay)

                grad_to_latent = self._get_transform(
                    p, group, "grad_to_latent", self._per_param_grad_to_latent
                )
                latent_to_param = self._get_transform(
                    p, group, "latent_to_param", self._per_param_latent_to_param
                )
                post_step_proj = self._get_transform(
                    p, group, "post_step_proj", self._per_param_post_step_proj
                )

                if grad_to_latent is None or latent_to_param is None:
                    raise RuntimeError(
                        "PAdam requires both grad_to_latent and latent_to_param "
                        "for every optimized parameter."
                    )

                # g_z = L^{-T} g_m
                g_z = grad_to_latent(p, g_m, self.state[p], group)
                if g_z is None:
                    raise RuntimeError("grad_to_latent must return a Tensor")

                state = self.state[p]

                # Lazy state init in latent coordinates
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(g_z)
                    state["exp_avg_sq"] = torch.zeros_like(g_z)
                    if amsgrad:
                        state["max_exp_avg_sq"] = torch.zeros_like(g_z)

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                state["step"] += 1
                step_t = state["step"]

                # Adam moments in latent coordinates
                exp_avg.mul_(beta1).add_(g_z, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(g_z, g_z, value=1.0 - beta2)

                if amsgrad:
                    max_exp_avg_sq = state["max_exp_avg_sq"]
                    torch.maximum(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    denom = max_exp_avg_sq.sqrt().add_(eps)
                else:
                    denom = exp_avg_sq.sqrt().add_(eps)

                bias_correction1 = 1.0 - beta1 ** step_t
                bias_correction2 = 1.0 - beta2 ** step_t

                # Adam step in latent coordinates
                exp_avg_hat = exp_avg / bias_correction1
                denom_hat = denom / (bias_correction2 ** 0.5)
                delta_z = exp_avg_hat / denom_hat

                # Map latent step back to physical coordinates:
                # dm = L^{-1} delta_z
                delta_m = latent_to_param(p, delta_z, state, group)
                if delta_m is None:
                    raise RuntimeError("latent_to_param must return a Tensor")

                # Standard descent step in physical coordinates
                p.add_(delta_m, alpha=-lr)

                # Optional projection/clamp in physical space
                if post_step_proj is not None:
                    projected = post_step_proj(p, p, state, group)
                    if projected is not None:
                        p.copy_(projected)

        return loss


