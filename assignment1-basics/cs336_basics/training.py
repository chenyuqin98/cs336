"""Training primitives implemented with elementary tensor operations."""

import math

import numpy as np
import torch


def cross_entropy(inputs, targets):
    shifted = inputs - inputs.max(dim=-1, keepdim=True).values
    return (shifted.exp().sum(dim=-1).log() - shifted.gather(-1, targets.unsqueeze(-1)).squeeze(-1)).mean()


def gradient_clipping(parameters, max_l2_norm):
    if max_l2_norm < 0:
        raise ValueError("max_l2_norm must be nonnegative")
    grads = [p.grad for p in parameters if p.grad is not None]
    if not grads:
        return
    by_device = {}
    for grad in grads:
        by_device.setdefault(grad.device, []).append(grad.detach().float().square().sum())
    norm = sum(torch.stack(values).sum().item() for values in by_device.values()) ** 0.5
    scale = min(1.0, max_l2_norm / (norm + 1e-6))
    with torch.no_grad():
        for grad in grads:
            grad.mul_(scale)


class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        if lr < 0 or eps < 0 or weight_decay < 0 or any(not 0 <= b < 1 for b in betas):
            raise ValueError("Invalid AdamW hyperparameters")
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                if g.is_sparse:
                    raise RuntimeError("AdamW does not support sparse gradients")
                state = self.state[p]
                if not state:
                    state.update(t=0, m=torch.zeros_like(p), v=torch.zeros_like(p))
                state["t"] += 1
                t, m, v = state["t"], state["m"], state["v"]
                m.mul_(beta1).add_(g, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(g, g, value=1 - beta2)
                alpha = group["lr"] * math.sqrt(1 - beta2**t) / (1 - beta1**t)
                p.addcdiv_(m, v.sqrt().add_(group["eps"]), value=-alpha)
                p.mul_(1 - group["lr"] * group["weight_decay"])
        return loss


def get_lr_cosine_schedule(it, max_learning_rate, min_learning_rate, warmup_iters, cosine_cycle_iters):
    if warmup_iters < 0 or cosine_cycle_iters <= warmup_iters:
        raise ValueError("Require 0 <= warmup_iters < cosine_cycle_iters")
    if it < warmup_iters:
        return it / warmup_iters * max_learning_rate
    if it >= cosine_cycle_iters:
        return min_learning_rate
    progress = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
    return min_learning_rate + 0.5 * (1 + math.cos(math.pi * progress)) * (max_learning_rate - min_learning_rate)


def get_batch(dataset, batch_size, context_length, device):
    if dataset.ndim != 1 or len(dataset) <= context_length or context_length <= 0 or batch_size <= 0:
        raise ValueError("Need a 1D token array longer than a positive context_length and a positive batch_size")
    starts = np.random.randint(0, len(dataset) - context_length, size=batch_size)
    indices = starts[:, None] + np.arange(context_length + 1)
    batch = torch.tensor(dataset[indices].astype(np.int64), device=device)
    return batch[:, :-1], batch[:, 1:]


def save_checkpoint(model, optimizer, iteration, out):
    torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), iteration=iteration), out)


def load_checkpoint(src, model, optimizer):
    checkpoint = torch.load(src, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint["iteration"]
