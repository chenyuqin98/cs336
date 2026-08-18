import torch
import jaxtyping

torch.nn.RMSNorm


class RMSNorm(torch.nn.Module):
    def __init__(self, d_model, eps=1e-5, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones((d_model,), device=device, dtype=dtype))

    def forward(self, x: jaxtyping.Float[torch.Tensor, "... d_model"]) -> jaxtyping.Float[torch.Tensor, "... d_model"]:
        intype = x.dtype
        x = x.to(torch.float32)
        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        return ((x / rms) * self.weight).to(intype)
