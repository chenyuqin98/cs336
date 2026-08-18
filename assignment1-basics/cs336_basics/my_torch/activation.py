import torch
from .linear import Linear

torch.nn.SiLU


class SiLU(torch.nn.Module):
    def forward(self, x):
        # return x / (1 + torch.exp(-x))
        return x * torch.sigmoid(x)


class SwiGLU(torch.nn.Module):
    def __init__(self, d_model, d_ff, w1, w2, w3, device=None, dtype=None):
        super().__init__()
        self.linear1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.linear1.load_state_dict({"weight": w1})
        self.linear2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.linear2.load_state_dict({"weight": w2})
        self.linear3 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.linear3.load_state_dict({"weight": w3})
        self.silu = SiLU()

    def forward(self, x):
        return self.linear2(torch.mul(self.silu(self.linear1(x)), self.linear3(x)))
