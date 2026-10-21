import torch
from jaxtyping import Float, Int


class RoPE(torch.nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        i = torch.arange(max_seq_len, device=device).float()
        k = torch.arange(1, d_k // 2 + 1, device=device).float()
        # print(k)
        omega = torch.pow(theta, -(2 * k - 2) / d_k)
        # print(omega)
        r_sin = torch.sin(torch.einsum("i,j->ij", i, omega))
        r_cos = torch.cos(torch.einsum("i,j->ij", i, omega))
        self.register_buffer("r_sin", r_sin, persistent=False)
        self.register_buffer("r_cos", r_cos, persistent=False)

    def forward(
        self, x: Float[torch.Tensor, "... d_k"], token_positions: Int[torch.Tensor, "..."]
    ) -> Float[torch.Tensor, "... d_k"]:
        cos = self.r_cos[token_positions]
        sin = self.r_sin[token_positions]
        # Batched positions omit the attention head axis.
        if token_positions.ndim == x.ndim - 2:
            cos = cos.unsqueeze(-3)
            sin = sin.unsqueeze(-3)

        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]

        out = torch.empty_like(x)
        out[..., 0::2] = x_even * cos - x_odd * sin
        out[..., 1::2] = x_odd * cos + x_even * sin

        return out
