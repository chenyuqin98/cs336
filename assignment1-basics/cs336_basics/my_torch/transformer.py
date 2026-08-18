import torch
import jaxtyping

from .linear import Linear
from .rmsnorm import RMSNorm
from .attention import RopeMHA
from .activation import SwiGLU

class PreNormTransformerBlock(torch.nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, theta: float, max_seq_len: int, device=None, dtype=None):
        super().__init__()
        self.attn = RopeMHA(d_model, num_heads, theta, max_seq_len, device=device, dtype=dtype)
        self.ln1 = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype)
        self.ln2 = RMSNorm(d_model)

    def forward(self, x: jaxtyping.Float[torch.Tensor, "... d_model"], token_positions: jaxtyping.Float[torch.Tensor, "... sequence_length"]=None) -> jaxtyping.Float[torch.Tensor, "... d_model"]:
        if not token_positions:
            token_positions = torch.arange(x.size(-2), device=x.device)
        x = x + self.attn(self.ln1(x), token_positions)
        x = x + self.ffn(self.ln2(x))
        return x