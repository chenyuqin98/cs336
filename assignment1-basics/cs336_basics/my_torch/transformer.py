import torch
import jaxtyping

from .linear import Linear
from .embedding import Embedding
from .rmsnorm import RMSNorm
from .attention import RopeMHA
from .activation import SwiGLU


class PreNormTransformerBlock(torch.nn.Module):
    def __init__(
        self, d_model: int, num_heads: int, d_ff: int, theta: float, max_seq_len: int, device=None, dtype=None
    ):
        super().__init__()
        self.attn = RopeMHA(d_model, num_heads, theta, max_seq_len, device=device, dtype=dtype)
        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype)
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)

    def forward(
        self,
        x: jaxtyping.Float[torch.Tensor, "... d_model"],
        token_positions: jaxtyping.Int[torch.Tensor, "... sequence_length"] | None = None,
    ) -> jaxtyping.Float[torch.Tensor, "... d_model"]:
        if token_positions is None:
            token_positions = torch.arange(x.size(-2), device=x.device)
        x = x + self.attn(self.ln1(x), token_positions)
        x = x + self.ffn(self.ln2(x))
        return x


class TransformerLM(torch.nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.token_embeddings = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        # ModuleList 而非 Sequential：block 的 forward 还要收 token_positions，Sequential 只透传一个位置参数。
        self.layers = torch.nn.ModuleList(
            PreNormTransformerBlock(d_model, num_heads, d_ff, rope_theta, context_length, device=device, dtype=dtype)
            for _ in range(num_layers)
        )
        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(
        self,
        x: jaxtyping.Int[torch.Tensor, "... sequence_length"],
        token_positions: jaxtyping.Int[torch.Tensor, "... sequence_length"] | None = None,
    ) -> jaxtyping.Float[torch.Tensor, "... sequence_length vocab_size"]:
        x = self.token_embeddings(x)
        for layer in self.layers:
            x = layer(x, token_positions)
        x = self.ln_final(x)
        return self.lm_head(x)
