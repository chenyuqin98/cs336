import torch
from jaxtyping import Float, Bool, Int

from .softmax import softmax
from .linear import Linear
from .rope import RoPE


class scale_dot_product_attention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.softmax = softmax()

    def forward(
        self,
        query: Float[torch.Tensor, "... queries d_k"],
        key: Float[torch.Tensor, "... keys d_k"],
        value: Float[torch.Tensor, "... keys d_v"],
        mask: Bool[torch.Tensor, "... queries keys"] | None = None,
    ) -> Float[torch.Tensor, "... queries d_v"]:
        d_k = query.size(-1)
        scores: Float[torch.Tensor, "... queries keys"] = torch.matmul(query, key.transpose(-2, -1)) / d_k**0.5

        # mask 中 True 表示允许 attend，所以对 ~mask 的位置填 -inf。
        # 必须在 softmax 之前填，这样剩余位置的概率才会自动归一化到和为 1。
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))

        attention_weights = self.softmax(scores, dim=-1)
        return torch.matmul(attention_weights, value)


class MultiHeadSelfAttention(torch.nn.Module):
    def __init__(self, d_model: int, num_heads: int, device=None, dtype=None):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_k = d_model // num_heads

        # W_Q ∈ R^{(h*d_k) x d_model}，而 h*d_k == d_model，所以四个投影都是 (d_model, d_model)。
        # 每个 head 拥有其中连续的 d_k 行，一次 matmul 就算完所有 head。
        self.query_linear: Linear = Linear(d_model, d_model, device=device, dtype=dtype)
        self.key_linear: Linear = Linear(d_model, d_model, device=device, dtype=dtype)
        self.value_linear: Linear = Linear(d_model, d_model, device=device, dtype=dtype)
        self.out_linear: Linear = Linear(d_model, d_model, device=device, dtype=dtype)

        self.sdpa: scale_dot_product_attention = scale_dot_product_attention()

    def _split_heads(
        self, x: Float[torch.Tensor, "... sequence_length d_model"]
    ) -> Float[torch.Tensor, "... heads sequence_length d_k"]:
        # 把最后一维切成 h 段（view 不复制数据），再把 head 维挪到 seq 前面，
        # 让 matmul 的最后两维是 (sequence_length, d_k)，head 退化成 batch 维。
        x: Float[torch.Tensor, "... sequence_length heads d_k"] = x.view(*x.shape[:-1], self.num_heads, self.d_k)
        return x.transpose(-3, -2)

    def _position_encode(
        self,
        query: Float[torch.Tensor, "... heads sequence_length d_k"],
        key: Float[torch.Tensor, "... heads sequence_length d_k"],
        token_positions: Int[torch.Tensor, "... sequence_length"] | None,
    ) -> tuple[Float[torch.Tensor, "... heads sequence_length d_k"], Float[torch.Tensor, "... heads sequence_length d_k"]]:
        # 基类不做位置编码；RopeMHA 覆写这个钩子来旋转 q/k，forward 主体只写一份。
        return query, key

    def forward(
        self,
        x: Float[torch.Tensor, "... sequence_length d_model"],
        token_positions: Int[torch.Tensor, "... sequence_length"] | None = None,
    ) -> Float[torch.Tensor, "... sequence_length d_model"]:
        sequence_length = x.size(-2)

        query: Float[torch.Tensor, "... heads sequence_length d_k"] = self._split_heads(self.query_linear(x))
        key: Float[torch.Tensor, "... heads sequence_length d_k"] = self._split_heads(self.key_linear(x))
        value: Float[torch.Tensor, "... heads sequence_length d_v"] = self._split_heads(self.value_linear(x))

        query, key = self._position_encode(query, key, token_positions)

        # causal mask: 位置 i 只能 attend j <= i。形状 (seq, seq)，广播到所有 batch 和 head。
        mask = torch.tril(torch.ones(sequence_length, sequence_length, dtype=torch.bool, device=x.device))

        output = self.sdpa(query, key, value, mask)  # (..., heads, sequence_length, d_k)

        # 合并 head：transpose 后布局不再连续，必须用 reshape 而不是 view。
        output = output.transpose(-3, -2).reshape(*x.shape[:-1], self.d_model)
        return self.out_linear(output)


class RopeMHA(MultiHeadSelfAttention):
    """带 RoPE 的多头自注意力。

    直接继承 MultiHeadSelfAttention（而不是把它包一层），所以参数就是
    query_linear / key_linear / ... 本身，没有 `mha.mha.` 这种嵌套前缀。
    """

    def __init__(self, d_model: int, num_heads: int, theta: float, max_seq_len: int, device=None, dtype=None):
        super().__init__(d_model, num_heads, device=device, dtype=dtype)
        self.rope = RoPE(theta, self.d_k, max_seq_len, device=device)

    def _position_encode(
        self,
        query: Float[torch.Tensor, "... heads sequence_length d_k"],
        key: Float[torch.Tensor, "... heads sequence_length d_k"],
        token_positions: Int[torch.Tensor, "... sequence_length"] | None,
    ) -> tuple[Float[torch.Tensor, "... heads sequence_length d_k"], Float[torch.Tensor, "... heads sequence_length d_k"]]:
        if token_positions is None:
            token_positions = torch.arange(query.size(-2), device=query.device)
        # value 不转，只有 q/k 需要带上相对位置信息。
        return self.rope(query, token_positions), self.rope(key, token_positions)
