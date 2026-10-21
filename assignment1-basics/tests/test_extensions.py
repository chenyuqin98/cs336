"""Checks for cross-chunk tokenization, generation, and architecture options."""

import io

import pytest
import torch

from cs336_basics.generation import generate
from cs336_basics.my_torch import TransformerLM
from cs336_basics.my_torch.rope import RoPE
from cs336_basics.tokenizer import Tokenizer
from cs336_basics.training import AdamW, load_checkpoint, save_checkpoint


def test_streaming_arbitrary_chunk_boundaries(tmp_path):
    vocab = {i: bytes([i]) for i in range(256)}
    merges = [(b"a", b"b"), (b"ab", b"c")]
    vocab.update({256: b"ab", 257: b"abc"})
    tokenizer = Tokenizer(vocab, merges, ["<eos>", "<eos><eos>"])
    text = "abc  abc\n\n我们 🥐 can't I'll <eos><eos>abc\t\n"
    expected = tokenizer.encode(text)
    for width in range(1, len(text) + 1):
        chunks = [text[i : i + width] for i in range(0, len(text), width)]
        assert list(tokenizer.encode_iterable(chunks)) == expected
    path = tmp_path / "tokenizer.json"
    tokenizer.save(path)
    loaded = Tokenizer.load(path)
    assert loaded.encode(text) == expected
    assert loaded.decode(expected) == text
    assert list(loaded.encode_iterable([])) == []


def test_batched_rope_positions():
    rope = RoPE(10000, 8, 32)
    x = torch.randn(2, 3, 5, 8)
    positions = torch.tensor([[0, 1, 2, 3, 4], [7, 8, 9, 10, 11]])
    expected = torch.stack([rope(x[b], positions[b]) for b in range(2)])
    torch.testing.assert_close(rope(x, positions), expected)
    torch.testing.assert_close(rope(x, positions).square().sum(-1), x.square().sum(-1))


@pytest.mark.parametrize(
    "norm_mode,use_rope,ffn_type",
    [
        ("pre", True, "swiglu"),
        ("post", True, "swiglu"),
        ("none", True, "swiglu"),
        ("pre", False, "swiglu"),
        ("pre", True, "silu"),
    ],
)
def test_architecture_causality_and_gradients(norm_mode, use_rope, ffn_type):
    model = TransformerLM(32, 8, 16, 1, 2, 32, 10000, norm_mode=norm_mode, use_rope=use_rope, ffn_type=ffn_type)
    tokens = torch.randint(0, 32, (2, 8))
    logits = model(tokens)
    modified = tokens.clone()
    modified[:, 4:] = (modified[:, 4:] + 1) % 32
    torch.testing.assert_close(logits[:, :4], model(modified)[:, :4])
    logits.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


class FixedLM(torch.nn.Module):
    def __init__(self, logits):
        super().__init__()
        self.register_buffer("logits", torch.tensor(logits))

    def forward(self, x):
        assert x.shape[1] <= 3
        return self.logits.expand(*x.shape, -1)


def test_sampling_nucleus_eos_and_context():
    model = FixedLM([4.0, 3.0, 2.0, 1.0])
    prompt = torch.tensor([[1, 2, 3, 1]])
    output = generate(model, prompt, 10, 3, top_p=0.1)
    assert output.shape == (1, 14)
    assert (output[:, 4:] == 0).all()
    assert model.training
    stopped = generate(model, prompt, 10, 3, temperature=0, eos_token_id=0)
    assert stopped.shape == (1, 5)
    with pytest.raises(ValueError):
        generate(model, prompt, 10, 3, top_p=0)


def test_checkpoint_resume_next_optimizer_step():
    torch.manual_seed(42)
    model = torch.nn.Linear(3, 2)
    optimizer = AdamW(model.parameters())
    x = torch.randn(4, 3)

    def step(model, optimizer):
        optimizer.zero_grad()
        model(x).square().mean().backward()
        optimizer.step()

    step(model, optimizer)
    buffer = io.BytesIO()
    save_checkpoint(model, optimizer, 1, buffer)
    other = torch.nn.Linear(3, 2)
    other_optimizer = AdamW(other.parameters())
    buffer.seek(0)
    assert load_checkpoint(buffer, other, other_optimizer) == 1
    step(model, optimizer)
    step(other, other_optimizer)
    for p, q in zip(model.parameters(), other.parameters()):
        torch.testing.assert_close(p, q, rtol=0, atol=0)
