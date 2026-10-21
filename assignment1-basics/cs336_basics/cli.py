"""Command-line tools for tokenizer preparation, training, and generation."""

import argparse
import itertools
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from .generation import generate
from .my_torch import TransformerLM
from .tokenizer import Tokenizer, train_bpe
from .training import AdamW, cross_entropy, get_batch, get_lr_cosine_schedule, gradient_clipping


def select_device(name):
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def train_tokenizer(args):
    started = time.perf_counter()
    vocab, merges = train_bpe(args.input, args.vocab_size, args.special_token)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Tokenizer(vocab, merges, args.special_token).save(args.output)
    print(
        json.dumps(
            dict(
                vocab_size=len(vocab),
                seconds=time.perf_counter() - started,
                longest_token=repr(max(vocab.values(), key=len)),
            )
        )
    )


def tokenize(args):
    tokenizer = Tokenizer.load(args.tokenizer)
    dtype = np.dtype(args.dtype)
    if max(tokenizer.vocab) > np.iinfo(dtype).max:
        raise ValueError("Token IDs do not fit in the selected dtype")
    count = 0
    started = time.perf_counter()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.input, encoding="utf-8") as src, open(args.output, "wb") as out:
        chunks = iter(lambda: src.read(65536), "")
        ids = tokenizer.encode_iterable(chunks)
        while batch := list(itertools.islice(ids, 65536)):
            np.asarray(batch, dtype=dtype).tofile(out)
            count += len(batch)
    print(json.dumps(dict(tokens=count, seconds=time.perf_counter() - started, dtype=str(dtype))))


def rng_state(device):
    state = np.random.get_state()
    result = dict(torch=torch.get_rng_state(), numpy=[state[0], state[1].tolist(), *state[2:]])
    if device.startswith("cuda"):
        result["cuda"] = torch.cuda.get_rng_state_all()
    elif device.startswith("mps"):
        result["mps"] = torch.mps.get_rng_state()
    return result


def restore_rng(state):
    torch.set_rng_state(state["torch"])
    name, keys, pos, gaussian, cached = state["numpy"]
    np.random.set_state((name, np.asarray(keys, dtype=np.uint32), pos, gaussian, cached))
    if "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])
    if "mps" in state:
        torch.mps.set_rng_state(state["mps"])


@torch.no_grad()
def evaluate(model, data, batch_size, context_length, device, batches):
    was_training = model.training
    model.eval()
    # Validation must not change which minibatches the training loop draws.
    state = np.random.get_state()
    np.random.seed(0)
    try:
        losses = []
        for _ in range(batches):
            x, y = get_batch(data, batch_size, context_length, device)
            losses.append(cross_entropy(model(x), y).item())
        return sum(losses) / len(losses)
    finally:
        np.random.set_state(state)
        model.train(was_training)


def train(args):
    if args.steps <= args.warmup_iters or min(args.eval_every, args.eval_batches, args.checkpoint_every) <= 0:
        raise ValueError("Require steps > warmup_iters and positive evaluation/checkpoint intervals")
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = select_device(args.device)
    train_data = np.memmap(args.train_data, dtype=args.dtype, mode="r")
    valid_data = np.memmap(args.valid_data, dtype=args.dtype, mode="r")
    config = dict(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
        norm_mode=args.norm_mode,
        use_rope=not args.no_rope,
        ffn_type=args.ffn_type,
    )
    checkpoint = torch.load(args.resume, map_location="cpu", weights_only=True) if args.resume else None
    if checkpoint:
        config = checkpoint["config"]
    for data in (train_data, valid_data):
        if len(data) <= config["context_length"]:
            raise ValueError("Dataset must contain more than context_length tokens")
        for start in range(0, len(data), 1 << 20):
            if int(data[start : start + (1 << 20)].max()) >= config["vocab_size"]:
                raise ValueError("Dataset has token IDs outside the model vocabulary")
    model = TransformerLM(**config, device=device)
    optimizer = AdamW(
        model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2), eps=args.eps, weight_decay=args.weight_decay
    )
    iteration, elapsed = 0, 0.0
    if checkpoint:
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        iteration = checkpoint["iteration"]
        elapsed = checkpoint["elapsed_seconds"]
        restore_rng(checkpoint["rng"])
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(json.dumps(dict(model=config, training=vars(args)), indent=2, default=str))
    started = time.perf_counter()

    def save():
        payload = dict(
            model=model.state_dict(),
            optimizer=optimizer.state_dict(),
            iteration=iteration,
            config=config,
            rng=rng_state(device),
            elapsed_seconds=elapsed + time.perf_counter() - started,
        )
        temporary = output / "checkpoint.tmp"
        torch.save(payload, temporary)
        temporary.replace(output / "checkpoint.pt")

    with (output / "metrics.jsonl").open("a" if checkpoint else "w") as metrics:

        def log(record):
            record.update(
                iteration=iteration,
                tokens=iteration * args.batch_size * config["context_length"],
                elapsed_seconds=elapsed + time.perf_counter() - started,
            )
            metrics.write(json.dumps(record, allow_nan=False) + "\n")
            metrics.flush()
            print(json.dumps(record), flush=True)

        initial_loss = evaluate(model, valid_data, args.batch_size, config["context_length"], device, args.eval_batches)
        log(dict(validation_loss=initial_loss))
        model.train()
        while iteration < args.steps:
            lr = get_lr_cosine_schedule(iteration, args.lr, args.min_lr, args.warmup_iters, args.steps)
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
            x, y = get_batch(train_data, args.batch_size, config["context_length"], device)
            loss = cross_entropy(model(x), y)
            if not torch.isfinite(loss).item():
                log(dict(diverged=True, learning_rate=lr))
                save()
                raise RuntimeError("Non-finite training loss")
            loss.backward()
            gradient_clipping(model.parameters(), args.max_grad_norm)
            optimizer.step()
            iteration += 1
            if iteration % args.log_every == 0 or iteration == args.steps:
                log(dict(train_loss=loss.item(), learning_rate=lr))
            if iteration % args.eval_every == 0 or iteration == args.steps:
                val = evaluate(model, valid_data, args.batch_size, config["context_length"], device, args.eval_batches)
                log(dict(validation_loss=val, perplexity=math.exp(min(val, 80))))
            if iteration % args.checkpoint_every == 0 or iteration == args.steps:
                save()


def sample(args):
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    device = select_device(args.device)
    tokenizer = Tokenizer.load(args.tokenizer)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = TransformerLM(**checkpoint["config"], device=device)
    model.load_state_dict(checkpoint["model"])
    ids = tokenizer.encode(args.prompt)
    if not ids:
        raise ValueError("Prompt must encode to at least one token")
    eos = tokenizer.byte_ids.get(args.eos_token.encode("utf-8"))
    result = generate(
        model,
        torch.tensor([ids], device=device),
        args.max_new_tokens,
        checkpoint["config"]["context_length"],
        args.temperature,
        args.top_p,
        eos,
    )
    print(tokenizer.decode(result[0].tolist()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("train-tokenizer")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--vocab-size", type=int, default=10000)
    p.add_argument("--special-token", action="append", default=None)
    p = commands.add_parser("tokenize")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--dtype", choices=["uint16", "uint32"], default="uint16")
    p = commands.add_parser("train")
    p.add_argument("--train-data", required=True)
    p.add_argument("--valid-data", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--resume")
    p.add_argument("--dtype", choices=["uint16", "uint32"], default="uint16")
    p.add_argument("--vocab-size", type=int, default=10000)
    p.add_argument("--context-length", type=int, default=256)
    p.add_argument("--d-model", type=int, default=512)
    p.add_argument("--d-ff", type=int, default=1344)
    p.add_argument("--num-layers", type=int, default=4)
    p.add_argument("--num-heads", type=int, default=16)
    p.add_argument("--rope-theta", type=float, default=10000)
    p.add_argument("--norm-mode", choices=["pre", "post", "none"], default="pre")
    p.add_argument("--no-rope", action="store_true")
    p.add_argument("--ffn-type", choices=["swiglu", "silu"], default="swiglu")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--warmup-iters", type=int, default=100)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--min-lr", type=float, default=3e-5)
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.95)
    p.add_argument("--eps", type=float, default=1e-8)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--eval-batches", type=int, default=10)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--checkpoint-every", type=int, default=500)
    p = commands.add_parser("generate")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--prompt", default="Once upon a time")
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--eos-token", default="<|endoftext|>")
    for name in ("train", "generate"):
        p = commands.choices[name]
        p.add_argument("--device", default="auto")
        p.add_argument("--seed", type=int, default=42)
        p.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.command == "train-tokenizer":
        args.special_token = args.special_token or ["<|endoftext|>"]
    handlers = {"train-tokenizer": train_tokenizer, "tokenize": tokenize, "train": train, "generate": sample}
    handlers[args.command](args)


if __name__ == "__main__":
    main()
