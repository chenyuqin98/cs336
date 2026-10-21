"""Run reproducible learning-rate, batch-size, and architecture comparisons."""

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-data", required=True)
    parser.add_argument("--valid-data", required=True)
    parser.add_argument("--output", default="runs/comparisons")
    parser.add_argument("--vocab-size", type=int, default=10000)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--small-model", action="store_true", help="Short integration experiments, not handout-scale results"
    )
    args = parser.parse_args()
    variants = [
        ("baseline", []),
        ("lr-low", ["--lr", "0.0003"]),
        ("lr-high", ["--lr", "0.01"]),
        ("lr-extreme", ["--lr", "100"]),
        ("batch-1", ["--batch-size", "1"]),
        ("batch-64", ["--batch-size", "64"]),
        ("batch-128", ["--batch-size", "128"]),
        ("no-norm", ["--norm-mode", "none"]),
        ("no-norm-low-lr", ["--norm-mode", "none", "--lr", "0.0003"]),
        ("post-norm", ["--norm-mode", "post"]),
        ("no-rope", ["--no-rope"]),
        ("silu", ["--ffn-type", "silu", "--d-ff", "576" if args.small_model else "2048"]),
    ]
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    summary = {}
    for name, extra in variants:
        output = root / name
        command = [
            sys.executable,
            "-m",
            "cs336_basics.cli",
            "train",
            "--train-data",
            args.train_data,
            "--valid-data",
            args.valid_data,
            "--output",
            str(output),
            "--vocab-size",
            str(args.vocab_size),
            "--steps",
            str(args.steps),
            "--warmup-iters",
            str(min(100, args.steps // 10)),
            "--device",
            args.device,
            "--lr",
            "0.001",
            "--min-lr",
            "0.00003",
            "--eval-every",
            str(max(1, args.steps // 5)),
            "--eval-batches",
            "5",
            "--log-every",
            str(max(1, args.steps // 10)),
        ]
        if args.small_model:
            command += [
                "--d-model",
                "128",
                "--d-ff",
                "384",
                "--num-layers",
                "2",
                "--num-heads",
                "4",
                "--context-length",
                "128",
                "--batch-size",
                "16",
            ]
        command += extra
        result = subprocess.run(command, check=False)
        metrics = output / "metrics.jsonl"
        records = [json.loads(line) for line in metrics.read_text().splitlines()] if metrics.exists() else []
        summary[name] = dict(command=command[3:], returncode=result.returncode, metrics=records)
        (root / "summary.json").write_text(json.dumps(summary, indent=2))
        print(f"{name}: return code {result.returncode}", flush=True)
    if any(v["returncode"] for k, v in summary.items() if k != "lr-extreme"):
        raise SystemExit("One or more experiments failed; inspect summary.json")


if __name__ == "__main__":
    main()
