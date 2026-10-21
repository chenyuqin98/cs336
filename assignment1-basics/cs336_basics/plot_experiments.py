"""Render saved experiment metrics (run with the optional matplotlib package)."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary")
    parser.add_argument("--output", default="curves.svg")
    parser.add_argument("--x-axis", choices=["iteration", "tokens", "elapsed_seconds"], default="tokens")
    args = parser.parse_args()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    summary = json.loads(Path(args.summary).read_text())
    groups = [
        ("Learning rate", ["baseline", "lr-low", "lr-high", "lr-extreme"]),
        ("Batch size", ["baseline", "batch-1", "batch-64", "batch-128"]),
        ("Architecture", ["baseline", "no-norm", "no-norm-low-lr", "post-norm", "no-rope", "silu"]),
    ]
    figure, axes = plt.subplots(1, 3, figsize=(16, 4.5), layout="constrained")
    for axis, (title, names) in zip(axes, groups):
        for name in names:
            if name not in summary:
                continue
            records = [r for r in summary[name]["metrics"] if "validation_loss" in r]
            axis.plot(
                [r[args.x_axis] for r in records], [r["validation_loss"] for r in records], marker=".", label=name
            )
        axis.set(title=title, xlabel=args.x_axis, ylabel="Validation cross-entropy (nats/token)")
        if title == "Learning rate":
            axis.set_yscale("log")
        axis.grid(alpha=0.3)
        axis.legend(fontsize="small")
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination)
    plt.close(figure)


if __name__ == "__main__":
    main()
