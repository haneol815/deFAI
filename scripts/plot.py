"""Plot a metric per round from one or more results/*/history.json files.

usage: python scripts/plot.py results/*/history.json -o results/acc.png
       python scripts/plot.py results/*/history.json -m dp-epsilon -o results/eps.png

Any key of a history row can be plotted, e.g. test_acc (default), test_loss, eval_acc,
train_loss, dp-epsilon, dp-clip-norm. Rounds without a (finite) value for the metric are skipped.
"""
import argparse
import json
import os

import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("files", nargs="+")
    p.add_argument("-o", "--out", default="results/acc.png")
    p.add_argument("-m", "--metric", default="test_acc")
    args = p.parse_args()

    for path in args.files:
        with open(path) as f:
            h = json.load(f)["history"]
        label = os.path.basename(os.path.dirname(path))
        rows = [x for x in h if x.get(args.metric) is not None]  # null = infinite / missing
        plt.plot([x["round"] for x in rows], [x[args.metric] for x in rows], label=label)

    plt.xlabel("Round")
    plt.ylabel(args.metric)
    plt.grid(alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    print(f"saved: {args.out}")


if __name__ == "__main__":
    main()
