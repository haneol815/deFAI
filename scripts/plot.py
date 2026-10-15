"""Plot test accuracy curves from one or more results/*/history.json files.

usage: python scripts/plot.py results/*/history.json -o results/acc.png
"""
import argparse
import json
import os

import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("files", nargs="+")
    p.add_argument("-o", "--out", default="results/acc.png")
    args = p.parse_args()

    for path in args.files:
        with open(path) as f:
            h = json.load(f)["history"]
        label = os.path.basename(os.path.dirname(path))
        plt.plot([x["round"] for x in h], [x["test_acc"] for x in h], label=label)

    plt.xlabel("Round")
    plt.ylabel("Test accuracy")
    plt.grid(alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    print(f"saved: {args.out}")


if __name__ == "__main__":
    main()
