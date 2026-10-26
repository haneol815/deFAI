"""Summarise results/*/history.json: one row per configuration.

Runs whose configuration differs only in ``seed`` (and ``output-dir``) are grouped, so repeated
runs show mean ± std over seeds.

usage: python scripts/compare.py results/*/history.json
       python scripts/compare.py results/*/history.json --target 0.95 --csv table.csv

columns: runs, final / best test accuracy, first round reaching --target accuracy (mean over the
runs that reached it, "-" if none did) and, for differentially private runs, the final epsilon.
"""
import argparse
import csv
import json
import statistics
import sys

IGNORED = ("seed", "output-dir")
IN_LABEL = ("strategy", "dataset", "partitioner", "attack", "num-malicious", "dp", "dp-side", "dp-noise-multiplier")


def load(path):
    with open(path) as f:
        data = json.load(f)
    return data["config"], data["history"], data.get("privacy")


def label(config):
    parts = [str(config["strategy"]), str(config["dataset"]), str(config["partitioner"])]
    if config.get("attack", "none") != "none" and config.get("num-malicious", 0) > 0:
        parts.append(f"{config['attack']}x{config['num-malicious']}")
    if config.get("dp", "none") != "none":
        parts.append(f"dp-{config['dp']}-{config['dp-side']}(z={config['dp-noise-multiplier']})")
    return " ".join(parts)


def run_stats(history, target):
    accs = [(r["round"], r["test_acc"]) for r in history if "test_acc" in r]
    reached = next((rnd for rnd, acc in accs if rnd > 0 and acc >= target), None)
    return {"final": accs[-1][1], "best": max(a for _, a in accs), "reached": reached}


def mean_std(values):
    if not values:
        return None, None
    return statistics.fmean(values), (statistics.stdev(values) if len(values) > 1 else 0.0)


def fmt(mean, std, runs):
    if mean is None:
        return "-"
    return f"{mean:.3f}" if runs == 1 else f"{mean:.3f} ± {std:.3f}"


def varying_settings(configs):
    """Settings that take different values across ``configs`` and are not already in the label."""
    keys = {k for c in configs for k in c if k not in IGNORED and k not in IN_LABEL}
    return sorted(k for k in keys if len({json.dumps(c.get(k)) for c in configs}) > 1)


def unique_labels(configs):
    """Label per config; configs with the same label get the settings that tell them apart."""
    base = [label(c) for c in configs]
    labels = list(base)
    for text in set(base):
        same = [i for i, b in enumerate(base) if b == text]
        if len(same) > 1:
            extra = varying_settings([configs[i] for i in same])
            for i in same:
                labels[i] = " ".join([text, *(f"{k}={configs[i].get(k)}" for k in extra)])
    return labels


def summarise(paths, target):
    groups = {}
    for path in paths:
        config, history, privacy = load(path)
        key = json.dumps({k: v for k, v in config.items() if k not in IGNORED}, sort_keys=True)
        groups.setdefault(key, {"config": config, "runs": []})["runs"].append((run_stats(history, target), privacy))
    groups = list(groups.values())
    names = unique_labels([g["config"] for g in groups])
    rows = []
    for name, group in zip(names, groups):
        runs = group["runs"]
        finals = [s["final"] for s, _ in runs]
        bests = [s["best"] for s, _ in runs]
        reached = [s["reached"] for s, _ in runs if s["reached"] is not None]
        eps = [p["epsilon"] for _, p in runs if p]
        rows.append({
            "run": name,
            "runs": len(runs),
            "final": fmt(*mean_std(finals), len(runs)),
            "best": fmt(*mean_std(bests), len(runs)),
            "rounds_to_target": "-" if not reached else f"{statistics.fmean(reached):.1f} ({len(reached)}/{len(runs)})",
            "epsilon": "-" if not eps else f"{statistics.fmean(eps):.4g}",
        })
    return sorted(rows, key=lambda r: r["run"])


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("files", nargs="+")
    p.add_argument("--target", type=float, default=0.9, help="accuracy for the rounds-to-target column")
    p.add_argument("--csv", help="also write the table to this CSV file")
    args = p.parse_args(argv)

    rows = summarise(args.files, args.target)
    columns = ["run", "runs", "final", "best", "rounds_to_target", "epsilon"]
    headers = {"run": "run", "runs": "runs", "final": "final acc", "best": "best acc",
               "rounds_to_target": f"rounds to {args.target:g}", "epsilon": "epsilon"}
    widths = {c: max(len(headers[c]), *(len(str(r[c])) for r in rows)) for c in columns}
    print("  ".join(headers[c].ljust(widths[c]) for c in columns))
    for r in rows:
        print("  ".join(str(r[c]).ljust(widths[c]) for c in columns))
    if args.csv:
        with open(args.csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
        print(f"saved: {args.csv}", file=sys.stderr)


if __name__ == "__main__":
    main()
