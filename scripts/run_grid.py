"""Run a grid of experiments (variants x seeds) with `flwr run`, resumably, and summarise them.

usage: python scripts/run_grid.py --output-dir results/noniid --seeds 1 2 3 --num-supernodes 10 \\
           --base "dataset='synthetic' partitioner='dirichlet' dirichlet-alpha=0.1 num-server-rounds=20" \\
           --vary "strategy='fedavg'" "strategy='scaffold'" "strategy='fednova'"

Every ``--vary`` entry is a run-config fragment that is appended to ``--base``; ``seed`` and ``output-dir``
are added per run. A run whose results already exist in ``--output-dir`` (same settings and seed) is skipped,
so an interrupted grid continues where it stopped. ``flwr run`` exits with 0 even when the ServerApp fails
(for example on an invalid config), so a run counts as successful only if it wrote a new history.json.
Logs go to ``<output-dir>/logs/``. At the end the grid is summarised with scripts/compare.py.

Set FLWR_DISABLE_RUNTIME_DEPENDENCY_INSTALLATION=1 (done here unless --install-deps is given) to reuse the
current environment instead of building a new one for every run.
"""
import argparse
import glob
import json
import os
import re
import shlex
import subprocess
import sys
import tomllib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import compare  # noqa: E402

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOKEN = re.compile(r"""(?:[^\s'"]|'[^']*'|"[^"]*")+""")  # whitespace-separated, quotes may contain spaces


def parse_assignments(fragment):
    """"a=1 b='x' c=true" -> {"a": 1, "b": "x", "c": True} (values use TOML syntax, like --run-config)."""
    out = {}
    for token in TOKEN.findall(fragment):
        key, sep, value = token.partition("=")
        if not sep or not key:
            raise ValueError(f"expected key=value, got {token!r}")
        out[key] = tomllib.loads(f"v = {value}")["v"]
    return out


def format_run_config(assignments):
    """Inverse of parse_assignments: strings as TOML literal strings, everything else as JSON (true/false, numbers)."""
    return " ".join(f"{k}={v!r}" if isinstance(v, str) else f"{k}={json.dumps(v)}" for k, v in assignments.items())


def histories(output_dir):
    """{path: modification time} of every history.json below output_dir."""
    paths = glob.glob(os.path.join(output_dir, "*", "history.json"))
    return {path: os.stat(path).st_mtime_ns for path in paths}


def read_config(path):
    """The config a finished run was made with, or None if the file is unreadable or half written."""
    try:
        with open(path) as f:
            return json.load(f)["config"]
    except (OSError, ValueError, KeyError):
        return None


def already_done(output_dir, assignments):
    """Whether a finished run with exactly these settings (including the seed) exists."""
    for path in histories(output_dir):
        config = read_config(path)
        if config and all(k in config and config[k] == v for k, v in assignments.items() if k != "output-dir"):
            return True
    return False


def build_command(num_supernodes, assignments, project_dir=PROJECT_DIR):
    return ["flwr", "run", project_dir, "--stream", "--federation-config", f"num-supernodes={num_supernodes}",
            "--run-config", format_run_config(assignments)]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--base", default="", help="run-config fragment shared by every run")
    p.add_argument("--vary", nargs="+", default=[""], help="run-config fragments, one run per fragment and seed")
    p.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    p.add_argument("--num-supernodes", type=int, default=10)
    p.add_argument("--timeout", type=float, default=1800, help="seconds allowed per run")
    p.add_argument("--no-resume", action="store_true", help="run everything again even if results exist")
    p.add_argument("--install-deps", action="store_true", help="let flwr build a fresh environment for each run")
    p.add_argument("--dry-run", action="store_true", help="print the commands only")
    p.add_argument("--target", type=float, default=0.9, help="accuracy for the rounds-to-target column")
    args = p.parse_args(argv)

    base = parse_assignments(args.base)
    variants = [parse_assignments(v) for v in args.vary]
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(os.path.join(output_dir, "logs"), exist_ok=True)
    env = dict(os.environ)
    if not args.install_deps:
        env["FLWR_DISABLE_RUNTIME_DEPENDENCY_INSTALLATION"] = "1"

    jobs = [(seed, vary) for seed in args.seeds for vary in variants]  # seed-major: partial grids stay comparable
    failed, skipped, done = [], 0, 0
    for number, (seed, vary) in enumerate(jobs, 1):
        assignments = {**base, **vary, "seed": seed}
        name = f"{number:03d}_seed{seed}"
        if not args.no_resume and already_done(output_dir, assignments):
            skipped += 1
            print(f"[{number}/{len(jobs)}] skip (done): {format_run_config(assignments)}")
            continue
        command = build_command(args.num_supernodes, {**assignments, "output-dir": output_dir})
        print(f"[{number}/{len(jobs)}] {format_run_config(assignments)}")
        if args.dry_run:
            print("   ", " ".join(shlex.quote(c) for c in command))
            continue
        before = histories(output_dir)
        log_path = os.path.join(output_dir, "logs", f"{name}.log")
        try:
            with open(log_path, "w") as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=env, timeout=args.timeout, check=False)
        except subprocess.TimeoutExpired:
            print(f"    timed out after {args.timeout:g}s (see {log_path})")
        if histories(output_dir) != before:  # a new file, or an old one rewritten
            done += 1
        else:
            failed.append(number)
            print(f"    FAILED: no results were written (see {log_path})")

    print(f"\n{done} run, {skipped} skipped, {len(failed)} failed" + (f" (runs {failed})" if failed else ""))
    finished = sorted(path for path in histories(output_dir) if read_config(path))
    if not args.dry_run and finished:
        compare.main([*finished, "--target", str(args.target)])
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
