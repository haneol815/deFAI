import importlib.util
import json
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_run(tmp_path, name, accs, seed=0, privacy=None, columns=None, **config):
    cfg = {"strategy": "fedavg", "dataset": "synthetic", "partitioner": "iid", "seed": seed,
           "output-dir": str(tmp_path), "attack": "none", "num-malicious": 0, "dp": "none", **config}
    history = [{"round": i, "test_acc": a} for i, a in enumerate(accs)]
    for key, values in (columns or {}).items():  # extra per-round metrics, one value (or None) per round
        for row, value in zip(history, values):
            row[key] = value
    path = tmp_path / name / "history.json"
    path.parent.mkdir()
    path.write_text(json.dumps({"config": cfg, "history": history, **({"privacy": privacy} if privacy else {})}))
    return str(path)


def test_compare_groups_seeds_and_reports_mean_std(tmp_path, capsys):
    compare = load_script("compare")
    paths = [
        write_run(tmp_path, "a0", [0.1, 0.5, 0.9], seed=0),
        write_run(tmp_path, "a1", [0.1, 0.7, 0.95], seed=1),
        write_run(tmp_path, "b", [0.1, 0.2, 0.3], strategy="scaffold"),
    ]
    rows = {r["run"]: r for r in compare.summarise(paths, target=0.9)}
    avg = rows["fedavg synthetic iid"]
    assert avg["runs"] == 2 and avg["final"] == "0.925 ± 0.035" and avg["best"] == "0.925 ± 0.035"
    assert avg["rounds_to_target"] == "2.0 (2/2)"
    assert rows["scaffold synthetic iid"]["rounds_to_target"] == "-"
    compare.main([*paths, "--target", "0.9", "--csv", str(tmp_path / "t.csv")])
    out = capsys.readouterr().out
    assert "fedavg synthetic iid" in out and "rounds to 0.9" in out
    assert (tmp_path / "t.csv").read_text().startswith("run,runs,final")


def test_compare_adds_columns_for_any_history_key(tmp_path, capsys):
    compare = load_script("compare")
    paths = [
        write_run(tmp_path, "a0", [0.1, 0.5], seed=0, columns={"eval_acc_min": [0.0, 0.2], "comm-mb-total": [0.0, 1.5]}),
        # the last round has no client-side value (null): the last recorded one is used
        write_run(tmp_path, "a1", [0.1, 0.7, 0.8], seed=1, columns={"eval_acc_min": [0.1, 0.4, None], "comm-mb-total": [0.0, 2.0, 2.5]}),
        write_run(tmp_path, "b", [0.1, 0.2], strategy="scaffold"),  # does not record the keys at all
    ]
    rows = {r["run"]: r for r in compare.summarise(paths, 0.9, extra=["eval_acc_min", "comm-mb-total"])}
    assert rows["fedavg synthetic iid"]["eval_acc_min"] == "0.300 ± 0.141"
    assert rows["fedavg synthetic iid"]["comm-mb-total"] == "2.000 ± 0.707"  # 1.5 and 2.5
    assert rows["scaffold synthetic iid"]["eval_acc_min"] == "-"
    compare.main([*paths, "--extra", "comm-mb-total", "--csv", str(tmp_path / "t.csv")])
    assert "comm-mb-total" in capsys.readouterr().out.splitlines()[0]
    assert (tmp_path / "t.csv").read_text().splitlines()[0].endswith(",comm-mb-total")


def test_compare_can_print_a_markdown_table(tmp_path, capsys):
    compare = load_script("compare")
    path = write_run(tmp_path, "a", [0.1, 0.8], columns={"eval_acc_min": [0.0, 0.5]})
    compare.main([path, "--extra", "eval_acc_min", "--markdown"])
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 3 and all(line.startswith("| ") and line.endswith(" |") for line in lines)
    assert lines[0].split("|")[1].strip() == "run" and lines[0].split("|")[-2].strip() == "eval_acc_min"
    assert set(lines[1].replace("|", "").replace(" ", "")) == {"-"}  # separator row
    assert lines[2].split("|")[1].strip() == "fedavg synthetic iid" and lines[2].split("|")[-2].strip() == "0.500"


def test_compare_tells_apart_groups_that_differ_only_in_other_settings(tmp_path):
    compare = load_script("compare")
    paths = [
        write_run(tmp_path, "r5", [0.1, 0.5], **{"num-server-rounds": 5, "learning-rate": 0.01}),
        write_run(tmp_path, "r10", [0.1, 0.5], **{"num-server-rounds": 10, "learning-rate": 0.01}),
    ]
    runs = [r["run"] for r in compare.summarise(paths, 0.9)]
    assert runs == ["fedavg synthetic iid num-server-rounds=10", "fedavg synthetic iid num-server-rounds=5"]
    # a setting shared by every group never clutters the label
    assert all("learning-rate" not in r for r in runs)


def test_compare_shows_attack_dp_and_epsilon(tmp_path):
    compare = load_script("compare")
    path = write_run(tmp_path, "c", [0.1, 0.8], attack="sign-flip", **{"num-malicious": 2, "dp": "fixed", "dp-side": "server", "dp-noise-multiplier": 1.0},
                     privacy={"epsilon": 12.5})
    row = compare.summarise([path], 0.9)[0]
    assert "sign-flipx2" in row["run"] and "dp-fixed-server(z=1.0)" in row["run"] and row["epsilon"] == "12.5"
    assert row["final"] == "0.800"  # a single run shows no ± term


def test_plot_script_accepts_any_history_key(tmp_path):
    pytest.importorskip("matplotlib")
    plot = load_script("plot")
    path = write_run(tmp_path, "d", [0.1, 0.5])
    data = json.loads(Path(path).read_text())
    data["history"][1]["dp-epsilon"] = 3.0
    Path(path).write_text(json.dumps(data))
    out = tmp_path / "eps.png"
    import sys
    old = sys.argv
    sys.argv = ["plot.py", path, "-m", "dp-epsilon", "-o", str(out)]
    try:
        plot.main()
    finally:
        sys.argv = old
    assert out.exists() and out.stat().st_size > 0


def test_scripts_handle_null_epsilon_from_unbounded_runs(tmp_path):
    compare, plot = load_script("compare"), load_script("plot")
    path = write_run(tmp_path, "inf", [0.1, 0.5], dp="fixed", **{"dp-side": "server", "dp-noise-multiplier": 0.0},
                     privacy={"epsilon": None, "epsilon_unbounded": True})
    data = json.loads(Path(path).read_text())
    data["history"][1]["dp-epsilon"] = None
    Path(path).write_text(json.dumps(data))
    assert compare.summarise([path], 0.9)[0]["epsilon"] == "inf"
    pytest.importorskip("matplotlib")
    import sys

    old, sys.argv = sys.argv, ["plot.py", path, "-m", "dp-epsilon", "-o", str(tmp_path / "e.png")]
    try:
        plot.main()  # all values null -> an empty curve, not a crash
    finally:
        sys.argv = old


def test_partition_stats_prints_a_table_and_the_skew(capsys):
    stats = load_script("partition_stats")
    stats.main(["--dataset", "synthetic", "--partitioner", "dirichlet", "--alpha", "0.1", "--num-partitions", "5"])
    out = capsys.readouterr().out.splitlines()
    assert out[0] == "dataset=synthetic partitioner=dirichlet alpha=0.1 clients=5 seed=42"
    rows = [line.split() for line in out[2:7]]
    assert [r[0] for r in rows] == ["0", "1", "2", "3", "4"] and sum(int(r[1]) for r in rows) == 3000
    assert out[-1].startswith("mean max-class share:") and float(out[-1].split(":")[1].split()[0]) > 0.4  # alpha 0.1 is skewed
    stats.main(["--partitioner", "iid", "--num-partitions", "5"])
    assert "partitioner=iid clients=5" in capsys.readouterr().out  # no alpha shown for IID


def test_partition_stats_can_save_a_heat_map(tmp_path, capsys):
    pytest.importorskip("matplotlib")
    stats = load_script("partition_stats")
    png = tmp_path / "p.png"
    stats.main(["--num-partitions", "4", "--plot", str(png)])
    assert png.exists() and png.stat().st_size > 1000


def test_run_grid_parses_run_config_fragments():
    grid = load_script("run_grid")
    parsed = grid.parse_assignments("strategy='fedavg' lr=0.5 rounds=3 on=true off=false path='a b'")
    assert parsed == {"strategy": "fedavg", "lr": 0.5, "rounds": 3, "on": True, "off": False, "path": "a b"}
    assert grid.parse_assignments("") == {}
    assert grid.format_run_config({"strategy": "fedavg", "lr": 1e-05, "on": True, "n": 3}) == "strategy='fedavg' lr=1e-05 on=true n=3"
    with pytest.raises(ValueError, match="key=value"):
        grid.parse_assignments("strategy")
    with pytest.raises(ValueError, match="key=value"):
        grid.parse_assignments("=3")


@pytest.fixture
def fake_flwr(monkeypatch):
    """Replace `flwr run` by a function that writes the history.json a successful ServerApp would write."""
    import tomllib

    calls = []

    def install(grid, writes=True):
        def run(command, stdout, stderr, env, timeout, check):
            run_config = command[command.index("--run-config") + 1]
            cfg = tomllib.loads(run_config.replace(" ", "\n"))  # independent of run_grid's own parser
            calls.append({"command": command, "env": env, "timeout": timeout, "cfg": cfg})
            full = {"strategy": "fedavg", "dataset": "synthetic", "partitioner": "iid", "attack": "none", "num-malicious": 0, "dp": "none", **cfg}  # what the server records
            stdout.write("fake flwr output\n")
            if writes:
                out = Path(cfg["output-dir"]) / f"run{len(calls)}"
                out.mkdir(parents=True)
                (out / "history.json").write_text(json.dumps({"config": full, "history": [{"round": 1, "test_acc": 0.5}]}))
            return None

        monkeypatch.setattr(grid.subprocess, "run", run)
        return calls

    return install


GRID_ARGS = ["--seeds", "1", "2", "--num-supernodes", "4", "--base", "dataset='synthetic' num-server-rounds=3",
             "--vary", "strategy='fedavg'", "strategy='scaffold' qfedavg-q=0.5"]


def test_run_grid_runs_every_variant_and_seed_and_resumes(tmp_path, fake_flwr, capsys):
    grid = load_script("run_grid")
    calls = fake_flwr(grid)
    assert grid.main(["--output-dir", str(tmp_path), *GRID_ARGS]) == 0
    assert [(c["cfg"]["seed"], c["cfg"]["strategy"]) for c in calls] == [(1, "fedavg"), (1, "scaffold"), (2, "fedavg"), (2, "scaffold")]
    first = calls[1]
    assert first["command"][:3] == ["flwr", "run", grid.PROJECT_DIR] and "--stream" in first["command"]
    assert first["command"][first["command"].index("--federation-config") + 1] == "num-supernodes=4"
    assert first["cfg"] == {"dataset": "synthetic", "num-server-rounds": 3, "strategy": "scaffold", "qfedavg-q": 0.5,
                            "seed": 1, "output-dir": str(tmp_path)}
    assert first["env"]["FLWR_DISABLE_RUNTIME_DEPENDENCY_INSTALLATION"] == "1" and first["timeout"] == 1800
    assert (tmp_path / "logs" / "002_seed1.log").read_text() == "fake flwr output\n"
    out = capsys.readouterr().out
    assert "4 run, 0 skipped, 0 failed" in out and "fedavg synthetic" in out  # summarised with compare.py

    # the same grid again does nothing; one more seed only runs the new seed
    assert grid.main(["--output-dir", str(tmp_path), *GRID_ARGS]) == 0
    assert len(calls) == 4 and "0 run, 4 skipped" in capsys.readouterr().out
    assert grid.main(["--output-dir", str(tmp_path), *GRID_ARGS[:1], "1", "2", "3", *GRID_ARGS[3:]]) == 0
    assert [c["cfg"]["seed"] for c in calls[4:]] == [3, 3]
    # --no-resume repeats everything, even though the history.json files get overwritten or added
    assert grid.main(["--output-dir", str(tmp_path), "--no-resume", *GRID_ARGS]) == 0
    assert len(calls) == 10 and "4 run, 0 skipped" in capsys.readouterr().out


def test_run_grid_install_deps_leaves_the_environment_alone(tmp_path, fake_flwr, monkeypatch):
    grid = load_script("run_grid")
    monkeypatch.delenv("FLWR_DISABLE_RUNTIME_DEPENDENCY_INSTALLATION", raising=False)
    calls = fake_flwr(grid)
    grid.main(["--output-dir", str(tmp_path), "--seeds", "1", "--install-deps", "--timeout", "7"])
    assert "FLWR_DISABLE_RUNTIME_DEPENDENCY_INSTALLATION" not in calls[0]["env"] and calls[0]["timeout"] == 7


def test_run_grid_reports_runs_that_wrote_no_results(tmp_path, fake_flwr, capsys):
    # `flwr run` exits with 0 even when the ServerApp fails, so only a new history.json counts as success
    grid = load_script("run_grid")
    calls = fake_flwr(grid, writes=False)
    assert grid.main(["--output-dir", str(tmp_path), *GRID_ARGS]) == 1
    out = capsys.readouterr().out
    assert len(calls) == 4 and "0 run, 0 skipped, 4 failed (runs [1, 2, 3, 4])" in out and "FAILED" in out
    assert "compare" not in out  # nothing to summarise


def test_run_grid_survives_a_timeout_and_unreadable_results(tmp_path, monkeypatch, capsys):
    grid = load_script("run_grid")

    def hang(command, **kwargs):
        raise grid.subprocess.TimeoutExpired(command, 1)

    monkeypatch.setattr(grid.subprocess, "run", hang)
    (tmp_path / "broken").mkdir()
    (tmp_path / "broken" / "history.json").write_text("{ half written")  # must not stop a resumed grid
    assert grid.main(["--output-dir", str(tmp_path), "--seeds", "1", "--timeout", "1"]) == 1
    assert "timed out after 1s" in capsys.readouterr().out


def test_run_grid_dry_run_only_prints_the_commands(tmp_path, fake_flwr, capsys):
    grid = load_script("run_grid")
    calls = fake_flwr(grid)
    assert grid.main(["--output-dir", str(tmp_path), "--dry-run", *GRID_ARGS]) == 0
    out = capsys.readouterr().out
    assert calls == [] and out.count("flwr run") == 4 and "seed=2" in out
    assert "0 run, 0 skipped, 0 failed" in out and not list(tmp_path.glob("*/history.json"))


def test_run_grid_only_skips_runs_with_the_same_seed_and_settings(tmp_path, fake_flwr):
    grid = load_script("run_grid")
    write_run(tmp_path, "old", [0.5], seed=1, **{"num-server-rounds": 3})
    assert grid.already_done(str(tmp_path), {"strategy": "fedavg", "seed": 1, "num-server-rounds": 3})
    assert not grid.already_done(str(tmp_path), {"strategy": "fedavg", "seed": 2, "num-server-rounds": 3})
    assert not grid.already_done(str(tmp_path), {"strategy": "fedavg", "seed": 1, "num-server-rounds": 5})
    assert not grid.already_done(str(tmp_path), {"strategy": "fedavg", "seed": 1, "not-in-config": 1})


def test_run_grid_counts_a_rewritten_history_as_success(tmp_path, monkeypatch, capsys):
    # two runs that get the same results directory name: the second overwrites the first one's history.json
    import os

    grid = load_script("run_grid")
    path = tmp_path / "same_name" / "history.json"
    path.parent.mkdir()
    cfg = {"strategy": "fedavg", "dataset": "synthetic", "partitioner": "iid", "attack": "none", "num-malicious": 0, "dp": "none"}

    def run(command, stdout, **kwargs):
        path.write_text(json.dumps({"config": cfg, "history": [{"round": 1, "test_acc": 0.5}]}))
        os.utime(path, ns=(1 + len(calls), 1 + len(calls)))  # a different mtime each time, whatever the clock resolution
        calls.append(command)

    calls = []
    monkeypatch.setattr(grid.subprocess, "run", run)
    path.write_text("{}")
    assert grid.main(["--output-dir", str(tmp_path), "--seeds", "1", "--no-resume"]) == 0
    assert grid.main(["--output-dir", str(tmp_path), "--seeds", "1", "--no-resume"]) == 0
    assert len(calls) == 2 and "1 run, 0 skipped, 0 failed" in capsys.readouterr().out
