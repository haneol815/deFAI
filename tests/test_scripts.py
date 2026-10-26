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


def write_run(tmp_path, name, accs, seed=0, privacy=None, **config):
    cfg = {"strategy": "fedavg", "dataset": "synthetic", "partitioner": "iid", "seed": seed,
           "output-dir": str(tmp_path), "attack": "none", "num-malicious": 0, "dp": "none", **config}
    history = [{"round": i, "test_acc": a} for i, a in enumerate(accs)]
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
