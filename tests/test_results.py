from flwr.app import MetricRecord
from flwr.serverapp.strategy import Result

from defai.results import build_history, run_name

BASE = {"dataset": "mnist", "strategy": "fedavg", "partitioner": "iid", "dp": "none", "dp-side": "server",
        "attack": "none", "num-malicious": 0}


def test_run_name_lists_only_the_axes_that_are_switched_on():
    assert run_name(BASE, "T") == "mnist_fedavg_iid_T"
    assert run_name({**BASE, "dp": "adaptive", "dp-side": "client"}, "T") == "mnist_fedavg_iid_dp-adaptive-client_T"
    assert run_name({**BASE, "attack": "sign-flip", "num-malicious": 2}, "T") == "mnist_fedavg_iid_sign-flipx2_T"
    assert run_name({**BASE, "attack": "sign-flip", "num-malicious": 0}, "T") == "mnist_fedavg_iid_T"  # no attacker, no tag
    assert run_name(BASE).startswith("mnist_fedavg_iid_20")  # timestamp by default


def test_build_history_merges_the_three_metric_sources_per_round():
    result = Result()
    result.evaluate_metrics_serverapp = {0: MetricRecord({"test_acc": 0.1}), 1: MetricRecord({"test_acc": 0.5})}
    result.train_metrics_clientapp = {1: MetricRecord({"train_loss": 2.0})}
    result.evaluate_metrics_clientapp = {1: MetricRecord({"eval_acc": 0.4})}
    assert build_history(result) == [
        {"round": 0, "test_acc": 0.1},
        {"round": 1, "test_acc": 0.5, "train_loss": 2.0, "eval_acc": 0.4},
    ]
