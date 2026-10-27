"""Error paths and edge cases of the FedAvg base strategy (replies, sampling, empty rounds)."""

import numpy as np
import pytest
from flwr.app import ConfigRecord, Error, Message, MessageType, MetricRecord, RecordDict
from flwr.proto.node_pb2 import NodeInfo

from conftest import FakeGrid, QuadraticClients
from defai.records import to_record
from defai.strategies import FedAvg, Scaffold
from defai.strategies.base import ordered_node_ids, weighted_metrics

DIM = 3
N = 4
TARGETS = {n: np.full(DIM, float(n)) for n in range(1, N + 1)}
COMMON = dict(fraction_train=1.0, fraction_evaluate=1.0, seed=0)
START = to_record({"w": np.zeros(DIM, np.float32)})


def configured(strategy, num_nodes=N):
    """Run configure_train once and return (messages, grid)."""
    grid = FakeGrid(num_nodes, QuadraticClients(TARGETS))
    return list(strategy.configure_train(1, START, ConfigRecord({"lr": 0.1}), grid)), grid


def good_reply(msg, **overrides):
    content = {
        "arrays": to_record({"w": np.ones(DIM, np.float32)}),
        "metrics": MetricRecord({"num-examples": 10, "train_loss": 1.0}),
    }
    content.update(overrides)
    return Message(content=RecordDict({k: v for k, v in content.items() if v is not None}), reply_to=msg)


def aggregate(strategy, replies):
    arrays, metrics = strategy.aggregate_train(1, replies)
    return (None if arrays is None else arrays["w"].numpy()), metrics


# ------------------------------------------------------------------------------ malformed replies
@pytest.mark.parametrize(
    "reply_kwargs,reason",
    [
        (dict(metrics=None), "missing MetricRecord"),
        (dict(metrics=MetricRecord({"train_loss": 1.0})), "positive finite number"),
        (dict(arrays=None), "missing ArrayRecord 'arrays'"),
        (dict(arrays=to_record({"v": np.ones(DIM, np.float32)})), "parameter names differ"),
        (dict(arrays=to_record({"w": np.ones(DIM + 1, np.float32)})), "shape mismatch"),
    ],
    ids=["no-metrics", "no-weight", "no-arrays", "renamed-parameter", "wrong-shape"],
)
def test_a_malformed_reply_is_dropped_with_its_reason_and_the_rest_still_aggregate(reply_kwargs, reason, caplog):
    import logging

    strategy = FedAvg(**COMMON)
    messages, _ = configured(strategy)
    replies = [good_reply(messages[0], **reply_kwargs)] + [good_reply(m) for m in messages[1:]]
    with caplog.at_level(logging.WARNING, logger="flwr"):
        w, metrics = aggregate(strategy, replies)
    assert reason in caplog.text
    np.testing.assert_allclose(w, 1.0)  # the three good replies all send ones
    assert metrics["train_loss"] == pytest.approx(1.0)


def test_error_replies_are_counted_as_failures_and_skipped(caplog):
    import logging

    strategy = FedAvg(**COMMON)
    messages, _ = configured(strategy)
    error = Message(Error(code=1, reason="client crashed"), reply_to=messages[0])
    with caplog.at_level(logging.INFO, logger="flwr"):
        w, _ = aggregate(strategy, [error] + [good_reply(m) for m in messages[1:]])
    assert "client crashed" in caplog.text and "3 results and 1 failures" in caplog.text
    np.testing.assert_allclose(w, 1.0)


def test_a_round_where_every_client_fails_produces_no_aggregate():
    """No aggregate means Result.arrays stays empty (server_app then keeps the initial model)."""

    class Failing(QuadraticClients):
        def __call__(self, msg):
            return Message(Error(code=1, reason="boom"), reply_to=msg)

    result = FedAvg(**COMMON).start(grid=FakeGrid(N, Failing(TARGETS)), initial_arrays=START, num_rounds=3)
    assert len(result.arrays) == 0
    assert result.train_metrics_clientapp == {} and result.evaluate_metrics_clientapp == {}


def test_a_scaffold_reply_without_its_control_variate_delta_is_dropped(caplog):
    import logging

    strategy = Scaffold(**COMMON)
    messages, _ = configured(strategy)
    with caplog.at_level(logging.WARNING, logger="flwr"):
        w, _ = aggregate(strategy, [good_reply(m) for m in messages])  # nobody sends 'c-delta'
    assert w is None and "missing ArrayRecord 'c-delta'" in caplog.text


def test_aggregating_before_configuring_is_reported_not_crashed(caplog):
    import logging

    strategy = FedAvg(**COMMON)
    msg = Message(content=RecordDict({}), message_type=MessageType.TRAIN, dst_node_id=1)
    with caplog.at_level(logging.WARNING, logger="flwr"):
        arrays, metrics = strategy.aggregate_train(1, [good_reply(msg)])
    assert arrays is None and metrics is None and "before configure_train" in caplog.text


# ---------------------------------------------------------------------------------- empty rounds
def test_zero_fractions_skip_training_and_evaluation_entirely():
    strategy = FedAvg(fraction_train=0.0, fraction_evaluate=0.0, seed=0)
    grid = FakeGrid(N, QuadraticClients(TARGETS))
    assert strategy.min_train_nodes == strategy.min_evaluate_nodes == 0
    assert list(strategy.configure_train(1, START, ConfigRecord({}), grid)) == []
    assert list(strategy.configure_evaluate(1, START, ConfigRecord({}), grid)) == []
    assert strategy.aggregate_train(1, []) == (None, None)
    assert strategy.aggregate_evaluate(1, []) is None
    result = strategy.start(grid=grid, initial_arrays=START, num_rounds=2)
    assert grid.sent == [] and len(result.arrays) == 0


def test_evaluate_only_federation_still_reports_metrics():
    strategy = FedAvg(fraction_train=0.0, fraction_evaluate=1.0, seed=0)
    grid = FakeGrid(N, QuadraticClients(TARGETS))
    result = strategy.start(grid=grid, initial_arrays=START, num_rounds=2)
    assert sorted(result.evaluate_metrics_clientapp) == [1, 2]
    assert {m.metadata.message_type for m in grid.sent} == {MessageType.EVALUATE}


# ---------------------------------------------------------------------------------------- sampling
def test_the_strategy_waits_until_enough_nodes_have_connected(monkeypatch):
    grid = FakeGrid(1, QuadraticClients(TARGETS))
    sleeps = []

    def connect_another_node(_seconds):
        sleeps.append(1)
        grid.nodes.append(len(grid.nodes) + 1)

    monkeypatch.setattr("defai.strategies.base.time.sleep", connect_another_node)
    strategy = FedAvg(fraction_train=1.0, fraction_evaluate=1.0, min_available_nodes=3, seed=0)
    messages = list(strategy.configure_train(1, START, ConfigRecord({}), grid))
    assert len(sleeps) == 2 and len(messages) == 3


def test_sample_size_follows_the_fraction_and_the_minimum():
    grid = FakeGrid(10, QuadraticClients(TARGETS))
    sized = lambda **kw: len(list(FedAvg(seed=0, **kw).configure_train(1, START, ConfigRecord({}), grid)))  # noqa: E731
    assert sized(fraction_train=0.5) == 5
    assert sized(fraction_train=0.1, min_train_nodes=2) == 2  # the minimum wins over 10% of 10
    assert sized(fraction_train=1.0) == 10
    # Asking for more nodes than are connected makes the strategy wait for them (see the test above),
    # exactly like Flower's own sampling, so that case is deliberately not exercised with a real sleep.


def test_sampling_is_reproducible_and_changes_with_the_seed():
    def picks(seed):
        grid = FakeGrid(20, QuadraticClients(TARGETS))
        s = FedAvg(fraction_train=0.25, seed=seed)
        return [sorted(m.metadata.dst_node_id for m in s.configure_train(r, START, ConfigRecord({}), grid)) for r in (1, 2, 3)]

    assert picks(1) == picks(1) and picks(1) != picks(2)


def test_ordering_falls_back_to_node_id_when_the_grid_has_no_registration_times():
    class NoTimes(FakeGrid):
        def __init__(self):
            super().__init__(0, None)
            self.nodes = [30, 10, 20]

        def get_nodes(self):  # like Grid.get_nodes() by default: ids only, registered_at stays empty
            return [NodeInfo(node_id=n) for n in self.nodes]

    assert ordered_node_ids(NoTimes()) == [10, 20, 30]
    assert ordered_node_ids(FakeGrid(3, None)) == [1, 2, 3]  # grids without get_nodes at all


# -------------------------------------------------------------------------------------- metrics
def test_weighted_metrics_averages_numbers_and_lists_and_skips_the_rest():
    metrics = [
        {"num-examples": 1, "loss": 1.0, "per-class": [1.0, 3.0], "note": "a", "flag": True},
        {"num-examples": 3, "loss": 5.0, "per-class": [5.0, 7.0], "note": "b", "flag": False},
    ]
    out = weighted_metrics(metrics, [1.0, 3.0], skip="num-examples")
    assert out["loss"] == pytest.approx(4.0)
    assert out["per-class"] == pytest.approx([4.0, 6.0])
    assert out["flag"] == pytest.approx(0.25)
    assert "note" not in out and "num-examples" not in out
    assert weighted_metrics([{"a": 1.0}, {"b": 2.0}], [1.0, 1.0], skip="x") == {}  # keys not shared by all


# ------------------------------------------------------------------------- fairness across clients
def eval_reply(msg, acc, n=10):
    return Message(content=RecordDict({"metrics": MetricRecord({"eval_acc": acc, "eval_loss": 1.0 - acc, "num-examples": n})}), reply_to=msg)


def test_evaluation_reports_the_spread_of_accuracy_across_clients_next_to_the_weighted_mean():
    from defai.strategies.base import client_spread

    strategy = FedAvg(**COMMON)
    grid = FakeGrid(N, QuadraticClients(TARGETS))
    messages = list(strategy.configure_evaluate(1, START, ConfigRecord({}), grid))
    accs = [1.0, 0.9, 0.5, 0.1]
    sizes = [10, 10, 10, 70]  # the big, badly served client dominates the weighted mean ...
    out = dict(strategy.aggregate_evaluate(1, [eval_reply(m, a, n) for m, a, n in zip(messages, accs, sizes)]))
    assert out["eval_acc"] == pytest.approx((1.0 * 10 + 0.9 * 10 + 0.5 * 10 + 0.1 * 70) / 100)
    # ... while the unweighted spread shows that one client is served far worse than the rest
    assert out["eval_acc_min"] == pytest.approx(0.1) and out["eval_acc_max"] == pytest.approx(1.0)
    assert out["eval_acc_std"] == pytest.approx(float(np.std(accs)))
    assert client_spread([{"x": 1.0}], "eval_acc") == {}  # nothing to report without the metric
    assert client_spread([{"eval_acc": True}], "eval_acc") == {}  # booleans are flags, not accuracies


def test_a_single_evaluating_client_has_zero_spread():
    strategy = FedAvg(**COMMON)
    grid = FakeGrid(N, QuadraticClients(TARGETS))
    (msg, *_) = strategy.configure_evaluate(1, START, ConfigRecord({}), grid)
    out = dict(strategy.aggregate_evaluate(1, [eval_reply(msg, 0.7)]))
    assert out["eval_acc_std"] == 0.0 and out["eval_acc_min"] == out["eval_acc_max"] == pytest.approx(0.7)
