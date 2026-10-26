import numpy as np
import pytest
from defai.privacy.dp import FixedClipping
from defai.strategies import (
    Bulyan,
    FedAdagrad,
    FedAdam,
    FedAvg,
    FedAvgM,
    FedMedian,
    FedProx,
    FedTrimmedAvg,
    FedYogi,
    Krum,
    Moon,
    MultiKrum,
    Scaffold,
)
from conftest import QuadraticClients, run_strategy

N = 8
DIM = 4
RNG = np.random.default_rng(0)
TARGETS = {n: RNG.normal(0, 1.0, DIM) for n in range(1, N + 1)}
OPT = np.mean(list(TARGETS.values()), axis=0)  # optimum of the average objective
COMMON = dict(fraction_train=1.0, fraction_evaluate=1.0, seed=0)


def distance(w):
    return float(np.linalg.norm(w - OPT))


def run(strategy, rounds=40, **client_kwargs):
    clients = QuadraticClients(TARGETS, **client_kwargs)
    return run_strategy(strategy, clients, N, rounds, DIM)


# ---------------------------------------------------------------------------- FedAvg family
def test_fedavg_one_round_is_example_weighted_mean_of_client_models():
    num_examples = {n: n for n in TARGETS}  # unequal weights 1..8
    clients = QuadraticClients(TARGETS, num_examples=num_examples, lr=0.5, steps=1)
    w, result, _ = run_strategy(FedAvg(**COMMON), clients, N, rounds=1, dim=DIM)
    models = {n: 0.5 * TARGETS[n] for n in TARGETS}  # one step of lr 0.5 from w=0
    weights = np.array([num_examples[n] for n in TARGETS], dtype=float)
    expected = sum(w_ * models[n] for w_, n in zip(weights / weights.sum(), TARGETS))
    np.testing.assert_allclose(w, expected, atol=1e-6)
    assert result.train_metrics_clientapp[1]["train_loss"] == pytest.approx(
        sum(w_ * 0.5 * np.sum(TARGETS[n] ** 2) for w_, n in zip(weights / weights.sum(), TARGETS))
    )


def test_fedavg_converges_to_the_optimum():
    w, _, _ = run(FedAvg(**COMMON))
    assert distance(w) < 1e-4


def test_server_learning_rate_scales_the_update():
    w1, _, _ = run(FedAvg(server_learning_rate=1.0, **COMMON), rounds=1, lr=0.5, steps=1)
    w2, _, _ = run(FedAvg(server_learning_rate=0.5, **COMMON), rounds=1, lr=0.5, steps=1)
    np.testing.assert_allclose(w2, 0.5 * w1, atol=1e-6)


def test_fedprox_sends_mu_and_converges_to_the_same_optimum():
    strategy = FedProx(proximal_mu=0.5, **COMMON)
    w, _, grid = run(strategy, rounds=80)
    assert grid.sent[0].content["config"]["proximal-mu"] == 0.5
    assert distance(w) < 1e-3


def test_moon_sends_its_hyperparameters():
    _, _, grid = run(Moon(mu=2.0, temperature=0.3, **COMMON), rounds=1)
    cfg = grid.sent[0].content["config"]
    assert (cfg["moon-mu"], cfg["moon-temperature"]) == (2.0, 0.3)


def test_scaffold_converges_and_sends_the_control_variate():
    w, _, grid = run(Scaffold(**COMMON), rounds=60)
    assert "c" in grid.sent[0].content  # control variate travels with the model
    assert distance(w) < 1e-4


def test_scaffold_server_control_variate_is_the_average_client_control_variate():
    clients = QuadraticClients(TARGETS)
    strategy = Scaffold(**COMMON)
    run_strategy(strategy, clients, N, rounds=5, dim=DIM)
    # full participation: c = mean_i c_i exactly (c_i starts at 0, c += |S|/N * mean(dc_i))
    np.testing.assert_allclose(strategy._c["w"], np.mean(list(clients.control.values()), axis=0), atol=1e-5)


def test_scaffold_removes_the_client_sampling_noise_floor_that_fedavg_keeps():
    """Clients fully fit their own target (large lr * steps) and only half participate each round.

    FedAvg then jumps to the mean of whichever clients were sampled and never settles;
    SCAFFOLD's control variates cancel the drift, so it converges to the true optimum.
    """
    kwargs = dict(lr=0.3, steps=30)
    part = dict(fraction_train=0.5, fraction_evaluate=0.5)
    fedavg_err = np.mean([distance(run(FedAvg(seed=s, **part), rounds=150, **kwargs)[0]) for s in range(3)])
    scaffold_err = np.mean([distance(run(Scaffold(seed=s, **part), rounds=150, **kwargs)[0]) for s in range(3)])
    assert fedavg_err > 0.1
    assert scaffold_err < 0.2 * fedavg_err


def test_scaffold_rejects_dp():
    with pytest.raises(ValueError):
        Scaffold(dp=FixedClipping(1.0, 1.0))


# ------------------------------------------------------------------------ server optimisers
def test_fedavgm_matches_torch_sgd_momentum_convention():
    clients = QuadraticClients(TARGETS, lr=0.5, steps=1)
    strategy = FedAvgM(momentum=0.9, server_learning_rate=1.0, **COMMON)
    # reference: replay FedAvg deltas by hand with torch.optim.SGD's momentum convention
    x = np.zeros(DIM)
    v = None
    for _ in range(3):
        # client models after one lr=0.5 step from x
        models = [x + 0.5 * (TARGETS[n] - x) for n in TARGETS]
        delta = np.mean(models, axis=0) - x
        grad = -delta
        v = grad if v is None else 0.9 * v + grad
        x = x - 1.0 * v
    w, _, _ = run_strategy(strategy, clients, N, rounds=3, dim=DIM)
    np.testing.assert_allclose(w, x, atol=1e-5)


FEDOPT_HPARAMS = dict(beta_1=0.9, beta_2=0.9, tau=0.05, server_learning_rate=0.2)


def reference_fedopt(kind, rounds):
    """Reddi et al. (2021), Algorithm 2, replayed with plain numpy on the quadratic clients.

    Written independently of the library code: v starts at tau^2, no bias correction.
    """
    b1, b2, tau, eta = (FEDOPT_HPARAMS[k] for k in ("beta_1", "beta_2", "tau", "server_learning_rate"))
    x, m, v = np.zeros(DIM), np.zeros(DIM), np.full(DIM, tau**2)
    for _ in range(rounds):
        models = [x + 0.5 * (TARGETS[n] - x) for n in TARGETS]  # one local step, lr 0.5
        delta = np.mean(models, axis=0) - x
        m = b1 * m + (1 - b1) * delta
        if kind == "adagrad":
            v = v + delta**2
        elif kind == "adam":
            v = b2 * v + (1 - b2) * delta**2
        else:
            v = v - (1 - b2) * delta**2 * np.sign(v - delta**2)
        x = x + eta * m / (np.sqrt(v) + tau)
    return x


@pytest.mark.parametrize("cls,kind", [(FedAdagrad, "adagrad"), (FedAdam, "adam"), (FedYogi, "yogi")])
def test_fedopt_matches_the_paper_over_several_rounds(cls, kind):
    clients = QuadraticClients(TARGETS, lr=0.5, steps=1)
    w, _, _ = run_strategy(cls(**FEDOPT_HPARAMS, **COMMON), clients, N, rounds=6, dim=DIM)
    expected = reference_fedopt(kind, 6)
    np.testing.assert_allclose(w, expected, rtol=1e-4, atol=1e-6)


def test_the_fedopt_references_are_distinguishable():
    """Guard: the replay above only proves something if the three rules give different results."""
    ref = {k: reference_fedopt(k, 6) for k in ("adagrad", "adam", "yogi")}
    for a, b in [("adagrad", "adam"), ("adam", "yogi"), ("adagrad", "yogi")]:
        assert np.linalg.norm(ref[a] - ref[b]) > 1e-3, (a, b)


@pytest.mark.parametrize(
    "strategy",
    [
        FedAvgM(momentum=0.5, **COMMON),
        FedAdam(server_learning_rate=0.05, **COMMON),
        FedYogi(server_learning_rate=0.05, **COMMON),
        FedAdagrad(server_learning_rate=0.1, **COMMON),
    ],
    ids=lambda s: type(s).__name__,
)
def test_server_optimisers_make_progress_towards_the_optimum(strategy):
    start = distance(np.zeros(DIM))
    w, _, _ = run(strategy, rounds=100)
    assert distance(w) < 0.2 * start


# ------------------------------------------------------------------------ robust aggregation
@pytest.mark.parametrize(
    "strategy",
    [
        FedMedian(**COMMON),
        FedTrimmedAvg(beta=0.25, **COMMON),
        Krum(num_byzantine=2, **COMMON),
        MultiKrum(num_byzantine=2, num_select=4, **COMMON),
    ],
    ids=lambda s: type(s).__name__,
)
def test_robust_rules_survive_byzantine_clients_that_break_fedavg(strategy):
    bad = (1, 2)
    honest_opt = np.mean([TARGETS[n] for n in TARGETS if n not in bad], axis=0)
    w, _, _ = run(strategy, rounds=40, bad=bad)
    fedavg_w, _, _ = run(FedAvg(**COMMON), rounds=40, bad=bad)
    assert np.linalg.norm(fedavg_w - honest_opt) > 100  # 2/8 clients sending 1e3 ruin FedAvg
    # median-like rules are biased by the spread of honest targets, but stay in range
    assert np.linalg.norm(w - honest_opt) < 1.5


def test_krum_selects_exactly_one_honest_clients_model():
    bad = (1, 2)
    w, _, _ = run(Krum(num_byzantine=2, **COMMON), rounds=1, bad=bad)
    # one round of 5 GD steps (lr 0.1) from w=0 gives honest client n the model t_n * (1 - 0.9**5)
    honest = [TARGETS[n] * (1 - 0.9**5) for n in TARGETS if n not in bad]
    assert any(np.allclose(w, h, atol=1e-5) for h in honest)


@pytest.mark.parametrize("cls", [FedMedian, FedTrimmedAvg, Krum, MultiKrum])
def test_robust_rules_reject_dp(cls):
    with pytest.raises(ValueError):
        cls(dp=FixedClipping(1.0, 1.0))


def test_trimmed_avg_warns_when_it_trims_nothing(caplog):
    import logging

    with caplog.at_level(logging.WARNING, logger="flwr"):
        run(FedTrimmedAvg(beta=0.05, **COMMON), rounds=1)  # int(0.05 * 8) == 0
    assert "trims nothing" in caplog.text


def test_bulyan_survives_byzantine_clients_and_selects_theta_updates():
    n = 11  # 4f + 3 with f = 2
    rng = np.random.default_rng(1)
    targets = {i: rng.normal(0, 0.3, DIM) for i in range(1, n + 1)}
    bad = (1, 2)
    honest_mean = np.mean([targets[i] for i in targets if i not in bad], axis=0)
    common = dict(fraction_train=1.0, fraction_evaluate=1.0, seed=0)

    def final(strategy):
        w, result, _ = run_strategy(strategy, QuadraticClients(targets, bad=bad), n, 30, DIM)
        return w, result

    w_avg, _ = final(FedAvg(**common))
    w_bulyan, result = final(Bulyan(num_byzantine=2, **common))
    assert np.linalg.norm(w_avg - honest_mean) > 100
    assert np.linalg.norm(w_bulyan - honest_mean) < 0.5
    assert np.all(np.isfinite(w_bulyan))


def test_bulyan_warns_and_degrades_with_too_few_replies(caplog):
    import logging

    with caplog.at_level(logging.WARNING, logger="flwr"):
        w, _, _ = run(Bulyan(num_byzantine=2, **COMMON), rounds=2)  # 8 replies < 4*2+3
    assert "robustness guarantee does not hold" in caplog.text and np.all(np.isfinite(w))


def test_bulyan_rejects_dp():
    with pytest.raises(ValueError):
        Bulyan(dp=FixedClipping(1.0, 1.0))


# ------------------------------------------------------------------------- reproducibility
def partitions_sampled(seed, grid_seed, rounds=4, n=10):
    """Registration positions (== partition ids in the simulation) picked in each round."""
    from flwr.app import ConfigRecord as Cfg
    from conftest import RegistrationOrderGrid
    from defai.records import to_record as rec

    targets = {i: np.zeros(DIM) for i in range(n)}
    grid = RegistrationOrderGrid(n, lambda m: None, grid_seed)
    position = {node: i for i, node in enumerate(grid.nodes)}
    strategy = FedAvg(fraction_train=0.5, fraction_evaluate=0.5, seed=seed)
    picked = []
    for rnd in range(1, rounds + 1):
        messages = strategy.configure_train(rnd, rec({"w": np.zeros(DIM, np.float32)}), Cfg({"lr": 0.1}), grid)
        picked.append(sorted(position[m.metadata.dst_node_id] for m in messages))
    assert targets  # keep the helper's signature honest
    return picked


def test_client_sampling_picks_the_same_partitions_whatever_the_random_node_ids():
    first = partitions_sampled(seed=3, grid_seed=1)
    assert first == partitions_sampled(seed=3, grid_seed=2)  # different random node ids, same partitions
    assert first != partitions_sampled(seed=4, grid_seed=1)  # a different seed samples differently
    assert all(len(p) == 5 for p in first)


def test_aggregation_order_does_not_depend_on_reply_arrival_order():
    from flwr.app import ConfigRecord as Cfg
    from conftest import RegistrationOrderGrid
    from defai.records import to_record as rec

    seen = []

    class Spy(FedAvg):
        def aggregate_updates(self, updates):
            seen.append([u.node_id for u in updates])
            return super().aggregate_updates(updates)

    n = 8
    grid = RegistrationOrderGrid(n, lambda m: None, 5)
    targets = dict(zip(grid.nodes, [np.full(DIM, float(i)) for i in range(n)]))
    clients = QuadraticClients(targets, lr=0.5, steps=1, num_examples={node: 10 for node in grid.nodes})
    strategy = Spy(fraction_train=1.0, fraction_evaluate=1.0, seed=0)
    messages = list(strategy.configure_train(1, rec({"w": np.zeros(DIM, np.float32)}), Cfg({"lr": 0.5}), grid))
    replies = [clients(m) for m in messages]
    rng = np.random.default_rng(0)
    for _ in range(3):
        rng.shuffle(replies)
        strategy.aggregate_train(1, list(replies))
    assert seen[0] == grid.nodes  # registration order, not id order and not arrival order
    assert all(order == seen[0] for order in seen)
