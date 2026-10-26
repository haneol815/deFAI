"""Byzantine-robust aggregation: FedMedian, FedTrimmedAvg, Krum and Multi-Krum.

* FedMedian / FedTrimmedAvg: Yin et al., "Byzantine-Robust Distributed Learning:
  Towards Optimal Statistical Rates", ICML 2018.
* Krum / Multi-Krum: Blanchard et al., "Machine Learning with Adversaries:
  Byzantine Tolerant Gradient Descent", NeurIPS 2017.

All rules operate on client *updates*, which gives the same result as operating on
the client models (the global model is a common shift) and lets us reuse the
server-learning-rate hook. None of them is compatible with the DP mechanism, whose
sensitivity analysis assumes a (clipped) mean.
"""

from logging import INFO, WARNING

from flwr.supercore import log

from defai.aggregation import Arrays, coordinate_median, multi_krum_select, trimmed_mean, weighted_mean
from defai.strategies.base import ClientUpdate, FedAvg


class FedMedian(FedAvg):
    """Coordinate-wise median of the client updates."""

    name = "fedmedian"
    supports_dp = False

    def aggregate_updates(self, updates: list[ClientUpdate]) -> Arrays:
        return coordinate_median([u.delta for u in updates])


class FedTrimmedAvg(FedAvg):
    """Coordinate-wise mean after dropping the ``floor(beta * n)`` largest and smallest values."""

    name = "fedtrimmedavg"
    supports_dp = False

    def __init__(self, *, beta: float = 0.2, **kwargs) -> None:
        super().__init__(**kwargs)
        if not 0.0 <= beta < 0.5:
            raise ValueError("beta must be in [0, 0.5)")
        self.beta = beta

    def aggregate_updates(self, updates: list[ClientUpdate]) -> Arrays:
        return trimmed_mean([u.delta for u in updates], self.beta)

    def summary(self) -> None:
        super().summary()
        log(INFO, "\t└──> Trim ratio beta: %s", self.beta)


class MultiKrum(FedAvg):
    """Average the ``num_select`` updates with the lowest Krum score.

    A client's score is the sum of squared distances to its ``n - f - 2`` nearest
    neighbours (``f`` = ``num_byzantine``). The guarantee needs ``n >= 2f + 3``.
    Training metrics are aggregated over the selected clients only.
    """

    name = "multikrum"
    supports_dp = False

    def __init__(self, *, num_byzantine: int = 1, num_select: int = 3, **kwargs) -> None:
        super().__init__(**kwargs)
        if num_byzantine < 0 or num_select < 1:
            raise ValueError("num_byzantine must be >= 0 and num_select >= 1")
        self.num_byzantine = num_byzantine
        self.num_select = num_select

    def select_updates(self, updates: list[ClientUpdate]) -> list[ClientUpdate]:
        n, f = len(updates), self.num_byzantine
        if n < 2 * f + 3:
            log(WARNING, "\tKrum: %d replies < 2f+3 = %d; the robustness guarantee does not hold", n, 2 * f + 3)
        chosen = multi_krum_select([u.delta for u in updates], f, self.num_select)
        log(INFO, "\tKrum selected nodes %s", [updates[i].node_id for i in chosen])
        return [updates[i] for i in chosen]

    def aggregate_updates(self, updates: list[ClientUpdate]) -> Arrays:
        return weighted_mean([u.delta for u in updates], [1.0] * len(updates))

    def summary(self) -> None:
        super().summary()
        log(INFO, "\t└──> Krum: f=%d, selecting %d", self.num_byzantine, self.num_select)


class Krum(MultiKrum):
    """Classic Krum: the single update with the lowest score."""

    name = "krum"

    def __init__(self, *, num_byzantine: int = 1, **kwargs) -> None:
        super().__init__(num_byzantine=num_byzantine, num_select=1, **kwargs)
