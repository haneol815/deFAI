"""Algorithms for non-IID data: FedProx, SCAFFOLD and MOON.

The server halves live here; the matching local-training halves are in
``defai.task.train`` / ``defai.client_app`` and are switched on by what the server
puts into the train message (``proximal-mu``, ``moon-mu``, the ``c`` control variate).

* FedProx:  Li et al., "Federated Optimization in Heterogeneous Networks", MLSys 2020.
* SCAFFOLD: Karimireddy et al., "SCAFFOLD: Stochastic Controlled Averaging for
            Federated Learning", ICML 2020 (option II control variates).
* MOON:     Li et al., "Model-Contrastive Federated Learning", CVPR 2021.
"""

from logging import INFO

import numpy as np
from flwr.app import ArrayRecord, ConfigRecord
from flwr.supercore import log

from defai.aggregation import Arrays, weighted_mean
from defai.records import KEY_C, KEY_C_DELTA, to_record
from defai.strategies.base import ClientUpdate, FedAvg


class FedProx(FedAvg):
    """FedAvg on the server; clients minimise ``loss + mu/2 * ||w - w_global||^2``."""

    name = "fedprox"

    def __init__(self, *, proximal_mu: float = 0.01, **kwargs) -> None:
        super().__init__(**kwargs)
        if proximal_mu < 0:
            raise ValueError("proximal_mu must be non-negative")
        self.proximal_mu = proximal_mu

    def _extend_train_config(self, server_round: int, config: ConfigRecord) -> None:
        config["proximal-mu"] = self.proximal_mu

    def summary(self) -> None:
        super().summary()
        log(INFO, "\t└──> Proximal mu: %s", self.proximal_mu)


class Moon(FedAvg):
    """FedAvg on the server; clients add the model-contrastive loss

    ``mu * -log( e^{sim(z, z_glob)/tau} / (e^{sim(z, z_glob)/tau} + e^{sim(z, z_prev)/tau}) )``

    where ``z`` is the representation of the model being trained, ``z_glob`` that of
    the received global model and ``z_prev`` that of the client's previous local model.
    """

    name = "moon"

    def __init__(self, *, mu: float = 1.0, temperature: float = 0.5, **kwargs) -> None:
        super().__init__(**kwargs)
        if mu < 0 or temperature <= 0:
            raise ValueError("mu must be non-negative and temperature positive")
        self.mu = mu
        self.temperature = temperature

    def _extend_train_config(self, server_round: int, config: ConfigRecord) -> None:
        config["moon-mu"] = self.mu
        config["moon-temperature"] = self.temperature

    def summary(self) -> None:
        super().summary()
        log(INFO, "\t└──> MOON: mu=%s temperature=%s", self.mu, self.temperature)


class Scaffold(FedAvg):
    """SCAFFOLD: control variates correct client drift.

    Server state: global control variate ``c`` (zeros initially). Each sampled client
    ``i`` runs corrected SGD ``y <- y - lr * (g - c_i + c)``, then returns its model and
    ``dc_i = c_i^+ - c_i`` where ``c_i^+ = c_i - c + (x - y) / (K * lr)``. The server does::

        x <- x + server_lr * mean_i(y_i - x)
        c <- c + (|S| / N) * mean_i(dc_i)

    Plain (uniform) averaging is used, as in the paper. Not compatible with DP: the
    control-variate updates are not clipped, so the sensitivity bound would not hold.
    """

    name = "scaffold"
    supports_dp = False
    extra_reply_records = (KEY_C_DELTA,)

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._c: Arrays | None = None

    def _extra_train_records(self, server_round: int) -> dict[str, ArrayRecord]:
        if self._c is None:
            self._c = {k: np.zeros(v.shape, dtype=np.float64) for k, v in self._global.items()}
        return {KEY_C: to_record({k: v.astype(self._global[k].dtype) for k, v in self._c.items()})}

    def aggregate_updates(self, updates: list[ClientUpdate]) -> Arrays:
        uniform = [1.0] * len(updates)
        mean_dc = weighted_mean([u.extras[KEY_C_DELTA] for u in updates], uniform)
        share = len(updates) / self._population  # |S| / N
        for k, dc in mean_dc.items():
            self._c[k] = self._c[k] + share * dc.astype(np.float64)
        log(INFO, "\tSCAFFOLD: |c| = %.4g", float(np.sqrt(sum(np.sum(c**2) for c in self._c.values()))))
        return weighted_mean([u.delta for u in updates], uniform)
