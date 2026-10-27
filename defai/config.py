"""Fail-fast validation of the run config (``[tool.flwr.app.config]`` plus ``--run-config`` overrides).

Without it a typo such as ``attack='signflip'`` silently runs a clean experiment and an
out-of-range value only fails deep inside a round.
"""

from logging import WARNING

from flwr.supercore import log

from defai.attacks import ATTACKS
from defai.strategies import STRATEGIES
from defai.task import DATASETS

CHOICES = {
    "strategy": tuple(STRATEGIES),
    "dataset": tuple(DATASETS),
    "partitioner": ("iid", "dirichlet"),
    "attack": ATTACKS,
    "dp": ("none", "fixed", "adaptive"),
    "dp-side": ("server", "client"),
}

# key -> (predicate, description)
RANGES = {
    "num-server-rounds": (lambda v: v >= 1, ">= 1"),
    "fraction-train": (lambda v: 0 <= v <= 1, "in [0, 1]"),
    "fraction-evaluate": (lambda v: 0 <= v <= 1, "in [0, 1]"),
    "local-epochs": (lambda v: v >= 1, ">= 1"),
    "batch-size": (lambda v: v >= 1, ">= 1"),
    "learning-rate": (lambda v: v > 0, "> 0"),
    "lr-decay": (lambda v: 0 < v <= 1, "in (0, 1]"),
    "dirichlet-alpha": (lambda v: v > 0, "> 0"),
    "proximal-mu": (lambda v: v >= 0, ">= 0"),
    "moon-mu": (lambda v: v >= 0, ">= 0"),
    "qfedavg-q": (lambda v: v >= 0, ">= 0"),
    "moon-temperature": (lambda v: v > 0, "> 0"),
    "server-learning-rate": (lambda v: v > 0, "> 0"),
    "server-momentum": (lambda v: 0 <= v < 1, "in [0, 1)"),
    "fedopt-learning-rate": (lambda v: v > 0, "> 0"),
    "beta-1": (lambda v: 0 <= v < 1, "in [0, 1)"),
    "beta-2": (lambda v: 0 <= v < 1, "in [0, 1)"),
    "tau": (lambda v: v > 0, "> 0"),
    "trim-ratio": (lambda v: 0 <= v < 0.5, "in [0, 0.5)"),
    "num-byzantine": (lambda v: v >= 0, ">= 0"),
    "multikrum-select": (lambda v: v >= 1, ">= 1"),
    "num-malicious": (lambda v: v >= 0, ">= 0"),
    "dropout-rate": (lambda v: 0 <= v <= 1, "in [0, 1]"),
    "dp-noise-multiplier": (lambda v: v >= 0, ">= 0"),
    "dp-clip-norm": (lambda v: v > 0, "> 0"),
    "dp-delta": (lambda v: 0 < v < 1, "in (0, 1)"),
    "dp-max-epsilon": (lambda v: v >= 0, ">= 0 (0 = no limit)"),
    "dp-target-quantile": (lambda v: 0 <= v <= 1, "in [0, 1]"),
    "dp-clip-lr": (lambda v: v > 0, "> 0"),
    "dp-clipped-count-stddev": (lambda v: v >= 0, ">= 0 (0 = default)"),
}


def validate_config(cfg) -> None:
    """Raise ``ValueError`` listing every problem found in ``cfg``."""
    errors = []
    for key, allowed in CHOICES.items():
        if key in cfg and cfg[key] not in allowed:
            errors.append(f"{key}={cfg[key]!r}: expected one of {', '.join(map(str, allowed))}")
    for key, (ok, expected) in RANGES.items():
        if key in cfg and not ok(cfg[key]):
            errors.append(f"{key}={cfg[key]!r}: expected {expected}")

    strategy = STRATEGIES.get(cfg.get("strategy"))
    if cfg.get("dp", "none") != "none" and strategy is not None and not strategy.supports_dp:
        errors.append(f"dp={cfg['dp']!r} cannot be combined with strategy={cfg['strategy']!r} (needs a mean-based strategy)")
    if errors:
        raise ValueError("invalid run config:\n  - " + "\n  - ".join(errors))

    if cfg.get("attack", "none") == "none" and cfg.get("num-malicious", 0) > 0:
        log(WARNING, "num-malicious=%d has no effect while attack='none'", cfg["num-malicious"])
    if cfg.get("attack", "none") != "none" and cfg.get("num-malicious", 0) == 0:
        log(WARNING, "attack=%r has no effect while num-malicious=0", cfg["attack"])
