"""Run naming and result (history.json) construction for the ServerApp."""

import time

from flwr.serverapp.strategy import Result


def run_name(cfg, timestamp=None) -> str:
    """results/ directory name: dataset, strategy, partitioner and the non-default experiment axes."""
    name = f"{cfg['dataset']}_{cfg['strategy']}_{cfg['partitioner']}"
    if cfg["dp"] != "none":
        name += f"_dp-{cfg['dp']}-{cfg['dp-side']}"
    if cfg["attack"] != "none" and cfg["num-malicious"] > 0:
        name += f"_{cfg['attack']}x{cfg['num-malicious']}"
    return f"{name}_{timestamp or time.strftime('%Y%m%d-%H%M%S')}"


def build_history(result: Result) -> list[dict]:
    """One row per round: server-side evaluation, aggregated train and client-side evaluation metrics."""
    history = []
    for r in sorted(result.evaluate_metrics_serverapp):
        row = {"round": r, **result.evaluate_metrics_serverapp[r]}
        row.update(result.train_metrics_clientapp.get(r, {}))
        row.update(result.evaluate_metrics_clientapp.get(r, {}))
        history.append(row)
    return history
