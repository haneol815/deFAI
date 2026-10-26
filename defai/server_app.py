"""ServerApp: orchestrates rounds, aggregates updates, evaluates the global model."""

import json
import os
import time

import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp

from defai.config import validate_config
from defai.strategies import build_strategy
from defai.task import build_model, get_device, load_centralized_testset, test

app = ServerApp()


def _run_name(cfg):
    name = f"{cfg['dataset']}_{cfg['strategy']}_{cfg['partitioner']}"
    if cfg["dp"] != "none":
        name += f"_dp-{cfg['dp']}-{cfg['dp-side']}"
    if cfg["attack"] != "none" and cfg["num-malicious"] > 0:
        name += f"_{cfg['attack']}x{cfg['num-malicious']}"
    return f"{name}_{time.strftime('%Y%m%d-%H%M%S')}"


@app.main()
def main(grid: Grid, context: Context) -> None:
    cfg = context.run_config
    validate_config(cfg)
    torch.manual_seed(cfg["seed"])
    model = build_model(cfg["dataset"])
    testloader = load_centralized_testset(cfg["dataset"])
    device = get_device()

    def global_evaluate(server_round: int, arrays: ArrayRecord) -> MetricRecord:
        model.load_state_dict(arrays.to_torch_state_dict())
        loss, acc = test(model, testloader, device)
        return MetricRecord({"test_loss": loss, "test_acc": acc})

    strategy = build_strategy(cfg)
    result = strategy.start(
        grid=grid,
        initial_arrays=ArrayRecord(model.state_dict()),
        train_config=ConfigRecord({"lr": cfg["learning-rate"]}),
        num_rounds=cfg["num-server-rounds"],
        evaluate_fn=global_evaluate,
    )

    out_dir = os.path.join(cfg["output-dir"], _run_name(cfg))
    os.makedirs(out_dir, exist_ok=True)

    history = []
    for r in sorted(result.evaluate_metrics_serverapp):
        row = {"round": r, **result.evaluate_metrics_serverapp[r]}
        row.update(result.train_metrics_clientapp.get(r, {}))
        row.update(result.evaluate_metrics_clientapp.get(r, {}))
        history.append(row)
    summary = {"config": dict(cfg), "history": history}
    if strategy.dp is not None:
        summary["privacy"] = {
            "mechanism": strategy.dp.describe(),
            "epsilon": strategy.dp.epsilon,
            "delta": strategy.dp.delta,
            "rounds_accounted": strategy.dp.accountant.num_releases,
        }
        print(f"\nPrivacy spent: epsilon={strategy.dp.epsilon:.4f} at delta={strategy.dp.delta:g}")
    with open(os.path.join(out_dir, "history.json"), "w") as f:
        json.dump(summary, f, indent=2)

    if cfg["save-model"]:
        torch.save(result.arrays.to_torch_state_dict(), os.path.join(out_dir, "final_model.pt"))
    print(f"\nSaved results to {os.path.abspath(out_dir)}")
