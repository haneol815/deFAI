"""ServerApp: orchestrates rounds, aggregates updates, evaluates the global model."""

import json
import os

import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp

from defai.config import validate_config
from defai.results import build_history, run_name
from defai.strategies import build_strategy
from defai.task import build_model, get_device, load_centralized_testset, test

app = ServerApp()


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

    out_dir = os.path.join(cfg["output-dir"], run_name(cfg))
    os.makedirs(out_dir, exist_ok=True)

    summary = {"config": dict(cfg), "history": build_history(result)}
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
