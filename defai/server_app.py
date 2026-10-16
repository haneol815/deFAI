"""ServerApp: orchestrates rounds, aggregates updates, evaluates the global model."""

import json
import os
import time

import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg, FedProx

from defai.task import build_model, get_device, load_centralized_testset, test

app = ServerApp()


def build_strategy(cfg):
    common = dict(
        fraction_train=cfg["fraction-train"],
        fraction_evaluate=cfg["fraction-evaluate"],
    )
    if cfg["strategy"] == "fedavg":
        return FedAvg(**common)
    if cfg["strategy"] == "fedprox":
        return FedProx(**common, proximal_mu=cfg["proximal-mu"])
    # Add your own strategy here (e.g. subclass FedAvg and override aggregate_train)
    raise ValueError(f"unknown strategy: {cfg['strategy']}")


@app.main()
def main(grid: Grid, context: Context) -> None:
    cfg = context.run_config
    torch.manual_seed(cfg["seed"])
    model = build_model(cfg["dataset"])
    testloader = load_centralized_testset(cfg["dataset"])
    device = get_device()

    def global_evaluate(server_round: int, arrays: ArrayRecord) -> MetricRecord:
        model.load_state_dict(arrays.to_torch_state_dict())
        loss, acc = test(model, testloader, device)
        return MetricRecord({"test_loss": loss, "test_acc": acc})

    result = build_strategy(cfg).start(
        grid=grid,
        initial_arrays=ArrayRecord(model.state_dict()),
        train_config=ConfigRecord({"lr": cfg["learning-rate"]}),
        num_rounds=cfg["num-server-rounds"],
        evaluate_fn=global_evaluate,
    )

    run_name = f"{cfg['dataset']}_{cfg['strategy']}_{cfg['partitioner']}_{time.strftime('%Y%m%d-%H%M%S')}"
    out_dir = os.path.join(cfg["output-dir"], run_name)
    os.makedirs(out_dir, exist_ok=True)

    history = []
    for r in sorted(result.evaluate_metrics_serverapp):
        row = {"round": r, **result.evaluate_metrics_serverapp[r]}
        row.update(result.train_metrics_clientapp.get(r, {}))
        row.update(result.evaluate_metrics_clientapp.get(r, {}))
        history.append(row)
    with open(os.path.join(out_dir, "history.json"), "w") as f:
        json.dump({"config": dict(cfg), "history": history}, f, indent=2)

    if cfg["save-model"]:
        torch.save(result.arrays.to_torch_state_dict(), os.path.join(out_dir, "final_model.pt"))
    print(f"\nSaved results to {os.path.abspath(out_dir)}")
