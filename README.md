# defFAI — Federated Learning Assignment

A federated learning (FL) assignment template built on [Flower](https://flower.ai).
It simulates multiple clients (SuperNodes) on a single machine with the Flower Simulation Engine,
and the same code can be deployed to a real distributed setup with the Deployment Engine.

- Aggregation strategies: FedAvg, FedProx (Flower built-in strategies)
- Datasets: MNIST, CIFAR-10 (downloaded and partitioned with [Flower Datasets](https://flower.ai/docs/datasets/))
- Partitioning: IID, Dirichlet non-IID
- Evaluation: federated (on each client's local data) + centralized (on the full test set at the server)

## Structure
```
.
├── pyproject.toml          # Dependencies + Flower app settings (run config)
├── deffai/
│   ├── server_app.py       # ServerApp: strategy selection, rounds, centralized eval, result saving
│   ├── client_app.py       # ClientApp: local training / evaluation
│   └── task.py             # Model, data partitioning/loading, train/test functions
├── scripts/plot.py         # Accuracy curve plotting
├── docs/report.md          # Report template
└── results/                # Experiment outputs (git-ignored)
```

## Installation
Requires Python 3.11 or later.
```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

## Usage
```bash
# Default run (MNIST, IID, FedAvg, 10 rounds, 10 SuperNodes)
flwr run . --stream --federation-config "num-supernodes=10"

# Non-IID (Dirichlet α=0.5) + FedProx
flwr run . --stream --federation-config "num-supernodes=10" \
  --run-config "partitioner='dirichlet' strategy='fedprox' proximal-mu=0.01"

# Change number of rounds, learning rate, etc.
flwr run . --stream --federation-config "num-supernodes=10" \
  --run-config "num-server-rounds=30 learning-rate=0.05"
```

- All configurable values are listed under `[tool.flwr.app.config]` in `pyproject.toml`.
- Since Flower 1.32 the default number of simulated SuperNodes is 2, so set it with `--federation-config`.
  To change the default permanently instead:
  `flwr federation simulation-config --num-supernodes 10`
- Results are saved to `results/<dataset>_<strategy>_<partitioner>_<timestamp>/history.json`.

```bash
pip install -e ".[plot]"
python scripts/plot.py results/*/history.json -o results/acc.png
```

## Extending
- **New aggregation algorithm**: subclass `flwr.serverapp.strategy.FedAvg`, override `aggregate_train()` etc., and register it in `build_strategy()` in `server_app.py`.
  Flower also ships with strategies such as FedAdam, FedYogi, FedMedian, and Krum.
- **New local training method**: `train()` in `task.py` and `train()` in `client_app.py`
- **New dataset/model**: `DATASETS` and `build_model()` in `task.py`
- **Real distributed deployment**: [Deployment Engine guide](https://flower.ai/docs/framework/how-to-run-flower-with-deployment-engine.html)

## License
[MIT](LICENSE)
