import argparse
import json
import os
import time

import numpy as np

from fl.client import Client
from fl.data import load_dataset, make_client_datasets
from fl.models import build_model
from fl.server import Server
from fl.utils import get_device, load_config, set_seed


def parse_args():
    p = argparse.ArgumentParser(description="Federated learning simulation")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--rounds", type=int, help="override fl.rounds")
    p.add_argument("--partition", choices=["iid", "dirichlet"], help="override data.partition")
    p.add_argument("--algorithm", choices=["fedavg", "fedprox"], help="override fl.algorithm")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    if args.rounds is not None:
        cfg["fl"]["rounds"] = args.rounds
    if args.partition:
        cfg["data"]["partition"] = args.partition
    if args.algorithm:
        cfg["fl"]["algorithm"] = args.algorithm

    set_seed(cfg["seed"])
    device = get_device(cfg["device"])
    print(f"device: {device}")

    train_set, test_set = load_dataset(cfg["data"]["dataset"], cfg["data"]["root"])
    client_sets = make_client_datasets(train_set, cfg, cfg["seed"])
    clients = [Client(i, ds, cfg, device) for i, ds in enumerate(client_sets)]
    print("client sizes:", [c.num_samples for c in clients])

    server = Server(build_model(cfg["data"]["dataset"]), clients, cfg, device,
                    np.random.default_rng(cfg["seed"]))

    history = []
    for r in range(1, cfg["fl"]["rounds"] + 1):
        t0 = time.time()
        train_loss, selected = server.run_round()
        test_loss, test_acc = server.evaluate(test_set)
        history.append({"round": r, "train_loss": train_loss,
                        "test_loss": test_loss, "test_acc": test_acc})
        print(f"[round {r:3d}] clients={selected} train_loss={train_loss:.4f} "
              f"test_loss={test_loss:.4f} test_acc={test_acc:.4f} ({time.time() - t0:.1f}s)")

    run_name = f"{cfg['data']['dataset']}_{cfg['fl']['algorithm']}_{cfg['data']['partition']}_{time.strftime('%Y%m%d-%H%M%S')}"
    out_dir = os.path.join(cfg["output_dir"], run_name)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "history.json"), "w") as f:
        json.dump({"config": cfg, "history": history}, f, indent=2)
    print(f"saved: {out_dir}/history.json")


if __name__ == "__main__":
    main()
