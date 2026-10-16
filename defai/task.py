"""Model, data loading, and local train/test logic shared by ServerApp and ClientApp."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import DirichletPartitioner, IidPartitioner
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor

DATASETS = {
    "mnist": {
        "hf_name": "ylecun/mnist",
        "image_key": "image",
        "channels": 1,
        "img_size": 28,
        "norm": ((0.1307,), (0.3081,)),
    },
    "cifar10": {
        "hf_name": "uoft-cs/cifar10",
        "image_key": "img",
        "channels": 3,
        "img_size": 32,
        "norm": ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    },
}


class Net(nn.Module):
    """Small CNN used in the FedAvg paper (McMahan et al., 2017)."""

    def __init__(self, in_channels=1, num_classes=10, img_size=28):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, 5, padding=2), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, padding=2), nn.ReLU(), nn.MaxPool2d(2),
        )
        feat = 64 * (img_size // 4) ** 2
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(feat, 512), nn.ReLU(), nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def build_model(dataset):
    spec = DATASETS[dataset]
    return Net(in_channels=spec["channels"], img_size=spec["img_size"])


def get_device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _make_transform(dataset):
    spec = DATASETS[dataset]
    tf = Compose([ToTensor(), Normalize(*spec["norm"])])
    key = spec["image_key"]

    def apply(batch):
        batch["x"] = [tf(img) for img in batch.pop(key)]
        batch["y"] = batch.pop("label")
        return batch

    return apply


_fds = None  # cache FederatedDataset per ClientApp process


def load_client_data(partition_id, num_partitions, cfg):
    """Return (trainloader, valloader) for this client's partition."""
    global _fds
    dataset = cfg["dataset"]
    if _fds is None:
        if cfg["partitioner"] == "iid":
            partitioner = IidPartitioner(num_partitions=num_partitions)
        elif cfg["partitioner"] == "dirichlet":
            partitioner = DirichletPartitioner(
                num_partitions=num_partitions,
                partition_by="label",
                alpha=cfg["dirichlet-alpha"],
                seed=cfg["seed"],
            )
        else:
            raise ValueError(f"unknown partitioner: {cfg['partitioner']}")
        _fds = FederatedDataset(
            dataset=DATASETS[dataset]["hf_name"],
            partitioners={"train": partitioner},
        )
    partition = _fds.load_partition(partition_id)
    # Each client holds out 20% of its local data for federated evaluation
    split = partition.train_test_split(test_size=0.2, seed=cfg["seed"])
    split = split.with_transform(_make_transform(dataset))
    trainloader = DataLoader(split["train"], batch_size=cfg["batch-size"], shuffle=True)
    valloader = DataLoader(split["test"], batch_size=cfg["batch-size"])
    return trainloader, valloader


def load_centralized_testset(dataset, batch_size=256):
    """Full test split, used by the ServerApp for centralized evaluation."""
    test = load_dataset(DATASETS[dataset]["hf_name"], split="test")
    return DataLoader(test.with_transform(_make_transform(dataset)), batch_size=batch_size)


def train(model, trainloader, epochs, lr, device, proximal_mu=0.0):
    """Local SGD. If proximal_mu > 0, adds the FedProx proximal term."""
    model.to(device)
    model.train()
    global_params = [p.detach().clone() for p in model.parameters()]
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    total_loss, steps = 0.0, 0
    for _ in range(epochs):
        for batch in trainloader:
            x, y = batch["x"].to(device), batch["y"].to(device)
            optimizer.zero_grad()
            loss = F.cross_entropy(model(x), y)
            if proximal_mu > 0:
                prox = sum(((p - g) ** 2).sum() for p, g in zip(model.parameters(), global_params))
                loss = loss + 0.5 * proximal_mu * prox
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            steps += 1
    return total_loss / max(steps, 1)


@torch.no_grad()
def test(model, loader, device):
    model.to(device)
    model.eval()
    correct, loss, total = 0, 0.0, 0
    for batch in loader:
        x, y = batch["x"].to(device), batch["y"].to(device)
        out = model(x)
        loss += F.cross_entropy(out, y, reduction="sum").item()
        correct += (out.argmax(1) == y).sum().item()
        total += y.size(0)
    return loss / total, correct / total
