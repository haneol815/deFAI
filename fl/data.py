import numpy as np
from torch.utils.data import Subset
from torchvision import datasets, transforms

_TRANSFORMS = {
    "mnist": transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ]),
    "cifar10": transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ]),
}

_DATASETS = {"mnist": datasets.MNIST, "cifar10": datasets.CIFAR10}


def load_dataset(name, root):
    cls, tf = _DATASETS[name], _TRANSFORMS[name]
    train = cls(root, train=True, download=True, transform=tf)
    test = cls(root, train=False, download=True, transform=tf)
    return train, test


def partition_iid(num_samples, num_clients, rng):
    idx = rng.permutation(num_samples)
    return [part.tolist() for part in np.array_split(idx, num_clients)]


def partition_dirichlet(labels, num_clients, alpha, rng):
    """Non-IID split: per class, draw client proportions from Dir(alpha)."""
    labels = np.asarray(labels)
    client_idx = [[] for _ in range(num_clients)]
    for c in np.unique(labels):
        idx_c = rng.permutation(np.where(labels == c)[0])
        props = rng.dirichlet([alpha] * num_clients)
        cuts = (np.cumsum(props) * len(idx_c)).astype(int)[:-1]
        for i, part in enumerate(np.split(idx_c, cuts)):
            client_idx[i].extend(part.tolist())
    return client_idx


def make_client_datasets(train_set, cfg, seed):
    rng = np.random.default_rng(seed)
    n = cfg["fl"]["num_clients"]
    if cfg["data"]["partition"] == "iid":
        parts = partition_iid(len(train_set), n, rng)
    elif cfg["data"]["partition"] == "dirichlet":
        parts = partition_dirichlet(train_set.targets, n, cfg["data"]["alpha"], rng)
    else:
        raise ValueError(f"unknown partition: {cfg['data']['partition']}")
    return [Subset(train_set, p) for p in parts]
