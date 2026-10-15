import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


def fedavg(updates):
    """Weighted average of client state_dicts. updates: list of (state_dict, num_samples)."""
    total = sum(n for _, n in updates)
    avg = {}
    for k in updates[0][0]:
        avg[k] = sum(sd[k].float() * (n / total) for sd, n in updates).to(updates[0][0][k].dtype)
    return avg


class Server:
    def __init__(self, model, clients, cfg, device, rng):
        self.model = model.to(device)
        self.clients = clients
        self.cfg = cfg
        self.device = device
        self.rng = rng

    def sample_clients(self):
        k = self.cfg["fl"]["clients_per_round"]
        ids = self.rng.choice(len(self.clients), size=k, replace=False)
        return [self.clients[i] for i in ids]

    def run_round(self):
        selected = self.sample_clients()
        results = [c.train(self.model) for c in selected]
        new_state = fedavg([(sd, n) for sd, n, _ in results])
        self.model.load_state_dict(new_state)
        train_loss = sum(l * n for _, n, l in results) / sum(n for _, n, _ in results)
        return train_loss, [c.cid for c in selected]

    @torch.no_grad()
    def evaluate(self, test_set, batch_size=256):
        self.model.eval()
        loader = DataLoader(test_set, batch_size=batch_size)
        correct, loss, total = 0, 0.0, 0
        for x, y in loader:
            x, y = x.to(self.device), y.to(self.device)
            out = self.model(x)
            loss += F.cross_entropy(out, y, reduction="sum").item()
            correct += (out.argmax(1) == y).sum().item()
            total += y.size(0)
        return loss / total, correct / total
