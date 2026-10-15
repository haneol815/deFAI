import copy

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


class Client:
    def __init__(self, cid, dataset, cfg, device):
        self.cid = cid
        self.dataset = dataset
        self.cfg = cfg
        self.device = device

    @property
    def num_samples(self):
        return len(self.dataset)

    def train(self, global_model):
        """Run local training from the global weights; return (state_dict, num_samples, loss)."""
        c = self.cfg["client"]
        model = copy.deepcopy(global_model).to(self.device)
        model.train()
        global_params = [p.detach().clone() for p in model.parameters()]

        loader = DataLoader(self.dataset, batch_size=c["batch_size"], shuffle=True)
        opt = torch.optim.SGD(model.parameters(), lr=c["lr"], momentum=c["momentum"])
        use_prox = self.cfg["fl"]["algorithm"] == "fedprox"
        mu = self.cfg["fl"].get("mu", 0.0)

        total_loss, steps = 0.0, 0
        for _ in range(c["local_epochs"]):
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                opt.zero_grad()
                loss = F.cross_entropy(model(x), y)
                if use_prox:
                    prox = sum(((p - g) ** 2).sum() for p, g in zip(model.parameters(), global_params))
                    loss = loss + 0.5 * mu * prox
                loss.backward()
                opt.step()
                total_loss += loss.item()
                steps += 1

        state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        return state, self.num_samples, total_loss / max(steps, 1)
