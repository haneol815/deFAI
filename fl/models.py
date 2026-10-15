import torch.nn as nn


class CNN(nn.Module):
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
    if dataset == "mnist":
        return CNN(in_channels=1, img_size=28)
    if dataset == "cifar10":
        return CNN(in_channels=3, img_size=32)
    raise ValueError(f"unknown dataset: {dataset}")
