import torch
from torch import nn
from dataclasses import dataclass
import logging

from ..utils import CosineAnnealingWithWarmupScheduler

@dataclass
class BenchmarkConfig:
    height: int = 32
    width: int = 32
    num_classes: int = 10

    # Optimizer parameters
    lr : float = 0.001
    betas: tuple[float, float] = (0.9, 0.999)
    weight_decay: float = 0.0001
    min_lr: float = 1e-5
    warmup_steps: int = 3




class BenchmarkClassifier(nn.Module):
    def __init__(self, config: BenchmarkConfig):
        super().__init__()
        self.config = config

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(64 * (self.config.height // 8) * (self.config.width // 8), 64)
        self.fc2 = nn.Linear(64, self.config.num_classes)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm1d(64)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x.shape = (B, C, H, W)
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.relu(self.conv3(x))
        x = self.pool(x)
        x = self.flatten(x)

        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.relu(self.bn(self.fc1(x)))
        x = self.fc2(x)

        # Return the logits
        return x

    def configure_optimizer(self, num_epochs: int) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.config.lr,
            betas=self.config.betas,
            weight_decay=self.config.weight_decay,
        )

        lr_scheduler = CosineAnnealingWithWarmupScheduler(
            optimizer,
            warmup_steps=self.config.warmup_steps,
            max_steps=num_epochs,
            max_lr=self.config.lr,
            min_lr=self.config.min_lr,
        )

        return optimizer, lr_scheduler
