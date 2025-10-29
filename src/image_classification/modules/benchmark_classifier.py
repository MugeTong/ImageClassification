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
    act_func: str = 'gelu'
    use_res: bool = False
    fc_dropout: float = 0
    conv_dropout: float = 0

    # Optimizer parameters
    lr : float = 0.001
    betas: tuple[float, float] = (0.9, 0.999)
    weight_decay: float = 0.01
    min_lr: float = 1e-5
    warmup_steps: int = 3




class BenchmarkClassifier(nn.Module):
    def __init__(self, config: BenchmarkConfig):
        super().__init__()
        self.config = config

        # Base model has referred the ResNet18 for expandability
        self.conv1 = nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=2)

        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.conv5 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.avg_pool = nn.AvgPool2d(kernel_size=8)
        self.act = nn.ReLU(inplace=True) if config.act_func == 'relu' else nn.GELU()


        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, self.config.num_classes)
        self.bn = nn.BatchNorm1d(64)
        self.conv_dropout = nn.Dropout2d(config.conv_dropout) if config.conv_dropout > 0 else nn.Identity()
        self.fc_dropout = nn.Dropout(config.fc_dropout) if config.fc_dropout > 0 else nn.Identity()


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.conv1(x))
        x = self.conv_dropout(x)

        if self.config.use_res:
            x += self.act(self.conv3(self.act(self.conv2(x))))
            x = self.fc_dropout(x)
            x += self.act(self.conv5(self.act(self.conv4(x))))
        else:
            x = self.act(self.conv3(self.act(self.conv2(x))))
            x = self.fc_dropout(x)
            x = self.act(self.conv5(self.act(self.conv4(x))))
        # Average the image to point.
        x = self.avg_pool(x)

        # x.shape = (B, C, H, W)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.fc_dropout(x)
        x = self.act(self.bn(self.fc1(x)))
        x = self.fc_dropout(x)
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
