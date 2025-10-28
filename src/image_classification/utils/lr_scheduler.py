import math
import torch

from torch import optim


class CosineAnnealingWithWarmupScheduler(optim.lr_scheduler._LRScheduler):
    """
    Cosine Annealing Learning Rate Scheduler with Warmup Steps.

    Args:
        optimizer (torch.optim.Optimizer): The optimizer for which to schedule the learning rate.
        warmup_steps (int): Number of steps for the warmup phase.
        max_steps (int): Total number of training steps.
        max_lr (float): The maximum learning rate.
        min_lr (float): The minimum learning rate after decay. Default is 1e-6.
        last_epoch (int): The index of the last epoch. Default is -1.
    """
    def __init__(self, optimizer, warmup_steps, max_steps, max_lr, min_lr: float=1e-6, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.max_lr = max_lr
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        current_step = self.last_epoch + 1

        # If the current step is smaller than the warmup_steps, in the warmup stage
        if current_step < self.warmup_steps:
            return [self.max_lr * current_step / self.warmup_steps for _ in self.optimizer.param_groups]
        elif self.warmup_steps <= current_step <= self.max_steps:
            # Else, enter the cosine annealing stage
            progress = (current_step - self.warmup_steps) / (self.max_steps - self.warmup_steps)

            cos_anneal_lr = self.min_lr + (self.max_lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))

            return [cos_anneal_lr for _ in self.optimizer.param_groups]
        else:
            return [self.min_lr for _ in self.optimizer.param_groups]
