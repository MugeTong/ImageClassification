from collections import defaultdict
import time
import torch
from torch import Generator, nn
from torch.utils.tensorboard.writer import SummaryWriter
from torch.utils.data import DataLoader, Dataset
import os
from box import Box
import logging

from .modules import BenchmarkClassifier
from .utils import seconds2hms

class Trainer:
    """A simple image classifier class."""

    def __init__(self, options: Box, model: BenchmarkClassifier, train_dataset: Dataset, val_dataset: Dataset):
        super().__init__()
        self.opt = options

        self.generator = Generator().manual_seed(self.opt.seed)
        self.device = torch.device('cuda' if torch.cuda.is_available() and not self.opt.no_cuda else 'cpu')
        # Log writers
        self.writers = {
            'train': SummaryWriter(os.path.join(self.opt.log_dir, self.opt.name, 'train')),
            'val': SummaryWriter(os.path.join(self.opt.log_dir, self.opt.name, 'val')),
        }

        self.model = model.to(self.device)
        self.optimizer, self.lr_scheduler = self.model.configure_optimizer(self.opt.num_epochs)

        logging.info("Preparing datasets and dataloaders...")
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.opt.batch_size,
            shuffle=True,
            num_workers=self.opt.num_workers,
            generator=self.generator,
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.opt.batch_size,
            shuffle=False,
            num_workers=self.opt.num_workers,
            generator=self.generator,
        )
        self.val_iter = iter(self.val_loader)
        self.num_total_steps = self.opt.num_epochs * len(self.train_loader)

        # Total epoch steps and batch steps
        self.epoch_start = 0
        self.batch_start = 0
        self.epoch_step = 0
        self.batch_step = 0
        self.load_weights(self.opt.weights_path)

    def save_weights(self):
        """Save model weights to the specified path."""
        model_meta = {
            'state_dict': self.model.state_dict(),
            'epoch': self.epoch_step,
            'batch': self.batch_step,
            'optimizer': self.optimizer.state_dict(),
        }
        torch.save(model_meta, f'{self.opt.log_dir}/{self.opt.name}/checkpoint_epoch_{self.epoch_step}.pth')

    def load_weights(self, path: str):
        """Load model weights from the specified path."""
        if path is None: return
        logging.info(f"Loading weights from {path}...")
        model_meta = torch.load(path, map_location='cpu')
        self.model.load_state_dict(model_meta['state_dict'])
        self.epoch_start = model_meta['epoch']
        self.batch_start = model_meta['batch'] - 1
        self.batch_step = self.batch_start

        # Update the optimizer state
        self.optimizer.load_state_dict(model_meta['optimizer'])

    def run(self):
        """Run the training process."""
        logging.info("Starting training...")
        self.time_start = time.time()
        for _ in range(self.epoch_start):
            self.lr_scheduler.step()

        # Training loop
        for self.epoch_step in range(self.epoch_start, self.opt.num_epochs):
            self.run_epoch()  # Training the whole epoch
            self.lr_scheduler.step()  # Change learning rate

            if (self.epoch_step + 1) % self.opt.save_freq == 0:
                self.save_weights()

        # Close log writers
        for writer in self.writers.values():
            writer.close()

        logging.info("Training completed.")

    def run_epoch(self):
        """Process one training epoch."""
        self.model.train()
        logging.info(f"Starting epoch {self.epoch_step + 1}/{self.opt.num_epochs}...")
        for batch_idx, (inputs, targets) in enumerate(self.train_loader):
            # Move data to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            targets = targets.to(self.device)

            # Forward pass
            outputs = self.process_train_batch(inputs)
            loss = nn.functional.cross_entropy(outputs, targets)
            # Backward pass and optimization
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.batch_step += 1

            # Logging
            if (batch_idx + 1) % self.opt.log_freq == 0:
                self.log_progress(batch_idx + 1, loss.item())
                self.log('train', inputs, outputs, loss, targets)

                self.validate()


    def process_train_batch(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        """Process a single training batch."""
        outputs = self.model(inputs['img'])

        return outputs

    def validate(self):
        """Run validation on a batch from the validation set."""
        self.model.eval()
        try:
            inputs, targets = next(self.val_iter)
        except StopIteration:
            self.val_iter = iter(self.val_loader)
            inputs, targets = next(self.val_iter)

        # Move data to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        targets = targets.to(self.device)

        with torch.no_grad():
            outputs = self.model(inputs['img'])
            loss = nn.functional.cross_entropy(outputs, targets)

        self.log('val', inputs, outputs, loss, targets)
        self.model.train()


    def log_progress(self, batch_idx: int, loss: float):
        """Log the training progress.

        Args:
            batch_idx: The batch index for current epoch.
            loss: Loss value.
        """
        time_elapsed = time.time() - self.time_start  # Get the elapsed time since training
        # Compute average speed
        delta_steps = self.batch_step + 1 - self.batch_start
        speed = delta_steps * self.opt.batch_size / time_elapsed
        # Estimate time left
        time_left = ((self.num_total_steps - self.batch_start) / delta_steps - 1.0) * time_elapsed \
            if delta_steps != 1 else 0

        logging.info(f"Epoch {self.epoch_step:>2} | Batch {batch_idx:>4} | example/s: {speed:4.1f} | loss: {loss:.5f} | "
                     f"lr: {self.optimizer.param_groups[0]['lr']:.6f} | "
                     f"Elapsed: {seconds2hms(time_elapsed)} | ETA: {seconds2hms(time_left)}")

    def log(self, phase: str, inputs: dict[str, float], outputs, loss, targets):
        """Log metrics to TensorBoard.

        Args:
            phase: 'train' or 'val'.
            metrics: A dictionary of metric names and their values.
            step: The current training step.
        """
        writer = self.writers[phase]
        writer.add_scalar('loss', loss.item(), self.batch_step)
        _, preds = torch.max(outputs, 1)
        acc = torch.sum(preds == targets).item() / targets.size(0)
        writer.add_scalar('accuracy', acc, self.batch_step)
