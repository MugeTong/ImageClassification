from dataclasses import dataclass
import random
import torch
from torchvision import datasets, transforms


@dataclass
class CIFARConfig:
    data_dir: str
    download: bool = True
    allow_flip: bool = False
    allow_aug: bool = False


class CIFAR10Dataset(datasets.CIFAR10):
    def __init__(self, config: CIFARConfig, train: bool = False) -> None:
        super().__init__(
            root=config.data_dir,
            train=train,
            download=True,
            transform=None, # Use custom transform below
            target_transform=None, # Use custom target transform below
        )
        # Find the mean and std at https://dev59.com/jL_qa4cB1Zd3GeqPSe1D
        self.mean = (0.49139968, 0.48215827, 0.44653124)
        self.std = (0.24703233, 0.24348505, 0.26158768)

        # Record data augmentation option
        self.allow_flip = config.allow_flip and train
        self.allow_aug = config.allow_aug and train

        self.base_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std),
        ])

        self.flip_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std),
        ])

    def __getitem__(self, index: int) -> tuple[dict[str, torch.Tensor], int]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (inputs, target) where target is index of the target class.
        """
        img, label =  super().__getitem__(index)
        inputs = {}

        if self.allow_flip and random.random() > 0.5:
            inputs['img'] = self.flip_transform(img)
        else:
            inputs['img'] = self.base_transform(img)

        return inputs, label


class CIFAR100Dataset(datasets.CIFAR100):
    pass
