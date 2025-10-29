from dataclasses import dataclass
import random
import torch
from torchvision import datasets, transforms


@dataclass
class CIFARConfig:
    data_dir: str
    download: bool = True
    allow_flip: bool = False
    allow_rotation: bool = False
    allow_aug: bool = False
    allow_crop: bool = False


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
        self.allow_rotation = config.allow_rotation and train
        self.allow_crop = config.allow_crop and train

        self.brightness = (0.8, 1.2)
        self.contrast = (0.8, 1.2)
        self.saturation = (0.8, 1.2)
        self.hue = (-0.1, 0.1)

        # Record transforms
        transforms_lst = []
        if self.allow_flip:
            transforms_lst.append(transforms.RandomHorizontalFlip())
        if self.allow_rotation:
            transforms_lst.append(transforms.RandomRotation(15))
        if self.allow_crop:
            transforms_lst.append(transforms.RandomCrop(32, padding=4))
        if self.allow_aug:
            transforms_lst.append(transforms.ColorJitter(
                brightness=self.brightness,
                contrast=self.contrast,
                saturation=self.saturation,
                hue=self.hue,
            ))

        self.cus_transform = transforms.Compose([
            *transforms_lst,
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

        inputs['img'] = self.cus_transform(img)

        return inputs, label


class CIFAR100Dataset(datasets.CIFAR100):
    def __init__(self, config: CIFARConfig, train: bool = False) -> None:
        super().__init__(
            root=config.data_dir,
            train=train,
            download=True,
            transform=None, # Use custom transform below
            target_transform=None, # Use custom target transform below
        )
        # Find the mean and std at https://blog.csdn.net/qq_45589658/article/details/109440786
        self.mean = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
        self.std = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)

        # Record data augmentation option
        self.allow_flip = config.allow_flip and train
        self.allow_aug = config.allow_aug and train
        self.allow_rotation = config.allow_rotation and train
        self.allow_crop = config.allow_crop and train

        self.brightness = (0.8, 1.2)
        self.contrast = (0.8, 1.2)
        self.saturation = (0.8, 1.2)
        self.hue = (-0.1, 0.1)

        # Record transforms
        transforms_lst = []
        if self.allow_flip:
            transforms_lst.append(transforms.RandomHorizontalFlip())
        if self.allow_rotation:
            transforms_lst.append(transforms.RandomRotation(15))
        if self.allow_crop:
            transforms_lst.append(transforms.RandomCrop(32, padding=4))
        if self.allow_aug:
            transforms_lst.append(transforms.ColorJitter(
                brightness=self.brightness,
                contrast=self.contrast,
                saturation=self.saturation,
                hue=self.hue,
            ))

        self.cus_transform = transforms.Compose([
            *transforms_lst,
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

        inputs['img'] = self.cus_transform(img)

        return inputs, label
