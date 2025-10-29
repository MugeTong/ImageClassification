# Image Classification

Various explorations and ablation experiments for image classification tasks.

## Introduction

This repository contains code and experiments related to image classification using deep learning techniques. It includes implementations of various architectures, training strategies, and evaluation metrics.

## Quick Start
To get started with the code, clone the repository and install the required dependencies:

```bash
git clone https://github.com/MugeTong/ImageClassification.git
cd ImageClassification
pip install -r requirements.txt
```
> We strongly recommend using a virtual environment to manage dependencies.
> You may need to install `Pytorch` seperately before you run this project.

We use `Make` to manage the code running.

```bash
make run
```
For RTX4090, it is reasonable to finish the basic model training in one minute. It costs three minutes to train one improved model.

### Train and validate the basic model
```bash
make run
make eval
```

### Train and validate the improved model
```bash
make run resnet
make eval resnet

```

## Train and validate on the CIFAR100 dataset

```bash
make run data_aug -- --dataset_name CIFAR100 --dconf.data_dir ./data/cifar100 --mconf.num_classes 100 --num_epochs 50

make eval data_aug -- --dataset_name CIFAR100 --dconf.data_dir ./data/cifar100 --mconf.num_classes 100 --weights_path ./logs/data_aug/checkpoint_epoch_49.pth
```


## Thanks
- [Shanghai Jiao Tong University](https://www.sjtu.edu.cn/)
