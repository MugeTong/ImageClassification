# Image Classification

图像分类任务的各种探索和消融实验。（用来完成课程作业）

## 介绍
本仓库包含与使用深度学习技术进行图像分类相关的代码和实验。它包括各种架构、训练策略和评估指标的实现。

## 快速开始
要开始使用代码，请克隆仓库并安装所需的依赖项：

```bash
git clone https://github.com/MugeTong/ImageClassification.git
cd ImageClassification
pip install .
```
> [!IMPORTANT]
> 1. 强烈建议使用虚拟环境来管理依赖包。
> 2. 在运行之前，你可能需要单独安装 `PyTorch`。

使用`Make`管理代码运行：
```bash
make run
```
对于一张4090显卡来讲，合理的基准模型训练时间应该在一分钟左右。改进模型训练在3分钟左右。

### 训练和验证基准模型
```bash
make run
make eval
```

### 训练和验证改进模型
```bash
make run resnet
make eval resnet

```

## 在CIFAR00数据集上训练和验证

```bash
make run data_aug -- --dataset_name CIFAR100 --dconf.data_dir ./data/cifar100 --mconf.num_classes 100 --num_epochs 50

make eval data_aug -- --dataset_name CIFAR100 --dconf.data_dir ./data/cifar100 --mconf.num_classes 100 --weights_path ./logs/data_aug/checkpoint_epoch_49.pth
```

## 致谢
- [上海交通大学](https://www.sjtu.edu.cn/)
