import logging
from box import Box
from image_classification import Trainer
from image_classification.utils import ArgumentParser, init_logging, init_random_state, deep_update
from image_classification.modules import BenchmarkClassifier, BenchmarkConfig
from image_classification.datasets import CIFAR10Dataset, CIFAR100Dataset, CIFARConfig


def log(trainer: Trainer):
    print(f"Epoch {trainer.opt.current_epoch} completed.")


def main():
    # Read the default yaml settings and command args
    args = ArgumentParser().parse_args()
    options = Box.from_yaml(filename=args.config)
    deep_update(options, args)

    # Initialize logging and random state
    init_logging(options.log_dir, options.name)
    init_random_state(seed=options.seed)

    # Create dataset, model
    logging.info("Creating model...")
    models = {'benchmark': (BenchmarkClassifier, BenchmarkConfig)}
    model = models[options.model_name][0](models[options.model_name][1](**options.mconf))

    logging.info("Preparing datasets...")
    datasets = {'CIFAR10': (CIFAR10Dataset, CIFARConfig),
                'CIFAR100': (CIFAR100Dataset, CIFARConfig)}
    train_dataset = datasets[options.dataset_name][0](datasets[options.dataset_name][1](**options.dconf), train=True)
    val_dataset = datasets[options.dataset_name][0](datasets[options.dataset_name][1](**options.dconf), train=False)

    # Create trainer and run training
    trainer = Trainer(options, model, train_dataset, val_dataset)
    trainer.run()


if __name__ == '__main__':
    main()
