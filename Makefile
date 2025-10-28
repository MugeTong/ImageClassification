# Define commands
CONFIG := $(if $(word 2,$(MAKECMDGOALS)),$(word 2,$(MAKECMDGOALS)),default)
ARGS := $(wordlist 3,$(words $(MAKECMDGOALS)),$(MAKECMDGOALS))


run:
	python ./scripts/train.py --config ./configs/$(CONFIG).yaml $(ARGS)

infer:
	python ./experiments/infer.py --config ./configs/$(CONFIG).yaml $(ARGS)

format:
	yapf -ir --style=./.style.yapf .

logs:
	python -m tensorboard.main --logdir=./logs


# To prevent make from attempting to build a second target, add the catch-all rule
%:
	@:
