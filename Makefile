bronze:
	python -m src.bronze.load_bronze

silver: bronze
	python -m src.silver.build_silver

gold: silver
	python -m src.gold.build_gold

train-sklearn: gold
	python -m src.models.train_sklearn --model all --split all

train-pytorch: gold
	python -m src.models.train_pytorch --model all --split all

train: train-sklearn train-pytorch

test:
	pytest --cov=src --cov-report=term-missing

lint:
	ruff check src/ tests/
	basedpyright src/ tests/

reset-db:
	python -m src.db

reset-ml:
	rm -rf mlruns/* models/*

reset: reset-db reset-ml
