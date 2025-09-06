.PHONY: install train-all train-rf-smote train-rf-weights train-isolation clean

install:
	@echo "Installing dependencies..."
	pip install --upgrade pip
	pip install -r requirements.txt
	@echo "Dependencies installed"

train-all: train-rf-smote train-rf-weights train-isolation

train-rf-smote:
	@echo "Training Random Forest with SMOTE..."
	python scripts/run_pipeline.py --config src/configs/random_forest_smote.yaml

train-rf-weights:
	@echo "Training Random Forest with Class Weights..."
	python scripts/run_pipeline.py --config src/configs/random_forest_class_weights.yaml

train-isolation:
	@echo "Training Isolation Forest..."
	python scripts/run_pipeline.py --config src/configs/isolation_forest.yaml

clean:
	@echo "Cleaning experiment outputs..."
	rm -rf experiments/*
	@echo "Clean complete"
