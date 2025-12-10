# Model Development

This directory contains model training, development, and experimentation code.

## Structure

- `configs/` - Training configurations
- `scripts/` - Training and development scripts
- `notebooks/` - Experiment notebooks
- `experiments/` - Experiment results and artifacts

## Usage

Train a model:

```bash
python model_development/scripts/train_gemini.py
```

Run hyperparameter tuning:

```bash
python model_development/scripts/hyperparameter_tuning.py
```

Track experiments with MLflow:

```bash
python model_development/scripts/train_with_tracking.py
```

## Model Registry

Models are tracked and stored in MLflow. Access the model registry:

```bash
mlflow ui
```
