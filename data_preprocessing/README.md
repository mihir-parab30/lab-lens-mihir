# Data Preprocessing Pipeline

This directory contains the data preprocessing pipeline for Lab Lens, including data acquisition, cleaning, validation, feature engineering, and bias detection.

## Structure

- `configs/` - Configuration files for the preprocessing pipeline
- `scripts/` - Preprocessing scripts
- `notebooks/` - Jupyter notebooks for data exploration
- `tests/` - Unit tests for preprocessing components

## Usage

Run the complete preprocessing pipeline:

```bash
python data_preprocessing/scripts/main_pipeline.py
```

Run individual components:

```bash
# Data acquisition
python data_preprocessing/scripts/data_acquisition.py

# Preprocessing
python data_preprocessing/scripts/preprocessing.py

# Validation
python data_preprocessing/scripts/validation.py

# Feature engineering
python data_preprocessing/scripts/feature_engineering.py

# Bias detection
python data_preprocessing/scripts/bias_detection.py
```

## Configuration

Edit `configs/pipeline_config.json` to configure the pipeline behavior.
