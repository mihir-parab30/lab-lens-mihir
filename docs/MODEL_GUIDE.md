# Complete Model Development & Testing Guide

Complete guide for model development, testing, validation, and deployment in the Lab Lens platform.

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Model Development](#model-development)
4. [Model Testing](#model-testing)
5. [Requirements Checklist](#requirements-checklist)
6. [Model Alternatives](#model-alternatives)
7. [Best Practices](#best-practices)
8. [Troubleshooting](#troubleshooting)

---

## Overview

This guide covers the complete model development lifecycle:

1. **Loading Data from Data Pipeline** - Seamless integration with versioned data
2. **Training and Selecting Best Model** - Model training with performance-based selection
3. **Model Validation** - Comprehensive validation with ROUGE and BLEU metrics
4. **Model Bias Detection** - Slicing techniques across demographic groups
5. **Hyperparameter Tuning** - Bayesian optimization with Optuna
6. **Experiment Tracking** - MLflow integration for all experiments
7. **Sensitivity Analysis** - Feature importance and hyperparameter sensitivity
8. **Model Registry** - Version control and reproducibility
9. **CI/CD Pipeline** - Automated training, validation, and bias detection
10. **Testing** - Comprehensive testing procedures

---

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Up API Key

```bash
python scripts/setup_gemini_api_key.py
```

### 3. Run Complete Training Pipeline

```bash
python src/training/train_with_tracking.py \
  --data-path data-pipeline/data/processed/processed_discharge_summaries.csv \
  --config configs/gemini_config.json \
  --output-dir models/gemini \
  --enable-tuning \
  --run-name "experiment-1"
```

### 4. Test Models

```bash
# Test all functionalities
python scripts/test_all_models.py

# Test with image
python scripts/test_all_models.py --image-path /path/to/chest_xray.jpg
```

---

## Model Development

### 1. Data Loading from Pipeline

The `CompleteModelTrainer` automatically loads data from the data pipeline with proper train/validation/test splits:

```python
from src.training import CompleteModelTrainer

trainer = CompleteModelTrainer()
train_df, val_df, test_df = trainer.load_data_from_pipeline(
    data_path='data-pipeline/data/processed/processed_discharge_summaries.csv',
    train_split=0.8,
    val_split=0.1
)
```

### 2. Model Training

Training with automatic hyperparameter optimization:

```python
from src.training import CompleteModelTrainer

trainer = CompleteModelTrainer(
    enable_hyperparameter_tuning=True,
    enable_bias_detection=True,
    enable_sensitivity_analysis=True
)

results = trainer.train_and_evaluate(
    data_path='data-pipeline/data/processed/processed_discharge_summaries.csv'
)
```

### 3. Model Validation

Validation with ROUGE and BLEU metrics:

```python
from src.training import ModelValidator

validator = ModelValidator()

# Validate from lists
metrics = validator.validate_model(predictions, references)

# Validate from DataFrame
metrics = validator.validate_from_dataframe(
    df,
    prediction_column='gemini_summary',
    reference_column='cleaned_text'
)

print(f"ROUGE-L F1: {metrics['rougeL_f']:.4f}")
print(f"BLEU: {metrics['bleu']:.4f}")
```

**Metrics Calculated:**
- ROUGE-1 (precision, recall, F1)
- ROUGE-2 (precision, recall, F1)
- ROUGE-L (precision, recall, F1)
- ROUGE-Lsum (precision, recall, F1)
- BLEU score
- Overall score (weighted combination)

### 4. Hyperparameter Tuning

Bayesian optimization using Optuna:

```python
from src.training import HyperparameterTuner

tuner = HyperparameterTuner(
    api_key=os.getenv('GOOGLE_API_KEY'),
    model_name='gemini-2.0-flash-exp',
    n_trials=20
)

study = tuner.optimize(
    train_data=train_df,
    val_data=val_df,
    input_column='cleaned_text',
    reference_column='cleaned_text'
)

best_params = tuner.get_best_hyperparameters(study)
print(f"Best temperature: {best_params['temperature']}")
print(f"Best max_output_tokens: {best_params['max_output_tokens']}")
```

**Hyperparameters Tuned:**
- `temperature`: 0.1 - 1.0 (sampling temperature)
- `max_output_tokens`: 100 - 500 (output length)
- `max_length`: 50 - 200 (summary length)

### 5. Experiment Tracking with MLflow

Automatic experiment tracking:

```python
from src.training import MLflowTracker

with MLflowTracker(experiment_name="gemini-experiments") as tracker:
    # Log hyperparameters
    tracker.log_hyperparameters({
        'temperature': 0.3,
        'max_output_tokens': 2048
    })
    
    # Log metrics
    tracker.log_metrics({
        'rougeL_f': 0.45,
        'bleu': 0.38
    })
    
    # Log model
    tracker.log_model(model)
    
    # Log artifacts
    tracker.log_artifact('bias_report.json')
```

**Tracked Information:**
- Hyperparameters
- Metrics (ROUGE, BLEU, bias scores)
- Model configurations
- Artifacts (reports, plots)
- Tags and metadata

### 6. Model Bias Detection

Bias detection using slicing techniques:

```python
from src.training import ModelBiasDetector

detector = ModelBiasDetector(bias_threshold=0.1)

bias_report = detector.detect_bias(
    df=df_with_predictions,
    prediction_column='gemini_summary',
    reference_column='cleaned_text',
    demographic_columns=['gender', 'ethnicity_clean', 'age_group']
)

print(f"Overall bias score: {bias_report['overall_bias_score']:.4f}")
print(f"Bias alerts: {len(bias_report['bias_alerts'])}")

# Save report
detector.save_bias_report(bias_report, 'logs/bias_report.json')
```

**Slicing Analysis:**
- Performance metrics per demographic group
- Disparity calculations (coefficient of variation, max difference)
- Automatic bias alerts when thresholds exceeded
- Comprehensive bias report with visualizations

### 7. Sensitivity Analysis

Feature importance and hyperparameter sensitivity:

```python
from src.training import SensitivityAnalyzer

analyzer = SensitivityAnalyzer()

# Hyperparameter sensitivity
sensitivity = analyzer.analyze_hyperparameter_sensitivity(
    optimization_history=study_history_df
)

print(f"Most important hyperparameter: {list(sensitivity['hyperparameter_importance'].keys())[0]}")

# Create visualizations
plot_paths = analyzer.create_sensitivity_plots(
    optimization_history=study_history_df,
    output_dir='logs/sensitivity_plots'
)
```

**Analysis Types:**
- Hyperparameter sensitivity (correlation analysis)
- Feature importance (SHAP/LIME for text features)
- Optimization history visualization

### 8. Model Registry

Register models for version control:

```python
from src.training import ModelRegistry

# MLflow Model Registry
registry = ModelRegistry(registry_type='mlflow')
registration = registry.register_model(
    run_id='abc123',
    model_name='gemini-medical-summarization',
    stage='Production'
)

# GCP Artifact Registry
registry_gcp = ModelRegistry(
    registry_type='gcp',
    gcp_project='your-project',
    gcp_location='us-central1',
    gcp_repository='lab-lens-models'
)
registration = registry_gcp.register_model(
    model_path='models/gemini',
    model_version='v1.0.0',
    metadata={'description': 'Best model from experiment 1'}
)
```

### 9. CI/CD Pipeline

The GitHub Actions workflow (`.github/workflows/model_training_ci.yml`) automatically:

1. **Triggers on code changes** to training code
2. **Runs model training** with validation
3. **Performs bias detection** and checks thresholds
4. **Uploads artifacts** (MLflow runs, models, logs)
5. **Sends notifications** on failures

**Setup CI/CD:**
1. Add GitHub secret: `GOOGLE_API_KEY`
2. Push code to trigger workflow
3. Check Actions tab for results

---

## Model Testing

### Test Scripts

**Main Test Script:** `scripts/test_all_models.py`

This script tests all three functionalities:
1. **Discharge Summary Generation/Simplification**
2. **Risk Prediction from Discharge Summaries**
3. **Disease Detection from Biomedical Images**

### Usage

#### Test Discharge Summary + Risk Prediction

```bash
python scripts/test_all_models.py
```

#### Test All 3 (Including Image Disease Detection)

```bash
# Test with image
python scripts/test_all_models.py --image-path /path/to/chest_xray.jpg

# Test with custom patient info
python scripts/test_all_models.py \
    --image-path /path/to/chest_xray.jpg \
    --age 65 \
    --gender M \
    --symptoms "Chest pain, shortness of breath"
```

### Command Line Options

- `--image-path`: Path to medical image (chest X-ray, CT scan, etc.)
- `--age`: Patient age for risk prediction and image analysis
- `--gender`: Patient gender (M/F)
- `--symptoms`: Patient symptoms
- `--output`: Output file for test results (default: `test_results.json`)

### Testing Individual Functionalities

#### Test 1: Discharge Summary Generation

**What it tests:**
- Simplified summary generation from discharge summaries
- Patient-friendly language translation
- Key information extraction

**Expected output:**
- Simplified summary (150-200 words)
- Clear, non-technical language
- Key diagnoses, medications, and follow-up instructions

#### Test 2: Risk Prediction

**What it tests:**
- Risk level prediction (Low/Medium/High)
- Risk score calculation (0-100)
- Risk factor identification
- Recommendations generation

**Expected output:**
- Risk level: LOW/MEDIUM/HIGH
- Risk score: 0-100
- Risk factors: List of identified factors
- Recommendations: Suggested actions

**Sample patient info:**
```python
patient_info = {
    'age': 65,
    'gender': 'M',
    'cleaned_text': 'discharge summary text...',
    'abnormal_lab_count': 5,
    'diagnosis_count': 4,
    'length_of_stay': 3
}
```

#### Test 3: Image Disease Detection

**What it tests:**
- Disease detection from chest X-rays
- Abnormality identification
- Severity assessment
- Clinical impression generation

**Expected output:**
- Diseases detected: List of identified conditions
- Severity: Normal/Mild/Moderate/Severe
- Findings: Detailed findings from image
- Impression: Clinical interpretation

**Supported image types:**
- Chest X-rays (`.jpg`, `.jpeg`, `.png`)
- CT scans
- MRIs
- Other medical images

### Test Results

Test results are saved to `test_results.json` by default.

**Result Structure:**
```json
{
  "timestamp": "2024-01-15T10:30:00",
  "discharge_summary_test": {
    "status": "success",
    "summary": "...",
    "summary_length": 185
  },
  "risk_prediction_test": {
    "status": "success",
    "risk_level": "high",
    "risk_score": 75,
    "risk_factors": {...},
    "recommendations": [...]
  },
  "image_disease_detection_test": {
    "status": "success",
    "diseases_detected": [...],
    "severity": "moderate",
    "findings": [...],
    "has_disease": true
  },
  "overall_status": "success"
}
```

---

## Requirements Checklist

### ✅ Completed Requirements

#### 1. Loading Data from Data Pipeline
- **Status**: ✅ **COMPLETE**
- **Location**: `src/training/train_with_tracking.py`, `src/training/train_gemini.py`
- **Details**: Code loads data from `data-pipeline/data/processed/processed_discharge_summaries.csv` with proper versioning

#### 2. Training and Selecting Best Model
- **Status**: ✅ **COMPLETE**
- **Location**: `src/training/train_gemini.py`, `src/training/train_with_tracking.py`
- **Details**: Model training with performance-based selection using validation metrics

#### 3. Model Validation
- **Status**: ✅ **COMPLETE**
- **Location**: `src/training/model_validation.py`
- **Details**: 
  - ROUGE metrics (ROUGE-1, ROUGE-2, ROUGE-L)
  - BLEU scores
  - Validation on hold-out dataset
  - Performance metrics tracking

#### 4. Model Bias Detection (Using Slicing Techniques)
- **Status**: ✅ **COMPLETE**
- **Location**: `src/training/model_bias_detection.py`, `data-pipeline/scripts/bias_detection.py`
- **Details**:
  - Demographic slicing (gender, ethnicity, age)
  - Performance metrics across slices
  - Statistical significance testing
  - Bias visualization

#### 5. Code to Check for Bias
- **Status**: ✅ **COMPLETE**
- **Location**: `src/training/model_bias_detection.py`
- **Details**:
  - Automated bias checking across dataset slices
  - Bias reports and visualizations
  - Mitigation strategy suggestions

#### 6. Hyperparameter Tuning
- **Status**: ✅ **COMPLETE**
- **Location**: `src/training/hyperparameter_tuning.py`
- **Details**:
  - Optuna-based Bayesian optimization
  - Tunes: temperature, max_output_tokens, max_length
  - Search space documentation
  - Best parameter selection

#### 7. Experiment Tracking and Results
- **Status**: ✅ **COMPLETE**
- **Location**: `src/training/mlflow_tracking.py`
- **Details**:
  - MLflow integration
  - Tracks: hyperparameters, metrics, model versions
  - Visualizations (bar plots, confusion matrices)
  - Model comparison and selection

#### 8. Model Sensitivity Analysis
- **Status**: ✅ **COMPLETE**
- **Location**: `src/training/sensitivity_analysis.py`
- **Details**:
  - Feature importance (SHAP/LIME)
  - Hyperparameter sensitivity analysis
  - Impact analysis on model performance

#### 9. Pushing Model to Artifact/Model Registry
- **Status**: ✅ **COMPLETE**
- **Location**: `src/training/model_registry.py`
- **Details**:
  - MLflow Model Registry
  - GCP Artifact Registry support
  - Model versioning
  - Reproducibility

#### 10. CI/CD Pipeline Automation
- **Status**: ✅ **COMPLETE**
- **Location**: `.github/workflows/model_training_ci.yml`
- **Details**:
  - Automated model training on code push
  - Automated validation
  - Automated bias detection
  - Model registry push
  - Artifact uploads

### ⚠️ Needs Enhancement

#### 11. Rollback Mechanism
- **Status**: ⚠️ **PARTIAL** - Needs implementation
- **Current**: Model registry tracks versions but no automatic rollback
- **Required**: Implement rollback when new model performs worse

#### 12. Docker/RAG Format
- **Status**: ❌ **MISSING**
- **Required**: Docker containerization for reproducibility

---

## Model Alternatives

### Problem
Gemini 2.5 Pro may misclassify diseased chest X-rays as normal, which is a critical issue for medical diagnosis.

### Medical-Specific Model Options

#### 1. **Med-PaLM 2** (Google - Medical-Specific)
- **Specialization**: Specifically trained on medical data
- **Access**: Via Google Cloud Vertex AI
- **Advantages**: 
  - Trained on medical literature and images
  - Better medical domain knowledge
  - May have better safety filters for medical use
- **Disadvantages**: Requires Vertex AI setup
- **API**: `vertexai.generative_models.GenerativeModel`

#### 2. **Hugging Face Medical Models**

**a. CheXNet / CheXpert Models**
- **Model**: `microsoft/resnet-50` (trained on CheXpert)
- **Specialization**: Chest X-ray pathology detection
- **Available**: Hugging Face Hub
- **Diseases**: 14 common chest X-ray findings
- **Use**: Can be loaded via `transformers` library

**b. ChestX-ray14 Models**
- Pre-trained on NIH ChestX-ray14 dataset
- Detects multiple pathologies
- Available on Hugging Face

**c. RadImage Models**
- Medical imaging specific
- Various architectures available

#### 3. **Clarifai Medical Imaging API**
- Medical-specific API
- Trained on medical datasets
- Easy API integration
- **Cost**: Pay-per-use

#### 4. **AWS HealthImaging**
- Amazon's medical imaging service
- DICOM support
- May require enterprise setup

#### 5. **Custom Model Training**
- Train on your own medical datasets
- Fine-tune existing models
- More control but requires data and compute

### Recommended Approach

#### Option 1: Med-PaLM 2 (Best for Google Cloud users)
```python
from vertexai.generative_models import GenerativeModel

model = GenerativeModel("med-palm-2")
# Use similar to Gemini but with medical specialization
```

#### Option 2: Hugging Face CheXNet (Best for local/self-hosted)
```python
from transformers import pipeline

chest_xray_classifier = pipeline(
    "image-classification",
    model="microsoft/resnet-50",  # CheXpert trained
    # or other chest X-ray models
)
```

#### Option 3: Hybrid Approach
- Use Gemini for general analysis
- Use medical-specific model for disease detection
- Combine results

### Implementation Plan

1. **Add model selection option** to `MedicalImageAnalyzer`
2. **Implement Med-PaLM 2 integration**
3. **Implement Hugging Face model integration**
4. **Add fallback mechanism** (try medical model first, fallback to Gemini)
5. **Compare results** from different models

---

## Best Practices

### 1. Data Versioning

Use DVC for data versioning:
```bash
dvc add data-pipeline/data/processed/processed_discharge_summaries.csv
git add data-pipeline/data/processed/processed_discharge_summaries.csv.dvc
```

### 2. Experiment Naming

Use descriptive run names:
```python
run_name = f"experiment-{datetime.now().strftime('%Y%m%d')}-tuning-v1"
```

### 3. Model Selection

Select best model based on validation metrics:
```python
from src.training import get_best_run

best_run = get_best_run(
    experiment_name='gemini-medical-summarization',
    metric='rougeL_f',
    ascending=False  # Higher is better
)
```

### 4. Bias Mitigation

If bias detected:
1. Review bias report
2. Adjust training data (re-sampling, re-weighting)
3. Retrain with bias-aware techniques
4. Re-validate and check bias again

### 5. Hyperparameter Search Space

Adjust search space based on domain knowledge:
```python
# In HyperparameterTuner.objective()
temperature = trial.suggest_float('temperature', 0.1, 0.5, step=0.1)  # Narrower range
```

### 6. Testing Best Practices

- Always test on hold-out test set
- Test all three functionalities (summary, risk, images)
- Compare results across different models
- Document test results and metrics
- Set up automated testing in CI/CD

---

## Troubleshooting

### Development Issues

#### Issue: MLflow tracking fails
**Solution:** Ensure `mlruns/` directory is writable or set custom tracking URI

#### Issue: Bias detection finds no demographic columns
**Solution:** Check that DataFrame has columns like 'gender', 'ethnicity_clean', 'age_group'

#### Issue: Hyperparameter tuning is slow
**Solution:** Reduce `sample_size` parameter or `n_trials` in `HyperparameterTuner`

#### Issue: Validation metrics are low
**Solution:** 
- Check data quality
- Adjust hyperparameters
- Try different prompt engineering
- Increase training data

### Testing Issues

#### Issue: "GOOGLE_API_KEY not found"
**Solution:** Set the API key in environment:
```bash
export GOOGLE_API_KEY="your-key"
# Or add to .env file
echo "GOOGLE_API_KEY=your-key" >> .env
```

#### Issue: "Image file not found"
**Solution:** Provide correct path:
```bash
# Use absolute path
python scripts/test_all_models.py --image-path /absolute/path/to/image.jpg

# Or relative path from project root
python scripts/test_all_models.py --image-path data-pipeline/data/raw/images/xray.jpg
```

#### Issue: NumPy compatibility errors
**Solution:** 
```bash
pip install "numpy<2"
# Or
pip install --upgrade --force-reinstall pandas scikit-learn
```

#### Issue: Import errors
**Solution:** Install missing dependencies:
```bash
pip install -r requirements.txt
```

---

## Output Structure

```
models/gemini/
├── gemini_config.json          # Model configuration
├── training_results.json       # Training results
├── bias_report.json           # Bias detection report
├── optimization_history.csv    # Hyperparameter tuning history
└── sensitivity_plots/         # Sensitivity analysis plots
    ├── hyperparameter_sensitivity.png
    └── optimization_history.png

mlruns/                        # MLflow tracking
└── gemini-medical-summarization/
    └── runs/
        └── [run_id]/
            ├── metrics/
            ├── params/
            └── artifacts/
```

---

## Next Steps

1. **Production Deployment**: Set up model serving infrastructure
2. **Monitoring**: Implement model performance monitoring
3. **A/B Testing**: Compare model versions in production
4. **Automated Retraining**: Schedule periodic model updates
5. **Rollback Implementation**: Add automatic rollback mechanism
6. **Docker Containerization**: Create Docker images for reproducibility

---

## References

- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [Optuna Documentation](https://optuna.readthedocs.io/)
- [Fairlearn Documentation](https://fairlearn.org/)
- [SHAP Documentation](https://shap.readthedocs.io/)
- [Gemini API Documentation](https://ai.google.dev/gemini-api/docs)
