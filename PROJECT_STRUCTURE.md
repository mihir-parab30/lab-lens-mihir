# Lab Lens - Project Structure

## Overview

This repository follows standard MLOps best practices with a clear separation of concerns across different stages of the machine learning lifecycle.

## Directory Structure

```
lab-lens/
├── README.md                      # Main project documentation
├── LICENSE                        # Project license
├── requirements.txt               # Main Python dependencies
├── .gitignore                     # Git ignore rules
│
├── data/                          # Data storage (gitignored)
│   ├── raw/                       # Raw data from sources
│   ├── processed/                 # Processed/cleaned data
│   ├── external/                  # External datasets
│   └── .gitkeep
│
├── data_preprocessing/            # Data preprocessing pipeline
│   ├── __init__.py
│   ├── README.md                  # Preprocessing documentation
│   ├── configs/                   # Preprocessing configurations
│   │   └── pipeline_config.json
│   ├── scripts/                   # Preprocessing scripts
│   │   ├── __init__.py
│   │   ├── data_acquisition.py   # Data acquisition from BigQuery
│   │   ├── preprocessing.py      # Data cleaning and preprocessing
│   │   ├── validation.py         # Data validation
│   │   ├── feature_engineering.py # Feature engineering
│   │   ├── bias_detection.py     # Bias detection
│   │   ├── automated_bias_handler.py # Bias mitigation
│   │   └── main_pipeline.py      # Main orchestration
│   ├── notebooks/                 # Exploration notebooks
│   │   └── data_acquisition.ipynb
│   └── tests/                     # Preprocessing tests
│       ├── __init__.py
│       ├── test_preprocessing.py
│       └── test_validation.py
│
├── model_development/             # Model training and development
│   ├── __init__.py
│   ├── README.md                  # Model development docs
│   ├── configs/                   # Training configurations
│   ├── scripts/                   # Training scripts
│   │   ├── __init__.py
│   │   ├── train_gemini.py       # Gemini model training
│   │   ├── train_with_tracking.py # Training with MLflow
│   │   ├── hyperparameter_tuning.py # Hyperparameter optimization
│   │   ├── model_validation.py   # Model validation
│   │   ├── model_registry.py     # Model registry integration
│   │   └── ...
│   ├── notebooks/                 # Training notebooks
│   └── experiments/               # Experiment results
│
├── model_deployment/              # Model deployment
│   ├── __init__.py
│   ├── README.md                  # Deployment documentation
│   ├── api/                       # API deployment (FastAPI)
│   │   ├── __init__.py
│   │   ├── app.py                # FastAPI application
│   │   └── summarizer.py         # Summarization model
│   ├── web/                       # Web interface (Streamlit)
│   │   ├── __init__.py
│   │   └── file_qa_web.py        # Streamlit web app
│   ├── containerization/          # Container configs
│   │   └── ...
│   └── scripts/                   # Deployment scripts
│       └── deploy-to-cloud-run.sh
│
├── monitoring/                    # Monitoring and observability
│   ├── __init__.py
│   ├── README.md                  # Monitoring documentation
│   ├── metrics.py                 # Metrics collection
│   ├── logging/                   # Logging configurations
│   └── dashboards/                # Monitoring dashboards
│
├── infrastructure/                # Infrastructure as code
│   ├── README.md                  # Infrastructure docs
│   ├── docker/                    # Docker configurations
│   │   ├── Dockerfile
│   │   ├── Dockerfile.cloudrun
│   │   ├── docker-compose.yml
│   │   └── cloudbuild.yaml
│   ├── kubernetes/                # Kubernetes manifests (if needed)
│   ├── terraform/                 # Terraform configs (if needed)
│   └── ci_cd/                     # CI/CD workflows
│       └── .github/
│           └── workflows/
│
├── src/                           # Source code library
│   ├── __init__.py
│   ├── data/                      # Data utilities
│   │   └── __init__.py
│   ├── rag/                       # RAG system
│   │   ├── __init__.py
│   │   ├── rag_system.py         # Core RAG implementation
│   │   ├── file_qa.py            # File Q&A system
│   │   ├── patient_qa.py         # Patient Q&A interface
│   │   ├── document_processor.py # Document processing
│   │   └── vector_db.py          # Vector database
│   └── utils/                     # Shared utilities
│       ├── __init__.py
│       ├── logging_config.py     # Logging configuration
│       ├── error_handling.py     # Error handling
│       └── medical_utils.py      # Medical utilities
│
├── notebooks/                     # Jupyter notebooks
│   ├── exploration/               # Data exploration
│   ├── experiments/               # Experiment notebooks
│   └── .gitkeep
│
├── tests/                         # Test suite
│   ├── __init__.py
│   ├── unit/                      # Unit tests
│   │   └── __init__.py
│   ├── integration/              # Integration tests
│   │   └── __init__.py
│   └── e2e/                       # End-to-end tests
│       └── __init__.py
│
├── configs/                       # Global configurations
│   └── ...
│
├── scripts/                       # Utility scripts
│   ├── setup.sh
│   ├── setup_gcp_auth.sh
│   └── ...
│
└── docs/                          # Documentation
    ├── README.md
    ├── deployment/                # Deployment guides
    ├── api/                       # API documentation
    └── ...
```

## Key Principles

1. **Separation of Concerns**: Each directory has a clear, single responsibility
2. **Modularity**: Code is organized into reusable modules
3. **Testability**: Tests are organized by type (unit, integration, e2e)
4. **Documentation**: Each major component has its own README
5. **Infrastructure as Code**: All infrastructure configs are version controlled

## Workflow

1. **Data Preprocessing**: `data_preprocessing/` - Clean and prepare data
2. **Model Development**: `model_development/` - Train and validate models
3. **Model Deployment**: `model_deployment/` - Deploy models to production
4. **Monitoring**: `monitoring/` - Monitor deployed models
5. **Infrastructure**: `infrastructure/` - Manage deployment infrastructure

## Import Paths

After restructuring, update imports to use the new structure:

```python
# Old
from src.utils.logging_config import get_logger
from data-pipeline.scripts.preprocessing import MIMICPreprocessor

# New
from src.utils.logging_config import get_logger
from data_preprocessing.scripts.preprocessing import MIMICPreprocessor
```
