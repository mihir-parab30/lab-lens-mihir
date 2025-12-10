# Repository Restructuring Plan

## Target Structure (Standard MLOps)

```
lab-lens/
├── README.md
├── LICENSE
├── .gitignore
├── requirements.txt
│
├── data/                          # Data storage (gitignored)
│   ├── raw/                      # Raw data
│   ├── processed/                # Processed data
│   ├── external/                  # External datasets
│   └── .gitkeep
│
├── data_preprocessing/            # Data preprocessing pipeline
│   ├── __init__.py
│   ├── README.md
│   ├── configs/                   # Preprocessing configs
│   │   └── pipeline_config.json
│   ├── scripts/                   # Preprocessing scripts
│   │   ├── __init__.py
│   │   ├── data_acquisition.py
│   │   ├── preprocessing.py
│   │   ├── validation.py
│   │   ├── feature_engineering.py
│   │   ├── bias_detection.py
│   │   ├── automated_bias_handler.py
│   │   └── main_pipeline.py
│   ├── notebooks/                 # Exploration notebooks
│   │   └── data_acquisition.ipynb
│   └── tests/                     # Preprocessing tests
│       ├── __init__.py
│       ├── test_preprocessing.py
│       └── test_validation.py
│
├── model_development/             # Model training and development
│   ├── __init__.py
│   ├── README.md
│   ├── configs/                   # Training configs
│   ├── scripts/                   # Training scripts
│   │   ├── __init__.py
│   │   ├── train_gemini.py
│   │   ├── train_with_tracking.py
│   │   ├── hyperparameter_tuning.py
│   │   └── ...
│   ├── notebooks/                 # Training notebooks
│   └── experiments/               # Experiment results
│
├── model_deployment/              # Model deployment
│   ├── __init__.py
│   ├── README.md
│   ├── api/                       # API code
│   │   ├── __init__.py
│   │   ├── app.py                 # FastAPI app
│   │   └── summarizer.py
│   ├── web/                       # Web interface
│   │   ├── __init__.py
│   │   └── file_qa_web.py
│   ├── containerization/          # Docker files
│   │   ├── Dockerfile
│   │   ├── Dockerfile.cloudrun
│   │   └── docker-compose.yml
│   └── scripts/                   # Deployment scripts
│       └── deploy-to-cloud-run.sh
│
├── monitoring/                    # Monitoring and observability
│   ├── __init__.py
│   ├── README.md
│   ├── metrics.py
│   ├── logging/                   # Logging configs
│   └── dashboards/                # Monitoring dashboards
│
├── infrastructure/                # Infrastructure as code
│   ├── README.md
│   ├── docker/                    # Docker configs
│   │   ├── Dockerfile
│   │   └── .dockerignore
│   ├── kubernetes/                # K8s manifests (if needed)
│   ├── terraform/                 # Terraform configs (if needed)
│   └── ci_cd/                     # CI/CD workflows
│       └── .github/
│           └── workflows/
│
├── src/                           # Source code library
│   ├── __init__.py
│   ├── data/                      # Data utilities
│   │   └── __init__.py
│   ├── preprocessing/             # Preprocessing modules
│   │   ├── __init__.py
│   │   └── ...
│   ├── models/                    # Model definitions
│   │   ├── __init__.py
│   │   └── ...
│   ├── rag/                       # RAG system
│   │   ├── __init__.py
│   │   ├── rag_system.py
│   │   ├── file_qa.py
│   │   ├── patient_qa.py
│   │   ├── document_processor.py
│   │   └── vector_db.py
│   ├── training/                  # Training utilities
│   │   ├── __init__.py
│   │   ├── gemini_inference.py
│   │   ├── gemini_model.py
│   │   ├── mlflow_tracking.py
│   │   ├── model_registry.py
│   │   └── ...
│   └── utils/                     # Shared utilities
│       ├── __init__.py
│       ├── logging_config.py
│       ├── error_handling.py
│       └── medical_utils.py
│
├── notebooks/                     # Jupyter notebooks
│   ├── exploration/               # Data exploration
│   ├── experiments/               # Experiment notebooks
│   └── .gitkeep
│
├── tests/                         # Integration tests
│   ├── __init__.py
│   ├── unit/                      # Unit tests
│   ├── integration/               # Integration tests
│   └── e2e/                       # End-to-end tests
│
├── configs/                       # Global configs
│   └── ...
│
├── scripts/                       # Utility scripts
│   ├── setup.sh
│   ├── setup_gcp_auth.sh
│   └── ...
│
├── docs/                          # Documentation
│   ├── README.md
│   ├── deployment/
│   ├── api/
│   └── ...
│
└── .github/                       # GitHub configs
    └── workflows/
```

## Migration Steps

1. Create new directory structure
2. Move files to appropriate locations
3. Update all import paths
4. Update configuration files
5. Update documentation
6. Test all functionality
7. Update CI/CD workflows
