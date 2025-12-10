# Migration Guide - Repository Restructuring

This guide helps you migrate from the old repository structure to the new standardized MLOps structure.

## Key Changes

### Directory Reorganization

| Old Location | New Location |
|-------------|-------------|
| `data-pipeline/scripts/` | `data_preprocessing/scripts/` |
| `data-pipeline/configs/` | `data_preprocessing/configs/` |
| `model-development/scripts/` | `model_development/scripts/` |
| `model-deployment/deployment_pipeline/` | `model_deployment/api/` |
| `scripts/file_qa_web.py` | `model_deployment/web/file_qa_web.py` |
| `src/training/` | `model_development/scripts/` |
| `Dockerfile*` | `infrastructure/docker/` |
| `.github/workflows/` | `infrastructure/ci_cd/.github/workflows/` |

### Import Path Updates

Update your imports to reflect the new structure:

#### Data Preprocessing

```python
# Old
from data-pipeline.scripts.preprocessing import MIMICPreprocessor
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

# New
from data_preprocessing.scripts.preprocessing import MIMICPreprocessor
from src.utils.logging_config import get_logger
```

#### Model Development

```python
# Old
from src.training.gemini_inference import GeminiInference

# New
from model_development.scripts.gemini_inference import GeminiInference
# OR if moved to src/
from src.training.gemini_inference import GeminiInference
```

#### Model Deployment

```python
# Old
from deployment_pipeline.summarizer import MedicalSummarizer

# New
from model_deployment.api.summarizer import MedicalSummarizer
```

### Configuration Updates

Update paths in configuration files:

1. **Pipeline Config** (`data_preprocessing/configs/pipeline_config.json`):
   - Update data paths to use `data/raw` and `data/processed`

2. **Dockerfiles** (`infrastructure/docker/`):
   - Update COPY paths to reflect new structure
   - Update WORKDIR if needed

3. **CI/CD Workflows** (`infrastructure/ci_cd/.github/workflows/`):
   - Update file paths in workflow triggers
   - Update build contexts

### Script Updates

Update scripts that reference old paths:

1. **Deployment Scripts**:
   ```bash
   # Old
   streamlit run scripts/file_qa_web.py
   
   # New
   streamlit run model_deployment/web/file_qa_web.py
   ```

2. **Docker Builds**:
   ```bash
   # Old
   docker build -f Dockerfile.cloudrun .
   
   # New
   docker build -f infrastructure/docker/Dockerfile.cloudrun .
   ```

## Migration Steps

1. **Update Imports**: Search and replace import statements
2. **Update Paths**: Update file paths in scripts and configs
3. **Test**: Run tests to ensure everything works
4. **Update Documentation**: Update README and docs
5. **Commit**: Commit the restructuring

## Testing After Migration

Run these commands to verify the migration:

```bash
# Test data preprocessing
python data_preprocessing/scripts/main_pipeline.py --help

# Test model training
python model_development/scripts/train_gemini.py --help

# Test API
cd model_deployment/api && python app.py

# Test web interface
streamlit run model_deployment/web/file_qa_web.py
```

## Rollback

If you need to rollback, the old structure is preserved in git history. You can:

```bash
git checkout <old-commit> -- data-pipeline/
git checkout <old-commit> -- model-deployment/
```

## Questions?

Refer to `PROJECT_STRUCTURE.md` for the complete new structure.
