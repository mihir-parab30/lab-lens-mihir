# CI/CD Setup Guide

This document describes the CI/CD pipeline setup for Lab Lens.

## Overview

The CI/CD pipeline consists of multiple workflows that handle different aspects of the development lifecycle:

1. **CI (Continuous Integration)** - Testing and code quality
2. **Build and Test** - Docker image building and validation
3. **Deployment** - Automated deployment to Cloud Run
4. **Data Pipeline CI** - Data preprocessing validation
5. **Model Training CI** - Model development validation
6. **Release** - Automated release management

## Workflows

### 1. CI - Test and Lint (`ci.yml`)

**Triggers:**
- Pull requests to `main` or `develop`
- Pushes to `main` or `develop`

**Jobs:**
- **Lint**: Code formatting (Black, isort) and linting (flake8)
- **Test**: Unit tests with Python 3.11 and 3.12
- **Integration Test**: Integration tests
- **Security Scan**: Security vulnerability scanning (Bandit)

**Required Secrets:** None

### 2. Build and Test (`build-and-test.yml`)

**Triggers:**
- Pull requests to `main`
- Pushes to `main` or `develop`

**Jobs:**
- **Build Docker**: Builds Docker image and validates dependencies
- **Test Deployment**: Tests deployment script imports

**Required Secrets:** None

### 3. Deploy to Cloud Run (`deploy-cloud-run.yml`)

**Triggers:**
- Pushes to `main` branch
- Manual workflow dispatch

**Jobs:**
- **Deploy**: Builds and deploys to Google Cloud Run

**Required Secrets:**
- `GCP_PROJECT_ID` - Your GCP project ID
- `GCP_SA_KEY` - Service account JSON key

**Environments:**
- `production` (default)
- `staging` (optional)

### 4. Data Pipeline CI (`data-pipeline-ci.yml`)

**Triggers:**
- Changes to `data_preprocessing/` directory
- Manual workflow dispatch

**Jobs:**
- **Test Preprocessing**: Runs preprocessing tests
- **Lint Preprocessing**: Lints preprocessing code

**Required Secrets:** None

### 5. Model Training CI (`model-training-ci.yml`)

**Triggers:**
- Changes to `model_development/` directory
- Manual workflow dispatch (with optional training)

**Jobs:**
- **Test Model Development**: Validates model development code
- **Train Model**: Optional model training (manual trigger)

**Required Secrets:**
- `GEMINI_API_KEY` - For model training
- `MLFLOW_TRACKING_URI` - MLflow tracking server (optional)

### 6. Release (`release.yml`)

**Triggers:**
- Tag pushes (e.g., `v1.0.0`)
- Manual workflow dispatch

**Jobs:**
- **Create Release**: Creates GitHub release
- **Deploy Release**: Deploys release to production

**Required Secrets:**
- `GCP_PROJECT_ID`
- `GCP_SA_KEY`

## Setup Instructions

### 1. Configure GitHub Secrets

Go to your repository → Settings → Secrets and variables → Actions

Add the following secrets:

#### Required for Deployment:
- `GCP_PROJECT_ID`: Your Google Cloud project ID
  ```
  gen-lang-client-0006590375
  ```

- `GCP_SA_KEY`: Service account JSON key
  ```json
  {
    "type": "service_account",
    "project_id": "...",
    ...
  }
  ```

#### Optional:
- `GEMINI_API_KEY`: For model training and API features
- `MLFLOW_TRACKING_URI`: MLflow tracking server URI (if using)

### 2. Configure GitHub Environments (Optional)

For staging/production separation:

1. Go to Settings → Environments
2. Create `production` environment
3. Create `staging` environment (optional)
4. Add environment-specific secrets if needed

### 3. Set Up GCP Service Account

Create a service account with these roles:
- Cloud Run Admin
- Service Account User
- Storage Admin
- Cloud Build Editor

```bash
# Create service account
gcloud iam service-accounts create github-actions \
  --display-name="GitHub Actions Service Account"

# Grant roles
gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
  --member="serviceAccount:github-actions@YOUR_PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/run.admin"

gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
  --member="serviceAccount:github-actions@YOUR_PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/iam.serviceAccountUser"

gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
  --member="serviceAccount:github-actions@YOUR_PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/storage.admin"

# Create and download key
gcloud iam service-accounts keys create key.json \
  --iam-account=github-actions@YOUR_PROJECT_ID.iam.gserviceaccount.com
```

Copy the contents of `key.json` to GitHub secret `GCP_SA_KEY`.

### 4. Enable Required APIs

```bash
gcloud services enable \
  cloudbuild.googleapis.com \
  run.googleapis.com \
  containerregistry.googleapis.com
```

## Workflow Triggers

### Automatic Triggers

- **On Pull Request**: CI, Build and Test, Data Pipeline CI, Model Training CI
- **On Push to Main**: All workflows (including deployment)
- **On Tag Push**: Release workflow

### Manual Triggers

All workflows support `workflow_dispatch` for manual execution:
- Go to Actions tab
- Select workflow
- Click "Run workflow"

## Monitoring

### View Workflow Runs

1. Go to **Actions** tab in GitHub
2. Select a workflow to see runs
3. Click on a run to see detailed logs

### View Deployment Status

```bash
# Check Cloud Run service status
gcloud run services describe lab-lens-web \
  --region us-central1 \
  --project YOUR_PROJECT_ID

# View logs
gcloud run services logs tail lab-lens-web \
  --region us-central1 \
  --project YOUR_PROJECT_ID
```

## Troubleshooting

### Deployment Fails

1. **Check secrets**: Verify `GCP_PROJECT_ID` and `GCP_SA_KEY` are set
2. **Check permissions**: Ensure service account has required roles
3. **Check logs**: View workflow logs in GitHub Actions

### Tests Fail

1. **Check Python version**: Ensure tests run on supported Python versions
2. **Check dependencies**: Verify `requirements.txt` is up to date
3. **Check imports**: Ensure all imports are correct after restructuring

### Build Fails

1. **Check Dockerfile path**: Verify path in `cloudbuild.yaml`
2. **Check build context**: Ensure all required files are in the repository
3. **Check timeout**: Increase timeout if build takes longer

## Best Practices

1. **Always test locally** before pushing
2. **Use feature branches** and create PRs
3. **Review CI results** before merging
4. **Tag releases** for production deployments
5. **Monitor deployments** after they complete

## Workflow Status Badge

Add to your README.md:

```markdown
![CI](https://github.com/your-org/lab-lens/workflows/CI%20-%20Test%20and%20Lint/badge.svg)
![Deploy](https://github.com/your-org/lab-lens/workflows/Deploy%20to%20Google%20Cloud%20Run/badge.svg)
```
