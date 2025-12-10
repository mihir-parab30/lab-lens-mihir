# CI/CD Pipeline Overview

This repository uses GitHub Actions for continuous integration and deployment.

## 🚀 Quick Start

### Required GitHub Secrets

Configure these in: **Settings → Secrets and variables → Actions**

1. **GCP_PROJECT_ID** - Your Google Cloud project ID
2. **GCP_SA_KEY** - Service account JSON key (for deployments)

### Optional Secrets

- **GEMINI_API_KEY** - For model training and API features
- **MLFLOW_TRACKING_URI** - MLflow tracking server

## 📋 Available Workflows

### 1. CI - Test and Lint
**File:** `.github/workflows/ci.yml`

Runs on every PR and push to main/develop:
- ✅ Code formatting checks (Black, isort)
- ✅ Linting (flake8)
- ✅ Unit tests (Python 3.11 & 3.12)
- ✅ Integration tests
- ✅ Security scanning (Bandit)

### 2. Build and Test
**File:** `.github/workflows/build-and-test.yml`

Validates Docker builds:
- ✅ Builds Docker image
- ✅ Tests Docker image
- ✅ Validates deployment scripts

### 3. Deploy to Cloud Run
**File:** `.github/workflows/deploy-cloud-run.yml`

Deploys to Google Cloud Run:
- ✅ Builds container image
- ✅ Pushes to GCR
- ✅ Deploys to Cloud Run
- ✅ Health check validation

**Triggers:**
- Push to `main` branch
- Manual workflow dispatch

### 4. Data Pipeline CI
**File:** `.github/workflows/data-pipeline-ci.yml`

Tests data preprocessing:
- ✅ Preprocessing tests
- ✅ Code linting
- ✅ Import validation

### 5. Model Training CI
**File:** `.github/workflows/model-training-ci.yml`

Validates model development:
- ✅ Model development tests
- ✅ MLflow validation
- ✅ Optional model training

### 6. Release
**File:** `.github/workflows/release.yml`

Creates releases and deploys:
- ✅ Creates GitHub release
- ✅ Deploys to production

**Triggers:**
- Tag push (e.g., `v1.0.0`)
- Manual workflow dispatch

### 7. Scheduled Tests
**File:** `.github/workflows/schedule-tests.yml`

Runs nightly test suite:
- ✅ Full test coverage
- ✅ Daily at 2 AM UTC

### 8. Dependency Review
**File:** `.github/workflows/dependency-review.yml`

Reviews dependencies in PRs:
- ✅ Security vulnerability checks
- ✅ Dependency updates

## 🔧 Setup Instructions

### Step 1: Configure GCP Service Account

```bash
# Create service account
gcloud iam service-accounts create github-actions \
  --display-name="GitHub Actions"

# Grant required roles
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

### Step 2: Add GitHub Secrets

1. Go to repository → **Settings** → **Secrets and variables** → **Actions**
2. Add secrets:
   - `GCP_PROJECT_ID`: Your project ID
   - `GCP_SA_KEY`: Contents of `key.json` file

### Step 3: Enable GCP APIs

```bash
gcloud services enable \
  cloudbuild.googleapis.com \
  run.googleapis.com \
  containerregistry.googleapis.com
```

## 📊 Workflow Status

View workflow runs in the **Actions** tab:
- ✅ Green = Success
- ❌ Red = Failed
- 🟡 Yellow = In progress

## 🎯 Workflow Triggers

| Workflow | Trigger | Branch |
|----------|---------|--------|
| CI | PR/Push | main, develop |
| Build & Test | PR/Push | main, develop |
| Deploy | Push | main |
| Data Pipeline CI | PR/Push | main |
| Model Training CI | PR/Push | main |
| Release | Tag/Manual | - |
| Scheduled Tests | Schedule | - |

## 🔍 Monitoring

### View Workflow Logs

1. Go to **Actions** tab
2. Click on a workflow
3. Click on a run to see logs

### View Deployment Status

```bash
gcloud run services describe lab-lens-web \
  --region us-central1 \
  --project YOUR_PROJECT_ID
```

## 🐛 Troubleshooting

### Common Issues

1. **Deployment fails**: Check secrets are set correctly
2. **Tests fail**: Verify Python version and dependencies
3. **Build fails**: Check Dockerfile path and build context

### Debug Steps

1. Check workflow logs in GitHub Actions
2. Verify secrets are set
3. Test locally first
4. Check GCP service account permissions

## 📚 Documentation

For detailed setup instructions, see:
- [CI/CD Setup Guide](./docs/CI_CD_SETUP.md)
- [Deployment Guide](./DEPLOYMENT_GUIDE.md)

## ✅ Best Practices

1. ✅ Always test locally before pushing
2. ✅ Create feature branches for changes
3. ✅ Review CI results before merging
4. ✅ Use semantic versioning for releases
5. ✅ Monitor deployments after completion
