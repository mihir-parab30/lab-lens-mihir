# Infrastructure

This directory contains infrastructure as code configurations.

## Structure

- `docker/` - Docker configurations and Dockerfiles
- `kubernetes/` - Kubernetes manifests (if applicable)
- `terraform/` - Terraform configurations (if applicable)
- `ci_cd/` - CI/CD workflow definitions

## Docker

Build images:

```bash
cd infrastructure/docker
docker build -f Dockerfile.cloudrun -t lab-lens:latest .
```

## CI/CD

GitHub Actions workflows are located in `ci_cd/.github/workflows/`.

Workflows:
- `deploy-cloud-run.yml` - Deploy to Google Cloud Run
- `model_training_ci.yml` - Model training CI pipeline
