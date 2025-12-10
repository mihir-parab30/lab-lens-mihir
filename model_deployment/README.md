# Model Deployment

This directory contains code for deploying models to production.

## Structure

- `api/` - FastAPI application for model serving
- `web/` - Streamlit web interface
- `containerization/` - Docker configurations
- `scripts/` - Deployment automation scripts

## API Deployment

Deploy the FastAPI application:

```bash
cd model_deployment/api
uvicorn app:app --host 0.0.0.0 --port 8000
```

## Web Interface Deployment

Run the Streamlit web app:

```bash
cd model_deployment/web
streamlit run file_qa_web.py
```

## Cloud Run Deployment

Deploy to Google Cloud Run:

```bash
./model_deployment/scripts/deploy-to-cloud-run.sh <project-id>
```

## Docker

Build and run with Docker:

```bash
cd infrastructure/docker
docker build -f Dockerfile.cloudrun -t lab-lens:latest .
docker run -p 8501:8501 lab-lens:latest
```
