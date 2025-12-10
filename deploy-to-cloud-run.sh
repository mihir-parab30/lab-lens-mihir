#!/bin/bash

# Lab Lens - Google Cloud Run Deployment Script
# Usage: ./deploy-to-cloud-run.sh [project-id]

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
PROJECT_ID="${1:-your-project-id}"
SERVICE_NAME="lab-lens-web"
REGION="us-central1"
IMAGE_NAME="gcr.io/$PROJECT_ID/$SERVICE_NAME"

echo -e "${GREEN}🚀 Lab Lens Cloud Run Deployment${NC}"
echo "=================================="
echo "Project ID: $PROJECT_ID"
echo "Service: $SERVICE_NAME"
echo "Region: $REGION"
echo ""

# Check if gcloud is installed
if ! command -v gcloud &> /dev/null; then
    echo -e "${RED}❌ gcloud CLI not found. Please install it first.${NC}"
    exit 1
fi

# Check if logged in
if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | grep -q .; then
    echo -e "${YELLOW}⚠️  Not logged in. Running 'gcloud auth login'...${NC}"
    gcloud auth login
fi

# Set project
echo -e "${YELLOW}📦 Setting project...${NC}"
gcloud config set project $PROJECT_ID

# Build and push using Cloud Build config
echo -e "${YELLOW}🔨 Building container...${NC}"
gcloud builds submit \
  --config=cloudbuild.yaml \
  --timeout=20m

# Check if secret exists
if ! gcloud secrets describe gemini-api-key &> /dev/null; then
    echo -e "${YELLOW}⚠️  Secret 'gemini-api-key' not found.${NC}"
    echo "Please create it with:"
    echo "  echo -n 'your-api-key' | gcloud secrets create gemini-api-key --data-file=-"
    echo ""
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Deploy
echo -e "${YELLOW}🚢 Deploying to Cloud Run...${NC}"
gcloud run deploy $SERVICE_NAME \
  --image $IMAGE_NAME \
  --platform managed \
  --region $REGION \
  --allow-unauthenticated \
  --port 8501 \
  --memory 4Gi \
  --cpu 2 \
  --timeout 600 \
  --max-instances 10 \
  --min-instances 0 \
  --set-env-vars HF_HOME=/root/.cache/huggingface,TRANSFORMERS_CACHE=/root/.cache/huggingface \
  --set-secrets GEMINI_API_KEY=gemini-api-key:latest,GOOGLE_API_KEY=gemini-api-key:latest

# Get URL
SERVICE_URL=$(gcloud run services describe $SERVICE_NAME --region $REGION --format='value(status.url)')

echo ""
echo -e "${GREEN}✅ Deployment successful!${NC}"
echo "=================================="
echo -e "🌐 Service URL: ${GREEN}$SERVICE_URL${NC}"
echo ""
echo "📊 View logs:"
echo "  gcloud run services logs tail $SERVICE_NAME --region $REGION"
echo ""
echo "📈 View monitoring:"
echo "  https://console.cloud.google.com/run/detail/$REGION/$SERVICE_NAME/metrics?project=$PROJECT_ID"
