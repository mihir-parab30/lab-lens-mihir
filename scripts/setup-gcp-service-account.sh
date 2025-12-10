#!/bin/bash

# Setup GCP Service Account for GitHub Actions CI/CD
# This script creates a service account and generates a key for GitHub Actions

set -e

# Configuration
PROJECT_ID="${1:-gen-lang-client-0006590375}"
SA_NAME="github-actions"
SA_EMAIL="${SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"
KEY_FILE="github-actions-key.json"

echo "🔧 Setting up GCP Service Account for GitHub Actions"
echo "=================================================="
echo "Project ID: $PROJECT_ID"
echo "Service Account: $SA_EMAIL"
echo ""

# Check if gcloud is installed
if ! command -v gcloud &> /dev/null; then
    echo "❌ Error: gcloud CLI not found. Please install it first."
    echo "   Install: brew install --cask google-cloud-sdk"
    exit 1
fi

# Check if logged in
if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | grep -q .; then
    echo "⚠️  Not logged in. Running 'gcloud auth login'..."
    gcloud auth login
fi

# Set project
echo "📦 Setting project..."
gcloud config set project $PROJECT_ID

# Check if service account exists
if gcloud iam service-accounts describe $SA_EMAIL --project=$PROJECT_ID &>/dev/null; then
    echo "✅ Service account already exists: $SA_EMAIL"
    read -p "Do you want to create a new key? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Skipping service account creation."
    else
        # Create new service account (will fail if exists, that's OK)
        gcloud iam service-accounts create $SA_NAME \
            --display-name="GitHub Actions Service Account" \
            --description="Service account for GitHub Actions CI/CD" \
            --project=$PROJECT_ID 2>/dev/null || echo "Service account already exists, continuing..."
    fi
else
    # Create service account
    echo "🔨 Creating service account..."
    gcloud iam service-accounts create $SA_NAME \
        --display-name="GitHub Actions Service Account" \
        --description="Service account for GitHub Actions CI/CD" \
        --project=$PROJECT_ID
    echo "✅ Service account created"
fi

# Grant roles
echo "🔐 Granting required roles..."

echo "  - Cloud Run Admin..."
gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:${SA_EMAIL}" \
    --role="roles/run.admin" \
    --condition=None 2>/dev/null || echo "    (Role already granted)"

echo "  - Service Account User..."
gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:${SA_EMAIL}" \
    --role="roles/iam.serviceAccountUser" \
    --condition=None 2>/dev/null || echo "    (Role already granted)"

echo "  - Storage Admin..."
gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:${SA_EMAIL}" \
    --role="roles/storage.admin" \
    --condition=None 2>/dev/null || echo "    (Role already granted)"

echo "  - Cloud Build Editor..."
gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:${SA_EMAIL}" \
    --role="roles/cloudbuild.builds.editor" \
    --condition=None 2>/dev/null || echo "    (Role already granted)"

echo "✅ Roles granted"

# Enable required APIs
echo "🔌 Enabling required APIs..."
gcloud services enable cloudbuild.googleapis.com --project=$PROJECT_ID 2>/dev/null || echo "  (Already enabled)"
gcloud services enable run.googleapis.com --project=$PROJECT_ID 2>/dev/null || echo "  (Already enabled)"
gcloud services enable containerregistry.googleapis.com --project=$PROJECT_ID 2>/dev/null || echo "  (Already enabled)"
echo "✅ APIs enabled"

# Create key
echo "🔑 Creating service account key..."
if [ -f "$KEY_FILE" ]; then
    echo "⚠️  Key file $KEY_FILE already exists."
    read -p "Overwrite? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Keeping existing key file."
        exit 0
    fi
    rm -f $KEY_FILE
fi

gcloud iam service-accounts keys create $KEY_FILE \
    --iam-account=$SA_EMAIL \
    --project=$PROJECT_ID

echo "✅ Key created: $KEY_FILE"
echo ""

# Display key (for easy copying)
echo "📋 Service Account Key (copy this to GitHub secret GCP_SA_KEY):"
echo "================================================================"
cat $KEY_FILE
echo ""
echo "================================================================"
echo ""

# Instructions
echo "📝 Next Steps:"
echo "=============="
echo ""
echo "1. Go to GitHub repository: https://github.com/kamalshahidnu/lab-lens"
echo "2. Navigate to: Settings → Secrets and variables → Actions"
echo "3. Add these secrets:"
echo ""
echo "   Secret Name: GCP_PROJECT_ID"
echo "   Secret Value: $PROJECT_ID"
echo ""
echo "   Secret Name: GCP_SA_KEY"
echo "   Secret Value: (Copy the entire JSON above, or run: cat $KEY_FILE)"
echo ""
echo "4. After adding to GitHub, delete the local key file:"
echo "   rm $KEY_FILE"
echo ""
echo "5. Add to .gitignore to prevent accidental commits:"
echo "   echo '$KEY_FILE' >> .gitignore"
echo ""
echo "✅ Setup complete!"
