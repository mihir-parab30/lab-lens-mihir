# GCP Service Account Setup for CI/CD

This guide shows you how to create and configure a Google Cloud service account for GitHub Actions CI/CD.

## Step 1: Create Service Account

```bash
# Set your project ID
export PROJECT_ID="gen-lang-client-0006590375"

# Create service account
gcloud iam service-accounts create github-actions \
  --display-name="GitHub Actions Service Account" \
  --description="Service account for GitHub Actions CI/CD" \
  --project=$PROJECT_ID
```

## Step 2: Grant Required Roles

The service account needs these roles for Cloud Run deployment:

```bash
# Cloud Run Admin - to deploy services
gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:github-actions@${PROJECT_ID}.iam.gserviceaccount.com" \
  --role="roles/run.admin"

# Service Account User - to use service accounts
gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:github-actions@${PROJECT_ID}.iam.gserviceaccount.com" \
  --role="roles/iam.serviceAccountUser"

# Storage Admin - for Container Registry
gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:github-actions@${PROJECT_ID}.iam.gserviceaccount.com" \
  --role="roles/storage.admin"

# Cloud Build Editor - to build containers
gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:github-actions@${PROJECT_ID}.iam.gserviceaccount.com" \
  --role="roles/cloudbuild.builds.editor"
```

## Step 3: Create and Download Key

```bash
# Create key file
gcloud iam service-accounts keys create github-actions-key.json \
  --iam-account=github-actions@${PROJECT_ID}.iam.gserviceaccount.com \
  --project=$PROJECT_ID

# Display the key (you'll copy this to GitHub)
cat github-actions-key.json
```

**⚠️ Important:** The key file contains sensitive credentials. Keep it secure!

## Step 4: Add Key to GitHub Secrets

1. **Go to your GitHub repository**
   - Navigate to: `https://github.com/kamalshahidnu/lab-lens`

2. **Go to Settings**
   - Click **Settings** tab in your repository

3. **Navigate to Secrets**
   - Click **Secrets and variables** → **Actions**

4. **Add GCP_PROJECT_ID secret**
   - Click **New repository secret**
   - Name: `GCP_PROJECT_ID`
   - Value: `gen-lang-client-0006590375`
   - Click **Add secret**

5. **Add GCP_SA_KEY secret**
   - Click **New repository secret** again
   - Name: `GCP_SA_KEY`
   - Value: Copy the **entire contents** of `github-actions-key.json`
     - It should start with `{` and end with `}`
     - Include everything between
   - Click **Add secret**

## Step 5: Verify Setup

Test the service account:

```bash
# Authenticate with the service account
gcloud auth activate-service-account \
  github-actions@${PROJECT_ID}.iam.gserviceaccount.com \
  --key-file=github-actions-key.json

# Test permissions
gcloud run services list --project=$PROJECT_ID
```

## Step 6: Clean Up Local Key File

After adding to GitHub, remove the local key file:

```bash
# Remove the key file (it's now in GitHub secrets)
rm github-actions-key.json

# Add to .gitignore to prevent accidental commits
echo "github-actions-key.json" >> .gitignore
```

## Alternative: Using gcloud CLI

If you prefer using gcloud directly:

```bash
# Get the full service account email
SA_EMAIL="github-actions@${PROJECT_ID}.iam.gserviceaccount.com"

# Create key and output to stdout (then copy to GitHub)
gcloud iam service-accounts keys create - \
  --iam-account=$SA_EMAIL \
  --project=$PROJECT_ID
```

Copy the entire JSON output and paste it into the GitHub secret.

## Troubleshooting

### Error: Permission Denied

If you get permission errors:
1. Check you're using the correct project ID
2. Verify you have Owner or IAM Admin role
3. Check service account was created successfully

### Error: Service Account Not Found

```bash
# List service accounts to verify
gcloud iam service-accounts list --project=$PROJECT_ID
```

### Error: Insufficient Permissions

If deployment fails, verify roles:
```bash
# Check service account permissions
gcloud projects get-iam-policy $PROJECT_ID \
  --flatten="bindings[].members" \
  --filter="bindings.members:github-actions@${PROJECT_ID}.iam.gserviceaccount.com"
```

## Security Best Practices

1. ✅ **Never commit the key file** to git
2. ✅ **Use least privilege** - only grant necessary roles
3. ✅ **Rotate keys regularly** (every 90 days recommended)
4. ✅ **Monitor service account usage** in GCP Console
5. ✅ **Use separate service accounts** for different environments

## Quick Reference

**Service Account Email:**
```
github-actions@gen-lang-client-0006590375.iam.gserviceaccount.com
```

**Required Roles:**
- `roles/run.admin`
- `roles/iam.serviceAccountUser`
- `roles/storage.admin`
- `roles/cloudbuild.builds.editor`

**GitHub Secrets:**
- `GCP_PROJECT_ID` = `gen-lang-client-0006590375`
- `GCP_SA_KEY` = (JSON key file contents)
