# GitHub Secrets Setup - Quick Guide

## ✅ Service Account Key Created!

Your GCP service account key has been created. Follow these steps to add it to GitHub:

## Step 1: Go to GitHub Repository

1. Open: https://github.com/kamalshahidnu/lab-lens
2. Click **Settings** tab (top menu)
3. Click **Secrets and variables** → **Actions** (left sidebar)

## Step 2: Add GCP_PROJECT_ID Secret

1. Click **New repository secret**
2. **Name:** `GCP_PROJECT_ID`
3. **Secret:** `gen-lang-client-0006590375`
4. Click **Add secret**

## Step 3: Add GCP_SA_KEY Secret

1. Click **New repository secret** again
2. **Name:** `GCP_SA_KEY`
3. **Secret:** Copy the **entire JSON** from the terminal output above
   - It starts with `{` and ends with `}`
   - Include everything between
4. Click **Add secret**

## Step 4: Verify Secrets

You should now see two secrets:
- ✅ `GCP_PROJECT_ID`
- ✅ `GCP_SA_KEY`

## Step 5: Clean Up Local Key File

After adding to GitHub, delete the local key file:

```bash
cd "/Users/shahidkamal/Documents/MS/MLOps Project/lab-lens"
rm github-actions-key.json
```

## ✅ Done!

Your CI/CD pipeline is now configured. The next push to `main` will automatically:
- Run tests and linting
- Build Docker image
- Deploy to Cloud Run

## Test the Setup

1. Make a small change and push to `main`
2. Go to **Actions** tab in GitHub
3. Watch the workflows run automatically

## Troubleshooting

If deployment fails:
- Check secrets are set correctly
- Verify service account has required roles
- Check workflow logs in Actions tab
