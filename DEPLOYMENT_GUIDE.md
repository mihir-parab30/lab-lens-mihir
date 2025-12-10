# 🚀 Lab Lens Frontend Deployment Guide

Complete guide for deploying the Streamlit web application (`file_qa_web.py`).

## 📋 Table of Contents

1. [Streamlit Community Cloud](#1-streamlit-community-cloud-easiest) ⭐ Recommended
2. [Docker Deployment](#2-docker-deployment)
3. [Cloud Platforms](#3-cloud-platform-deployment)
4. [Self-Hosted VPS](#4-self-hosted-vps-deployment)

---

## 1. Streamlit Community Cloud (Easiest) ⭐

**Best for:** Quick deployment, free tier available, automatic HTTPS

### Prerequisites
- GitHub account
- Repository pushed to GitHub

### Steps

1. **Push your code to GitHub**
   ```bash
   git add .
   git commit -m "Add Streamlit web app"
   git push origin main
   ```

2. **Go to Streamlit Community Cloud**
   - Visit: https://share.streamlit.io/
   - Click "Sign in with GitHub"
   - Authorize Streamlit

3. **Deploy your app**
   - Click "New app"
   - Select your repository: `lab-lens`
   - Main file path: `scripts/file_qa_web.py`
   - App URL: `https://your-app-name.streamlit.app`

4. **Set Environment Variables**
   In the Streamlit Cloud dashboard, add:
   ```
   GEMINI_API_KEY=your-gemini-api-key
   GOOGLE_API_KEY=your-gemini-api-key
   ```

5. **Deploy**
   - Click "Deploy"
   - Wait for build to complete (~2-3 minutes)
   - Your app will be live!

### Requirements File for Streamlit Cloud
Create `packages.txt` (for system packages) if needed:
```txt
# packages.txt (optional)
```

The `requirements.txt` will be automatically detected.

---

## 2. Docker Deployment

**Best for:** Consistent deployments, containerization, production environments

### Step 1: Create Dockerfile for Streamlit App

Create `Dockerfile.streamlit`:

```dockerfile
FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install additional dependencies for PDF processing
RUN pip install --no-cache-dir pdfplumber

# Copy application code
COPY . .

# Expose Streamlit port
EXPOSE 8501

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Run Streamlit
ENTRYPOINT ["streamlit", "run", "scripts/file_qa_web.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### Step 2: Create .dockerignore

Ensure `.dockerignore` includes:
```
__pycache__
*.pyc
*.pyo
*.pyd
.Python
.env
.venv
models/
data/
logs/
*.log
.git
.gitignore
```

### Step 3: Build and Run Docker Container

```bash
# Build image
docker build -f Dockerfile.streamlit -t lab-lens-web:latest .

# Run container
docker run -d \
  -p 8501:8501 \
  -e GEMINI_API_KEY=your-api-key \
  -e GOOGLE_API_KEY=your-api-key \
  --name lab-lens-app \
  lab-lens-web:latest

# Check logs
docker logs -f lab-lens-app
```

### Step 4: Deploy to Cloud with Docker

#### Docker Hub
```bash
# Login to Docker Hub
docker login

# Tag image
docker tag lab-lens-web:latest yourusername/lab-lens-web:latest

# Push to Docker Hub
docker push yourusername/lab-lens-web:latest
```

#### AWS ECS / Google Cloud Run / Azure Container Instances
Use the Docker image with their container services.

---

## 3. Cloud Platform Deployment

### Option A: Google Cloud Run (Recommended for GCP users)

**Best for:** Serverless, auto-scaling, pay-per-use

1. **Create Dockerfile** (use the one above)

2. **Build and push to Google Container Registry**
   ```bash
   # Authenticate
   gcloud auth login
   gcloud config set project YOUR_PROJECT_ID

   # Build and push
   gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/lab-lens-web
   ```

3. **Deploy to Cloud Run**
   ```bash
   gcloud run deploy lab-lens-web \
     --image gcr.io/YOUR_PROJECT_ID/lab-lens-web \
     --platform managed \
     --region us-central1 \
     --allow-unauthenticated \
     --port 8501 \
     --memory 2Gi \
     --set-env-vars GEMINI_API_KEY=your-key
   ```

4. **Get URL**
   ```bash
   gcloud run services describe lab-lens-web --region us-central1
   ```

### Option B: AWS Elastic Beanstalk

1. **Install EB CLI**
   ```bash
   pip install awsebcli
   ```

2. **Initialize EB**
   ```bash
   eb init -p python-3.12 lab-lens-web
   eb create lab-lens-env
   ```

3. **Set environment variables**
   ```bash
   eb setenv GEMINI_API_KEY=your-key
   ```

4. **Deploy**
   ```bash
   eb deploy
   ```

### Option C: Azure App Service

1. **Install Azure CLI**
   ```bash
   az login
   az group create --name lab-lens-rg --location eastus
   ```

2. **Create App Service**
   ```bash
   az webapp create \
     --resource-group lab-lens-rg \
     --plan lab-lens-plan \
     --name lab-lens-web \
     --runtime "PYTHON:3.12"
   ```

3. **Configure environment variables**
   ```bash
   az webapp config appsettings set \
     --resource-group lab-lens-rg \
     --name lab-lens-web \
     --settings GEMINI_API_KEY=your-key
   ```

4. **Deploy**
   ```bash
   az webapp up --name lab-lens-web --resource-group lab-lens-rg
   ```

### Option D: Heroku

1. **Install Heroku CLI**
   ```bash
   # Mac
   brew tap heroku/brew && brew install heroku
   ```

2. **Create Procfile**
   ```
   web: streamlit run scripts/file_qa_web.py --server.port=$PORT --server.address=0.0.0.0
   ```

3. **Create setup.sh**
   ```bash
   mkdir -p ~/.streamlit/

   echo "\
   [server]\n\
   headless = true\n\
   port = $PORT\n\
   enableCORS = false\n\
   \n\
   " > ~/.streamlit/config.toml
   ```

4. **Deploy**
   ```bash
   heroku login
   heroku create lab-lens-web
   heroku config:set GEMINI_API_KEY=your-key
   git push heroku main
   ```

---

## 4. Self-Hosted VPS Deployment

**Best for:** Full control, custom configurations

### Using Systemd Service

1. **Install dependencies on server**
   ```bash
   sudo apt update
   sudo apt install python3.12 python3-pip nginx
   ```

2. **Clone repository**
   ```bash
   git clone https://github.com/your-username/lab-lens.git
   cd lab-lens
   pip3 install -r requirements.txt
   pip3 install pdfplumber
   ```

3. **Create systemd service**
   Create `/etc/systemd/system/lab-lens.service`:
   ```ini
   [Unit]
   Description=Lab Lens Streamlit App
   After=network.target

   [Service]
   Type=simple
   User=your-username
   WorkingDirectory=/path/to/lab-lens
   Environment="GEMINI_API_KEY=your-key"
   Environment="GOOGLE_API_KEY=your-key"
   ExecStart=/usr/bin/streamlit run scripts/file_qa_web.py --server.port=8501 --server.address=0.0.0.0
   Restart=always

   [Install]
   WantedBy=multi-user.target
   ```

4. **Start service**
   ```bash
   sudo systemctl daemon-reload
   sudo systemctl enable lab-lens
   sudo systemctl start lab-lens
   sudo systemctl status lab-lens
   ```

5. **Configure Nginx reverse proxy** (optional)
   Create `/etc/nginx/sites-available/lab-lens`:
   ```nginx
   server {
       listen 80;
       server_name your-domain.com;

       location / {
           proxy_pass http://127.0.0.1:8501;
           proxy_http_version 1.1;
           proxy_set_header Upgrade $http_upgrade;
           proxy_set_header Connection "upgrade";
           proxy_set_header Host $host;
           proxy_set_header X-Real-IP $remote_addr;
           proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
           proxy_set_header X-Forwarded-Proto $scheme;
       }
   }
   ```

   Enable and restart:
   ```bash
   sudo ln -s /etc/nginx/sites-available/lab-lens /etc/nginx/sites-enabled/
   sudo nginx -t
   sudo systemctl restart nginx
   ```

6. **Set up SSL with Let's Encrypt** (optional)
   ```bash
   sudo apt install certbot python3-certbot-nginx
   sudo certbot --nginx -d your-domain.com
   ```

---

## 🔐 Environment Variables

Required environment variables for all deployments:

```bash
# Gemini API Key (required)
GEMINI_API_KEY=your-gemini-api-key-here
GOOGLE_API_KEY=your-gemini-api-key-here  # Alias for compatibility

# Optional configurations
MODEL_ID=asadwaraich/bart-medical-discharge-summarizer  # For MedicalSummarizer
LOG_LEVEL=INFO
```

---

## ✅ Pre-Deployment Checklist

- [ ] All dependencies in `requirements.txt`
- [ ] Environment variables configured
- [ ] API keys set securely (not in code)
- [ ] PDF processing library (`pdfplumber`) installed
- [ ] Tested locally before deploying
- [ ] `.env` file in `.gitignore` (never commit secrets)
- [ ] Health check endpoint working

---

## 🐛 Troubleshooting

### Issue: App won't start
- Check logs: `docker logs <container>` or cloud platform logs
- Verify environment variables are set
- Ensure port 8501 is accessible

### Issue: PDF processing fails
- Verify `pdfplumber` is installed: `pip install pdfplumber`
- Check file permissions

### Issue: Memory errors
- Increase memory allocation (Cloud Run: 2Gi+, Docker: --memory=2g)
- MedicalSummarizer (BART) requires ~1GB RAM

### Issue: Slow model loading
- MedicalSummarizer loads on first use (lazy loading)
- Consider pre-warming the service
- Use GPU for faster inference (if available)

---

## 📊 Recommended Deployment Options by Use Case

| Use Case | Recommended Option | Cost | Difficulty |
|----------|-------------------|------|------------|
| Quick demo/prototype | Streamlit Community Cloud | Free | ⭐ Easy |
| Production (small scale) | Google Cloud Run | Pay-per-use | ⭐⭐ Medium |
| Production (large scale) | AWS ECS / GKE | Variable | ⭐⭐⭐ Hard |
| Full control | Self-hosted VPS | Fixed | ⭐⭐ Medium |
| Enterprise | Kubernetes | Variable | ⭐⭐⭐⭐ Very Hard |

---

## 🔗 Quick Links

- Streamlit Cloud: https://share.streamlit.io/
- Docker Hub: https://hub.docker.com/
- Google Cloud Run: https://cloud.google.com/run
- AWS Elastic Beanstalk: https://aws.amazon.com/elasticbeanstalk/

---

## 📝 Notes

- **Security**: Never commit API keys. Use environment variables or secrets management.
- **Performance**: First request may be slow (model loading). Subsequent requests are faster.
- **Scaling**: Streamlit apps can handle multiple concurrent users, but for high traffic, consider load balancing.
