# Google Cloud Run Quick Reference

## 🚀 Quick Deploy
```bash
# One-command deployment
./deploy-to-cloud-run.sh your-project-id
```

## 📋 Essential Commands

### Deploy
```bash
# Manual build and deploy
gcloud builds submit --tag gcr.io/PROJECT_ID/lab-lens-web
gcloud run deploy lab-lens-web \
  --image gcr.io/PROJECT_ID/lab-lens-web \
  --region us-central1 \
  --allow-unauthenticated
```

### View Logs
```bash
# Real-time logs
gcloud run services logs tail lab-lens-web --region us-central1

# Recent logs
gcloud run services logs read lab-lens-web --region us-central1 --limit 100
```

### Get Service Info
```bash
# Get URL
gcloud run services describe lab-lens-web --region us-central1 --format='value(status.url)'

# Get full details
gcloud run services describe lab-lens-web --region us-central1
```

### Update Configuration
```bash
# Update environment variables
gcloud run services update lab-lens-web \
  --region us-central1 \
  --set-env-vars KEY=value

# Update memory/CPU
gcloud run services update lab-lens-web \
  --region us-central1 \
  --memory 4Gi \
  --cpu 2

# Update scaling
gcloud run services update lab-lens-web \
  --region us-central1 \
  --min-instances 1 \
  --max-instances 20
```

### Manage Secrets
```bash
# Create secret
echo -n "your-api-key" | gcloud secrets create gemini-api-key --data-file=-

# Update secret
echo -n "new-api-key" | gcloud secrets versions add gemini-api-key --data-file=-

# List secrets
gcloud secrets list

# Grant Cloud Run access
PROJECT_NUMBER=$(gcloud projects describe PROJECT_ID --format='value(projectNumber)')
gcloud secrets add-iam-policy-binding gemini-api-key \
  --member="serviceAccount:$PROJECT_NUMBER-compute@developer.gserviceaccount.com" \
  --role="roles/secretmanager.secretAccessor"
```

### Rollback
```bash
# List revisions
gcloud run revisions list --service lab-lens-web --region us-central1

# Route to specific revision
gcloud run services update-traffic lab-lens-web \
  --region us-central1 \
  --to-revisions REVISION_NAME=100
```

### Delete Service
```bash
gcloud run services delete lab-lens-web --region us-central1
```

## 🔍 Monitoring URLs

- **Service Dashboard**: https://console.cloud.google.com/run?project=PROJECT_ID
- **Logs**: https://console.cloud.google.com/logs?project=PROJECT_ID
- **Metrics**: https://console.cloud.google.com/monitoring?project=PROJECT_ID
- **Traces**: https://console.cloud.google.com/traces?project=PROJECT_ID

## 🚨 Troubleshooting

### Container fails to start
```bash
# Check logs for errors
gcloud run services logs read lab-lens-web --region us-central1 --limit 50

# Common issues:
# - Missing environment variables
# - Port mismatch (should be $PORT or 8501)
# - Out of memory (increase to 2Gi or 4Gi)
```

### 502 Bad Gateway
- Service taking too long to start (increase memory/CPU)
- App crashing on startup (check logs)
- Health check failing (verify /_stcore/health endpoint)

### Out of Memory
```bash
# Increase memory allocation
gcloud run services update lab-lens-web \
  --region us-central1 \
  --memory 4Gi
```

### Slow Cold Starts
```bash
# Keep at least one instance warm
gcloud run services update lab-lens-web \
  --region us-central1 \
  --min-instances 1
```

## 💰 Cost Optimization

- **Use minimum instances = 0** for development (default)
- **Use minimum instances = 1** for production (eliminates cold starts)
- **Set max instances** to control costs: `--max-instances 10`
- **Monitor usage** in Billing section of Cloud Console

## 🔐 Security Best Practices

1. ✅ Use Secret Manager for API keys (never environment variables)
2. ✅ Enable Cloud Armor for DDoS protection (if needed)
3. ✅ Use IAM for authentication (instead of --allow-unauthenticated)
4. ✅ Regularly rotate secrets
5. ✅ Set up VPC connector for private services

## 📊 Monitoring Metrics

Key metrics to monitor:
- **Request count**: Total requests per minute
- **Request latency**: P50, P95, P99 response times
- **Error rate**: 4xx and 5xx errors
- **Container CPU/Memory utilization**
- **Cold start latency**

## 🎯 Performance Tips

1. **Use lazy loading** for ML models (load on first use)
2. **Cache frequently accessed data** (use Redis if needed)
3. **Optimize Docker image size** (use multi-stage builds)
4. **Set appropriate timeout** (default 300s, max 3600s)
5. **Use HTTP/2** for better performance (enabled by default)
