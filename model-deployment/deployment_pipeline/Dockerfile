FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y gcc g++ curl && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire model directory into the container
COPY models/bart-medical /app/models/bart-medical

# Copy application code
COPY app.py summarizer.py ./

# Environment variables
ENV TRANSFORMERS_CACHE=/app/.cache
ENV HF_HOME=/app/.cache

EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl --fail http://localhost:8080/health || exit 1

# Run FastAPI
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080", "--timeout-keep-alive", "300"]