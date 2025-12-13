FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc g++ curl git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app.py summarizer.py ./

# Set environment for model caching
ENV TRANSFORMERS_CACHE=/app/.cache
ENV HF_HOME=/app/.cache
ENV MODEL_ID=asadwaraich/bart-medical-discharge-summarizer

# Pre-download model from Hugging Face during build
RUN python3 -c "from transformers import AutoTokenizer, AutoModelForSeq2SeqLM; \
    import os; \
    model_id = os.getenv('MODEL_ID'); \
    print(f'Downloading {model_id}...'); \
    tokenizer = AutoTokenizer.from_pretrained(model_id); \
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id); \
    print('✅ Model downloaded')"

EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl --fail http://localhost:8080/health || exit 1

# Run FastAPI
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080", "--timeout-keep-alive", "300"]