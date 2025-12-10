# Dockerfile for Lab Lens Model Training and Inference
# Multi-stage build for optimized image size

FROM python:3.9-slim as base

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Create necessary directories
RUN mkdir -p data-pipeline/data/raw \
    data-pipeline/data/processed \
    data-pipeline/logs \
    models/gemini \
    mlruns \
    logs

# Set environment variables
ENV PYTHONPATH=/app
ENV MLFLOW_TRACKING_URI=./mlruns

# Default command (can be overridden)
CMD ["python", "--version"]

# ============================================
# Stage 2: Training Image
# ============================================
FROM base as training

LABEL stage="training"
LABEL description="Lab Lens model training environment"

# Training-specific setup
WORKDIR /app

# Expose MLflow UI port (optional)
EXPOSE 5000

# Default training command
CMD ["python", "src/training/train_with_tracking.py", \
     "--data-path", "data-pipeline/data/processed/processed_discharge_summaries.csv", \
     "--config", "configs/gemini_config.json"]

# ============================================
# Stage 3: Inference Image
# ============================================
FROM base as inference

LABEL stage="inference"
LABEL description="Lab Lens model inference environment"

# Inference-specific setup
WORKDIR /app

# Expose API port (if you add an API server)
EXPOSE 8000

# Default inference command
CMD ["python", "src/training/example_usage.py"]

# ============================================
# Stage 4: Production Image (minimal)
# ============================================
FROM python:3.9-slim as production

WORKDIR /app

# Install only runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy only necessary files
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    google-generativeai \
    python-dotenv \
    mlflow \
    pandas \
    numpy

# Copy minimal code
COPY src/training/gemini_model.py src/training/
COPY src/training/gemini_inference.py src/training/
COPY src/utils/ src/utils/
COPY configs/ configs/

# Create directories
RUN mkdir -p models/gemini logs

ENV PYTHONPATH=/app

CMD ["python", "src/training/gemini_inference.py"]




