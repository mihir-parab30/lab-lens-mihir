# Monitoring and Observability

This directory contains monitoring, logging, and observability code for Lab Lens.

## Structure

- `metrics.py` - Metrics collection utilities
- `logging/` - Logging configurations
- `dashboards/` - Monitoring dashboard configurations

## Usage

Collect metrics:

```python
from monitoring.metrics import collect_metrics

metrics = collect_metrics()
```

## Logging

Configure logging:

```python
from src.utils.logging_config import get_logger

logger = get_logger(__name__)
logger.info("Application started")
```

## Cloud Monitoring

Metrics are automatically sent to Google Cloud Monitoring when deployed to Cloud Run.
