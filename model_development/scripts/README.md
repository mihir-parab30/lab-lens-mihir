# Gemini 1.5 Pro Training Module

This module provides Gemini 1.5 Pro integration for medical text summarization on MIMIC-III discharge summaries.

## Overview

Gemini 1.5 Pro is Google's advanced multimodal AI model optimized for complex reasoning tasks. It can process large amounts of text (up to 2M tokens) and is excellent for medical text summarization without requiring fine-tuning.

## Components

### 1. `gemini_model.py`
- `GeminiSummarizer`: Main model class for Gemini-based summarization
- `load_gemini_model()`: Utility function to load Gemini model

### 2. `train_gemini.py`
- `GeminiTrainer`: Setup class for prompt engineering and processing
- Main processing script with command-line interface

### 3. `gemini_inference.py`
- `GeminiInference`: Inference class for generating summaries
- Batch processing capabilities
- DataFrame integration

## Quick Start

### 1. Install Dependencies

```bash
pip install google-generativeai
```

### 2. Get API Key

1. Go to: https://aistudio.google.com/app/apikey
2. Create an API key
3. Set environment variable:
   ```bash
   export GOOGLE_API_KEY="your-api-key-here"
   ```

### 3. Process Data with Gemini

```bash
python src/training/train_gemini.py \
    --data-path data-pipeline/data/processed/processed_discharge_summaries.csv \
    --config configs/gemini_config.json \
    --output-dir models/gemini
```

### 4. Generate Summaries

```bash
# Single text
python src/training/gemini_inference.py \
    --api-key YOUR_API_KEY \
    --input "Patient admitted with chest pain..."

# Batch processing (CSV)
python src/training/gemini_inference.py \
    --api-key YOUR_API_KEY \
    --input data-pipeline/data/processed/processed_discharge_summaries.csv \
    --output data-pipeline/data/processed/summarized_discharge_summaries.csv \
    --input-column cleaned_text \
    --output-column gemini_summary
```

## Configuration

Edit `configs/gemini_config.json` to customize:

- **Model settings**: Model name, temperature, output tokens
- **Processing settings**: Batch size, delays, system prompts
- **Data settings**: Column names, output column
- **API settings**: Rate limiting, API key configuration

## Model Features

- **No Training Required**: Gemini 1.5 Pro is pre-trained and ready to use
- **Large Context**: Can process up to 2M tokens (much larger than BERT's 512)
- **Medical Understanding**: Excellent at understanding medical terminology
- **Abstractive Summarization**: Generates new summaries, not just extracts
- **API-Based**: No local model files needed

## Integration with Pipeline

You can integrate Gemini into your existing pipeline:

```python
from src.training import GeminiTrainer, GeminiInference

# Process data
trainer = GeminiTrainer(
    config_path='configs/gemini_config.json',
    output_dir='models/gemini'
)
results = trainer.process_data(
    data_path='data-pipeline/data/processed/processed_discharge_summaries.csv'
)

# Use for inference
inferencer = GeminiInference(api_key=os.getenv('GOOGLE_API_KEY'))
summary = inferencer.summarize(discharge_text)
```

## Model Variants

Available Gemini models:
- `gemini-2.0-flash-exp` (default) - Latest model, best performance
- `gemini-1.5-pro` (fallback) - Best for complex tasks, large context
- `gemini-1.5-flash` - Faster, lower cost, good for simpler tasks
- `gemini-pro` - Previous generation

## API Costs

- **Free Tier**: 15 requests per minute
- **Paid Tier**: Pay per token (input + output)
- **Rate Limits**: Built-in delays to respect API limits

## Future Enhancements

- [ ] Few-shot learning with examples
- [ ] Custom prompt templates
- [ ] ROUGE and BLEU evaluation metrics
- [ ] Cost tracking and optimization
- [ ] Caching for repeated queries
- [ ] Integration with MLflow for experiment tracking

## Notes

- Requires internet connection (API-based)
- API key required (free tier available)
- Rate limits apply (script includes delays)
- Costs apply for large-scale usage (check Google AI pricing)

## References

- Gemini API Documentation: https://ai.google.dev/gemini-api/docs
- Google AI Studio: https://aistudio.google.com/
- Gemini Models: https://ai.google.dev/gemini-api/docs/models/gemini
