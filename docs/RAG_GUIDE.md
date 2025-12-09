# RAG-Based Patient Q&A Guide

## Overview

The RAG (Retrieval-Augmented Generation) system enables patients to ask questions about their discharge summaries. The system:

1. **Retrieves** relevant information from discharge summaries using semantic search
2. **Generates** patient-friendly answers using Gemini AI
3. **Provides** source citations so patients can see where answers come from

## Features

- ✅ **Semantic Search**: Finds relevant sections in discharge summaries using embeddings
- ✅ **Context-Aware Answers**: Uses retrieved context to generate accurate answers
- ✅ **Patient-Friendly**: Answers in simple, understandable language
- ✅ **Source Citations**: Shows which sections of the report were used
- ✅ **Multi-Question Support**: Answer multiple questions at once
- ✅ **Patient Filtering**: Filter questions to specific patient records
- ✅ **Single-Patient Mode**: Load only one patient's data for faster, more secure Q&A

## Quick Start

### Installation

Install required dependencies:

```bash
pip install sentence-transformers faiss-cpu
```

**Note**: `faiss-cpu` is optional but recommended for faster similarity search. The system will work without it using numpy-based search (slower).

### Basic Usage

#### Single-Patient Mode (Recommended)

For patient-specific Q&A, use single-patient mode which loads only that patient's record:

```bash
# Interactive mode for one patient
python scripts/patient_qa_single.py --hadm-id 130656

# Single question
python scripts/patient_qa_single.py --hadm-id 130656 --question "What are my diagnoses?"
```

**Benefits:**
- Fast: Only processes one patient's record (5-10 seconds)
- Efficient: Minimal memory usage (<100 MB)
- Secure: Only loads that patient's data

#### Interactive Mode (All Records)

Start an interactive Q&A session with all records:

```bash
python scripts/patient_qa_interactive.py
```

For a specific patient:

```bash
python scripts/patient_qa_interactive.py --hadm-id 130656
```

#### Quick Test

Test with a small dataset first:

```bash
# Create test dataset (first 10 records)
python -c "
import pandas as pd
df = pd.read_csv('data-pipeline/data/processed/processed_discharge_summaries.csv')
df.head(10).to_csv('data-pipeline/data/processed/test_discharge_summaries.csv', index=False)
print('Created test dataset with 10 records')
"

# Test with smaller dataset
python scripts/quick_test_rag.py --hadm-id 130656 --question "What are my diagnoses?" --data-path data-pipeline/data/processed/test_discharge_summaries.csv
```

## Example Questions

Patients can ask questions like:

- **Diagnoses**: "What are my diagnoses?" / "What conditions do I have?"
- **Medications**: "What medications do I need to take?" / "What are my discharge medications?"
- **Follow-up**: "When is my follow-up appointment?" / "Do I need a follow-up?"
- **Lab Results**: "What were my lab results?" / "Were there any abnormal labs?"
- **Hospital Stay**: "What happened during my hospital stay?" / "What was my hospital course?"
- **Instructions**: "What are my discharge instructions?" / "What should I do at home?"
- **Symptoms**: "What should I watch for?" / "What are warning signs I should look for?"
- **Explanations**: "Can you explain my condition in simple terms?" / "What does this diagnosis mean?"

## How It Works

### 1. Document Processing

- Discharge summaries are split into chunks (500 characters with 50 character overlap)
- Each chunk is embedded using a sentence transformer model
- Embeddings are stored in a vector database (FAISS or numpy-based)

### 2. Question Processing

- Patient's question is embedded using the same model
- Semantic similarity search finds the most relevant chunks
- Top K chunks (default: 5) are retrieved

### 3. Answer Generation

- Retrieved chunks are provided as context to Gemini
- Gemini generates a patient-friendly answer based on the context
- Answer is returned with source citations

## Architecture

```
Patient Question
      ↓
  Embedding Model
      ↓
  Vector Search (FAISS/numpy)
      ↓
  Retrieve Relevant Chunks
      ↓
  Build Context
      ↓
  Gemini AI (Answer Generation)
      ↓
  Patient-Friendly Answer
```

## Advanced Usage

### Python API

Use the RAG system programmatically:

```python
from src.rag.patient_qa import PatientQA

# Single-patient mode (recommended)
qa = PatientQA(
    data_path="data-pipeline/data/processed/processed_discharge_summaries.csv",
    hadm_id=130656  # Loads only this patient's data
)

# Ask questions (no hadm_id needed - already filtered)
result = qa.ask_question("What are my diagnoses?")
print(result['answer'])

# All records mode
qa = PatientQA(
    data_path="data-pipeline/data/processed/processed_discharge_summaries.csv"
)

# Ask with patient filter
result = qa.ask_question(
    question="What are my diagnoses?",
    hadm_id=130656  # Optional: filter to specific patient
)
```

### Custom Configuration

```python
from src.rag.patient_qa import PatientQA

qa = PatientQA(
    data_path="path/to/data.csv",
    embedding_model="all-mpnet-base-v2",  # Different embedding model
    gemini_model="gemini-1.5-pro",        # Different Gemini model
    rag_k=10  # Retrieve more chunks for context
)
```

### Direct RAG System Access

```python
from src.rag.rag_system import RAGSystem

# Initialize RAG system
rag = RAGSystem(
    embedding_model="all-MiniLM-L6-v2",
    chunk_size=500,
    chunk_overlap=50,
    data_path="data-pipeline/data/processed/processed_discharge_summaries.csv",
    hadm_id=130656  # Single-patient mode
)

# Retrieve relevant chunks
results = rag.retrieve(
    query="What medications are prescribed?",
    k=5,
    min_score=0.3  # Minimum similarity threshold
)

for result in results:
    print(f"Score: {result['score']:.3f}")
    print(f"Chunk: {result['chunk'][:100]}...")
```

## Configuration Options

### Embedding Models

Available sentence transformer models:

- `all-MiniLM-L6-v2` (default) - Fast, good quality, 384 dimensions
- `all-mpnet-base-v2` - Better quality, slower, 768 dimensions
- `all-MiniLM-L12-v2` - Balanced option

### Chunking Strategy

Default settings:
- **Chunk Size**: 500 characters
- **Chunk Overlap**: 50 characters

Adjust in `RAGSystem`:
```python
rag = RAGSystem(
    chunk_size=750,      # Larger chunks
    chunk_overlap=100    # More overlap
)
```

### Retrieval Parameters

```python
results = rag.retrieve(
    query="question",
    k=10,                # Number of chunks to retrieve
    min_score=0.4,       # Minimum similarity score (0-1)
)
```

## Caching

Embeddings are automatically cached to speed up subsequent runs:

- **Cache Location**: `models/rag_embeddings/embeddings_{filename}.pkl`
- **First Run**: Creates embeddings (takes a few minutes for full dataset)
- **Subsequent Runs**: Loads from cache (seconds)

To rebuild embeddings:
```python
rag.load_data("data.csv", force_rebuild=True)
```

## Performance

### Single-Patient Mode
- **First run**: 5-10 seconds (create embeddings for one patient)
- **Subsequent runs**: 1-2 seconds (load cached embeddings)
- **Memory**: <100 MB

### All Records Mode
- **Embedding Generation** (first run): ~2-5 minutes for 300K records
- **Embedding Loading** (cached): ~5-10 seconds
- **Question Answering**: ~2-5 seconds per question
- **Memory**: 2-4 GB

## Troubleshooting

### Import Errors

**Error**: `No module named 'sentence_transformers'`

**Solution**:
```bash
pip install sentence-transformers
```

### FAISS Not Available

**Warning**: `faiss-cpu not available. Using numpy-based similarity search.`

**Solution** (optional but recommended):
```bash
pip install faiss-cpu
```

**Note**: System works without FAISS, but search will be slower.

### Memory Issues

For very large datasets:

1. Use single-patient mode (recommended)
2. Use smaller chunk sizes
3. Use CPU-only FAISS: `pip install faiss-cpu` (instead of `faiss-gpu`)
4. Process in batches

### Gemini API Errors

**Error**: `GOOGLE_API_KEY not found`

**Solution**:
```bash
export GOOGLE_API_KEY="your-api-key"
# Or
python scripts/setup_gemini_api_key.py
```

See [API Setup Guide](API_SETUP.md) for more details.

### First Run is Slow

- First run creates embeddings (2-5 minutes for large datasets)
- Subsequent runs load cached embeddings (5-10 seconds)
- Use single-patient mode for faster testing

## Best Practices

### 1. Use Single-Patient Mode

For patient-specific Q&A, always use single-patient mode:

```bash
python scripts/patient_qa_single.py --hadm-id 130656
```

### 2. Clear Questions

Encourage patients to ask clear, specific questions:
- ✅ "What are my discharge medications?"
- ❌ "meds?"

### 3. Multiple Follow-ups

Use interactive mode for natural conversation:

```bash
python scripts/patient_qa_single.py --hadm-id 130656
```

### 4. Verify Sources

Always check source citations to verify answers come from the report.

## Integration Examples

### Flask API Integration

```python
from flask import Flask, request, jsonify
from src.rag.patient_qa import PatientQA

app = Flask(__name__)
qa_system = PatientQA()

@app.route('/ask', methods=['POST'])
def ask_question():
    data = request.json
    question = data.get('question')
    hadm_id = data.get('hadm_id')
    
    # Use single-patient mode if hadm_id provided
    if hadm_id:
        qa_system = PatientQA(hadm_id=hadm_id)
    
    result = qa_system.ask_question(question)
    return jsonify(result)

if __name__ == '__main__':
    app.run()
```

### Batch Processing

```python
from src.rag.patient_qa import PatientQA

qa = PatientQA(hadm_id=130656)  # Single-patient mode

questions = [
    "What are my diagnoses?",
    "What medications do I need?",
    "When is my follow-up?"
]

results = qa.ask_multiple_questions(questions)

for question, result in zip(questions, results):
    print(f"Q: {question}")
    print(f"A: {result['answer']}\n")
```

## Related Documentation

- [API Setup Guide](API_SETUP.md) - Setting up Gemini API
- [Model Development Guide](MODEL_DEVELOPMENT_GUIDE.md) - Model development
- [Testing Guide](MODEL_TESTING_GUIDE.md) - Testing procedures
