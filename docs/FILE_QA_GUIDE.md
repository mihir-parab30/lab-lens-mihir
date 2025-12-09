# File-Based Q&A System Guide

## Overview

The File Q&A system allows you to upload documents (text files, PDFs, or images) and ask questions about them using RAG (Retrieval-Augmented Generation).

## Supported File Types

- **Text files**: `.txt`, `.md`
- **PDF files**: `.pdf`
- **Image files**: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`, `.tif`
  - Uses OCR (Optical Character Recognition) to extract text
  - Optionally uses Gemini Vision API for medical image analysis

## Installation

Install required dependencies:

```bash
source .venv/bin/activate
pip install PyPDF2 pdfplumber pytesseract Pillow
```

For OCR (Tesseract), you also need to install the Tesseract binary:
- **macOS**: `brew install tesseract`
- **Linux**: `sudo apt-get install tesseract-ocr`
- **Windows**: Download from https://github.com/UB-Mannheim/tesseract/wiki

## Usage

### Interactive Mode

**Load a single file:**
```bash
python scripts/file_qa_interactive.py --file report.pdf
```

**Load multiple files:**
```bash
python scripts/file_qa_interactive.py --files report1.pdf report2.txt scan.jpg
```

**Load raw text:**
```bash
python scripts/file_qa_interactive.py --text "Patient discharge summary text here..."
```

**Start without loading (load later):**
```bash
python scripts/file_qa_interactive.py
# Then type 'reload' in interactive mode to load a file
```

### Non-Interactive Mode

Answer a single question:
```bash
python scripts/file_qa_interactive.py --file report.pdf --question "What is the main diagnosis?"
```

## Interactive Commands

Once in interactive mode, you can:

- **Ask questions**: Type any question and press Enter
- **Type `help`**: See example questions
- **Type `reload`**: Load a new file (will prompt for file path)
- **Type `exit` or `quit`**: End the session

## Example Questions

- "What is the main diagnosis?"
- "What medications are mentioned?"
- "What are the key findings?"
- "What are the lab results?"
- "Can you summarize the document?"
- "What procedures were performed?"
- "What are the discharge instructions?"

## How It Works

1. **Document Processing**:
   - Text files: Directly read
   - PDF files: Extract text using PyPDF2 or pdfplumber
   - Images: Extract text using OCR (Tesseract) and optionally analyze with Gemini Vision

2. **RAG Processing**:
   - Documents are split into chunks
   - Chunks are converted to embeddings
   - A vector index is built for fast similarity search

3. **Question Answering**:
   - Your question is converted to an embedding
   - Most relevant document chunks are retrieved
   - Context is sent to Gemini to generate an answer

## Configuration

You can customize models:

```bash
# Use different embedding model
python scripts/file_qa_interactive.py --file report.pdf --embedding-model all-mpnet-base-v2

# Use different Gemini model
python scripts/file_qa_interactive.py --file report.pdf --gemini-model gemini-1.5-pro
```

## Requirements

- **GOOGLE_API_KEY**: Must be set for Gemini answer generation
  - Set via: `export GOOGLE_API_KEY='your-key'`
  - Or in `.env` file: `GOOGLE_API_KEY=your-key`

## Troubleshooting

### PDF extraction fails
- Try installing `pdfplumber` (better than PyPDF2): `pip install pdfplumber`
- Some PDFs may be image-based (scanned) - try converting to image first

### OCR doesn't work
- Install Tesseract binary (see Installation section)
- For better OCR, ensure images are high quality and well-lit
- Medical images may need Gemini Vision instead of OCR

### Image analysis
- Set `GOOGLE_API_KEY` to use Gemini Vision for better medical image analysis
- Without API key, only OCR text extraction will work

### Memory issues with large files
- Large PDFs or images may use significant memory
- Consider splitting very large documents into smaller files

## Examples

### Medical Report PDF
```bash
python scripts/file_qa_interactive.py --file patient_report.pdf
# Then ask: "What are the patient's diagnoses?"
```

### Lab Results Image
```bash
python scripts/file_qa_interactive.py --file lab_results.jpg
# Then ask: "What are the abnormal lab values?"
```

### Multiple Documents
```bash
python scripts/file_qa_interactive.py --files discharge_summary.pdf lab_report.txt xray.jpg
# Then ask: "What are all the findings across these documents?"
```

## API Usage

You can also use the FileQA class programmatically:

```python
from src.rag.file_qa import FileQA

# Initialize
qa = FileQA()

# Load a file
qa.load_file("report.pdf")

# Ask questions
result = qa.ask_question("What is the diagnosis?")
print(result['answer'])
```

## Performance Tips

- **Large documents**: Consider splitting into smaller files
- **Multiple files**: Batch processing works, but may take time for embeddings
- **Image processing**: OCR can be slow; Gemini Vision is faster but requires API key
- **Caching**: Embeddings are generated on-the-fly; consider saving for reuse


