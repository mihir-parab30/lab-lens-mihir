# Web Interface Guide - File Q&A System

## Overview

A modern, chat-based web interface for the File Q&A system, similar to ChatGPT-style interfaces.

## Installation

Install Streamlit:
```bash
source .venv/bin/activate
pip install streamlit
```

## Running the Web Interface

```bash
streamlit run scripts/file_qa_web.py
```

This will:
- Start a local web server
- Open your browser automatically
- Display the chat interface

## Features

### 📁 Document Upload
- **File Upload**: Drag and drop or browse for files
  - Supports: `.txt`, `.pdf`, `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`, `.md`
  - Multiple files can be uploaded at once
- **Text Input**: Paste raw text directly into the text area

### 💬 Chat Interface
- **Modern Chat UI**: Clean, ChatGPT-style interface
- **Question Input**: Type questions in the chat input at the bottom
- **Conversation History**: All questions and answers are saved in the session
- **Source Citations**: Click "Sources" to see which parts of documents were used

### 🎯 Usage Flow

1. **Upload Documents**:
   - Click "Upload files" in the sidebar
   - Select one or more files (PDF, text, images)
   - OR paste text in the text area
   - Click "📥 Load Documents"

2. **Ask Questions**:
   - Type your question in the chat input
   - Press Enter or click send
   - View the answer and sources

3. **Continue Conversation**:
   - Ask follow-up questions
   - All context is maintained

4. **Clear Session**:
   - Click "🗑️ Clear All" to reset

## Interface Elements

### Sidebar
- **File Uploader**: Upload documents
- **Text Input**: Paste raw text
- **Load Button**: Process uploaded documents
- **Clear Button**: Reset the session
- **Status**: Shows if documents are loaded

### Main Chat Area
- **Chat Messages**: Displays conversation history
- **User Messages**: Your questions
- **Assistant Messages**: AI-generated answers
- **Sources**: Expandable sections showing document sources

### Chat Input
- Located at the bottom
- Placeholder: "Ask a question about your documents..."
- Press Enter to send

## Example Questions

- "What is the main diagnosis?"
- "What medications are mentioned?"
- "What are the key findings?"
- "Can you summarize the document?"
- "What are the lab results?"

## Customization

### Change Port
```bash
streamlit run scripts/file_qa_web.py --server.port 8502
```

### Change Theme
Edit the script to modify colors and styling in the CSS section.

## Troubleshooting

### Port Already in Use
```bash
# Use a different port
streamlit run scripts/file_qa_web.py --server.port 8502
```

### Documents Not Loading
- Check file format is supported
- Ensure GOOGLE_API_KEY is set
- Check console for error messages

### Slow Response
- Large files take time to process
- First question may be slower (embedding generation)
- Subsequent questions are faster

## Advanced Usage

### Multiple Sessions
Open multiple browser tabs for different document sets.

### API Integration
The underlying `FileQA` class can be used programmatically:
```python
from src.rag.file_qa import FileQA

qa = FileQA()
qa.load_file("report.pdf")
result = qa.ask_question("What is the diagnosis?")
```


