import streamlit as st

# Document processing imports
import PyPDF2
from docx import Document

# Page config
st.set_page_config(
    page_title="Medical Discharge Summarizer",
    layout="centered"
)

# Supported file formats
SUPPORTED_FORMATS = {
    "application/pdf": ("PDF Document", "pdf"),
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ("Word Document", "docx"),
    "text/plain": ("Text File", "txt"),
}

def extract_text_from_pdf(file) -> str:
    """Extract text from PDF file."""
    try:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text.strip()
    except Exception as e:
        raise ValueError(f"Failed to extract text from PDF: {str(e)}")

def extract_text_from_docx(file) -> str:
    """Extract text from Word document."""
    try:
        doc = Document(file)
        text = "\n".join([para.text for para in doc.paragraphs])
        return text.strip()
    except Exception as e:
        raise ValueError(f"Failed to extract text from Word document: {str(e)}")

def extract_text_from_txt(file) -> str:
    """Extract text from plain text file."""
    try:
        return file.read().decode("utf-8").strip()
    except Exception as e:
        raise ValueError(f"Failed to read text file: {str(e)}")

def get_file_format(uploaded_file) -> tuple:
    """Detect and validate file format."""
    mime_type = uploaded_file.type
    file_name = uploaded_file.name.lower()
    
    if mime_type in SUPPORTED_FORMATS:
        return SUPPORTED_FORMATS[mime_type]
    
    if file_name.endswith(".pdf"):
        return ("PDF Document", "pdf")
    elif file_name.endswith(".docx"):
        return ("Word Document", "docx")
    elif file_name.endswith(".txt"):
        return ("Text File", "txt")
    else:
        raise ValueError(f"Unsupported file format: {uploaded_file.name}\nSupported formats: PDF, DOCX, TXT")

def extract_text(uploaded_file, format_key: str) -> str:
    """Extract text based on file format."""
    if format_key == "pdf":
        return extract_text_from_pdf(uploaded_file)
    elif format_key == "docx":
        return extract_text_from_docx(uploaded_file)
    elif format_key == "txt":
        return extract_text_from_txt(uploaded_file)
    else:
        raise ValueError(f"Unknown format: {format_key}")

import requests

# API endpoint
API_URL = "https://medical-summarizer-api-616478754804.us-central1.run.app"

# Test API connection
@st.cache_data(ttl=60)
def test_api_connection():
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

if not test_api_connection():
    st.warning("⚠️ API connection issue. Please check backend service.")

# Header
st.title("Medical Discharge Summarizer")
st.markdown("Convert complex medical discharge notes into **patient-friendly** summaries.")

st.divider()

# Input method selection
input_method = st.radio(
    "Select Input Method",
    ["Paste Text", "Upload Document"],
    horizontal=True
)

input_text = ""

if input_method == "Upload Document":
    st.subheader("Upload Discharge Document")
    st.markdown("Supported formats: PDF, DOCX, TXT")
    
    uploaded_file = st.file_uploader(
        "Upload discharge document",
        type=["pdf", "docx", "txt"],
        label_visibility="collapsed"
    )
    
    if uploaded_file:
        try:
            format_name, format_key = get_file_format(uploaded_file)
            
            with st.spinner(f"Extracting text from {format_name}..."):
                input_text = extract_text(uploaded_file, format_key)
            
            with st.expander("Preview Extracted Text", expanded=False):
                st.text(input_text[:2000] + ("..." if len(input_text) > 2000 else ""))
            
            st.success(f"Successfully extracted {len(input_text):,} characters from {format_name}.")
            
        except ValueError as e:
            st.error(str(e))
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

else:
    st.subheader("Paste Discharge Notes")
    input_text = st.text_area(
        label="discharge_notes",
        label_visibility="collapsed",
        height=200,
        placeholder="Paste the medical discharge summary here...\n\nExample: Patient is a 65 year old male admitted with chest pain. Diagnosis: Acute myocardial infarction..."
    )

st.divider()

# Summarize button
if st.button("Generate Summary", type="primary", use_container_width=True):
    if not input_text or len(input_text.strip()) < 50:
        st.error("Please provide at least 50 characters of medical text.")
    else:
        with st.spinner("Generating patient-friendly summary... (first request may take 60 seconds)"):
            try:
                # Call API instead of loading model
                response = requests.post(
                    f"{API_URL}/summarize",
                    json={"text": input_text},
                    timeout=120  # 2 minute timeout for first request
                )
                
                if response.status_code == 200:
                    result = response.json()
                    
                    st.divider()
                    st.subheader("Patient-Friendly Summary")
                    st.markdown(result["summary"])  # API returns "summary" not "final_summary"
                else:
                    st.error(f"API error: {response.status_code} - {response.text}")
                        
            except requests.Timeout:
                st.error("Request timed out. The model may be loading (first request takes ~60s). Please try again.")
            except Exception as e:
                st.error(f"Error: {str(e)}")

# Footer
st.divider()
st.caption("Disclaimer: This tool is for informational purposes only. Always consult healthcare providers for medical advice.")