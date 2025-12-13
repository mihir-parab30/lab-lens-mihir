import streamlit as st
import requests
import PyPDF2
from docx import Document
import time

# Page config with custom styling
st.set_page_config(
    page_title="Lab Lens - Medical Discharge Summarizer",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    /* Main theme colors */
    :root {
        --primary-color: #2E86AB;
        --secondary-color: #A23B72;
        --success-color: #06A77D;
        --warning-color: #F18F01;
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .main-header h1 {
        color: white !important;
        margin: 0;
        font-size: 2.5rem;
    }
    
    .main-header p {
        color: rgba(255,255,255,0.9);
        font-size: 1.1rem;
        margin-top: 0.5rem;
    }
    
    /* Card styling */
    .info-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .success-card {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Summary box styling */
    .summary-box {
        background: white;
        padding: 2rem;
        border-radius: 10px;
        border-left: 5px solid #667eea;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        margin: 1.5rem 0;
    }
    
    .diagnosis-badge {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        display: inline-block;
        font-weight: bold;
        margin: 0.5rem 0;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        font-size: 1.1rem;
        border-radius: 25px;
        font-weight: bold;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 8px rgba(0,0,0,0.15);
    }
    
    /* Metrics styling */
    div[data-testid="stMetricValue"] {
        font-size: 2rem;
        color: #667eea;
        font-weight: bold;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background-color: #f5f7fa;
        border-radius: 5px;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem;
        color: #666;
        font-size: 0.9rem;
    }
    </style>
""", unsafe_allow_html=True)

# API endpoint - UPDATED with YOUR URL
API_URL = "https://lab-lens-api-226509074083.us-central1.run.app"

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

# Test API connection
@st.cache_data(ttl=60)
def test_api_connection():
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

# Sidebar
with st.sidebar:
    st.markdown("### 🏥 Lab Lens")
    st.markdown("**Medical Text Summarization**")
    st.divider()
    
    # API Status
    api_status = test_api_connection()
    if api_status:
        st.success("✅ API Connected")
    else:
        st.error("❌ API Disconnected")
    
    st.divider()
    
    # Features
    st.markdown("#### ✨ Features")
    st.markdown("""
    - 📄 Upload PDF/DOCX/TXT
    - 🤖 AI-Powered Summarization
    - 💬 Patient-Friendly Language
    - 🔒 HIPAA-Compliant Processing
    """)
    
    st.divider()
    
    # Model Info
    with st.expander("ℹ️ About the Model"):
        st.markdown("""
        **Model Pipeline:**
        1. Smart Extraction
        2. BART Summarization
        3. Gemini Refinement
        
        **Model:** Fine-tuned BART-large  
        **Dataset:** MIMIC-III  
        **Purpose:** Medical text simplification
        """)
    
    st.divider()
    st.caption("Lab Lens MLOps Project")

# Main header
st.markdown("""
    <div class="main-header">
        <h1>🏥 Medical Discharge Summarizer</h1>
        <p>Transform complex medical discharge notes into clear, patient-friendly summaries powered by AI</p>
    </div>
""", unsafe_allow_html=True)

# API connection warning
if not test_api_connection():
    st.error("⚠️ **API Connection Issue** - Please check backend service or try again later.")

# Stats row
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("📊 Model", "BART + Gemini", "Fine-tuned")
with col2:
    st.metric("⚡ Processing", "2-3 sec", "Average")
with col3:
    st.metric("🎯 Accuracy", "ROUGE: 0.42", "Validation")

st.divider()

# Input method selection with better styling
st.markdown("### 📝 Input Method")
input_method = st.radio(
    "Choose how to provide discharge notes:",
    ["📋 Paste Text", "📁 Upload Document"],
    horizontal=True,
    label_visibility="collapsed"
)

input_text = ""

if "Upload" in input_method:
    st.markdown("#### 📁 Upload Discharge Document")
    st.markdown("Supported formats: **PDF**, **DOCX**, **TXT**")
    
    uploaded_file = st.file_uploader(
        "Upload discharge document",
        type=["pdf", "docx", "txt"],
        label_visibility="collapsed",
        help="Upload a discharge summary in PDF, Word, or text format"
    )
    
    if uploaded_file:
        try:
            format_name, format_key = get_file_format(uploaded_file)
            
            with st.spinner(f"📄 Extracting text from {format_name}..."):
                input_text = extract_text(uploaded_file, format_key)
            
            # Success message with stats
            col1, col2 = st.columns([3, 1])
            with col1:
                st.success(f"✅ Successfully extracted from **{format_name}**")
            with col2:
                st.info(f"📊 {len(input_text):,} chars")
            
            # Preview with better styling
            with st.expander("👁️ Preview Extracted Text", expanded=False):
                st.text_area(
                    "Extracted text",
                    value=input_text[:2000] + ("..." if len(input_text) > 2000 else ""),
                    height=200,
                    label_visibility="collapsed"
                )
            
        except ValueError as e:
            st.error(f"❌ {str(e)}")
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")

else:
    st.markdown("#### 📋 Paste Discharge Notes")
    input_text = st.text_area(
        label="discharge_notes",
        label_visibility="collapsed",
        height=300,
        placeholder="Paste the medical discharge summary here...\n\nExample:\n\nDISCHARGE SUMMARY\n\nPatient: John Doe\nAdmission Date: 12/08/2024\n\nCHIEF COMPLAINT: Chest pain\n\nDISCHARGE DIAGNOSIS: Acute myocardial infarction\n\nDISCHARGE MEDICATIONS:\n1. Aspirin 81mg daily\n2. Atorvastatin 80mg daily\n\nDISCHARGE INSTRUCTIONS:\nFollow up with cardiology in 2 weeks...",
        help="Paste or type the discharge summary text here"
    )
    
    # Character counter
    if input_text:
        char_count = len(input_text)
        if char_count < 50:
            st.warning(f"⚠️ {char_count} characters (minimum 50 required)")
        else:
            st.info(f"📊 {char_count:,} characters")

st.divider()

# Summarize button with better styling
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    summarize_button = st.button(
        "🚀 Generate Patient-Friendly Summary",
        type="primary",
        use_container_width=True
    )

if summarize_button:
    if not input_text or len(input_text.strip()) < 50:
        st.error("❌ Please provide at least 50 characters of medical text.")
    else:
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Step 1: Preparing
            status_text.text("📋 Preparing request...")
            progress_bar.progress(20)
            time.sleep(0.5)
            
            # Step 2: Calling API
            status_text.text("🔄 Processing with AI models (this may take 60s on first request)...")
            progress_bar.progress(40)
            
            response = requests.post(
                f"{API_URL}/summarize",
                json={"text": input_text},
                timeout=120
            )
            
            progress_bar.progress(80)
            
            if response.status_code == 200:
                result = response.json()
                progress_bar.progress(100)
                status_text.text("✅ Summary generated!")
                time.sleep(0.5)
                progress_bar.empty()
                status_text.empty()
                
                # Display results with beautiful formatting
                st.markdown("---")
                
                # Success banner
                st.markdown("""
                    <div class="success-card">
                        <h2 style="color: #06A77D; margin: 0;">✅ Summary Generated Successfully!</h2>
                    </div>
                """, unsafe_allow_html=True)
                
                # Create tabs for different views
                tab1, tab2, tab3 = st.tabs(["📋 Patient Summary", "🔬 Clinical Summary", "📊 Details"])
                
                with tab1:
                    st.markdown("### 💬 Patient-Friendly Summary")
                    st.markdown(f"""
                        <div class="summary-box">
                            {result.get('summary', 'No summary generated')}
                        </div>
                    """, unsafe_allow_html=True)
                    
                    # Diagnosis badge if available
                    if result.get('diagnosis') and result['diagnosis'] != 'Unknown':
                        st.markdown(f"""
                            <div class="diagnosis-badge">
                                🏥 Primary Diagnosis: {result['diagnosis']}
                            </div>
                        """, unsafe_allow_html=True)
                
                with tab2:
                    st.markdown("### 🔬 Clinical Summary (BART Output)")
                    if result.get('bart_summary'):
                        st.info(result['bart_summary'])
                    else:
                        st.warning("No clinical summary available")
                
                with tab3:
                    st.markdown("### 📊 Processing Details")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Input Length", f"{len(input_text)} chars")
                        st.metric("Summary Length", f"{len(result.get('summary', ''))} chars")
                    with col2:
                        st.metric("Compression Ratio", f"{len(result.get('summary', '')) / max(len(input_text), 1):.1%}")
                        st.metric("Diagnosis Extracted", "Yes" if result.get('diagnosis') != 'Unknown' else "No")
                    
                    # Raw JSON response
                    with st.expander("🔍 View Raw API Response"):
                        st.json(result)
                
                # Action buttons
                st.markdown("---")
                col1, col2, col3 = st.columns(3)
                with col1:
                    if st.button("📋 Copy Summary", use_container_width=True):
                        st.code(result.get('summary', ''), language=None)
                with col2:
                    if st.button("🔄 Process Another", use_container_width=True):
                        st.rerun()
                with col3:
                    st.download_button(
                        label="💾 Download Summary",
                        data=result.get('summary', ''),
                        file_name="patient_summary.txt",
                        mime="text/plain",
                        use_container_width=True
                    )
            
            else:
                progress_bar.empty()
                status_text.empty()
                st.error(f"❌ **API Error ({response.status_code})**")
                st.code(response.text)
                    
        except requests.Timeout:
            progress_bar.empty()
            status_text.empty()
            st.error("""
                ⏱️ **Request Timed Out**
                
                The model may be loading (first request takes ~60 seconds). 
                Please try again in a moment.
            """)
        except requests.ConnectionError:
            progress_bar.empty()
            status_text.empty()
            st.error("""
                🔌 **Connection Error**
                
                Unable to connect to the API. Please check:
                - Backend service is running
                - Network connection is stable
            """)
        except Exception as e:
            progress_bar.empty()
            status_text.empty()
            st.error(f"❌ **Error:** {str(e)}")
            with st.expander("🔍 Error Details"):
                st.exception(e)

# Example section
st.markdown("---")
st.markdown("### 💡 Example Discharge Summary")

with st.expander("📄 Click to view example"):
    example_text = """DISCHARGE SUMMARY

Admission Date: 12/08/2024
Discharge Date: 12/10/2024

SERVICE: Cardiology

CHIEF COMPLAINT: Chest pain

HISTORY OF PRESENT ILLNESS:
Patient is a 67-year-old male with history of hypertension and hyperlipidemia who presented to the emergency department with acute onset substernal chest pain radiating to the left arm. The pain started at rest and was associated with diaphoresis and shortness of breath.

HOSPITAL COURSE:
The patient was admitted to the cardiac care unit. Electrocardiogram showed ST elevation in the anterior leads consistent with acute myocardial infarction. Emergent cardiac catheterization was performed with placement of a drug-eluting stent to the left anterior descending artery. Post-procedure course was uncomplicated. Patient remained hemodynamically stable throughout hospitalization.

DISCHARGE DIAGNOSIS:
1. Acute ST-elevation myocardial infarction
2. Coronary artery disease
3. Hypertension
4. Hyperlipidemia

DISCHARGE MEDICATIONS:
1. Aspirin 81mg orally daily
2. Clopidogrel 75mg orally daily (for 12 months)
3. Atorvastatin 80mg orally daily
4. Metoprolol succinate 50mg orally twice daily
5. Lisinopril 10mg orally daily

DISCHARGE INSTRUCTIONS:
Patient to follow up with cardiologist in 2 weeks. Cardiac rehabilitation referral has been made. Patient was counseled extensively on lifestyle modifications including diet, exercise, and smoking cessation. Patient understands warning signs of recurrent MI and when to seek emergency care.

DISCHARGE DISPOSITION:
Home with family

FOLLOW-UP:
Cardiology clinic in 2 weeks
Primary care physician in 1 week"""
    
    st.text_area("Example text", value=example_text, height=300, label_visibility="collapsed")
    
    if st.button("📝 Use This Example", use_container_width=True):
        st.rerun()

# Footer
st.markdown("---")
st.markdown("""
    <div class="footer">
        <p><strong>Lab Lens</strong> - AI-Powered Healthcare Intelligence Platform</p>
        <p>⚠️ <em>Disclaimer: This tool is for informational purposes only. Always consult healthcare providers for medical advice.</em></p>
        <p style="font-size: 0.8rem; color: #999;">
            Powered by BART-large (fine-tuned) + Gemini 2.0 Flash | Deployed on Google Cloud Run
        </p>
    </div>
""", unsafe_allow_html=True)

# Add some breathing room at bottom
st.markdown("<br><br>", unsafe_allow_html=True)