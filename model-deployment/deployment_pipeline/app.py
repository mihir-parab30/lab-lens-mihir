from dotenv import load_dotenv
load_dotenv()  # Load .env before anything else
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from summarizer import MedicalSummarizer
import logging
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Medical Discharge Summarizer API",
    description="AI-powered medical discharge summary generation",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request and Response models
class DischargeRequest(BaseModel):
    text: str
    
class SummaryResponse(BaseModel):
    summary: str
    diagnosis: str
    bart_summary: str

# Global variable - model loads on first request
summarizer = None
model_loading = False

def get_summarizer():
    """Lazy load the model on first request"""
    global summarizer, model_loading
    
    if summarizer is not None:
        return summarizer
    
    if model_loading:
        raise HTTPException(status_code=503, detail="Model is currently loading, please retry in 30 seconds")
    
    try:
        model_loading = True
        logger.info("📄 Loading Medical Summarizer (first request)...")
        summarizer = MedicalSummarizer(use_gpu=False)
        logger.info("✅ Model loaded successfully!")
        model_loading = False
        return summarizer
    except Exception as e:
        model_loading = False
        logger.error(f"❌ Failed to load model: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")

@app.get("/")
def root():
    """Root endpoint"""
    return {
        "message": "Medical Discharge Summarizer API",
        "status": "running",
        "model_status": "loaded" if summarizer else "not loaded (will load on first request)",
        "endpoints": {
            "health": "/health",
            "summarize": "/summarize",
            "info": "/info",
            "docs": "/docs"
        }
    }

@app.get("/health")
def health():
    """Health check endpoint - always returns healthy (model loads lazily)"""
    return {
        "status": "healthy",
        "model_loaded": summarizer is not None,
        "model_loading": model_loading,
        "gemini_configured": bool(os.getenv("GEMINI_API_KEY"))
    }

@app.post("/summarize", response_model=SummaryResponse)
def summarize(request: DischargeRequest):
    """Generate patient-friendly summary from discharge text"""
    # Load model on first request
    current_summarizer = get_summarizer()
    
    if not request.text or len(request.text.strip()) < 50:
        raise HTTPException(status_code=400, detail="Text too short (minimum 50 characters)")
    
    try:
        logger.info(f"Processing summary request (text length: {len(request.text)} chars)")
        
        # Generate summary
        result = current_summarizer.generate_summary(request.text)
        
        # FIXED: Extract values matching what summarizer.py actually returns
        summary = result.get('summary', '')              # Changed from 'final_summary'
        diagnosis = result.get('diagnosis', 'Unknown')    # Changed from nested 'extracted_data'
        bart_summary = result.get('bart_summary', '')     # Changed from 'raw_bart_summary'
        
        logger.info(f"Summary generated: {len(summary)} chars")
        
        return SummaryResponse(
            summary=summary,
            diagnosis=diagnosis,
            bart_summary=bart_summary
        )
    
    except Exception as e:
        logger.error(f"Error during summarization: {e}")
        raise HTTPException(status_code=500, detail=f"Summarization failed: {str(e)}")

@app.get("/info")
def info():
    """Get API information"""
    return {
        "model_id": os.getenv("MODEL_ID", "asadwaraich/bart-medical-discharge-summarizer"),
        "device": "cpu",
        "gemini_enabled": bool(os.getenv("GEMINI_API_KEY")),
        "model_loaded": summarizer is not None,
        "version": "1.0.0"
    }