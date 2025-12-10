from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from model_deployment.api.summarizer import MedicalSummarizer
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

# Add CORS middleware for frontend integration
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

# Global variable to hold the model
summarizer = None

@app.on_event("startup")
async def load_model():
    """Load the model when the API starts"""
    global summarizer
    try:
        logger.info("Loading Medical Summarizer...")
        summarizer = MedicalSummarizer(use_gpu=False)
        logger.info("✅ Model loaded successfully!")
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        raise e

@app.get("/")
def root():
    """Root endpoint"""
    return {
        "message": "Medical Discharge Summarizer API",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "summarize": "/summarize",
            "docs": "/docs"
        }
    }

@app.get("/health")
def health():
    """Health check endpoint"""
    if summarizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "status": "healthy",
        "model_loaded": True,
        "gemini_configured": bool(os.getenv("GEMINI_API_KEY"))
    }

@app.post("/summarize", response_model=SummaryResponse)
def summarize(request: DischargeRequest):
    """Generate patient-friendly summary from discharge text"""
    if summarizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if not request.text or len(request.text.strip()) < 50:
        raise HTTPException(status_code=400, detail="Text too short (minimum 50 characters)")
    
    try:
        logger.info(f"Processing summary request (text length: {len(request.text)} chars)")
        
        # Generate summary
        result = summarizer.generate_summary(request.text)
        
        # Debug: Print what keys are in result
        logger.info(f"Result keys: {list(result.keys())}")
        
        # Extract values with safe fallbacks
        summary = result.get('final_summary', '')
        diagnosis = result.get('extracted_data', {}).get('diagnosis', 'Unknown')
        bart_summary = result.get('summary', '')  # Changed from 'bart_summary' to 'summary'
        
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
        "version": "1.0.0"
    }
