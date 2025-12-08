from fastapi import FastAPI
from summarizer import MedicalSummarizer  # Your moved script
import os

# Initialize once at startup
app = FastAPI(title="Medical Discharge Summarizer")
summarizer = None

@app.on_event("startup")
async def load_model():
    global summarizer
    model_id = os.getenv("MODEL_ID", "asadwaraich/bart-medical-discharge-summarizer")
    summarizer = MedicalSummarizer(model_id=model_id)