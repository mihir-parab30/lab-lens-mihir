"""
Medical Summarization System
Architecture: Smart Extraction -> BART-large-cnn -> Gemini Refinement
"""

import os
import re
import logging
import torch
from pathlib import Path
from typing import Dict, Optional
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from dotenv import load_dotenv

# Load .env
# Load .env from multiple possible locations
# Try current directory first, then parent
from pathlib import Path
import os
from dotenv import load_dotenv

# Try multiple paths
env_paths = [
    Path(__file__).parent / '.env',           # Same directory as summarizer.py
    Path(__file__).parent.parent / '.env',    # Parent directory
    Path.cwd() / '.env',                       # Current working directory
]

for env_path in env_paths:
    if env_path.exists():
        load_dotenv(dotenv_path=env_path, override=True)
        print(f"✅ Loaded .env from: {env_path}")
        break
else:
    print("⚠️  No .env file found, using environment variables")


import google.generativeai as genai

# Configuration
MODEL_ID = "/app/models/bart-medical"  # Use local path in container
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Medical term mappings
MEDICAL_TERMS = {
    "myocardial infarction": "heart attack",
    "cerebrovascular accident": "stroke",
    "cva": "stroke",
    "hypertension": "high blood pressure",
    "htn": "high blood pressure",
    "hyperlipidemia": "high cholesterol",
    "renal failure": "kidney failure",
    "pneumonia": "lung infection",
    "sepsis": "severe blood infection",
    "hemorrhage": "bleeding",
    "intracranial": "inside the brain",
    "subdural hematoma": "brain bleed",
    "fracture": "broken bone",
    "snf": "nursing home",
    "skilled nursing facility": "nursing home",
    "ambulate": "walk",
    "dyspnea": "shortness of breath",
    "syncope": "fainting",
    "intubated": "put on a breathing machine",
    "tracheostomy": "breathing tube in neck",
    "peg tube": "feeding tube",
    "cabg": "heart bypass surgery",
    "arrhythmia": "irregular heartbeat",
    "edema": "swelling",
    "embolism": "blood clot",
    "thrombosis": "blood clot",
    "anemia": "low blood count",
    "diabetic": "related to diabetes",
    "hyperglycemia": "high blood sugar",
    "hypoglycemia": "low blood sugar",
}

class MedicalSummarizer:
    """Simplified medical summarizer using BART + Gemini."""
    
    def __init__(self, use_gpu: bool = False):
        logger.info("Initializing Medical Summarizer...")
        
        # Device setup
        self.device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Load BART model
        logger.info(f"Loading model: {MODEL_ID}")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                MODEL_ID)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                MODEL_ID)
            self.model.to(self.device)
            self.model.eval()
            logger.info("✅ BART model loaded")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise e
        
        # Configure Gemini
        self.gemini_configured = False
        api_key = os.getenv('GEMINI_API_KEY')  # ← Check at runtime, not module level

        if GEMINI_API_KEY:
            try:
                genai.configure(api_key=GEMINI_API_KEY)
                self.gemini_model = genai.GenerativeModel('gemini-2.5-flash')
                self.gemini_configured = True
                logger.info("✅ Gemini configured")
            except Exception as e:
                logger.warning(f"Gemini setup failed: {e}")
        else:
            logger.warning("GEMINI_API_KEY not found - using BART output only")
        
        logger.info("✅ System Ready")

    def extract_key_sections(self, text: str) -> Dict[str, str]:
        """Extract key sections from discharge text."""
        sections = {'diagnosis': '', 'procedures': '', 'disposition': '', 'course': ''}
        text_lower = text.lower()
        
        patterns = {
            'diagnosis': r'(?:discharge|final|primary)?\s*diagnosis[:\s]+([^\n]+)',
            'disposition': r'(?:discharge)?\s*(?:disposition|to)[:\s]+([^\n]+)',
            'procedures': r'(?:major)?\s*(?:surgical)?\s*(?:procedures?|operations?)[:\s]+([^\n]+)',
            'course': r'hospital course[:\s]+(.{,500})'
        }
        
        for key, pattern in patterns.items():
            match = re.search(pattern, text_lower, re.DOTALL)
            if match:
                sections[key] = match.group(1).strip()
        
        return sections

    def smart_extract(self, text: str, max_tokens: int = 1024) -> str:
        """Extract most important parts to fit context window."""
        sections = self.extract_key_sections(text)
        parts = []
        
        if sections['diagnosis']: parts.append(f"Diagnosis: {sections['diagnosis']}")
        if sections['disposition']: parts.append(f"Disposition: {sections['disposition']}")
        if sections['procedures']: parts.append(f"Procedures: {sections['procedures']}")
        if sections['course']: parts.append(f"Course: {sections['course']}")
        
        if parts:
            structured = "\n".join(parts)
            remaining_chars = (max_tokens * 4) - len(structured)
            if remaining_chars > 0:
                structured += "\n" + text[:remaining_chars]
            return structured
        
        # Fallback: keyword extraction
        sentences = text.split('. ')
        keywords = ['diagnosis', 'admitted', 'discharge', 'treatment', 'plan', 'history']
        scored = [(sum(3 for k in keywords if k in s.lower()), s) for s in sentences]
        scored.sort(key=lambda x: x[0], reverse=True)
        return ". ".join([s[1] for s in scored[:15]])

    def _get_term_mappings(self, text: str) -> str:
        """Find medical terms in text and return mapping string."""
        text_lower = text.lower()
        found = []
        for medical, simple in MEDICAL_TERMS.items():
            if medical in text_lower:
                found.append(f"{medical} = {simple}")
        return ", ".join(found) if found else "None identified"

    def _call_gemini(self, prompt: str) -> Optional[str]:
        """Call Gemini API."""
        if not self.gemini_configured:
            return None
        try:
            response = self.gemini_model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            logger.error(f"Gemini error: {e}")
            return None

    def generate_summary(self, text: str) -> Dict[str, str]:
        """Main summarization pipeline."""
        logger.info("Starting summarization...")
        
        # Step 1: Smart extraction
        input_text = self.smart_extract(text)
        
        # Step 2: BART summarization
        inputs = self.tokenizer(
            input_text, 
            max_length=1024, 
            truncation=True, 
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            output_ids = self.model.generate(
                inputs.input_ids,
                max_length=150,
                min_length=40,
                num_beams=4,
                length_penalty=2.0,
                early_stopping=True
            )
        bart_summary = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        logger.info("✅ BART summary generated")
        
        # Step 3: Gemini refinement
        sections = self.extract_key_sections(text)
        term_mappings = self._get_term_mappings(text + " " + bart_summary)
        
        prompt = f"""Act as a medical patient advocate. Rewrite this clinical summary for a patient using simple, clear language (6th-grade reading level).

**Clinical Summary:** {bart_summary}
**Diagnosis:** {sections.get('diagnosis', 'See summary')}
**Discharge Plan:** {sections.get('disposition', 'See summary')}
**Medical terms to simplify:** {term_mappings}

Write exactly 3 sections:
**What Happened:** (1-2 sentences explaining the diagnosis simply)
**What Was Done:** (1-2 sentences about treatments)
**What is Next:** (1-2 sentences about follow-up care)"""

        final_summary = self._call_gemini(prompt)
        
        if not final_summary:
            logger.warning("Gemini unavailable, using BART output")
            final_summary = f"**Summary:** {bart_summary}"
            if sections.get('disposition'):
                final_summary += f"\n\n**Next Steps:** {sections['disposition']}"

        return {
            "final_summary": final_summary,
        }