"""
Medical Summarization System - PRODUCTION VERSION
Architecture: Smart Extraction -> BART -> Gemini Refinement
"""

import os
import re
import logging
import torch
from pathlib import Path
from typing import Dict, Optional
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from dotenv import load_dotenv

# Load environment variables from multiple possible locations
env_paths = [
    Path(__file__).parent / '.env',
    Path(__file__).parent.parent / '.env',
    Path.cwd() / '.env',
]

for env_path in env_paths:
    if env_path.exists():
        load_dotenv(dotenv_path=env_path, override=True)
        print(f"✅ Loaded .env from: {env_path}")
        break

import google.generativeai as genai

# Configuration
MODEL_ID = os.getenv("MODEL_ID", "asadwaraich/bart-medical-discharge-summarizer")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Comprehensive medical term mappings
MEDICAL_TERMS = {
    # Cardiac
    "myocardial infarction": "heart attack",
    "mi": "heart attack",
    "stemi": "severe heart attack",
    "nstemi": "heart attack",
    "cabg": "heart bypass surgery",
    "pci": "heart artery opening procedure",
    "cardiac catheterization": "heart artery examination",
    "arrhythmia": "irregular heartbeat",
    "atrial fibrillation": "irregular heartbeat",
    "afib": "irregular heartbeat",
    
    # Neurological
    "cerebrovascular accident": "stroke",
    "cva": "stroke",
    "intracranial": "in the brain",
    "subdural hematoma": "bleeding in the brain",
    "syncope": "fainting",
    
    # General conditions
    "hypertension": "high blood pressure",
    "htn": "high blood pressure",
    "hyperlipidemia": "high cholesterol",
    "diabetes mellitus": "diabetes",
    "renal failure": "kidney failure",
    "pneumonia": "lung infection",
    "sepsis": "serious blood infection",
    "hemorrhage": "bleeding",
    "fracture": "broken bone",
    "edema": "swelling",
    "anemia": "low blood count",
    
    # Procedures
    "intubated": "breathing tube placed",
    "tracheostomy": "breathing tube in neck",
    "peg tube": "feeding tube",
    "nasogastric tube": "feeding tube through nose",
    "foley catheter": "urinary catheter",
    
    # Facilities
    "snf": "nursing home",
    "skilled nursing facility": "nursing home",
    "icu": "intensive care unit",
    
    # Symptoms
    "dyspnea": "difficulty breathing",
    "ambulate": "walk",
    "diaphoresis": "sweating",
    "nausea": "feeling sick to stomach",
    
    # Medications
    "anticoagulation": "blood thinners",
    "antiplatelet": "blood thinners",
    "analgesic": "pain medication",
    "antihypertensive": "blood pressure medication",
    "statin": "cholesterol medication",
}

class MedicalSummarizer:
    """Production-grade medical summarizer with robust error handling."""
    
    def __init__(self, use_gpu: bool = False):
        logger.info("="*60)
        logger.info("Initializing Medical Summarizer...")
        logger.info("="*60)
        
        # Device setup
        self.device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
        logger.info(f"Device: {self.device}")
        
        # Load BART model
        logger.info(f"Loading model: {MODEL_ID}")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID)
            self.model.to(self.device)
            self.model.eval()
            logger.info("✅ BART model loaded successfully")
        except Exception as e:
            logger.error(f"❌ Failed to load BART model: {e}")
            raise e
        
        # Configure Gemini with robust error handling
        self.gemini_configured = False
        self.gemini_model = None
        
        if GEMINI_API_KEY and GEMINI_API_KEY != "your_gemini_api_key_here":
            try:
                genai.configure(api_key=GEMINI_API_KEY)
                
                # Use stable Gemini model
                self.gemini_model = genai.GenerativeModel(
                    'models/gemini-2.0-flash',  # Stable version
                    generation_config={
                        'temperature': 0.7,
                        'top_p': 0.95,
                        'top_k': 40,
                        'max_output_tokens': 1024,
                    },
                    safety_settings={
                        'HARM_CATEGORY_HATE_SPEECH': 'BLOCK_NONE',
                        'HARM_CATEGORY_HARASSMENT': 'BLOCK_NONE',
                        'HARM_CATEGORY_SEXUALLY_EXPLICIT': 'BLOCK_NONE',
                        'HARM_CATEGORY_DANGEROUS_CONTENT': 'BLOCK_NONE',
                    }
                )
                
                # Test Gemini with simple request
                test_response = self.gemini_model.generate_content("Say 'ready'")
                if test_response and test_response.text:
                    self.gemini_configured = True
                    logger.info("✅ Gemini configured and tested successfully")
                else:
                    logger.warning("⚠️  Gemini test failed, using BART only")
                    
            except Exception as e:
                logger.error(f"❌ Gemini setup failed: {e}")
                logger.error(f"Error type: {type(e).__name__}")
                logger.error(f"Error details: {str(e)}")
                self.gemini_configured = False
        else:
            logger.warning("⚠️  GEMINI_API_KEY not found or invalid")
        
        logger.info("="*60)
        logger.info(f"✅ System Ready | Gemini: {'ON' if self.gemini_configured else 'OFF'}")
        logger.info("="*60)

    def extract_key_sections(self, text: str) -> Dict[str, str]:
        """Extract key sections from discharge text with comprehensive patterns."""
        sections = {
            'diagnosis': '',
            'procedures': '',
            'medications': '',
            'disposition': '',
            'course': '',
            'chief_complaint': '',
            'history': ''
        }
        
        text_lower = text.lower()
        
        # Comprehensive extraction patterns
        patterns = {
            'chief_complaint': r'chief\s+complaint[:\s]+([^\n]{10,200})',
            'diagnosis': r'(?:discharge|final|primary)?\s*diagnos(?:is|es)[:\s]+([^\n]+(?:\n\d+\..*)?)',
            'disposition': r'(?:discharge)?\s*disposition[:\s]+([^\n]{5,100})',
            'procedures': r'(?:major\s+)?(?:surgical\s+)?procedures?[:\s]+([^\n]+)',
            'medications': r'(?:discharge\s+)?medications?[:\s]+([^\n]+(?:\n\d+\..*)?)',
            'course': r'hospital\s+course[:\s]+(.{50,1000}?)(?=\n\n|\ndischarge)',
            'history': r'history\s+of\s+present\s+illness[:\s]+(.{50,600}?)(?=\n\n|\nhospital)',
        }
        
        for key, pattern in patterns.items():
            match = re.search(pattern, text_lower, re.DOTALL | re.MULTILINE)
            if match:
                extracted = match.group(1).strip()
                sections[key] = extracted[:500]  # Limit length
                logger.debug(f"Extracted {key}: {len(extracted)} chars")
        
        return sections

    def simplify_medical_terms(self, text: str) -> str:
        """Replace medical jargon with patient-friendly terms."""
        if not text:
            return text
            
        simplified = text
        replacements_made = 0
        
        for medical, simple in MEDICAL_TERMS.items():
            # Use word boundaries for accurate replacement
            pattern = r'\b' + re.escape(medical) + r'\b'
            new_text = re.sub(pattern, simple, simplified, flags=re.IGNORECASE)
            if new_text != simplified:
                replacements_made += 1
                simplified = new_text
        
        logger.debug(f"Simplified {replacements_made} medical terms")
        return simplified

    def _call_gemini(self, prompt: str) -> Optional[str]:
        """Call Gemini with comprehensive error handling."""
        if not self.gemini_configured or not self.gemini_model:
            logger.warning("Gemini not configured, skipping")
            return None
        
        try:
            logger.info("Calling Gemini API...")
            response = self.gemini_model.generate_content(prompt)
            
            if not response or not response.text:
                logger.error("Gemini returned empty response")
                return None
            
            result = response.text.strip()
            logger.info(f"✅ Gemini response received ({len(result)} chars)")
            return result
            
        except Exception as e:
            logger.error(f"❌ Gemini API error: {type(e).__name__}")
            logger.error(f"Error message: {str(e)}")
            return None

    def generate_summary(self, text: str) -> Dict[str, str]:
        """Generate patient-friendly summary with full pipeline."""
        logger.info("="*60)
        logger.info(f"Starting summarization (input: {len(text)} chars)")
        logger.info("="*60)
        
        # Step 1: Extract structured sections
        sections = self.extract_key_sections(text)
        logger.info(f"Sections extracted: {sum(1 for v in sections.values() if v)}/7")
        
        # Step 2: Build structured input for BART
        structured_parts = []
        
        if sections['chief_complaint']:
            structured_parts.append(f"Chief Complaint: {sections['chief_complaint']}")
        if sections['diagnosis']:
            structured_parts.append(f"Diagnosis: {sections['diagnosis']}")
        if sections['history']:
            structured_parts.append(f"History: {sections['history'][:300]}")
        if sections['course']:
            structured_parts.append(f"Hospital Course: {sections['course'][:500]}")
        if sections['procedures']:
            structured_parts.append(f"Procedures: {sections['procedures']}")
        if sections['medications']:
            structured_parts.append(f"Medications: {sections['medications'][:300]}")
        if sections['disposition']:
            structured_parts.append(f"Disposition: {sections['disposition']}")
        
        # Use structured input if available, otherwise use raw text
        if structured_parts:
            input_text = "\n\n".join(structured_parts)
            logger.info(f"Using structured input ({len(input_text)} chars)")
        else:
            input_text = text[:1200]
            logger.info(f"Using raw text (first {len(input_text)} chars)")
        
        # Step 3: BART summarization
        logger.info("Generating BART summary...")
        inputs = self.tokenizer(
            input_text,
            max_length=1024,
            truncation=True,
            return_tensors="pt",
            padding=True
        ).to(self.device)
        
        with torch.no_grad():
            output_ids = self.model.generate(
                inputs.input_ids,
                max_length=250,           # Longer for better detail
                min_length=80,            # Ensure substantial output
                num_beams=5,              # More beams for better quality
                length_penalty=2.0,
                early_stopping=True,
                no_repeat_ngram_size=3,   # Avoid repetition
                repetition_penalty=1.2    # Discourage repetition
            )
        
        bart_output = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        logger.info(f"✅ BART output generated ({len(bart_output)} chars)")
        
        # Step 4: Simplify medical terms
        bart_simplified = self.simplify_medical_terms(bart_output)
        
        # Step 5: Gemini refinement with robust error handling
        final_summary = None
        
        if self.gemini_configured:
            logger.info("Attempting Gemini refinement...")
            
            # Enhanced prompt with clear instructions
            gemini_prompt = f"""You are a compassionate medical translator. Convert this clinical summary into warm, simple language for patients.

PATIENT INFORMATION:
{bart_simplified}

STRUCTURED DETAILS:
- Chief Complaint: {sections.get('chief_complaint', 'Not documented')}
- Diagnosis: {sections.get('diagnosis', 'See summary')}
- Procedures: {sections.get('procedures', 'None documented')}
- Medications: {sections.get('medications', 'See instructions')[:200]}
- Going Home To: {sections.get('disposition', 'Not specified')}

YOUR TASK:
Rewrite this as if you're a caring doctor talking directly to the patient. Use simple words a 6th grader would understand.

Create EXACTLY these 3 sections (be specific and detailed):

**What Happened:**
Explain in 2-3 clear sentences what brought them to the hospital and what the doctors found. Use "you" and "your". Be specific about the condition.

**What We Did:**
Explain in 2-3 sentences what treatments or procedures they received. Include specific details about medications if available.

**What's Next:**
Explain in 2-3 sentences what they need to do now - medications to take, doctor appointments, warning signs to watch for.

RULES:
- Use "you" and "your" (second person)
- NO medical jargon - use everyday words
- Be specific with medication names and instructions
- Include numbers and details (like "2 weeks", "81mg daily")
- Be warm and reassuring but accurate
- Each section should be 2-3 complete sentences"""

            final_summary = self._call_gemini(gemini_prompt)
        
        # Step 6: Fallback if Gemini fails
        if not final_summary:
            logger.warning("Using BART fallback (Gemini unavailable)")
            
            # Create structured fallback
            fallback_parts = []
            
            fallback_parts.append(f"**What Happened:**\n{bart_simplified}")
            
            if sections.get('procedures'):
                procedures_simple = self.simplify_medical_terms(sections['procedures'])
                fallback_parts.append(f"\n\n**What We Did:**\n{procedures_simple}")
            
            next_steps = []
            if sections.get('medications'):
                meds = sections['medications'][:300]
                meds_simple = self.simplify_medical_terms(meds)
                next_steps.append(f"Take your medications as prescribed: {meds_simple}")
            
            if sections.get('disposition'):
                disp_simple = self.simplify_medical_terms(sections['disposition'])
                next_steps.append(f"You're going to: {disp_simple}")
            
            if next_steps:
                fallback_parts.append(f"\n\n**What's Next:**\n{' '.join(next_steps)}")
            
            final_summary = "\n".join(fallback_parts)
        
        # Step 7: Extract clean diagnosis
        diagnosis = "Unknown"
        if sections.get('diagnosis'):
            # Take first line of diagnosis only
            diagnosis_lines = sections['diagnosis'].split('\n')
            diagnosis = diagnosis_lines[0].strip()
            # Remove numbering
            diagnosis = re.sub(r'^\d+\.\s*', '', diagnosis)
        
        logger.info("="*60)
        logger.info("✅ Summarization complete")
        logger.info(f"Output: {len(final_summary)} chars")
        logger.info("="*60)
        
        return {
            "summary": final_summary,
            "diagnosis": diagnosis,
            "bart_summary": bart_simplified
        }
    
    def _call_gemini(self, prompt: str) -> Optional[str]:
        """Call Gemini with comprehensive error handling and retry logic."""
        if not self.gemini_configured or not self.gemini_model:
            return None
        
        max_retries = 2
        for attempt in range(max_retries):
            try:
                logger.info(f"Gemini attempt {attempt + 1}/{max_retries}")
                
                response = self.gemini_model.generate_content(
                    prompt,
                    request_options={'timeout': 30}  # 30 second timeout
                )
                
                if not response:
                    logger.error("Gemini returned None response")
                    continue
                
                if not hasattr(response, 'text'):
                    logger.error(f"Gemini response has no text attribute: {type(response)}")
                    continue
                
                result_text = response.text
                
                if not result_text or len(result_text.strip()) < 20:
                    logger.error(f"Gemini returned empty/short text: '{result_text}'")
                    continue
                
                logger.info(f"✅ Gemini success on attempt {attempt + 1}")
                return result_text.strip()
                
            except Exception as e:
                logger.error(f"Gemini attempt {attempt + 1} failed: {type(e).__name__}")
                logger.error(f"Error details: {str(e)}")
                
                if attempt < max_retries - 1:
                    logger.info("Retrying...")
                    continue
                else:
                    logger.error("All Gemini attempts failed")
                    return None
        
        return None