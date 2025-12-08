"""
Production Medical Summarization System
Architecture: Smart Extraction -> BART-large-cnn -> RAG Simplification -> Gemini Refinement
"""

import os
import sys
from pathlib import Path

# Get the parent directory (model-deployment)
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

# Load .env from parent directory using absolute path
from dotenv import load_dotenv
abs_path = Path(__file__).parent.parent / '.env'
load_dotenv(dotenv_path=abs_path, override=True)

import re
import logging
import torch
import numpy as np
from typing import Dict, List, Optional
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
from google import genai
from google.genai.errors import APIError




# Model Configuration
MODEL_ID = os.getenv("MODEL_ID", "asadwaraich/bart-medical-discharge-summarizer")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MedicalSummarizer:
    """
    Production-ready medical summarizer.
    Selected Model: BART-large-cnn (Winner of BioBART comparison)
    """
    
    def __init__(self, use_gpu: bool = True, model_source: str = "huggingface"):
        logger.info("Initializing Production Medical Summarizer...")
        
        # 1. Device Setup
        self.device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # 2. Load The Winner Model (BART-large-cnn)
        logger.info("Loading Summarization Model...")
        try:
            # Determine model source
            if model_source == "huggingface":
                self.model_name = "asadwaraich/bart-medical-discharge-summarizer"
            elif model_source == "local":
                self.model_name = "../model_registry/fine_tuned_bart_large_cnn"
            else:
                self.model_name = MODEL_ID  # Use environment variable
            
            logger.info(f"Loading from: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
            self.model.to(self.device)
            logger.info("✅ Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise e
        
        # 3. Load RAG Components
        logger.info("Loading RAG Embedder (BioBERT)...")
        self.embedder = SentenceTransformer('dmis-lab/biobert-base-cased-v1.2')
        self.build_knowledge_base()
        self.create_retrieval_index()
        
        # 4. Check API Key
        if "GEMINI_API_KEY" not in os.environ:
            logger.warning("GEMINI_API_KEY not found. Gemini refinement will be skipped.")
        
        logger.info("✅ System Ready for Deployment")

    def build_knowledge_base(self):
        """Builds the simplification dictionary."""
        # Expanded list for production coverage
        self.medical_kb = [
            {"medical": "myocardial infarction MI", "simple": "heart attack", "context": "heart blockage"},
            {"medical": "cerebrovascular accident CVA stroke", "simple": "stroke", "context": "brain blockage"},
            {"medical": "hypertension HTN", "simple": "high blood pressure", "context": "chronic condition"},
            {"medical": "hyperlipidemia", "simple": "high cholesterol", "context": "blood fats"},
            {"medical": "renal failure", "simple": "kidney failure", "context": "organ failure"},
            {"medical": "pneumonia", "simple": "lung infection", "context": "respiratory infection"},
            {"medical": "sepsis", "simple": "severe blood infection", "context": "systemic infection"},
            {"medical": "hemorrhage", "simple": "bleeding", "context": "blood loss"},
            {"medical": "intracranial", "simple": "inside the brain", "context": "location"},
            {"medical": "subdural hematoma", "simple": "brain bleed", "context": "brain injury"},
            {"medical": "fracture", "simple": "broken bone", "context": "bone injury"},
            {"medical": "discharged to home", "simple": "sent home", "context": "disposition"},
            {"medical": "SNF skilled nursing facility", "simple": "nursing home", "context": "care facility"},
            {"medical": "ambulate", "simple": "walk", "context": "movement"},
            {"medical": "dyspnea", "simple": "shortness of breath", "context": "symptom"},
            {"medical": "syncope", "simple": "fainting", "context": "symptom"},
            {"medical": "unresponsive", "simple": "not waking up", "context": "status"},
            {"medical": "intubated", "simple": "put on a breathing machine", "context": "procedure"},
            {"medical": "tracheostomy", "simple": "breathing tube in neck", "context": "procedure"},
            {"medical": "PEG tube", "simple": "feeding tube", "context": "procedure"},
        ]
        
        # Pre-compute embeddings
        texts = [f"{entry['medical']} {entry['context']}" for entry in self.medical_kb]
        self.kb_embeddings = self.embedder.encode(texts)

    def create_retrieval_index(self):
        """Creates the search index."""
        self.retriever = NearestNeighbors(n_neighbors=5, metric='cosine')
        self.retriever.fit(self.kb_embeddings)

    def _call_gemini_api(self, prompt: str) -> str:
        """Securely calls Gemini API with error handling."""
        try:
            # Get the API key from environment
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                logger.warning("GEMINI_API_KEY not found in environment")
                return None
            
            # Configure the API with your key
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            
            # Create the model
            model = genai.GenerativeModel('gemini-2.5-flash')
            
            # Generate content
            response = model.generate_content(prompt)
            
            return response.text.strip()
            
        except Exception as e:
            logger.error(f"Gemini API Error: {e}")
            return None

    def extract_key_sections(self, text: str) -> Dict[str, str]:
        """Regex-based extraction of specific headers."""
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
        """
        Intelligently selects the most important parts of the text 
        to fit within the model's context window.
        """
        # 1. Try Structured Extraction First
        sections = self.extract_key_sections(text)
        parts = []
        
        if sections['diagnosis']: parts.append(f"Diagnosis: {sections['diagnosis']}")
        if sections['disposition']: parts.append(f"Disposition: {sections['disposition']}")
        if sections['procedures']: parts.append(f"Procedures: {sections['procedures']}")
        if sections['course']: parts.append(f"Course: {sections['course']}")
        
        if parts:
            structured = "\n".join(parts)
            # If we have space, add raw text start
            remaining_chars = (max_tokens * 4) - len(structured)
            if remaining_chars > 0:
                structured += "\n" + text[:remaining_chars]
            return structured
            
        # 2. Fallback: Keyword Sentence Extraction
        sentences = text.split('. ')
        keywords = ['diagnosis', 'admitted', 'discharge', 'treatment', 'plan', 'history']
        scored_sentences = []
        
        for sent in sentences:
            score = sum(3 for k in keywords if k in sent.lower())
            scored_sentences.append((score, sent))
            
        # Sort by score and reassemble chronologically
        scored_sentences.sort(key=lambda x: x[0], reverse=True)
        top_sentences = [s[1] for s in scored_sentences[:15]] # Take top 15 sentences
        
        return ". ".join(top_sentences)

    def generate_summary(self, text: str) -> Dict[str, str]:
        """Main pipeline execution."""
        logger.info("Starting summarization pipeline...")
        
        # Step 1: Smart Extraction
        input_text = self.smart_extract(text)
        
        # Step 2: BART Summarization
        inputs = self.tokenizer(input_text, max_length=1024, truncation=True, return_tensors="pt").to(self.device)
        with torch.no_grad():
            summary_ids = self.model.generate(
                inputs.input_ids, 
                max_length=150, 
                min_length=40, 
                num_beams=4, 
                length_penalty=2.0, 
                early_stopping=True
            )
        base_summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        
        # Step 3: RAG Simplification (Identify terms to simplify)
        query_vec = self.embedder.encode(base_summary)
        distances, indices = self.retriever.kneighbors([query_vec])
        
        relevant_terms = [self.medical_kb[idx] for idx in indices[0]]
        
        # Step 4: Gemini Refinement
        sections = self.extract_key_sections(text)
        
        prompt = f"""
        Act as a medical patient advocate. Rewrite this clinical summary into a clear, empathetic 3-part structure for a patient with a 6th-grade reading level.
        
        **Source Information:**
        - Primary Diagnosis: {sections.get('diagnosis', 'See summary')}
        - Discharge Plan: {sections.get('disposition', 'See summary')}
        - Clinical Summary: {base_summary}
        
        **Medical Terms to Simplify:**
        {", ".join([f"{t['medical']} -> {t['simple']}" for t in relevant_terms])}
        
        **Required Output Format:**
        **What Happened:** (Explain diagnosis and cause simply)
        **What Was Done:** (Explain treatments and hospital stay)
        **What is Next:** (Explain discharge plan and recovery)
        """
        
        final_summary = self._call_gemini_api(prompt)
        
        # Fallback if Gemini fails
        if not final_summary:
            logger.warning("Gemini failed, using fallback template.")
            final_summary = f"**Summary:** {base_summary}\n\n**Note:** {sections.get('disposition', '')}"

        return {
            "final_summary": final_summary,
            "raw_bart_summary": base_summary,
            "extracted_data": sections
        }

