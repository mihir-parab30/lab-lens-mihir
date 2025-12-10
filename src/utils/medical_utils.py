"""
Medical Utilities Module
Provides BioBERT embeddings and medical term simplification for enhanced RAG
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import re

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.utils.logging_config import get_logger

logger = get_logger(__name__)

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logger.warning("sentence-transformers not available. BioBERT embeddings disabled.")


class MedicalEmbedder:
    """
    Medical-specific embedder using BioBERT
    Better suited for medical terminology than generic embeddings
    """
    
    _instance = None
    _embedder = None
    _model_name = "dmis-lab/biobert-base-cased-v1.2"
    
    def __new__(cls):
        """Singleton pattern to avoid loading model multiple times"""
        if cls._instance is None:
            cls._instance = super(MedicalEmbedder, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize BioBERT embedder (lazy loading)"""
        if self._embedder is None and SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                logger.info(f"Loading BioBERT embedder: {self._model_name}")
                self._embedder = SentenceTransformer(self._model_name)
                logger.info("✅ BioBERT embedder loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load BioBERT: {e}")
                self._embedder = None
    
    @property
    def embedder(self):
        """Get the embedder instance"""
        if self._embedder is None and SENTENCE_TRANSFORMERS_AVAILABLE:
            self.__init__()
        return self._embedder
    
    def encode(self, texts: List[str], **kwargs):
        """Encode texts using BioBERT"""
        if not self.embedder:
            raise ValueError("BioBERT embedder not available. Install sentence-transformers.")
        return self.embedder.encode(texts, **kwargs)
    
    def get_embedding_dimension(self) -> Optional[int]:
        """Get the dimension of embeddings"""
        if not self.embedder:
            return None
        return self.embedder.get_sentence_embedding_dimension()


class MedicalTermSimplifier:
    """
    Medical term simplification using knowledge base
    Simplifies complex medical terminology for patient-friendly answers
    """
    
    def __init__(self):
        """Initialize medical knowledge base"""
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
            {"medical": "ED emergency department", "simple": "emergency room", "context": "location"},
            {"medical": "ICU intensive care unit", "simple": "intensive care", "context": "location"},
            {"medical": "cardiac arrest", "simple": "heart stopped", "context": "condition"},
            {"medical": "respiratory failure", "simple": "breathing problems", "context": "condition"},
            {"medical": "diabetes mellitus DM", "simple": "diabetes", "context": "condition"},
            {"medical": "chronic obstructive pulmonary disease COPD", "simple": "lung disease", "context": "condition"},
            {"medical": "congestive heart failure CHF", "simple": "heart failure", "context": "condition"},
            {"medical": "atrial fibrillation AFib", "simple": "irregular heartbeat", "context": "condition"},
            {"medical": "pulmonary embolism PE", "simple": "blood clot in lung", "context": "condition"},
            {"medical": "deep vein thrombosis DVT", "simple": "blood clot in leg", "context": "condition"},
        ]
        
        # Build lookup dictionaries for faster matching
        self._build_lookup_dicts()
        logger.info(f"Medical term simplifier initialized with {len(self.medical_kb)} terms")
    
    def _build_lookup_dicts(self):
        """Build lookup dictionaries for efficient term matching"""
        self.term_map = {}
        self.medical_patterns = []
        
        for entry in self.medical_kb:
            medical_terms = entry["medical"].lower().split()
            simple_term = entry["simple"]
            
            # Create pattern for each medical term variant
            for term in medical_terms:
                # Store longest match
                if term not in self.term_map or len(term) > len(self.term_map[term][0]):
                    self.term_map[term] = (term, simple_term)
            
            # Also store full phrase matches
            full_phrase = entry["medical"].lower()
            self.term_map[full_phrase] = (full_phrase, simple_term)
            
            # Create regex pattern for phrase matching
            pattern = r'\b' + re.escape(entry["medical"].lower()) + r'\b'
            self.medical_patterns.append((re.compile(pattern, re.IGNORECASE), simple_term))
    
    def simplify_text(self, text: str, aggressive: bool = False) -> str:
        """
        Simplify medical terms in text
        
        Args:
            text: Text containing medical terminology
            aggressive: If True, replace more aggressively (can affect context)
            
        Returns:
            Text with simplified medical terms
        """
        if not text:
            return text
        
        simplified = text
        
        # Sort patterns by length (longest first) to match longer phrases first
        sorted_patterns = sorted(self.medical_patterns, key=lambda x: len(x[0].pattern), reverse=True)
        
        # Apply substitutions (use parentheses to capture word boundaries)
        for pattern, replacement in sorted_patterns:
            # Replace with simple term, preserving case
            def replace_func(match):
                matched_text = match.group(0)
                # Preserve capitalization
                if matched_text.isupper():
                    return replacement.upper()
                elif matched_text[0].isupper():
                    return replacement.capitalize()
                else:
                    return replacement
            
            simplified = pattern.sub(replace_func, simplified)
        
        # Additional single-word replacements if aggressive
        if aggressive:
            words = simplified.split()
            simplified_words = []
            for word in words:
                word_lower = word.lower().rstrip('.,;:!?)')
                if word_lower in self.term_map:
                    simple = self.term_map[word_lower][1]
                    # Preserve punctuation and capitalization
                    if word[-1] in '.,;:!?)':
                        punctuation = word[-1]
                    else:
                        punctuation = ''
                    
                    if word[0].isupper():
                        simple = simple.capitalize()
                    simplified_words.append(simple + punctuation)
                else:
                    simplified_words.append(word)
            simplified = ' '.join(simplified_words)
        
        return simplified
    
    def find_medical_terms(self, text: str) -> List[Tuple[str, str]]:
        """
        Find medical terms in text that can be simplified
        
        Args:
            text: Text to analyze
            
        Returns:
            List of tuples (medical_term, simple_term) found in text
        """
        found_terms = []
        text_lower = text.lower()
        
        for entry in self.medical_kb:
            medical_phrase = entry["medical"].lower()
            if medical_phrase in text_lower:
                found_terms.append((entry["medical"], entry["simple"]))
        
        return found_terms
    
    def add_term(self, medical: str, simple: str, context: str = ""):
        """
        Add a custom medical term to the knowledge base
        
        Args:
            medical: Medical term
            simple: Simplified term
            context: Optional context
        """
        self.medical_kb.append({
            "medical": medical,
            "simple": simple,
            "context": context
        })
        self._build_lookup_dicts()
        logger.info(f"Added custom term: {medical} -> {simple}")


def get_medical_embedder() -> Optional[MedicalEmbedder]:
    """Get singleton instance of medical embedder"""
    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        return None
    return MedicalEmbedder()


def get_medical_simplifier() -> MedicalTermSimplifier:
    """Get instance of medical term simplifier"""
    return MedicalTermSimplifier()
