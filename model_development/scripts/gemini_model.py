#!/usr/bin/env python3
"""
Gemini 1.5 Pro Model for Medical Text Summarization
Uses Google's Gemini 1.5 Pro API for summarizing MIMIC-III discharge summaries
"""

import os
import sys
import time
from typing import Dict, List, Optional, Tuple
import pandas as pd
from pathlib import Path
import warnings

# Suppress gRPC warnings
os.environ['GRPC_VERBOSITY'] = 'ERROR'
os.environ['GRPC_PYTHON_LOG_LEVEL'] = 'ERROR'
# Suppress absl logging warnings (if available)
try:
    import absl.logging
    absl.logging.set_verbosity(absl.logging.ERROR)
except ImportError:
    pass  # absl not available, warnings will be suppressed by env vars

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

# Load environment variables from .env file if available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not required, but helpful

from utils.logging_config import get_logger
from utils.error_handling import ErrorHandler, safe_execute

logger = get_logger(__name__)

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    logger.warning("google-generativeai not installed. Install with: pip install google-generativeai")


class GeminiSummarizer:
    """
    Gemini 1.5 Pro-based model for medical text summarization
    
    Uses Google's Gemini 2.0 Flash Exp API for abstractive summarization
    of medical discharge summaries (with fallback to gemini-1.5-pro).
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gemini-2.0-flash-exp",  # Latest model, fallback to gemini-1.5-pro if not available
        temperature: float = 0.3,
        max_output_tokens: int = 2048
    ):
        """
        Initialize Gemini summarization model
        
        Args:
            api_key: Google AI API key (or set GOOGLE_API_KEY env var)
            model_name: Gemini model identifier
            temperature: Sampling temperature (0.0-1.0, lower = more deterministic)
            max_output_tokens: Maximum tokens in output
        """
        if not GEMINI_AVAILABLE:
            raise ImportError("google-generativeai package required. Install with: pip install google-generativeai")
        
        self.model_name = model_name
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        
        # Get API key
        self.api_key = api_key or os.getenv('GOOGLE_API_KEY')
        if not self.api_key:
            raise ValueError(
                "Google API key required. Set GOOGLE_API_KEY environment variable "
                "or pass api_key parameter. Get key from: https://aistudio.google.com/app/apikey"
            )
        
        # Configure Gemini
        genai.configure(api_key=self.api_key)
        
        # Initialize model (try latest first, fallback to 1.5-pro)
        try:
            self.model = genai.GenerativeModel(model_name)
            logger.info(f"Initialized Gemini model: {model_name}")
        except Exception as e:
            # Fallback to gemini-1.5-pro if newer model not available
            if model_name != "gemini-1.5-pro":
                logger.warning(f"Model {model_name} not available, trying gemini-1.5-pro...")
                try:
                    self.model = genai.GenerativeModel("gemini-1.5-pro")
                    self.model_name = "gemini-1.5-pro"
                    logger.info(f"Using fallback model: gemini-1.5-pro")
                except Exception as e2:
                    logger.error(f"Failed to initialize Gemini model: {e2}")
                    raise
            else:
                logger.error(f"Failed to initialize Gemini model: {e}")
                raise
    
    def summarize(
        self,
        text: str,
        max_length: int = 150,
        system_prompt: Optional[str] = None
    ) -> str:
        """
        Generate summary for input medical text
        
        Args:
            text: Input medical text to summarize
            max_length: Target maximum length of summary (approximate)
            system_prompt: Optional custom system prompt
            
        Returns:
            Generated summary text
        """
        # Default prompt for medical summarization
        if system_prompt is None:
            system_prompt = """You are a medical expert assistant. Your task is to create concise, 
accurate summaries of medical discharge summaries. Focus on:
- Chief complaint and primary diagnosis
- Key treatments and procedures
- Discharge medications
- Important follow-up instructions

Keep the summary clear, professional, and medically accurate."""
        
        prompt = f"""{system_prompt}

Please summarize the following medical discharge summary in approximately {max_length} words:

{text}

Summary:"""
        
        try:
            # Generate response (API has built-in timeout, but we add error handling)
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=self.temperature,
                    max_output_tokens=self.max_output_tokens,
                )
            )
            
            summary = response.text.strip()
            return summary
            
        except Exception as e:
            error_msg = str(e).lower()
            if 'timeout' in error_msg or 'timed out' in error_msg:
                logger.error(f"Timeout generating summary: {e}")
                raise TimeoutError(f"API call timed out: {e}")
            logger.error(f"Error generating summary: {e}")
            raise
    
    def batch_summarize(
        self,
        texts: List[str],
        max_length: int = 150,
        batch_size: int = 10,
        delay: float = 1.0
    ) -> List[str]:
        """
        Generate summaries for multiple texts
        
        Args:
            texts: List of input texts
            max_length: Target maximum length of summary
            batch_size: Number of texts to process before delay
            delay: Delay between batches (seconds) to respect rate limits
            
        Returns:
            List of generated summaries
        """
        logger.info(f"Summarizing {len(texts)} texts in batches of {batch_size}")
        
        summaries = []
        
        for i, text in enumerate(texts):
            try:
                summary = self.summarize(text, max_length=max_length)
                summaries.append(summary)
                
                # Rate limiting
                if (i + 1) % batch_size == 0:
                    logger.info(f"Processed {i + 1}/{len(texts)} texts...")
                    time.sleep(delay)
                    
            except Exception as e:
                logger.error(f"Error summarizing text {i + 1}: {e}")
                summaries.append("")  # Empty summary on error
        
        return summaries
    
    def answer_question(
        self,
        question: str,
        context: str,
        temperature: Optional[float] = None
    ) -> str:
        """
        Answer a question based on provided context (for Q&A tasks)
        
        Args:
            question: The question to answer
            context: Context/documentation to base answer on
            temperature: Optional temperature override (uses instance default if None)
            
        Returns:
            Generated answer text
        """
        temp = temperature if temperature is not None else self.temperature
        
        prompt = f"""You are a helpful medical assistant helping a patient understand their discharge summary. Answer the patient's question based ONLY on the information provided in their discharge summary below. 

Important guidelines:
1. Answer clearly and in simple, patient-friendly language
2. Only use information from the provided discharge summary
3. If the information is not in the summary, say so clearly
4. Do not provide medical advice beyond what's in the summary
5. For medication questions, refer to the medications listed
6. For diagnosis questions, refer to the diagnoses listed
7. Keep your answer concise but complete (2-4 sentences typically)

DISCHARGE SUMMARY CONTEXT:
{context}

PATIENT'S QUESTION: {question}

ANSWER (based only on the discharge summary above):"""
        
        try:
            # Generate response (API has built-in timeout, but we add error handling)
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=temp,
                    max_output_tokens=self.max_output_tokens,
                )
            )
            
            answer = response.text.strip()
            return answer
            
        except Exception as e:
            error_msg = str(e).lower()
            if 'timeout' in error_msg or 'timed out' in error_msg:
                logger.error(f"Timeout generating answer: {e}")
                raise TimeoutError(f"API call timed out: {e}")
            logger.error(f"Error generating answer: {e}")
            raise
    
    def process_dataframe(
        self,
        df: pd.DataFrame,
        input_column: str = 'cleaned_text',
        output_column: str = 'gemini_summary',
        max_length: int = 150,
        batch_size: int = 10,
        delay: float = 1.0
    ) -> pd.DataFrame:
        """
        Process DataFrame and add summaries
        
        Args:
            df: Input DataFrame
            input_column: Column name with input text
            output_column: Column name for output summaries
            max_length: Target maximum summary length
            batch_size: Number of texts to process before delay
            delay: Delay between batches (seconds) to respect rate limits
            
        Returns:
            DataFrame with added summary column
        """
        logger.info(f"Processing DataFrame with {len(df)} rows")
        
        if input_column not in df.columns:
            raise ValueError(f"Column '{input_column}' not found in DataFrame")
        
        texts = df[input_column].astype(str).tolist()
        summaries = self.batch_summarize(texts, max_length=max_length, batch_size=batch_size, delay=delay)
        
        df[output_column] = summaries
        
        logger.info(f"Added summaries to column '{output_column}'")
        
        return df


def load_gemini_model(
    api_key: Optional[str] = None,
    model_name: str = "gemini-2.0-flash-exp",  # Latest model
    temperature: float = 0.3,
    max_output_tokens: int = 2048
) -> GeminiSummarizer:
    """
    Load Gemini model
    
    Args:
        api_key: Google AI API key (or set GOOGLE_API_KEY env var)
        model_name: Gemini model identifier (default: gemini-2.0-flash-exp)
        temperature: Sampling temperature (0.0-1.0)
        max_output_tokens: Maximum tokens in output
        
    Returns:
        GeminiSummarizer instance
    """
    logger.info(f"Loading Gemini model: {model_name}")
    
    model = GeminiSummarizer(
        api_key=api_key,
        model_name=model_name,
        temperature=temperature,
        max_output_tokens=max_output_tokens
    )
    
    return model

