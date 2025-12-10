#!/usr/bin/env python3
"""
Gemini 1.5 Pro Inference Script
Use Gemini 1.5 Pro to generate summaries for medical texts
"""

import os
import sys
import argparse
import pandas as pd
from pathlib import Path
from typing import List, Optional, Dict

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.training.gemini_model import GeminiSummarizer, load_gemini_model
from src.utils.logging_config import get_logger
from src.utils.error_handling import ErrorHandler, safe_execute

logger = get_logger(__name__)


class GeminiInference:
    """
    Inference class for Gemini 1.5 Pro summarization
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gemini-2.0-flash-exp"  # Latest model
    ):
        """
        Initialize inference model
        
        Args:
            api_key: Google AI API key (or set GOOGLE_API_KEY env var)
            model_name: Gemini model identifier
        """
        logger.info(f"Initializing Gemini inference model: {model_name}")
        
        try:
            self.model = GeminiSummarizer(api_key=api_key, model_name=model_name)
            logger.info("Gemini model initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini model: {e}")
            raise
    
    def summarize(
        self,
        text: str,
        max_length: int = 150
    ) -> str:
        """
        Generate summary for input text
        
        Args:
            text: Input medical text
            max_length: Maximum summary length
            
        Returns:
            Generated summary
        """
        logger.debug(f"Summarizing text of length {len(text)}")
        
        summary = self.model.summarize(text, max_length=max_length)
        
        return summary
    
    def batch_summarize(
        self,
        texts: List[str],
        max_length: int = 150,
        batch_size: int = 10
    ) -> List[str]:
        """
        Generate summaries for multiple texts
        
        Args:
            texts: List of input texts
            max_length: Maximum summary length
            batch_size: Batch size for processing
            
        Returns:
            List of generated summaries
        """
        return self.model.batch_summarize(texts, max_length=max_length, batch_size=batch_size)
    
    def process_dataframe(
        self,
        df: pd.DataFrame,
        input_column: str = 'cleaned_text',
        output_column: str = 'gemini_summary',
        max_length: int = 150
    ) -> pd.DataFrame:
        """
        Process DataFrame and add summaries
        
        Args:
            df: Input DataFrame
            input_column: Column name with input text
            output_column: Column name for output summaries
            max_length: Maximum summary length
            
        Returns:
            DataFrame with added summary column
        """
        return self.model.process_dataframe(df, input_column, output_column, max_length)


def main():
    """Main entry point for inference script"""
    parser = argparse.ArgumentParser(description='Gemini 1.5 Pro inference for medical text summarization')
    parser.add_argument('--api-key', type=str,
                       help='Google AI API key (or set GOOGLE_API_KEY env var)')
    parser.add_argument('--model-name', type=str, default='gemini-2.0-flash-exp',
                       help='Gemini model name (default: gemini-2.0-flash-exp, also try: gemini-1.5-pro)')
    parser.add_argument('--input', type=str, required=True,
                       help='Input: text string or path to CSV file')
    parser.add_argument('--output', type=str,
                       help='Output path for CSV (if input is CSV)')
    parser.add_argument('--input-column', type=str, default='cleaned_text',
                       help='Input column name (for CSV)')
    parser.add_argument('--output-column', type=str, default='gemini_summary',
                       help='Output column name (for CSV)')
    parser.add_argument('--max-length', type=int, default=150,
                       help='Maximum summary length')
    
    args = parser.parse_args()
    
    try:
        # Initialize inference model
        inferencer = GeminiInference(
            api_key=args.api_key,
            model_name=args.model_name
        )
        
        # Check if input is a file or text
        if os.path.exists(args.input):
            # Process CSV file
            logger.info(f"Loading data from {args.input}")
            df = pd.read_csv(args.input)
            
            df = inferencer.process_dataframe(
                df,
                input_column=args.input_column,
                output_column=args.output_column,
                max_length=args.max_length
            )
            
            # Save output
            output_path = args.output or args.input.replace('.csv', '_summarized.csv')
            df.to_csv(output_path, index=False)
            logger.info(f"Saved results to {output_path}")
            
        else:
            # Process single text
            summary = inferencer.summarize(args.input, max_length=args.max_length)
            print("\n=== Summary ===")
            print(summary)
        
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()


