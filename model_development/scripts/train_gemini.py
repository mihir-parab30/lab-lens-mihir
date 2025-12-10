#!/usr/bin/env python3
"""
Training/Prompt Engineering for Gemini 1.5 Pro Medical Text Summarization
Since Gemini is an API model, this script sets up prompt engineering and few-shot examples
"""

import os
import sys
import argparse
import json
import pandas as pd
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.training.gemini_model import GeminiSummarizer, load_gemini_model
from src.utils.logging_config import get_logger
from src.utils.error_handling import ErrorHandler, safe_execute, ErrorContext

logger = get_logger(__name__)


class GeminiTrainer:
    """
    Trainer class for setting up Gemini 1.5 Pro with medical text summarization
    Since Gemini is an API model, this focuses on prompt engineering and few-shot examples
    """
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        api_key: Optional[str] = None,
        model_name: str = "gemini-2.0-flash-exp",  # Latest model
        output_dir: str = "models/gemini"
    ):
        """
        Initialize Gemini trainer
        
        Args:
            config_path: Path to training configuration file
            api_key: Google AI API key
            model_name: Gemini model identifier
            output_dir: Directory to save configuration and examples
        """
        self.config = self._load_config(config_path)
        # Use model_name from config if available, otherwise use parameter
        self.model_name = self.config.get('model_name', model_name)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.api_key = api_key or os.getenv('GOOGLE_API_KEY')
        
        self.error_handler = ErrorHandler(logger)
        
        logger.info(f"Initializing Gemini trainer")
        logger.info(f"Model: {self.model_name}")
        logger.info(f"Output directory: {self.output_dir}")
    
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Load training configuration"""
        default_config = {
            'max_input_length': 32000,  # Gemini 1.5 Pro supports up to 2M tokens
            'max_output_length': 150,
            'temperature': 0.3,
            'system_prompt': None,  # Will use default
            'few_shot_examples': 3,
            'batch_size': 10,
            'delay_between_batches': 1.0,
            'input_column': 'cleaned_text',
            'target_column': None,
            'train_split': 0.8,
            'val_split': 0.1
        }
        
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    user_config = json.load(f)
                
                # Handle nested config structure
                if 'model_config' in user_config:
                    default_config.update(user_config['model_config'])
                if 'processing_config' in user_config:
                    default_config.update(user_config['processing_config'])
                if 'data_config' in user_config:
                    default_config.update(user_config['data_config'])
                # Also update flat keys if present
                for key in ['input_column', 'target_column', 'batch_size', 'delay_between_batches', 'max_output_length']:
                    if key in user_config:
                        default_config[key] = user_config[key]
                
                logger.info(f"Loaded config from {config_path}")
            except Exception as e:
                logger.warning(f"Failed to load config: {e}")
        
        return default_config
    
    @safe_execute("load_data", logger, ErrorHandler(logger))
    def load_data(self, data_path: str) -> pd.DataFrame:
        """
        Load training data
        
        Args:
            data_path: Path to processed data CSV
            
        Returns:
            DataFrame with training data
        """
        logger.info(f"Loading data from {data_path}")
        
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}")
        
        df = pd.read_csv(data_path)
        logger.info(f"Loaded {len(df)} records")
        logger.info(f"Columns: {df.columns.tolist()}")
        
        return df
    
    def create_few_shot_examples(self, df: pd.DataFrame, num_examples: int = 3) -> str:
        """
        Create few-shot examples for prompt engineering
        
        Args:
            df: Training data
            num_examples: Number of examples to include
            
        Returns:
            Formatted few-shot examples string
        """
        examples = []
        
        for i in range(min(num_examples, len(df))):
            row = df.iloc[i]
            input_text = str(row.get(self.config['input_column'], ''))[:500]  # Truncate for examples
            
            # Create extractive summary for example
            sentences = input_text.split('. ')
            summary = '. '.join(sentences[:2]) + '.' if len(sentences) > 2 else input_text[:100]
            
            examples.append(f"""
Example {i + 1}:
Input: {input_text[:200]}...
Summary: {summary}
""")
        
        return '\n'.join(examples)
    
    def setup_model(self) -> GeminiSummarizer:
        """
        Set up Gemini model with optimized prompts
        
        Returns:
            Configured GeminiSummarizer
        """
        logger.info("Setting up Gemini model...")
        
        model = load_gemini_model(
            api_key=self.api_key,
            model_name=self.model_name
        )
        
        # Save configuration
        config = {
            'model_name': self.model_name,
            'temperature': self.config['temperature'],
            'max_output_length': self.config['max_output_length'],
            'config': self.config,
            'timestamp': datetime.now().isoformat()
        }
        
        config_path = self.output_dir / 'gemini_config.json'
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Saved configuration to {config_path}")
        
        return model
    
    def process_data(
        self,
        data_path: str,
        output_path: Optional[str] = None
    ) -> Dict:
        """
        Process data with Gemini model
        
        Args:
            data_path: Path to input data
            output_path: Optional output path for processed data
            
        Returns:
            Processing results dictionary
        """
        with ErrorContext("gemini_processing", logger, self.error_handler):
            logger.info("Starting Gemini processing")
            
            # Load data
            df = self.load_data(data_path)
            
            # Set up model
            model = self.setup_model()
            
            # Process data
            logger.info("Processing data with Gemini...")
            df_processed = model.process_dataframe(
                df,
                input_column=self.config['input_column'],
                output_column='gemini_summary',
                max_length=self.config['max_output_length'],
                batch_size=self.config.get('batch_size', 10),
                delay=self.config.get('delay_between_batches', 1.0)
            )
            
            # Save results
            if output_path is None:
                output_path = self.output_dir / 'processed_with_summaries.csv'
            
            df_processed.to_csv(output_path, index=False)
            logger.info(f"Saved processed data to {output_path}")
            
            # Generate report
            results = {
                'total_records': len(df_processed),
                'processed_records': len(df_processed[df_processed['gemini_summary'].notna()]),
                'output_path': str(output_path),
                'config': self.config,
                'timestamp': datetime.now().isoformat()
            }
            
            results_path = self.output_dir / 'processing_results.json'
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2)
            
            logger.info("Processing completed successfully!")
            return results


def main():
    """Main entry point for training script"""
    parser = argparse.ArgumentParser(description='Set up Gemini 1.5 Pro for medical text summarization')
    parser.add_argument('--data-path', type=str, required=True,
                       help='Path to processed data CSV file')
    parser.add_argument('--config', type=str,
                       help='Path to configuration JSON')
    parser.add_argument('--api-key', type=str,
                       help='Google AI API key (or set GOOGLE_API_KEY env var)')
    parser.add_argument('--model-name', type=str, default='gemini-2.0-flash-exp',
                       help='Gemini model identifier (default: gemini-2.0-flash-exp, also try: gemini-1.5-pro)')
    parser.add_argument('--output-dir', type=str, default='models/gemini',
                       help='Output directory for configuration and results')
    parser.add_argument('--output', type=str,
                       help='Output path for processed data')
    
    args = parser.parse_args()
    
    try:
        trainer = GeminiTrainer(
            config_path=args.config,
            api_key=args.api_key,
            model_name=args.model_name,
            output_dir=args.output_dir
        )
        
        results = trainer.process_data(
            data_path=args.data_path,
            output_path=args.output
        )
        
        print("\n=== Processing Summary ===")
        print(f"Total Records: {results['total_records']}")
        print(f"Processed Records: {results['processed_records']}")
        print(f"Output saved to: {results['output_path']}")
        
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
