#!/usr/bin/env python3
"""
Example usage of Gemini 1.5 Pro for medical text summarization
"""

import os
import sys
import pandas as pd

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.training import GeminiTrainer, GeminiInference, load_gemini_model
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


def example_setup():
    """Example: Set up Gemini model"""
    print("=" * 60)
    print("Example: Setting Up Gemini 2.0 Flash Exp")
    print("=" * 60)
    
    # Initialize trainer (sets up prompts and configuration)
    trainer = GeminiTrainer(
        config_path='configs/gemini_config.json',
        model_name='gemini-2.0-flash-exp',  # Latest model
        output_dir='models/gemini'
    )
    
    # Process data with Gemini
    results = trainer.process_data(
        data_path='data-pipeline/data/processed/processed_discharge_summaries.csv'
    )
    
    print(f"\nProcessing completed!")
    print(f"Total Records: {results['total_records']}")
    print(f"Processed Records: {results['processed_records']}")
    print(f"Output saved to: {results['output_path']}")


def example_inference():
    """Example: Use Gemini for inference"""
    print("=" * 60)
    print("Example: Gemini 2.0 Flash Exp Inference")
    print("=" * 60)
    
    # Sample medical text
    sample_text = """
    Patient is a 65-year-old male with a history of hypertension, diabetes mellitus type 2, 
    and coronary artery disease. He presented to the emergency department with acute onset 
    of chest pain and shortness of breath. EKG showed ST elevation in leads II, III, and aVF. 
    Cardiac enzymes were elevated. Patient was taken to the cardiac catheterization lab where 
    he underwent successful percutaneous coronary intervention with placement of a drug-eluting 
    stent to the right coronary artery. Post-procedure, patient was stable and transferred to 
    the cardiac care unit for monitoring. Discharge medications include aspirin, clopidogrel, 
    atorvastatin, metformin, and lisinopril. Patient is to follow up with cardiology in 2 weeks.
    """
    
    # Initialize inference model
    inferencer = GeminiInference(
        api_key=os.getenv('GOOGLE_API_KEY'),
        model_name='gemini-2.0-flash-exp'  # Latest model
    )
    
    # Generate summary
    summary = inferencer.summarize(sample_text, max_length=150)
    
    print(f"\nOriginal Text Length: {len(sample_text)} characters")
    print(f"\nGenerated Summary ({len(summary)} characters):")
    print("-" * 60)
    print(summary)
    print("-" * 60)


def example_batch_processing():
    """Example: Batch process DataFrame"""
    print("=" * 60)
    print("Example: Batch Processing with Gemini 2.0 Flash Exp")
    print("=" * 60)
    
    # Load data
    df = pd.read_csv('data-pipeline/data/processed/processed_discharge_summaries.csv')
    
    # Take a small sample for demonstration
    df_sample = df.head(10)
    
    # Initialize inference model
    inferencer = GeminiInference(
        api_key=os.getenv('GOOGLE_API_KEY'),
        model_name='gemini-2.0-flash-exp'  # Latest model
    )
    
    # Process DataFrame
    df_summarized = inferencer.process_dataframe(
        df_sample,
        input_column='cleaned_text',
        output_column='gemini_summary',
        max_length=150
    )
    
    # Display results
    print(f"\nProcessed {len(df_summarized)} records")
    print("\nSample summaries:")
    for idx, row in df_summarized.iterrows():
        print(f"\nRecord {idx + 1}:")
        print(f"Original: {row['cleaned_text'][:100]}...")
        print(f"Summary: {row['gemini_summary']}")
        print("-" * 60)
    
    # Save results
    output_path = 'data-pipeline/data/processed/summarized_sample.csv'
    df_summarized.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")


def example_load_model():
    """Example: Load and inspect Gemini model"""
    print("=" * 60)
    print("Example: Loading Gemini 2.0 Flash Exp Model")
    print("=" * 60)
    
    # Load model
    model = load_gemini_model(
        api_key=os.getenv('GOOGLE_API_KEY'),
        model_name='gemini-2.0-flash-exp'  # Latest model
    )
    
    print(f"Model: {model.model_name}")
    print(f"Temperature: {model.temperature}")
    print(f"Max Output Tokens: {model.max_output_tokens}")
    
    # Test summarization
    test_text = "Patient admitted with chest pain and shortness of breath. EKG showed ST elevation. Cardiac enzymes elevated. Underwent successful PCI with stent placement."
    summary = model.summarize(test_text, max_length=50)
    
    print(f"\nTest Text: {test_text}")
    print(f"Summary: {summary}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Gemini 1.5 Pro usage examples')
    parser.add_argument('--example', type=str, 
                       choices=['setup', 'inference', 'batch', 'load'],
                       default='load',
                       help='Which example to run')
    parser.add_argument('--api-key', type=str,
                       help='Google AI API key (or set GOOGLE_API_KEY env var)')
    
    args = parser.parse_args()
    
    # Set API key if provided
    if args.api_key:
        os.environ['GOOGLE_API_KEY'] = args.api_key
    
    if args.example == 'setup':
        example_setup()
    elif args.example == 'inference':
        example_inference()
    elif args.example == 'batch':
        example_batch_processing()
    elif args.example == 'load':
        example_load_model()
    else:
        print("Available examples:")
        print("  --example setup    : Set up Gemini model and process data")
        print("  --example inference: Generate summary for sample text")
        print("  --example batch    : Batch process DataFrame")
        print("  --example load     : Load and inspect model")


