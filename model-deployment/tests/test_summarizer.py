"""
Test script for production medical summarizer
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


import pytest
from deployment_pipeline.summarizer import MedicalSummarizer

# Load environment variables
load_dotenv(override=True)


def test_model_loading():
    """Test 1: Can we load the model?"""
    print("\n" + "="*50)
    print("TEST 1: Model Loading")
    print("="*50)
    
    try:
        # Test HuggingFace loading
        summarizer = MedicalSummarizer(use_gpu=False, model_source="huggingface")
        print("✅ Model loaded from HuggingFace successfully")
        return summarizer
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return None
@pytest.fixture
def summarizer():
    """Create a summarizer instance for tests"""
    return MedicalSummarizer(use_gpu=False, model_source="huggingface")

def test_basic_summarization(summarizer):
    """Test 2: Basic summarization without Gemini"""
    print("\n" + "="*50)
    print("TEST 2: Basic Summarization (No Gemini)")
    print("="*50)
    
    # Remove Gemini key temporarily
    os.environ.pop("GEMINI_API_KEY", None)
    
    test_text = """
    Admission Date: [**] Discharge Date: [**]

    Service: NEUROLOGY

    Chief Complaint:
    Found unresponsive on floor.

    History of Present Illness:
    Patient is 67 yo M with history of schizophrenia, dementia, anxiety, 
    found lying face-down unresponsive. O2 sat in 40s, pupils fixed.

    Hospital Course:
    Patient admitted to NeuroICU for management of brainstem hemorrhage. 
    Blood pressure maintained <160. Patient remained minimally responsive 
    for first 2 weeks. Day 14 showed improvement with eye opening. 
    Developed ventilator-associated pneumonia, treated with vancomycin 
    and cefepime 8-day course. Required tracheostomy and PEG placement 
    for long-term management.

    Discharge Diagnosis:
    Brainstem hemorrhage involving R midbrain, bilateral pons, and 
    4th ventricle, likely hypertensive etiology.

    Discharge Medications:
    [List medications...]

    Discharge Disposition:
    Extended Care Facility

    Discharge Instructions:
    Patient suffered major brain hemorrhage resulting in inability to 
    move, eat or speak. Requires ongoing medical care and therapy.
    """
    
    try:
        result = summarizer.generate_summary(test_text)
        print(f"✅ BART Summary: {result['raw_bart_summary'][:100]}...")
        print(f"✅ Extracted diagnosis: {result['extracted_data']['diagnosis']}")
        return True
    except Exception as e:
        print(f"❌ Summarization failed: {e}")
        return False

def test_with_gemini(summarizer):
    """Test 3: Full pipeline with Gemini"""
    print("\n" + "="*50)
    print("TEST 3: Full Pipeline with Gemini")
    print("="*50)
    # Reload .env since test 2 removed it!
    abs_path = Path(__file__).parent.parent / '.env'
    load_dotenv(dotenv_path=abs_path, override=True)
    

    # Set Gemini key if you have it
    if "GEMINI_API_KEY" in os.environ:
        test_text = """
        Discharge Diagnosis: Brainstem hemorrhage
        Discharge Disposition: Extended care facility
        Hospital Course: Patient found unresponsive. CT showed brainstem hemorrhage.
        Unable to walk or speak. Required tracheostomy.
        """
        
        try:
            result = summarizer.generate_summary(test_text)
            print(f"✅ Final Summary Generated:")
            print(result['final_summary'][:200])
            return True
        except Exception as e:
            print(f"⚠️ Gemini processing failed: {e}")
            return False
    else:
        print("⚠️ Skipping - GEMINI_API_KEY not set")
        return None

def test_performance():
    """Test 4: Performance metrics"""
    print("\n" + "="*50)
    print("TEST 4: Performance Metrics")
    print("="*50)
    
    import time
    
    summarizer = MedicalSummarizer(use_gpu=False)
    test_text = "Discharge Diagnosis: Pneumonia. Disposition: Home."
    
    start = time.time()
    result = summarizer.generate_summary(test_text)
    duration = time.time() - start
    
    print(f"✅ Inference time: {duration:.2f} seconds")
    print(f"✅ Expected: 2-3 seconds")
    
    if duration < 5:
        print("✅ Performance acceptable")
        return True
    else:
        print("⚠️ Performance slower than expected")
        return False

if __name__ == "__main__":
    print("🏥 MEDICAL SUMMARIZER TEST SUITE")
    print("="*50)
    
    # Run tests
    summarizer = test_model_loading()
    if summarizer:
        test_basic_summarization(summarizer)
        test_with_gemini(summarizer)
        test_performance()
    
    print("\n" + "="*50)
    print("Testing Complete!")


    