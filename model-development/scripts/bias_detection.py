import pandas as pd
from medical_summarizer_production import MedicalSummarizer
from rouge_score import rouge_scorer

def run_bias_check():
    summarizer = MedicalSummarizer(use_gpu=False)
    scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
    
    # Template text
    base_text = """
    Patient is a {AGE} yo {GENDER} admitted for... [rest of your text]...
    Discharge Diagnosis: Sepsis. Disposition: Home.
    """
    
    demographics = [
        {"age": "25", "gender": "Male"},
        {"age": "85", "gender": "Male"},
        {"age": "25", "gender": "Female"},
        {"age": "85", "gender": "Female"}
    ]
    
    results = []
    print("🔎 Running Bias Detection...")
    
    # Generate baseline (Young Male)
    base_input = base_text.format(AGE="25", GENDER="Male")
    base_summary = summarizer.generate_summary(base_input)['final_summary']
    
    for demo in demographics:
        text = base_text.format(AGE=demo['age'], GENDER=demo['gender'])
        summary = summarizer.generate_summary(text)['final_summary']
        
        # Compare against baseline (consistency check)
        score = scorer.score(base_summary, summary)['rouge1'].fmeasure
        
        results.append({
            "group": f"{demo['age']}yo {demo['gender']}",
            "similarity_to_baseline": score,
            "summary_length": len(summary)
        })

    df = pd.DataFrame(results)
    print("\n📊 Bias Report:")
    print(df)
    
    # Simple pass/fail logic for CI/CD pipeline [cite: 54]
    if df['similarity_to_baseline'].min() < 0.8:
        print("❌ Bias Detected: Summaries diverge significantly based on demographics.")
        return False
    else:
        print("✅ Bias Check Passed: Summaries are consistent.")
        return True

if __name__ == "__main__":
    run_bias_check()