#!/usr/bin/env python3
"""
Risk Prediction Module for Medical Discharge Summaries
Predicts risk levels based on conditions, diagnoses, and clinical factors
"""

import os
import sys
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import logging

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    import google.generativeai as genai
    from dotenv import load_dotenv
    load_dotenv()
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

logger = logging.getLogger(__name__)


class MedicalRiskPredictor:
    """
    Predicts medical risk levels based on discharge summaries and clinical data
    """
    
    # High-risk conditions and keywords
    HIGH_RISK_CONDITIONS = [
        'sepsis', 'shock', 'cardiac arrest', 'stroke', 'mi', 'myocardial infarction',
        'respiratory failure', 'renal failure', 'liver failure', 'multiorgan failure',
        'severe', 'critical', 'acute', 'emergency', 'icu', 'intensive care',
        'ventilator', 'intubation', 'dialysis', 'transfusion', 'surgery',
        'complication', 'infection', 'bleeding', 'hemorrhage'
    ]
    
    # Medium-risk conditions
    MEDIUM_RISK_CONDITIONS = [
        'hypertension', 'diabetes', 'copd', 'asthma', 'chf', 'congestive heart failure',
        'pneumonia', 'uti', 'urinary tract infection', 'dehydration',
        'electrolyte imbalance', 'anemia', 'fever', 'infection'
    ]
    
    # Risk factors from demographics and clinical data
    RISK_FACTORS = {
        'age_high_risk': 75,  # Age threshold for high risk
        'age_medium_risk': 65,
        'abnormal_labs_threshold': 3,  # Number of abnormal labs
        'diagnosis_count_threshold': 5,  # Multiple diagnoses
        'length_of_stay_high': 14,  # Days
        'length_of_stay_medium': 7
    }
    
    def __init__(self, use_gemini: bool = True, model_name: str = "gemini-2.5-flash"):
        """
        Initialize risk predictor
        
        Args:
            use_gemini: Whether to use Gemini for advanced risk prediction
            model_name: Gemini model name
        """
        self.use_gemini = use_gemini and GEMINI_AVAILABLE
        self.model = None
        
        if self.use_gemini:
            try:
                api_key = os.getenv('GOOGLE_API_KEY')
                if api_key:
                    genai.configure(api_key=api_key)
                    self.model = genai.GenerativeModel(model_name)
                    logger.info(f"✓ Initialized Gemini model for risk prediction: {model_name}")
                else:
                    logger.warning("GOOGLE_API_KEY not found, using rule-based prediction")
                    self.use_gemini = False
            except Exception as e:
                logger.warning(f"Failed to initialize Gemini: {e}, using rule-based prediction")
                self.use_gemini = False
    
    def extract_risk_factors(self, record: Dict) -> Dict:
        """
        Extract risk factors from a discharge record
        
        Args:
            record: Dictionary containing discharge summary data
            
        Returns:
            Dictionary of extracted risk factors
        """
        def safe_int(value, default=0):
            """Safely convert to int"""
            try:
                if value is None or value == '':
                    return default
                return int(float(str(value)))
            except (ValueError, TypeError):
                return default
        
        def safe_float(value, default=0.0):
            """Safely convert to float"""
            try:
                if value is None or value == '':
                    return default
                return float(str(value))
            except (ValueError, TypeError):
                return default
        
        risk_factors = {
            'age': self._parse_age(record.get('age_at_admission', '')),
            'abnormal_labs': safe_int(record.get('abnormal_count', 0)),
            'diagnosis_count': safe_int(record.get('diagnosis_count', 0)),
            'text_length': len(str(record.get('cleaned_text', '') or '')),
            'has_medications': bool(record.get('has_medications', False)),
            'has_follow_up': bool(record.get('has_follow_up', False)),
            'complexity_score': safe_float(record.get('complexity_score', 0)),
            'urgency_indicator': safe_int(record.get('urgency_indicator', 0))
        }
        
        # Extract text for analysis
        text = str(record.get('cleaned_text', '') or record.get('cleaned_text_final', '')).lower()
        diagnosis = str(record.get('discharge_diagnosis', '') or '').lower()
        
        # Count high-risk keywords
        high_risk_count = sum(1 for condition in self.HIGH_RISK_CONDITIONS 
                              if condition in text or condition in diagnosis)
        medium_risk_count = sum(1 for condition in self.MEDIUM_RISK_CONDITIONS 
                               if condition in text or condition in diagnosis)
        
        risk_factors['high_risk_keywords'] = high_risk_count
        risk_factors['medium_risk_keywords'] = medium_risk_count
        risk_factors['total_risk_keywords'] = high_risk_count + medium_risk_count
        
        return risk_factors
    
    def _parse_age(self, age_str: str) -> int:
        """Parse age from string"""
        try:
            if isinstance(age_str, (int, float)):
                return int(age_str)
            age_str = str(age_str).strip()
            # Extract first number
            match = re.search(r'\d+', age_str)
            if match:
                return int(match.group())
            return 0
        except:
            return 0
    
    def predict_risk_rule_based(self, risk_factors: Dict) -> Tuple[str, float, Dict]:
        """
        Predict risk using rule-based approach
        
        Args:
            risk_factors: Dictionary of extracted risk factors
            
        Returns:
            Tuple of (risk_level, risk_score, details)
        """
        score = 0.0
        details = {}
        
        # Age-based risk
        age = risk_factors.get('age', 0)
        if age >= self.RISK_FACTORS['age_high_risk']:
            score += 0.3
            details['age_risk'] = 'high'
        elif age >= self.RISK_FACTORS['age_medium_risk']:
            score += 0.15
            details['age_risk'] = 'medium'
        else:
            details['age_risk'] = 'low'
        
        # High-risk conditions
        high_risk_keywords = risk_factors.get('high_risk_keywords', 0)
        if high_risk_keywords >= 3:
            score += 0.4
            details['condition_risk'] = 'very_high'
        elif high_risk_keywords >= 2:
            score += 0.3
            details['condition_risk'] = 'high'
        elif high_risk_keywords >= 1:
            score += 0.2
            details['condition_risk'] = 'medium'
        else:
            details['condition_risk'] = 'low'
        
        # Abnormal labs
        abnormal_labs = risk_factors.get('abnormal_labs', 0)
        if abnormal_labs >= self.RISK_FACTORS['abnormal_labs_threshold']:
            score += 0.2
            details['lab_risk'] = 'high'
        elif abnormal_labs >= 1:
            score += 0.1
            details['lab_risk'] = 'medium'
        else:
            details['lab_risk'] = 'low'
        
        # Multiple diagnoses
        diagnosis_count = risk_factors.get('diagnosis_count', 0)
        if diagnosis_count >= self.RISK_FACTORS['diagnosis_count_threshold']:
            score += 0.15
            details['diagnosis_risk'] = 'high'
        elif diagnosis_count >= 3:
            score += 0.1
            details['diagnosis_risk'] = 'medium'
        else:
            details['diagnosis_risk'] = 'low'
        
        # Complexity and urgency
        complexity = risk_factors.get('complexity_score', 0)
        if complexity > 0.7:
            score += 0.1
        elif complexity > 0.5:
            score += 0.05
        
        urgency = risk_factors.get('urgency_indicator', 0)
        if urgency >= 2:
            score += 0.1
        elif urgency >= 1:
            score += 0.05
        
        # Normalize score to 0-1
        score = min(1.0, score)
        
        # Determine risk level
        if score >= 0.7:
            risk_level = 'HIGH'
        elif score >= 0.4:
            risk_level = 'MEDIUM'
        else:
            risk_level = 'LOW'
        
        details['total_score'] = score
        details['risk_level'] = risk_level
        
        return risk_level, score, details
    
    def predict_risk_gemini(self, record: Dict, risk_factors: Dict) -> Tuple[str, float, Dict]:
        """
        Predict risk using Gemini for advanced analysis
        
        Args:
            record: Discharge record
            risk_factors: Extracted risk factors
            
        Returns:
            Tuple of (risk_level, risk_score, details)
        """
        if not self.model:
            return self.predict_risk_rule_based(risk_factors)
        
        text = str(record.get('cleaned_text', '') or record.get('cleaned_text_final', ''))[:3000]
        diagnosis = str(record.get('discharge_diagnosis', '') or '')
        age = risk_factors.get('age', 0)
        abnormal_labs = risk_factors.get('abnormal_labs', 0)
        
        prompt = f"""You are a medical risk assessment expert. Analyze the following discharge summary and predict the patient's risk level for readmission or complications.

Patient Information:
- Age: {age} years
- Discharge Diagnosis: {diagnosis}
- Abnormal Lab Values: {abnormal_labs}

Discharge Summary (excerpt):
{text[:2000]}

Based on this information, assess the patient's risk level and provide:
1. Risk Level: LOW, MEDIUM, or HIGH
2. Risk Score: 0.0 to 1.0 (where 1.0 is highest risk)
3. Key Risk Factors: List 2-3 main factors contributing to the risk
4. Recommendations: Brief recommendation for follow-up care

Format your response as:
RISK_LEVEL: [LOW/MEDIUM/HIGH]
RISK_SCORE: [0.0-1.0]
KEY_FACTORS: [factor1, factor2, factor3]
RECOMMENDATION: [brief recommendation]
"""
        
        try:
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.2,  # Lower temperature for more consistent risk assessment
                    max_output_tokens=500,
                )
            )
            
            result_text = response.text.strip()
            
            # Parse response
            risk_level = 'MEDIUM'
            risk_score = 0.5
            key_factors = []
            recommendation = ""
            
            lines = result_text.split('\n')
            for line in lines:
                if 'RISK_LEVEL:' in line.upper():
                    level = line.split(':')[1].strip().upper()
                    if level in ['LOW', 'MEDIUM', 'HIGH']:
                        risk_level = level
                elif 'RISK_SCORE:' in line.upper():
                    try:
                        score_str = line.split(':')[1].strip()
                        risk_score = float(re.search(r'[\d.]+', score_str).group())
                        risk_score = max(0.0, min(1.0, risk_score))
                    except:
                        pass
                elif 'KEY_FACTORS:' in line.upper():
                    factors = line.split(':')[1].strip()
                    key_factors = [f.strip() for f in factors.split(',')]
                elif 'RECOMMENDATION:' in line.upper():
                    recommendation = line.split(':')[1].strip()
            
            details = {
                'method': 'gemini',
                'risk_level': risk_level,
                'risk_score': risk_score,
                'key_factors': key_factors,
                'recommendation': recommendation,
                'raw_response': result_text
            }
            
            return risk_level, risk_score, details
            
        except Exception as e:
            logger.warning(f"Gemini prediction failed: {e}, falling back to rule-based")
            return self.predict_risk_rule_based(risk_factors)
    
    def predict(self, record: Dict) -> Dict:
        """
        Predict risk for a single record
        
        Args:
            record: Discharge record dictionary
            
        Returns:
            Dictionary with risk prediction results
        """
        # Extract risk factors
        risk_factors = self.extract_risk_factors(record)
        
        # Predict risk
        if self.use_gemini:
            risk_level, risk_score, details = self.predict_risk_gemini(record, risk_factors)
        else:
            risk_level, risk_score, details = self.predict_risk_rule_based(risk_factors)
        
        # Combine results
        result = {
            'hadm_id': record.get('hadm_id', ''),
            'subject_id': record.get('subject_id', ''),
            'risk_level': risk_level,
            'risk_score': round(risk_score, 3),
            'risk_factors': risk_factors,
            'prediction_details': details,
            'timestamp': datetime.now().isoformat()
        }
        
        return result
    
    def predict_batch(self, records: List[Dict]) -> List[Dict]:
        """
        Predict risk for multiple records
        
        Args:
            records: List of discharge record dictionaries
            
        Returns:
            List of risk prediction results
        """
        results = []
        for i, record in enumerate(records, 1):
            try:
                result = self.predict(record)
                results.append(result)
                if i % 10 == 0:
                    logger.info(f"Processed {i}/{len(records)} records")
            except Exception as e:
                logger.error(f"Error predicting risk for record {i}: {e}")
                continue
        
        return results


def main():
    """Example usage"""
    import csv
    import argparse
    
    parser = argparse.ArgumentParser(description='Predict medical risk from discharge summaries')
    parser.add_argument('--input', type=str,
                       default='data-pipeline/data/processed/processed_discharge_summaries.csv',
                       help='Input CSV file')
    parser.add_argument('--output', type=str,
                       default='models/gemini/risk_predictions.csv',
                       help='Output CSV file')
    parser.add_argument('--limit', type=int, default=None,
                       help='Limit number of records')
    parser.add_argument('--use-gemini', action='store_true',
                       help='Use Gemini for advanced risk prediction')
    
    args = parser.parse_args()
    
    # Initialize predictor
    predictor = MedicalRiskPredictor(use_gemini=args.use_gemini)
    
    # Load data
    print(f"Loading data from {args.input}...")
    records = []
    with open(args.input, 'r') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if args.limit and i >= args.limit:
                break
            records.append(row)
    
    print(f"Predicting risk for {len(records)} records...")
    
    # Predict risk
    results = predictor.predict_batch(records)
    
    # Save results
    if results:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', newline='') as f:
            fieldnames = ['hadm_id', 'subject_id', 'risk_level', 'risk_score'] + \
                        [f'risk_factor_{k}' for k in results[0]['risk_factors'].keys()]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for result in results:
                row = {
                    'hadm_id': result['hadm_id'],
                    'subject_id': result['subject_id'],
                    'risk_level': result['risk_level'],
                    'risk_score': result['risk_score']
                }
                for k, v in result['risk_factors'].items():
                    row[f'risk_factor_{k}'] = v
                writer.writerow(row)
        
        print(f"✅ Saved {len(results)} risk predictions to {args.output}")
        
        # Print summary
        risk_counts = {}
        for r in results:
            level = r['risk_level']
            risk_counts[level] = risk_counts.get(level, 0) + 1
        
        print("\n📊 Risk Distribution:")
        for level in ['LOW', 'MEDIUM', 'HIGH']:
            count = risk_counts.get(level, 0)
            pct = (count / len(results)) * 100 if results else 0
            print(f"  {level}: {count} ({pct:.1f}%)")


if __name__ == '__main__':
    main()

