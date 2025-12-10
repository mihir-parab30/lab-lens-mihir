"""
MIMIC-III Feature Engineering Pipeline
=======================================

**What this script does:**
This script transforms clean clinical text into numeric, ML-ready features.
It's the bridge between human-readable medical notes and machine learning models.

**Input:**  data/processed/processed_discharge_summaries.csv (~9,600 records, 40+ columns)
**Output:** data/processed/mimic_features.csv (~9,600 records, 60+ numeric features)

**Core Strategy: Lexicon-Based Feature Engineering**

We use predefined medical vocabularies (lexicons) to quantify clinical content:

1. CHRONIC_DISEASE_TERMS: diabetes, hypertension, copd, etc.
2. SYMPTOM_TERMS: fever, cough, chest pain, etc.
3. MEDICATION_TERMS: insulin, metformin, lisinopril, etc.
4. HIGH_RISK_TERMS: sepsis, shock, critical, icu, etc.
5. POSITIVE_OUTCOME_TERMS: improved, stable, recovered, etc.

**The 6 Feature Categories:**

1. KEYWORD COUNT FEATURES
   - Simple counts: How many times does "diabetes" appear?
   - Examples: kw_chronic_disease, kw_symptoms, kw_medications
   - Purpose: Quantify disease burden and treatment complexity

2. READABILITY & QUALITY FEATURES
   - Flesch Reading Ease: Text difficulty score (0-100)
   - Vocabulary Richness: Unique words / total words
   - Documentation Completeness: % of sections present
   - Purpose: Measure note quality and case complexity

3. CLINICAL RATIO & DENSITY FEATURES
   - risk_outcome_ratio: high_risk_score / positive_outcome_score
   - acute_chronic_ratio: acute_terms / chronic_terms
   - disease_density: chronic_disease_count / sentence_count
   - medication_density: medication_count / sentence_count
   - Purpose: Context-aware severity indicators

4. TREATMENT COMPLEXITY FEATURES
   - polypharmacy_flag: 1 if patient on 5+ medications
   - treatment_intensity: Weighted score (meds + labs + comorbidities)
   - Purpose: Quantify treatment burden

5. LAB VALUE FEATURES
   - total_labs: Count of lab tests
   - abnormal_lab_count: Count of abnormal results
   - abnormal_lab_ratio: abnormal / total (0.0 to 1.0)
   - Purpose: Objective clinical severity measures

6. ONE-HOT ENCODED DEMOGRAPHICS
   - ethnicity_clean → eth_WHITE, eth_BLACK, eth_HISPANIC, eth_ASIAN, eth_OTHER
   - gender → gender_M, gender_F
   - insurance → ins_MEDICARE, ins_MEDICAID, ins_PRIVATE, etc.
   - Purpose: Make categorical data ML-compatible

**Key Technologies:**
- pandas: Data manipulation
- numpy: Numerical operations
- regex: Pattern matching for medical terms
- Custom logging utilities

**Pipeline Stage:** 2.5 of 4 (Acquisition → Preprocessing → **Feature Engineering** → Validation → Bias Detection)

**Example Transformation:**

Before:
  cleaned_text: "Patient with diabetes, hypertension admitted for chest pain..."
  
After:
  kw_chronic_disease: 2
  kw_symptoms: 1
  disease_density: 0.4
  high_risk_score: 0
  polypharmacy_flag: 1
  eth_WHITE: 1
  eth_BLACK: 0
  ...
"""

# Feature Engineering Pipeline for MIMIC-III Discharge Summaries
# Author: Lab Lens Team
# Description: Creates advanced features from preprocessed clinical text data

import argparse
import logging
import re
import json
import os
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================
# Helper functions for file paths, logging, and safe operations
# ============================================================================

def find_repo_root(start: Path = Path.cwd()) -> Path:
    """
    Find the repository root by looking for data-pipeline directory
    
    This walks up the directory tree until it finds a directory containing
    either 'data-pipeline/' or 'configs/' subdirectory.
    
    Args:
        start: Starting directory for search
        
    Returns:
        Path to repository root
    """
    cur = start
    while cur != cur.parent:
        if (cur / "data-pipeline").exists() or (cur / "configs").exists():
            return cur
        cur = cur.parent
    return start


def setup_logger(log_path: Path = None, level: int = logging.INFO) -> logging.Logger:
    """
    Configure logger with file and console handlers
    
    Creates a logger that writes to both:
    1. Console (stdout) for real-time monitoring
    2. Log file for permanent record
    
    Args:
        log_path: Path to log file (if None, only console logging)
        level: Logging level (default: INFO)
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger("feature_engineering")
    logger.setLevel(level)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # Console handler - prints to terminal
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler - saves to log file
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setLevel(level)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


def safe_len(x) -> int:
    """
    Safely get length of object, return 0 if not possible
    
    Handles NaN, None, and non-iterable objects gracefully
    
    Args:
        x: Any object
        
    Returns:
        Length of object or 0 if operation fails
    """
    try:
        return len(x)
    except Exception:
        return 0


# ============================================================================
# MEDICAL TERM LEXICONS FOR FEATURE EXTRACTION
# ============================================================================
# These vocabularies define what medical concepts we're looking for in text.
# They're hand-curated based on common medical terminology in MIMIC-III.
# ============================================================================

# ========================================================================
# Lexicon 1: CHRONIC DISEASE TERMS
# ========================================================================
# Purpose: Detect chronic conditions and comorbidities
# Used to calculate: kw_chronic_disease, disease_density, comorbidity_score
# ========================================================================
CHRONIC_DISEASE_TERMS = [
    "diabetes", "hypertension", "ckd", "chf", "cad", "copd", "asthma",
    "cirrhosis", "hepatitis", "hiv", "stroke", "afib", "atrial fibrillation",
    "hyperlipidemia", "hld", "hypothyroidism", "hyperthyroidism",
    "dementia", "alzheimer", "parkinson"
]

# ========================================================================
# Lexicon 2: SYMPTOM TERMS
# ========================================================================
# Purpose: Capture clinical presentation and patient complaints
# Used to calculate: kw_symptoms, symptom_density
# ========================================================================
SYMPTOM_TERMS = [
    "fever", "chills", "nausea", "vomiting", "diarrhea", "dyspnea",
    "sob", "cough", "chest pain", "fatigue", "dizziness", "syncope",
    "edema", "pain", "headache", "rash"
]

# ========================================================================
# Lexicon 3: MEDICATION TERMS
# ========================================================================
# Purpose: Identify specific medications for treatment complexity analysis
# Used to calculate: kw_medications, medication_density, polypharmacy_flag
# ========================================================================
MEDICATION_TERMS = [
    "insulin", "metformin", "lisinopril", "losartan", "amlodipine",
    "metoprolol", "atorvastatin", "simvastatin", "warfarin", "heparin",
    "aspirin", "pantoprazole", "omeprazole", "gabapentin", "oxycodone",
    "duloxetine", "citalopram", "midodrine", "furosemide", "spironolactone",
    "lactulose", "thiamine", "folic acid", "acetaminophen", "naproxen"
]

# ========================================================================
# Lexicon 4: MEDICATION SUFFIXES
# ========================================================================
# Purpose: Catch medications not in explicit list using common drug endings
# Examples: "enalapril" ends in "pril", "propranolol" ends in "olol"
# Used to calculate: kw_med_suffix_hits
# ========================================================================
MED_SUFFIXES = [
    "pril", "sartan", "olol", "dipine", "statin", "azole", "tidine", "caine",
    "cycline", "cillin", "mycin", "sone"
]

# ========================================================================
# Lexicon 5: HIGH-RISK CLINICAL TERMS
# ========================================================================
# Purpose: Detect severe/critical clinical situations
# Used to calculate: high_risk_score, risk_outcome_ratio
# ========================================================================
HIGH_RISK_TERMS = [
    "sepsis", "shock", "arrest", "failure", "critical", "icu",
    "intubat", "ventilat", "resuscitat", "unstable", "deteriorat",
    "code blue", "emergency", "urgent"
]

# ========================================================================
# Lexicon 6: POSITIVE OUTCOME TERMS
# ========================================================================
# Purpose: Detect recovery and improvement indicators
# Used to calculate: positive_outcome_score, risk_outcome_ratio
# ========================================================================
POSITIVE_OUTCOME_TERMS = [
    "improv", "stable", "recover", "discharg", "ambulat", 
    "independent", "normal", "resolved", "healing", "better"
]

# ========================================================================
# Lexicon 7: ACUTE PRESENTATION INDICATORS
# ========================================================================
# Purpose: Distinguish acute events from chronic conditions
# Used to calculate: acute_presentation_score, acute_chronic_ratio
# ========================================================================
ACUTE_TERMS = [
    "acute", "sudden", "rapid", "urgent", "emergency", "immediate", "new onset"
]

# ========================================================================
# Lexicon 8: CHRONIC CONDITION INDICATORS
# ========================================================================
# Purpose: Identify long-standing conditions
# Used to calculate: chronic_condition_score, acute_chronic_ratio
# ========================================================================
CHRONIC_TERMS = [
    "chronic", "longstanding", "history of", "ongoing", "persistent", "long term"
]

# ========================================================================
# Section Presence Patterns
# ========================================================================
# Purpose: Detect if specific clinical sections exist in the note
# Used to calculate: has_allergies_section, has_medications_section, etc.
# ========================================================================
SECTION_PATTERNS = {
    "has_allergies_section": [r"\ballerg(y|ies)\b", r"\ballergies:\b"],
    "has_medications_section": [r"\bmedications?\b", r"\bdischarge medications?\b"],
    "has_brief_hospital_course": [r"\bbrief hospital course\b"],
}

# ========================================================================
# Negation Tokens
# ========================================================================
# Purpose: Detect negative statements ("no fever", "denies pain")
# Used to calculate: negation_density
# ========================================================================
NEGATION_TOKENS = [" no ", " denies ", " without ", " not ", " none "]


# ============================================================================
# CORE FEATURE EXTRACTION FUNCTIONS
# ============================================================================
# These functions do the actual counting and pattern matching
# ============================================================================

def sentence_count(text: str) -> int:
    """
    Count sentences in text using punctuation delimiters
    
    Splits on: . ! ?
    
    Args:
        text: Input text string
        
    Returns:
        Number of sentences
    """
    if not isinstance(text, str) or not text:
        return 0
    parts = re.split(r"[\.!?]+", text)
    return sum(1 for p in parts if p.strip())


def count_terms(text: str, terms: List[str]) -> int:
    """
    Count occurrences of specific terms in text
    
    Handles both:
    - Single words: Uses word boundaries (\b) to avoid partial matches
      Example: "pain" matches "pain" but not "painful"
    - Multi-word phrases: Uses exact phrase matching
      Example: "chest pain" matches "chest pain" exactly
    
    Args:
        text: Input text string
        terms: List of terms to search for
        
    Returns:
        Total count of all terms found
    """
    if not isinstance(text, str) or not text:
        return 0
    tl = text.lower()
    total = 0
    for t in terms:
        if " " in t:
            # Multi-word phrases - exact match
            total += len(re.findall(re.escape(t), tl))
        else:
            # Single words - word boundary match
            total += len(re.findall(rf"\b{re.escape(t)}\b", tl))
    return total


def count_med_suffixes(text: str, suffixes: List[str]) -> int:
    """
    Count medication-like words based on common drug suffixes
    
    Strategy:
    1. Extract all words that look like medications (alphanumeric, 3+ chars)
    2. Check if each word ends with a common drug suffix
    3. Count matches
    
    Example:
    - "patient on lisinopril" → "lisinopril" ends in "pril" → count: 1
    - "atorvastatin 20mg" → "atorvastatin" ends in "statin" → count: 1
    
    Args:
        text: Input text string
        suffixes: List of medication suffixes to match
        
    Returns:
        Count of words matching medication patterns
    """
    if not isinstance(text, str) or not text:
        return 0
    # Extract potential medication words (alphanumeric, 3+ chars)
    tokens = re.findall(r"[a-zA-Z][a-zA-Z\-]{2,}", text.lower())
    return sum(any(tok.endswith(sfx) for sfx in suffixes) for tok in tokens)


def parse_icd_list(top_diagnoses: str) -> List[str]:
    """
    Parse comma-separated ICD codes into unique list
    
    Example Input: "401.9, 250.00, 401.9, 428.0"
    Example Output: ['250.00', '401.9', '428.0'] (unique and sorted)
    
    Args:
        top_diagnoses: Comma-separated string of ICD codes
        
    Returns:
        Sorted list of unique ICD codes
    """
    if not isinstance(top_diagnoses, str) or not top_diagnoses.strip():
        return []
    parts = [p.strip() for p in top_diagnoses.split(",") if p.strip()]
    return sorted(set(parts))


def one_hot_topk(df: pd.DataFrame, col: str, k: int, prefix: str) -> pd.DataFrame:
    """
    Create one-hot encoding for top-k categories, group rest as OTHER
    
    Strategy:
    1. Find the k most common values in the column
    2. Replace all other values with "OTHER"
    3. Create binary columns for each category
    
    Example:
    ethnicity_clean: ['WHITE', 'BLACK', 'WHITE', 'ASIAN', 'NATIVE AMERICAN']
    With k=3 → Top 3: WHITE, BLACK, ASIAN
    
    Output columns:
    - eth_WHITE:  [1, 0, 1, 0, 0]
    - eth_BLACK:  [0, 1, 0, 0, 0]
    - eth_ASIAN:  [0, 0, 0, 1, 0]
    - eth_OTHER:  [0, 0, 0, 0, 1]
    
    Args:
        df: Input DataFrame
        col: Column name to encode
        k: Number of top categories to keep
        prefix: Prefix for new column names
        
    Returns:
        DataFrame with one-hot encoded columns (original column removed)
    """
    if col not in df.columns:
        return df
    
    series = df[col].astype("string").fillna("UNKNOWN")
    topk = series.value_counts().index[:k]
    trimmed = series.where(series.isin(topk), "OTHER")
    dummies = pd.get_dummies(trimmed, prefix=prefix, dtype="int8")
    
    return pd.concat([df.drop(columns=[col]), dummies], axis=1)


# ============================================================================
# READABILITY AND TEXT QUALITY FEATURES
# ============================================================================
# These measure how complex and well-structured the clinical note is
# ============================================================================

def calculate_readability_scores(text: str) -> Dict[str, float]:
    """
    Calculate text readability metrics including Flesch score
    
    Flesch Reading Ease Formula:
    206.835 - 1.015 × (words/sentences) - 84.6 × (syllables/words)
    
    Score Interpretation:
    - 90-100: Very easy (5th grade)
    - 60-70: Standard (8th-9th grade)
    - 0-30: Very difficult (college graduate)
    
    Medical notes typically score 30-50 (difficult)
    
    Args:
        text: Input text string
        
    Returns:
        Dictionary with:
        - flesch_reading_ease: 0-100 scale (higher = easier)
        - avg_syllables_per_word: Average syllables per word
        - vocabulary_richness: Unique words / total words (0-1)
    """
    if not isinstance(text, str) or not text:
        return {
            'flesch_reading_ease': 0, 
            'avg_syllables_per_word': 0, 
            'vocabulary_richness': 0
        }
    
    # Count sentences, words, and syllables
    sentences = len([s for s in re.split(r'[.!?]+', text) if s.strip()])
    words = text.split()
    word_count = len(words)
    
    if word_count == 0 or sentences == 0:
        return {
            'flesch_reading_ease': 0,
            'avg_syllables_per_word': 0,
            'vocabulary_richness': 0
        }
    
    # Simple syllable counting by vowel clusters
    # Example: "patient" → "a", "ie" → 2 syllable groups → 2 syllables
    syllables = sum(
        max(1, len(re.findall(r'[aeiouy]+', word.lower()))) 
        for word in words
    )
    
    # Flesch Reading Ease score: 0-100, higher is easier to read
    flesch = 206.835 - 1.015 * (word_count / sentences) - 84.6 * (syllables / word_count)
    flesch = max(0, min(100, flesch))  # Clamp to 0-100 range
    
    # Vocabulary richness: unique words / total words
    # High richness = varied vocabulary (better documentation)
    # Low richness = repetitive (possibly template-based notes)
    unique_words = len(set(word.lower() for word in words))
    vocabulary_richness = unique_words / word_count
    
    return {
        'flesch_reading_ease': round(flesch, 2),
        'avg_syllables_per_word': round(syllables / word_count, 2),
        'vocabulary_richness': round(vocabulary_richness, 3)
    }


# ============================================================================
# DOCUMENTATION QUALITY FEATURES
# ============================================================================
# Measures how complete and well-structured the discharge summary is
# ============================================================================

def calculate_missingness_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate features based on missing clinical sections
    
    Documentation completeness is a proxy for:
    - Quality of medical documentation
    - Case complexity (complex cases get more detailed notes)
    - Provider thoroughness
    
    Creates two features:
    1. missing_section_count: Raw count of missing sections (0-5)
    2. documentation_completeness: Proportion of sections present (0.0-1.0)
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with missingness features added
    """
    # Key sections we expect from preprocessing
    section_cols = [
        'discharge_diagnosis', 'discharge_medications', 
        'follow_up', 'chief_complaint', 'hospital_course'
    ]
    
    existing_sections = [col for col in section_cols if col in df.columns]
    
    if existing_sections:
        # Count missing sections per record
        # Missing = either NaN or empty string
        missing_count = sum(
            df[col].isna() | (df[col] == '') 
            for col in existing_sections
        )
        
        df['missing_section_count'] = missing_count.astype('Int64')
        
        # Calculate completeness score (0-1)
        # 1.0 = all sections present, 0.0 = no sections present
        df['documentation_completeness'] = (
            (len(existing_sections) - missing_count) / len(existing_sections)
        ).clip(0, 1)
    else:
        # Fallback if no sections found
        df['missing_section_count'] = 0
        df['documentation_completeness'] = 1.0
    
    return df


# ============================================================================
# CLINICAL RISK AND SEVERITY FEATURES
# ============================================================================
# These quantify how sick the patient is and their prognosis
# ============================================================================

def calculate_clinical_risk_features(df: pd.DataFrame, text_col: str) -> pd.DataFrame:
    """
    Calculate clinical risk and outcome indicators from text
    
    Creates "smart" ratio features that provide context:
    
    1. risk_outcome_ratio: high_risk / positive_outcome
       - High ratio (e.g., 5.0) = many risk terms, few positive terms = concerning
       - Low ratio (e.g., 0.2) = few risk terms, many positive terms = good prognosis
    
    2. acute_chronic_ratio: acute_terms / chronic_terms
       - High ratio = acute event (e.g., heart attack)
       - Low ratio = chronic care (e.g., diabetes management)
    
    Args:
        df: Input DataFrame
        text_col: Name of text column to analyze
        
    Returns:
        DataFrame with clinical risk features added
    """
    # ---- High-Risk Term Count ----
    # Counts: sepsis, shock, critical, icu, etc.
    df["high_risk_score"] = df[text_col].map(
        lambda s: count_terms(s, HIGH_RISK_TERMS)
    ).astype("Int64")
    
    # ---- Positive Outcome Term Count ----
    # Counts: improved, stable, recovered, etc.
    df["positive_outcome_score"] = df[text_col].map(
        lambda s: count_terms(s, POSITIVE_OUTCOME_TERMS)
    ).astype("Int64")
    
    # ---- Risk to Outcome Ratio ----
    # Add 1 to denominator to avoid division by zero
    # Clip at 10.0 to prevent extreme outliers
    df["risk_outcome_ratio"] = (
        df["high_risk_score"] / (df["positive_outcome_score"] + 1)
    ).clip(upper=10.0)
    
    # ---- Acute Presentation Score ----
    # Counts: acute, sudden, emergency, etc.
    df["acute_presentation_score"] = df[text_col].map(
        lambda s: count_terms(s, ACUTE_TERMS)
    ).astype("Int64")
    
    # ---- Chronic Condition Score ----
    # Counts: chronic, longstanding, history of, etc.
    df["chronic_condition_score"] = df[text_col].map(
        lambda s: count_terms(s, CHRONIC_TERMS)
    ).astype("Int64")
    
    # ---- Acute to Chronic Ratio ----
    # Add 1 to denominator to avoid division by zero
    # Clip at 5.0 to prevent extreme outliers
    df["acute_chronic_ratio"] = (
        df["acute_presentation_score"] / (df["chronic_condition_score"] + 1)
    ).clip(upper=5.0)
    
    return df


def calculate_medical_density(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate density of medical terms per sentence
    
    Why density matters:
    - Raw counts favor longer notes
    - Density normalizes by note length
    - High density = information-rich note
    
    Example:
    - Note A: 10 disease terms, 50 sentences → density = 0.2
    - Note B: 10 disease terms, 10 sentences → density = 1.0
    - Note B is more clinically dense
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with medical density features added
    """
    # Use sentences as denominator, replace 0 with NaN to avoid division errors
    denom = df["sentences"].replace({0: np.nan})
    
    # ---- Disease Density ----
    # Chronic disease mentions per sentence
    df["disease_density"] = (
        df["kw_chronic_disease"] / denom
    ).fillna(0).clip(upper=5.0)
    
    # ---- Medication Density ----
    # Medication mentions per sentence
    df["medication_density"] = (
        df["kw_medications"] / denom
    ).fillna(0).clip(upper=5.0)
    
    # ---- Symptom Density ----
    # Symptom mentions per sentence
    df["symptom_density"] = (
        df["kw_symptoms"] / denom
    ).fillna(0).clip(upper=5.0)
    
    return df


def calculate_treatment_complexity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate treatment complexity and intensity measures
    
    Treatment complexity indicates:
    - Medication burden
    - Need for close monitoring (abnormal labs)
    - Multiple comorbidities requiring management
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with treatment complexity features added
    """
    # ---- Polypharmacy Flag ----
    # Binary indicator: 1 if patient on 5+ medications, 0 otherwise
    # Polypharmacy is clinically significant (drug interactions, adherence issues)
    df["polypharmacy_flag"] = (df["kw_medications"] >= 5).astype("int8")
    
    # ---- Overall Treatment Intensity Score ----
    # Weighted combination of:
    # - Medications (30% weight): More meds = more complex
    # - Abnormal labs (40% weight): Abnormal labs require intervention
    # - Comorbidities (30% weight): More diseases = more complex
    med_score = df["kw_medications"] * 0.3
    lab_score = df.get("abnormal_lab_count", 0) * 0.4
    comorbid_score = df.get("comorbidity_score", 0) * 0.3
    
    df["treatment_intensity"] = (
        med_score + lab_score + comorbid_score
    ).fillna(0).clip(upper=20.0)
    
    return df


# ============================================================================
# OPTIONAL ADVANCED FEATURES
# ============================================================================
# Additional features for more sophisticated analysis
# ============================================================================

def add_section_flags(df: pd.DataFrame, text_col: str = "cleaned_text") -> pd.DataFrame:
    """
    Add binary flags for presence of specific clinical sections
    
    Also calculates negation density:
    - Counts phrases like "no fever", "denies pain"
    - Important for accurate symptom/condition detection
    - High negation density may indicate rule-out diagnoses
    
    Args:
        df: Input DataFrame
        text_col: Name of text column to analyze
        
    Returns:
        DataFrame with section flag features added
    """
    if text_col not in df.columns:
        return df
    
    tl = df[text_col].astype("string").str.lower().fillna("")
    
    # ---- Check for Each Section Pattern ----
    for flag, patterns in SECTION_PATTERNS.items():
        df[flag] = (
            tl.map(lambda s: int(any(re.search(p, s) is not None for p in patterns)))
            .astype("int8")
        )
    
    # ---- Use Existing Word Count if Available ----
    text_tokens_col = 'word_count' if 'word_count' in df.columns else 'text_tokens'
    if text_tokens_col not in df.columns:
        df[text_tokens_col] = tl.str.split().map(safe_len).astype("Int64")
    
    # ---- Count Negation Tokens ----
    def neg_count(s: str) -> int:
        """Count negation phrases in text"""
        if not isinstance(s, str) or not s:
            return 0
        s_pad = f" {s} "  # Add padding to catch negations at start/end
        return sum(s_pad.count(tok) for tok in NEGATION_TOKENS)
    
    neg = tl.map(neg_count)
    denom = df[text_tokens_col].replace({0: np.nan})
    
    # ---- Negation Density ----
    # Proportion of words that are negation tokens
    # High density suggests many "rule-out" statements
    df["negation_density"] = (neg / denom).fillna(0.0)
    
    return df


# ============================================================================
# COLUMN NORMALIZATION
# ============================================================================
# Maps preprocessing output names to feature engineering expected names
# ============================================================================

def normalize_column_names(df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    """
    Map preprocessing output column names to expected feature engineering names
    
    Why this is needed:
    - Preprocessing might output 'text_length'
    - Feature engineering expects 'text_chars'
    - This function creates a bridge between the two
    
    Args:
        df: Input DataFrame
        logger: Logger instance
        
    Returns:
        DataFrame with normalized column names
    """
    column_mapping = {
        'text_length': 'text_chars',       # Character count
        'word_count': 'text_tokens',       # Word count
        'abnormal_count': 'abnormal_lab_count',  # Lab abnormality count
    }
    
    rename_dict = {old: new for old, new in column_mapping.items() if old in df.columns}
    if rename_dict:
        logger.info(f"Mapping columns: {rename_dict}")
        df = df.rename(columns=rename_dict)
    
    return df


# ============================================================================
# MAIN FEATURE ENGINEERING PIPELINE
# ============================================================================
# This orchestrates all feature creation in the correct order
# ============================================================================

def engineer_features(df: pd.DataFrame, logger: logging.Logger, with_sections: bool = False) -> pd.DataFrame:
    """
    Main feature engineering pipeline
    
    Execution Order (Critical):
    1. Normalize column names from preprocessing
    2. Calculate base text metrics (chars, tokens, sentences)
    3. Calculate readability scores
    4. Count medical keywords (diseases, symptoms, medications)
    5. Calculate clinical risk features
    6. Calculate documentation quality
    7. Process lab values and ratios
    8. Process diagnosis information
    9. Calculate medical density metrics
    10. Calculate treatment complexity
    11. Normalize text metrics
    12. (Optional) Add section flags
    13. One-hot encode demographics
    14. Order columns logically
    
    Args:
        df: Input DataFrame from preprocessing
        logger: Logger instance
        with_sections: Whether to include section-level features
        
    Returns:
        DataFrame with ~60+ engineered features
    """
    # ====================================================================
    # STEP 0: VALIDATE INPUT
    # ====================================================================
    if "cleaned_text" not in df.columns:
        raise ValueError("Missing column 'cleaned_text' in input dataset.")

    # ====================================================================
    # STEP 1: NORMALIZE COLUMN NAMES
    # ====================================================================
    df = normalize_column_names(df, logger)
    
    # ====================================================================
    # STEP 2: ENSURE BASE TEXT METRICS EXIST
    # ====================================================================
    logger.info("Ensuring base text metrics...")
    
    # Character count
    if "text_chars" not in df.columns:
        df["text_chars"] = df["cleaned_text"].astype(str).str.len()
    
    # Word count
    if "text_tokens" not in df.columns:
        df["text_tokens"] = df["cleaned_text"].astype(str).str.split().map(safe_len)
    
    df["text_tokens"] = pd.to_numeric(df["text_tokens"], errors="coerce").astype("Int64")
    df["text_chars"] = pd.to_numeric(df["text_chars"], errors="coerce").astype("Int64")

    # ====================================================================
    # STEP 3: CALCULATE SENTENCE COUNT
    # ====================================================================
    logger.info("Computing sentence counts...")
    if "sentence_count" in df.columns:
        df["sentences"] = pd.to_numeric(df["sentence_count"], errors="coerce").astype("Int64")
    else:
        df["sentences"] = df["cleaned_text"].map(sentence_count).astype("Int64")

    # ====================================================================
    # STEP 4: CALCULATE READABILITY SCORES
    # ====================================================================
    logger.info("Calculating readability scores...")
    readability = df["cleaned_text"].map(calculate_readability_scores)
    df["flesch_reading_ease"] = readability.map(lambda x: x['flesch_reading_ease'])
    df["avg_syllables_per_word"] = readability.map(lambda x: x['avg_syllables_per_word'])
    df["vocabulary_richness"] = readability.map(lambda x: x['vocabulary_richness'])

    # ====================================================================
    # STEP 5: COUNT MEDICAL KEYWORD FAMILIES
    # ====================================================================
    logger.info("Counting medical terms...")
    
    # Chronic disease count (diabetes, hypertension, etc.)
    df["kw_chronic_disease"] = df["cleaned_text"].map(
        lambda s: count_terms(s, CHRONIC_DISEASE_TERMS)
    ).astype("Int64")
    
    # Symptom count (fever, cough, etc.)
    df["kw_symptoms"] = df["cleaned_text"].map(
        lambda s: count_terms(s, SYMPTOM_TERMS)
    ).astype("Int64")
    
    # Explicit medication count (insulin, metformin, etc.)
    df["kw_medications"] = df["cleaned_text"].map(
        lambda s: count_terms(s, MEDICATION_TERMS)
    ).astype("Int64")
    
    # Medication suffix matches (-pril, -olol, etc.)
    df["kw_med_suffix_hits"] = df["cleaned_text"].map(
        lambda s: count_med_suffixes(s, MED_SUFFIXES)
    ).astype("Int64")

    # ====================================================================
    # STEP 6: CALCULATE CLINICAL RISK FEATURES
    # ====================================================================
    logger.info("Calculating clinical risk features...")
    df = calculate_clinical_risk_features(df, "cleaned_text")

    # ====================================================================
    # STEP 7: CALCULATE DOCUMENTATION QUALITY
    # ====================================================================
    logger.info("Calculating documentation quality...")
    df = calculate_missingness_features(df)

    # ====================================================================
    # STEP 8: PROCESS LAB VALUES AND RATIOS
    # ====================================================================
    logger.info("Deriving lab ratios...")
    if "total_labs" in df.columns and "abnormal_lab_count" in df.columns:
        denom = pd.to_numeric(df["total_labs"], errors="coerce").replace({0: np.nan})
        num = pd.to_numeric(df["abnormal_lab_count"], errors="coerce")
        
        # Abnormal lab ratio: abnormal / total (0.0 to 1.0)
        df["abnormal_lab_ratio"] = (num / denom).clip(lower=0.0, upper=1.0).fillna(0.0)
        df["total_labs"] = denom.fillna(0).astype("Int64")
        df["abnormal_lab_count"] = num.fillna(0).astype("Int64")
    else:
        # Fallback if lab columns missing
        df["total_labs"] = pd.Series([0] * len(df), dtype="Int64")
        df["abnormal_lab_count"] = pd.Series([0] * len(df), dtype="Int64")
        df["abnormal_lab_ratio"] = 0.0

    # ====================================================================
    # STEP 9: PROCESS DIAGNOSIS INFORMATION
    # ====================================================================
    logger.info("Processing diagnosis information...")
    
    # Count unique ICD codes
    if "top_diagnoses" in df.columns:
        df["diagnosis_unique_count"] = df["top_diagnoses"].map(
            lambda s: len(parse_icd_list(s))
        ).astype("Int64")
    else:
        df["diagnosis_unique_count"] = pd.Series([pd.NA] * len(df), dtype="Int64")
    
    # Comorbidity score (total diagnosis count)
    if "diagnosis_count" in df.columns:
        df["comorbidity_score"] = pd.to_numeric(df["diagnosis_count"], errors="coerce").astype("Int64")
    else:
        df["comorbidity_score"] = df.get("diagnosis_unique_count", pd.Series([pd.NA] * len(df), dtype="Int64"))

    # ====================================================================
    # STEP 10: CALCULATE MEDICAL DENSITY METRICS
    # ====================================================================
    logger.info("Calculating medical density...")
    df = calculate_medical_density(df)

    # ====================================================================
    # STEP 11: CALCULATE TREATMENT COMPLEXITY
    # ====================================================================
    logger.info("Calculating treatment complexity...")
    df = calculate_treatment_complexity(df)

    # ====================================================================
    # STEP 12: CALCULATE NORMALIZED TEXT METRICS
    # ====================================================================
    logger.info("Normalizing text metrics...")
    
    # Characters per word (average word length)
    denom_tok = df["text_tokens"].replace({0: np.nan})
    df["chars_per_token"] = (df["text_chars"] / denom_tok).fillna(0.0)
    
    # Long note flag (1 if ≥512 words, 0 otherwise)
    # 512 is a common token limit for NLP models
    df["long_note_flag"] = (
        pd.to_numeric(df["text_tokens"], errors="coerce").fillna(0) >= 512
    ).astype("int16")

    # ====================================================================
    # STEP 13: ADD OPTIONAL SECTION FLAGS
    # ====================================================================
    if with_sections:
        logger.info("Adding section flags and negation density...")
        df = add_section_flags(df, text_col="cleaned_text")

    # ====================================================================
    # STEP 14: ONE-HOT ENCODE DEMOGRAPHIC CATEGORIES
    # ====================================================================
    logger.info("One-hot encoding demographics...")
    
    # Encode multiple demographic columns
    # Format: (column_name, top_k_categories, prefix)
    for col, k, prefix in [
        ("gender", 2, "gender"),              # gender_M, gender_F
        ("ethnicity_clean", 6, "eth"),        # eth_WHITE, eth_BLACK, etc.
        ("insurance", 5, "ins"),              # ins_MEDICARE, ins_MEDICAID, etc.
        ("admission_type", 4, "adm"),         # adm_EMERGENCY, adm_ELECTIVE, etc.
        ("language", 4, "lang"),              # lang_ENGLISH, lang_SPANISH, etc.
    ]:
        if col in df.columns:
            df[col] = df[col].astype("string").fillna("UNKNOWN")
            df = one_hot_topk(df, col, k=k, prefix=prefix)

    # ====================================================================
    # STEP 15: ORDER COLUMNS LOGICALLY FOR OUTPUT
    # ====================================================================
    logger.info("Organizing output columns...")
    
    # ID columns first
    id_cols = [c for c in ["subject_id", "hadm_id"] if c in df.columns]
    
    # Base engineered features
    base_features = [
        "text_chars", "text_tokens", "sentences", "chars_per_token", "long_note_flag",
        "flesch_reading_ease", "avg_syllables_per_word", "vocabulary_richness",
        "kw_chronic_disease", "kw_symptoms", "kw_medications", "kw_med_suffix_hits",
        "high_risk_score", "positive_outcome_score", "risk_outcome_ratio",
        "acute_presentation_score", "chronic_condition_score", "acute_chronic_ratio",
        "missing_section_count", "documentation_completeness",
        "disease_density", "medication_density", "symptom_density",
        "polypharmacy_flag", "treatment_intensity",
        "total_labs", "abnormal_lab_count", "abnormal_lab_ratio",
        "diagnosis_count", "diagnosis_unique_count", "comorbidity_score",
    ]
    
    # Optional section features
    optional = [
        c for c in ["has_allergies_section", "has_medications_section", 
                   "has_brief_hospital_course", "negation_density"] 
        if c in df.columns
    ]
    
    # Combine all engineered features
    engineered = id_cols + [c for c in base_features if c in df.columns] + optional
    
    # One-hot encoded columns
    ohe_cols = [c for c in df.columns if c.startswith(("gender_", "eth_", "ins_", "adm_", "lang_"))]
    
    # Other columns not yet included
    others = [c for c in df.columns if c not in engineered + ohe_cols + ["cleaned_text", "cleaned_text_final"]]
    
    # Final column order: IDs → Engineered → Others → One-Hot
    ordered = engineered + others + ohe_cols
    df = df[[c for c in ordered if c in df.columns]].copy()

    logger.info(f"Feature engineering complete: {len(df.columns)} features created")
    return df


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================
# Allows running the script from terminal with custom arguments
# ============================================================================

def parse_args() -> argparse.Namespace:
    """Parse command line arguments"""
    p = argparse.ArgumentParser(description="Feature engineering for MIMIC-III discharge summaries")
    p.add_argument("--input", type=str, help="Input CSV path (default: from config)")
    p.add_argument("--output", type=str, help="Output CSV path (default: from config)")
    p.add_argument("--log", type=str, help="Log file path (default: from config)")
    p.add_argument("--with_sections", action="store_true", help="Include section-level features")
    return p.parse_args()


# ============================================================================
# MAIN EXECUTION FUNCTION
# ============================================================================
# This runs when you execute: python data-pipeline/scripts/feature_engineering.py
#
# What happens:
# 1. Finds repository root directory
# 2. Loads pipeline_config.json for paths
# 3. Sets up logging to file and console
# 4. Loads preprocessed data
# 5. Runs feature engineering pipeline
# 6. Saves feature matrix to CSV
# 7. Prints summary to console
#
# Expected runtime: 1-3 minutes for ~9,600 records
# ============================================================================

def main() -> None:
    """
    Main execution function
    
    Execution Steps:
    1. Find repository root
    2. Load configuration
    3. Setup logging
    4. Load preprocessed data
    5. Run feature engineering
    6. Save output
    7. Print summary
    """
    
    # ====================================================================
    # STEP 1: FIND REPOSITORY ROOT
    # ====================================================================
    repo = find_repo_root()
    
    # ====================================================================
    # STEP 2: LOAD CONFIGURATION
    # ====================================================================
    config_path = repo / "data-pipeline" / "configs" / "pipeline_config.json"
    
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
        
        # Use config paths
        default_input = repo / config['pipeline_config']['output_path'] / "processed_discharge_summaries.csv"
        default_output = repo / config['pipeline_config']['output_path'] / "mimic_features.csv"
        default_log = repo / config['pipeline_config']['logs_path'] / "feature_engineering.log"
    else:
        # Fallback paths if config not found
        default_input = repo / "data-pipeline" / "data" / "processed" / "processed_discharge_summaries.csv"
        default_output = repo / "data-pipeline" / "data" / "processed" / "mimic_features.csv"
        default_log = repo / "data-pipeline" / "logs" / "feature_engineering.log"

    # ====================================================================
    # STEP 3: PARSE COMMAND LINE ARGUMENTS
    # ====================================================================
    args = parse_args()
    input_path = Path(args.input) if args.input else default_input
    output_path = Path(args.output) if args.output else default_output
    log_path = Path(args.log) if args.log else default_log

    # ====================================================================
    # STEP 4: SETUP LOGGER
    # ====================================================================
    logger = setup_logger(log_path)
    logger.info("="*60)
    logger.info("FEATURE ENGINEERING PIPELINE")
    logger.info("="*60)
    logger.info(f"Repository root: {repo}")
    logger.info(f"Input file: {input_path}")
    logger.info(f"Output file: {output_path}")
    logger.info(f"Log file: {log_path}")
    logger.info(f"Section features: {args.with_sections}")

    # ====================================================================
    # STEP 5: VALIDATE INPUT FILE EXISTS
    # ====================================================================
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # ====================================================================
    # STEP 6: LOAD PREPROCESSED DATA
    # ====================================================================
    logger.info("Loading preprocessed data...")
    df = pd.read_csv(input_path, encoding="utf-8", low_memory=False)
    logger.info(f"Loaded {df.shape[0]} rows, {df.shape[1]} columns")

    # ====================================================================
    # STEP 7: RUN FEATURE ENGINEERING
    # ====================================================================
    df_features = engineer_features(df, logger=logger, with_sections=args.with_sections)

    # ====================================================================
    # STEP 8: SAVE FEATURES
    # ====================================================================
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_features.to_csv(output_path, index=False, encoding="utf-8")
    
    # ====================================================================
    # STEP 9: LOG COMPLETION
    # ====================================================================
    logger.info("="*60)
    logger.info("FEATURE ENGINEERING COMPLETE")
    logger.info("="*60)
    logger.info(f"Total features: {len(df_features.columns)}")
    logger.info(f"Output saved: {output_path}")
    logger.info(f"Final shape: {df_features.shape[0]} rows, {df_features.shape[1]} columns")
    logger.info("="*60)
    
    # ====================================================================
    # STEP 10: PRINT SUMMARY TO CONSOLE
    # ====================================================================
    print("\n" + "="*60)
    print("FEATURE ENGINEERING COMPLETE")
    print("="*60)
    print(f"✓ Records processed: {df_features.shape[0]}")
    print(f"✓ Features created: {df_features.shape[1]}")
    print(f"✓ Output file: {output_path}")
    print(f"✓ Log file: {log_path}")
    print("="*60)


if __name__ == "__main__":
    main()