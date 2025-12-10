"""
MIMIC-III Comprehensive Bias Detection Pipeline
================================================

**How to run (from project root):**
```bash
python data-pipeline/scripts/bias_detection.py
```

**IMPORTANT: Always run this command from the project root directory!**

**What this script does:**
This is the most advanced script in the pipeline. It acts as an intelligent auditor,
determining if demographic differences are legitimate (clinical reasons) or unfair (bias).

**Input:**  data/processed/mimic_features.csv (~9,600 records, 60+ features)
**Output:** 
- logs/bias_report.json (comprehensive statistical analysis)
- logs/bias_summary.csv (executive summary table)
- logs/bias_plots/ (visualization suite)

**The Three-Stage Investigation:**

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STAGE 1: RAW BIAS DETECTION ("The Simple Alarm")
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

What it does:
  Calculates surface-level averages to see if any "alarms" go off.
  No conclusions drawn yet - just detecting differences.

How it works:
  1. Groups data by: gender, ethnicity_clean, age_group
  2. Calculates average for many features:
     - Documentation: text_chars, documentation_completeness
     - Clinical Risk: high_risk_score, risk_outcome_ratio
     - Treatment: polypharmacy_flag, treatment_intensity
     - Labs: total_labs, abnormal_lab_count
     - Readability: flesch_reading_ease

Metric Used: Coefficient of Variation (CV)
  CV = (Standard Deviation / Mean) × 100
  
  Interpretation:
  - Low CV (2-5%): Good! Minimal variation across groups
  - Medium CV (5-15%): Moderate variation - needs investigation
  - High CV (>15%): RED FLAG! Triggers Stage 2 investigation

Example:
  If older patients have notes 20% longer than younger patients,
  CV = 20% → Alarm triggered → Move to Stage 2

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STAGE 2: ADJUSTED BIAS ANALYSIS ("The Detective" - MOST CRITICAL)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

The Big Question:
  "Are older patients' notes longer because they are older (bias),
   or because they are SICKER (legitimate)?"

How it works:
  1. Builds a statistical model (Linear Regression)
  2. Uses "sickness features" to predict note length:
     - comorbidity_score (number of diseases)
     - treatment_intensity (medication burden)
     - abnormal_lab_count (test abnormalities)
     - high_risk_score (clinical severity)
     - kw_chronic_disease (disease mentions)
     - kw_medications (medication count)
  
  3. Model predicts: "Based on how sick this patient is,
                      how long should the note be?"
  
  4. Calculates "residuals" = Actual length - Predicted length
     Residuals = unexplained differences after accounting for sickness

The Verdict:
  ✓ NO BIAS FOUND (LEGITIMATE_VARIATION):
    - Residuals are random and equal across all age groups
    - Sickness fully explains the note length difference
    - Example: Older patients' notes are longer, but they're also sicker
    - Action: NO mitigation needed - this is appropriate care
  
  ✗ BIAS FOUND (POTENTIAL_BIAS):
    - Residuals still differ after accounting for sickness
    - Unexplained systematic differences remain
    - Example: Even equally-sick patients have different note lengths by age
    - Action: IMMEDIATE audit and mitigation required

Real Example:
  Before adjustment:
    - 65+ patients: avg 12,000 characters
    - 18-35 patients: avg 8,000 characters
    - Raw CV: 25% (HIGH - alarm!)
  
  After adjustment (accounting for sickness):
    - 65+ patients: residual = +100 characters
    - 18-35 patients: residual = -80 characters
    - Adjusted CV: 3% (LOW - legitimate!)
  
  Conclusion: Older patients ARE sicker, so longer notes are justified.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STAGE 3: QUALITY PARITY ANALYSIS ("The Quality Check")
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

The Question:
  "Regardless of note length, does EVERY patient get complete documentation
   of critical items (diagnosis, medications, follow-up)?"

Why this matters:
  Note length doesn't matter if essential information is missing!
  A short note with all critical info is better than a long incomplete note.

How it works:
  Checks these critical quality indicators for every demographic group:
  - documentation_completeness: Overall completeness score (0.0 to 1.0)
  - has_diagnosis: Is diagnosis documented? (True/False)
  - has_medications: Are medications documented? (True/False)
  - has_follow_up: Are follow-up instructions documented? (True/False)

The Verdict:
  If one group (e.g., Hispanic patients) is missing follow-up instructions
  30% of the time while other groups miss it only 10% of the time:
  → HIGH-PRIORITY DISPARITY flagged, even if Stage 2 passed

Importance:
  Quality disparities are MORE concerning than text length variation!
  ALL patients deserve complete essential documentation.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

**Final Outputs:**

1. VISUALIZATIONS (bias_plots/ folder):
   - text_length_by_age.png: Shows raw differences
   - treatment_intensity_by_age.png: Shows "sickness" explanation
   - abnormal_labs_by_age.png: Another complexity indicator
   - completeness_by_ethnicity.png: Quality parity check

2. REPORTS:
   - bias_report.json: Detailed statistical results from all 3 stages
   - bias_summary.csv: Executive summary table (PASS/FAIL metrics)

3. CONSOLE OUTPUT:
   Human-readable interpretations with actionable verdicts:
   - "Verdict: LEGITIMATE CLINICAL VARIATION"
   - "Verdict: POTENTIAL DOCUMENTATION BIAS"
   - "Verdict: SERIOUS CONCERN - IMMEDIATE ACTION REQUIRED"

**Key Technologies:**
- pandas: Data manipulation
- numpy: Numerical operations
- scikit-learn: Linear regression for adjusted analysis
- scipy: Statistical hypothesis testing
- matplotlib/seaborn: Visualization
- Custom error handling & logging utilities

**Pipeline Stage:** 4 of 4 (Acquisition → Preprocessing → Feature Engineering → **Bias Detection**)

**Expected Runtime:** 2-5 minutes for ~9,600 records
"""

# Comprehensive Bias Detection Pipeline for MIMIC-III Data
# Author: Lab Lens Team
# Description: Advanced bias detection with statistical controls and quality analysis

import pandas as pd
import numpy as np
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any, Optional
from scipy import stats
from datetime import datetime
import sys
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from utils.logging_config import get_logger, log_data_operation
from utils.error_handling import (
    BiasDetectionError, ErrorHandler, safe_execute, 
    validate_dataframe, validate_file_path, ErrorContext
)

# Set up logging
logger = get_logger(__name__)


# ============================================================================
# MIMICBiasDetector Class
# ============================================================================
# This class implements the complete three-stage bias detection system.
#
# Stage 1: Raw Bias Detection - Measures direct demographic differences
# Stage 2: Adjusted Bias Analysis - Controls for clinical complexity
# Stage 3: Quality Parity Analysis - Ensures complete documentation for all
# ============================================================================

class MIMICBiasDetector:
    """
    Comprehensive bias detection for MIMIC-III medical records
    
    This class implements a three-stage bias detection approach:
    
    Stage 1 - Raw Bias Detection:
        Calculates coefficient of variation in text length and other metrics
        across demographic groups without adjusting for confounding factors.
        Think of this as "raising alarms" when differences are detected.
    
    Stage 2 - Adjusted Bias Analysis:
        Uses linear regression to control for clinical complexity factors
        (comorbidities, treatment intensity, etc.) to determine if observed
        differences are explained by legitimate clinical need.
        Think of this as "investigating the alarms" to determine legitimacy.
    
    Stage 3 - Quality Parity Analysis:
        Ensures all demographic groups receive equivalent documentation quality
        regardless of text length, checking for presence of critical sections.
        Think of this as "ensuring everyone gets the essentials."
    
    The combination of these analyses provides a complete picture of whether
    detected variation represents appropriate clinical care or systematic bias.
    """
    
    def __init__(self, input_path: str = 'data/processed', output_path: str = 'logs', config: Dict = None):
        """
        Initialize bias detector with paths and configuration
        
        Args:
            input_path: Path to processed/features data directory
            output_path: Path to save bias reports and visualizations
            config: Configuration dictionary with bias detection rules
        """
        self.input_path = input_path
        self.output_path = output_path
        self.error_handler = ErrorHandler(logger)
        
        try:
            os.makedirs(output_path, exist_ok=True)
            os.makedirs(os.path.join(output_path, 'bias_plots'), exist_ok=True)
            logger.info(f"Initialized bias detector with input: {input_path}, output: {output_path}")
        except Exception as e:
            raise self.error_handler.handle_file_error("directory_creation", output_path, e)
        
        # ========================================================================
        # Bias Detection Thresholds Configuration
        # ========================================================================
        # These thresholds define what variation is "acceptable" vs "concerning"
        #
        # CV (Coefficient of Variation) Thresholds:
        # - gender_cv_max: 5% - Gender should show minimal variation
        # - ethnicity_cv_max: 10% - Some ethnic variation acceptable
        # - age_cv_max: 20% - Age can vary due to clinical complexity
        # - age_cv_adjusted_max: 5% - After adjustment should be minimal
        # - quality_cv_max: 10% - Quality should be consistent
        #
        # Statistical Thresholds:
        # - min_sample_size: 30 - Minimum for reliable statistical tests
        # - significance_threshold: 0.05 - Standard alpha level (p < 0.05)
        # - min_regression_samples: 100 - Minimum for regression analysis
        # - residual_cv_threshold: 5% - Threshold for adjusted bias
        # ========================================================================
        
        # Start with comprehensive defaults
        default_thresholds = {
            'gender_cv_max': 5.0,              # Gender should show minimal variation
            'ethnicity_cv_max': 10.0,          # Some ethnic variation acceptable
            'age_cv_max': 20.0,                # Age can vary due to clinical complexity
            'age_cv_adjusted_max': 5.0,        # After adjusting should be minimal
            'quality_cv_max': 10.0,            # Quality should be consistent
            'overall_bias_score_max': 10.0,
            'min_sample_size': 30,             # Minimum for statistical tests
            'significance_threshold': 0.05,     # Alpha level for hypothesis tests
            'min_regression_samples': 100,      # Minimum samples for regression
            'residual_cv_threshold': 5.0,       # Threshold for adjusted bias
            'high_correlation_threshold': 0.8   # For feature correlation checks
        }
        
        # Merge config thresholds with defaults (config overrides defaults)
        if config and 'bias_detection_config' in config:
            config_thresholds = config['bias_detection_config'].get('alert_thresholds', {})
            self.thresholds = {**default_thresholds, **config_thresholds}
        else:
            self.thresholds = default_thresholds
        
        logger.info(f"Using bias detection thresholds: {self.thresholds}")
    
    @safe_execute("load_data", logger, ErrorHandler(logger))
    @log_data_operation(logger, "load_data")
    def load_data(self, filename: str = 'mimic_features.csv') -> pd.DataFrame:
        """
        Load feature-engineered data for bias detection with validation
        
        Args:
            filename: Name of features CSV file
            
        Returns:
            DataFrame with engineered features and demographics
            
        Raises:
            FileNotFoundError: If input file does not exist
            ValueError: If required columns are missing or data is insufficient
        """
        filepath = os.path.join(self.input_path, filename)
        
        validate_file_path(filepath, logger, must_exist=True)
        
        logger.info(f"Loading data from {filepath}")
        df = pd.read_csv(filepath)
        
        # Validate DataFrame structure
        required_columns = ['hadm_id', 'subject_id']
        validate_dataframe(df, required_columns, logger)
        
        # Validate sufficient data for analysis
        if len(df) < self.thresholds['min_regression_samples']:
            raise ValueError(
                f"Insufficient data for bias analysis: {len(df)} rows "
                f"(minimum required: {self.thresholds['min_regression_samples']})"
            )
        
        logger.info(f"Loaded {len(df)} records with {len(df.columns)} features for bias detection")
        return df
    
    def _validate_feature_quality(self, df: pd.DataFrame, features: List[str]) -> List[str]:
        """
        Validate that features have sufficient quality for statistical analysis
        
        Quality checks performed:
        1. Excessive missing values (>50%) - Too sparse for reliable analysis
        2. Zero variance (all same value) - Cannot contribute to predictions
        3. Extreme outliers (>5% beyond 3×IQR) - May skew results
        
        Args:
            df: Input DataFrame
            features: List of feature names to validate
            
        Returns:
            List of features that pass all quality checks
        """
        valid_features = []
        
        for feature in features:
            if feature not in df.columns:
                logger.warning(f"Feature {feature} not found in data, skipping")
                continue
            
            # ---- Check 1: Missing Values ----
            # More than 50% missing means the feature is too sparse
            missing_pct = df[feature].isna().sum() / len(df)
            if missing_pct > 0.5:
                logger.warning(f"Feature {feature} has {missing_pct*100:.1f}% missing values, excluding")
                continue
            
            # ---- Check 2: Variance ----
            # If all values are the same, feature provides no information
            if df[feature].nunique() < 2:
                logger.warning(f"Feature {feature} has no variation, excluding")
                continue
            
            # ---- Check 3: Outliers ----
            # Check if numeric and has reasonable range
            if pd.api.types.is_numeric_dtype(df[feature]):
                q1 = df[feature].quantile(0.25)
                q3 = df[feature].quantile(0.75)
                iqr = q3 - q1
                
                # Check for extreme outliers (beyond 3 IQR)
                outliers = ((df[feature] < q1 - 3*iqr) | (df[feature] > q3 + 3*iqr)).sum()
                outlier_pct = outliers / len(df)
                
                if outlier_pct > 0.05:
                    logger.info(f"Feature {feature} has {outlier_pct*100:.1f}% extreme outliers")
            
            valid_features.append(feature)
        
        logger.info(f"Validated {len(valid_features)}/{len(features)} features for analysis")
        return valid_features
    
    def _analyze_metric_by_demographics(self, 
                                       df: pd.DataFrame, 
                                       metric: str, 
                                       agg_funcs: List[str] = None) -> Dict[str, Any]:
        """
        Reusable helper to analyze any metric across demographic groups
        
        This consolidates repeated groupby operations throughout the codebase.
        Calculates aggregate statistics for a given metric split by demographics.
        
        Example:
          metric='text_chars', demo='age_group'
          Output: {'<18': mean=5000, '18-35': mean=7000, '65+': mean=12000, ...}
        
        Args:
            df: Input DataFrame
            metric: Column name of metric to analyze
            agg_funcs: List of aggregation functions (default: mean, median, std, count)
            
        Returns:
            Dictionary with statistics for each demographic column
        """
        if agg_funcs is None:
            agg_funcs = ['mean', 'median', 'std', 'count']
        
        results = {}
        
        if metric not in df.columns:
            logger.warning(f"Metric {metric} not found in data")
            return results
        
        for demo_col in ['gender', 'ethnicity_clean', 'age_group']:
            if demo_col not in df.columns:
                continue
            
            try:
                stats = df.groupby(demo_col)[metric].agg(agg_funcs)
                results[demo_col] = json.loads(stats.to_json())
            except Exception as e:
                logger.error(f"Error analyzing {metric} by {demo_col}: {str(e)}")
                continue
        
        return results
    
    def _perform_statistical_test(self, 
                                 df: pd.DataFrame, 
                                 metric: str, 
                                 demo_col: str) -> Optional[Dict[str, Any]]:
        """
        Perform appropriate statistical test for metric differences across groups
        
        Test Selection:
        - Two groups: Independent t-test
        - Multiple groups: One-way ANOVA (Analysis of Variance)
        
        Hypothesis Testing:
        - Null Hypothesis (H0): No difference between groups
        - Alternative (H1): Significant difference exists
        - If p-value < 0.05: Reject H0, difference is statistically significant
        
        Args:
            df: Input DataFrame
            metric: Column name of metric to test
            demo_col: Demographic column to group by
            
        Returns:
            Dictionary with test results or None if test cannot be performed
        """
        if metric not in df.columns or demo_col not in df.columns:
            return None
        
        # Get groups with sufficient sample size (minimum 30 per group)
        groups = []
        group_names = []
        
        for group_name, group_data in df.groupby(demo_col):
            group_values = group_data[metric].dropna()
            if len(group_values) >= self.thresholds['min_sample_size']:
                groups.append(group_values.values)
                group_names.append(group_name)
        
        if len(groups) < 2:
            logger.warning(f"Insufficient groups for statistical test on {metric} by {demo_col}")
            return None
        
        try:
            if len(groups) == 2:
                # ---- Two Groups: Use t-test ----
                # Tests if means of two groups are significantly different
                t_stat, p_value = stats.ttest_ind(groups[0], groups[1])
                test_type = 't-test'
                test_stat = float(t_stat)
                mean_diff = float(groups[0].mean() - groups[1].mean())
            else:
                # ---- Multiple Groups: Use ANOVA ----
                # Tests if means across multiple groups are significantly different
                f_stat, p_value = stats.f_oneway(*groups)
                test_type = 'ANOVA'
                test_stat = float(f_stat)
                mean_diff = None
            
            return {
                'test_type': test_type,
                'test_statistic': test_stat,
                'p_value': float(p_value),
                'significant': p_value < self.thresholds['significance_threshold'],
                'mean_difference': mean_diff,
                'groups_tested': group_names,
                'interpretation': (
                    'Significant difference detected' if p_value < self.thresholds['significance_threshold']
                    else 'No significant difference'
                )
            }
        except (ValueError, TypeError) as e:
            logger.error(f"Statistical test failed for {metric} by {demo_col}: {str(e)}")
            return None
    
    # ========================================================================
    # STAGE 1A: DOCUMENTATION BIAS DETECTION
    # ========================================================================
    # Measures raw differences in documentation length and quality.
    # High variation here doesn't necessarily mean bias - it could reflect
    # legitimate clinical differences that will be evaluated in Stage 2.
    # ========================================================================
    
    def detect_documentation_bias(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Stage 1A: Detect raw bias in documentation length and quality
        
        This measures simple differences in text length across demographics
        without accounting for confounding factors. High variation here does
        not necessarily indicate bias - it could reflect legitimate differences
        in clinical complexity that will be evaluated in the adjusted analysis.
        
        What we're measuring:
        - Average note length by gender, ethnicity, age
        - Statistical significance of differences
        - Documentation completeness scores
        
        Args:
            df: Input DataFrame with features
            
        Returns:
            Dictionary with documentation bias analysis results including:
            - Length statistics by demographic groups
            - Statistical test results
            - Quality metric distributions
        """
        bias_report = {
            'documentation_length_bias': {},
            'documentation_quality_bias': {},
            'statistical_tests': {}
        }
        
        if 'text_chars' not in df.columns:
            logger.warning("text_chars column not found, skipping documentation length analysis")
            return bias_report
        
        # Analyze documentation length across all demographics
        bias_report['documentation_length_bias'] = self._analyze_metric_by_demographics(
            df, 'text_chars', ['mean', 'median', 'std', 'count']
        )
        
        # Perform statistical tests for each demographic
        for demo_col in ['gender', 'ethnicity_clean', 'age_group']:
            if demo_col in df.columns:
                test_result = self._perform_statistical_test(df, 'text_chars', demo_col)
                if test_result:
                    bias_report['statistical_tests'][demo_col] = test_result
        
        # Analyze documentation quality metrics
        if 'documentation_completeness' in df.columns:
            bias_report['documentation_quality_bias'] = self._analyze_metric_by_demographics(
                df, 'documentation_completeness', ['mean', 'std', 'count']
            )
        
        return bias_report
    
    # ========================================================================
    # STAGE 1B-E: OTHER RAW BIAS DETECTION METHODS
    # ========================================================================
    # These methods analyze different aspects of care to build a complete
    # picture of potential bias before the adjusted analysis.
    # ========================================================================
    
    def detect_clinical_risk_bias(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Stage 1B: Detect raw bias in clinical risk assessment and scoring
        
        Analyzes whether risk scores and acuity assessments vary systematically
        across demographic groups. Variation here may be legitimate if certain
        populations have higher actual clinical risk.
        
        Args:
            df: Input DataFrame with features
            
        Returns:
            Dictionary with clinical risk bias analysis results
        """
        risk_bias_report = {}
        
        # Analyze key risk metrics
        risk_metrics = ['high_risk_score', 'risk_outcome_ratio', 'acute_chronic_ratio']
        
        for metric in risk_metrics:
            if metric in df.columns:
                risk_bias_report[metric] = self._analyze_metric_by_demographics(
                    df, metric, ['mean', 'median', 'std', 'count']
                )
        
        return risk_bias_report
    
    def detect_treatment_complexity_bias(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Stage 1C: Detect raw bias in treatment complexity and intensity measures
        
        Examines whether treatment patterns (medication counts, treatment intensity)
        differ across demographic groups. Important for understanding whether
        documentation differences reflect actual treatment differences.
        
        Args:
            df: Input DataFrame with features
            
        Returns:
            Dictionary with treatment complexity bias analysis results
        """
        treatment_bias = {}
        
        # Analyze polypharmacy rates (patients on 5+ medications)
        if 'polypharmacy_flag' in df.columns:
            treatment_bias['polypharmacy_rates'] = {}
            for demo_col in ['gender', 'ethnicity_clean', 'age_group']:
                if demo_col in df.columns:
                    poly_rate = df.groupby(demo_col)['polypharmacy_flag'].mean()
                    treatment_bias['polypharmacy_rates'][demo_col] = poly_rate.to_dict()
        
        # Treatment intensity analysis
        treatment_metrics = ['treatment_intensity', 'kw_medications']
        for metric in treatment_metrics:
            if metric in df.columns:
                treatment_bias[metric] = self._analyze_metric_by_demographics(
                    df, metric, ['mean', 'median', 'count']
                )
        
        return treatment_bias
    
    def detect_lab_testing_bias(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Stage 1D: Detect raw bias in laboratory testing patterns
        
        Analyzes whether lab testing frequency and abnormality patterns differ
        across demographics. This helps understand if documentation differences
        are accompanied by differences in diagnostic workup.
        
        Args:
            df: Input DataFrame with features
            
        Returns:
            Dictionary with lab testing bias analysis results
        """
        lab_bias_report = {}
        
        # Analyze lab-related metrics
        lab_metrics = ['total_labs', 'abnormal_lab_count', 'abnormal_lab_ratio']
        
        for metric in lab_metrics:
            if metric in df.columns:
                lab_bias_report[metric] = self._analyze_metric_by_demographics(
                    df, metric, ['mean', 'median', 'std', 'count']
                )
        
        return lab_bias_report
    
    def detect_readability_bias(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Stage 1E: Detect raw bias in documentation readability and complexity
        
        Examines whether documentation complexity varies across demographics.
        Differences here could indicate varying levels of detail or care in
        documentation practices.
        
        Args:
            df: Input DataFrame with features
            
        Returns:
            Dictionary with readability bias analysis results
        """
        readability_bias = {}
        
        # Analyze readability metrics
        readability_metrics = ['flesch_reading_ease', 'vocabulary_richness', 'disease_density']
        
        for metric in readability_metrics:
            if metric in df.columns:
                readability_bias[metric] = self._analyze_metric_by_demographics(
                    df, metric, ['mean', 'median', 'count']
                )
        
        return readability_bias
    
    # ========================================================================
    # STAGE 2: ADJUSTED BIAS ANALYSIS (THE CRITICAL INVESTIGATION)
    # ========================================================================
    # This is where we determine if raw bias is legitimate or problematic.
    # Uses regression to control for clinical complexity factors.
    # ========================================================================
    
    def detect_adjusted_bias(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Stage 2: Detect adjusted bias after controlling for clinical complexity
        
        *** THIS IS THE MOST CRITICAL ANALYSIS ***
        
        This determines whether raw bias detected in Stage 1 is legitimate.
        
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        THE METHOD: Statistical Regression Analysis
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        
        Step 1: Identify Clinical Complexity Features
          Features that explain why notes are longer/shorter:
          - comorbidity_score: Number of concurrent diseases
          - treatment_intensity: Overall treatment burden
          - abnormal_lab_count: Number of abnormal lab results
          - high_risk_score: Clinical risk assessment
          - kw_chronic_disease: Mentions of chronic conditions
          - kw_medications: Number of medications
        
        Step 2: Build Regression Model
          Model equation: text_length = β₀ + β₁(complexity) + β₂(age) + ε
          
          What this means:
          "Predict note length based on how sick the patient is"
        
        Step 3: Calculate Residuals
          Residual = Actual length - Predicted length
          
          What residuals represent:
          Text length variation NOT explained by clinical complexity
          
          If residuals are near zero: Sickness explains everything ✓
          If residuals vary by age: Unexplained age bias remains ✗
        
        Step 4: Analyze Residual Patterns
          Calculate CV of residuals by demographic groups
          
          Low residual CV (<5%): LEGITIMATE_VARIATION
            - Sickness explains the differences
            - No bias mitigation needed
          
          High residual CV (>5%): POTENTIAL_BIAS
            - Unexplained systematic differences remain
            - Bias mitigation required
        
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        REAL EXAMPLE:
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        
        Before Adjustment (Raw):
          65+ patients: 12,000 characters
          18-35 patients: 8,000 characters
          Difference: 4,000 characters (33%)
          Raw CV: 25% → ALARM!
        
        Regression Model Predicts:
          Based on comorbidity_score, treatment_intensity, etc.
          65+ patients: Predicted = 11,900 characters
          18-35 patients: Predicted = 8,100 characters
        
        Residuals (Unexplained Differences):
          65+ patients: 12,000 - 11,900 = +100 characters
          18-35 patients: 8,000 - 8,100 = -100 characters
          Residual difference: 200 characters (1.7%)
          Adjusted CV: 3% → NO BIAS!
        
        Interpretation:
          The 4,000 character difference is almost entirely explained by
          older patients being sicker. Only 200 characters (5%) remain
          unexplained, which is within normal variation.
          
          VERDICT: LEGITIMATE_VARIATION - No action needed
        
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        
        Args:
            df: Input DataFrame with features
            
        Returns:
            Dictionary with adjusted bias analysis results including:
            - Model performance metrics (R-squared)
            - Residual analysis by demographics
            - Interpretation of findings (LEGITIMATE_VARIATION or POTENTIAL_BIAS)
        """
        adjusted_bias_report = {
            'analysis_performed': False,
            'reason_skipped': None
        }
        
        # ====================================================================
        # STEP 1: PREPARE DATA
        # ====================================================================
        
        # Map age groups to numeric values for regression
        # (Regression needs numbers, not categories)
        age_map = {'<18': 10, '18-35': 26, '35-50': 42, '50-65': 57, '65+': 75}
        
        if 'age_group' not in df.columns:
            adjusted_bias_report['reason_skipped'] = "age_group column not found"
            logger.warning("Skipping adjusted bias analysis: age_group column missing")
            return adjusted_bias_report
        
        # ====================================================================
        # STEP 2: DEFINE CLINICAL COMPLEXITY FEATURES
        # ====================================================================
        # These features explain WHY documentation length varies
        # They represent how "sick" or "complex" a patient is
        
        complexity_features = [
            'comorbidity_score',      # Number of concurrent diseases
            'treatment_intensity',     # Overall treatment burden
            'abnormal_lab_count',     # Number of abnormal lab results
            'high_risk_score',        # Clinical risk assessment
            'kw_chronic_disease',     # Mentions of chronic conditions
            'kw_medications'          # Number of medications mentioned
        ]
        
        # ====================================================================
        # STEP 3: VALIDATE FEATURES
        # ====================================================================
        # Ensure features exist and meet quality standards
        available_features = self._validate_feature_quality(df, complexity_features)
        
        if len(available_features) < 2:
            adjusted_bias_report['reason_skipped'] = (
                f"Insufficient valid complexity features: only {len(available_features)} available"
            )
            logger.warning("Skipping adjusted bias analysis: insufficient features")
            return adjusted_bias_report
        
        # ====================================================================
        # STEP 4: PREPARE REGRESSION DATA
        # ====================================================================
        # Create clean dataset for regression
        df_temp = df[available_features + ['age_group', 'text_chars']].copy()
        df_temp['age_numeric'] = df_temp['age_group'].map(age_map)
        
        # X = predictor variables (complexity + age)
        # y = target variable (text length)
        X = df_temp[available_features + ['age_numeric']].fillna(0)
        y = df_temp['text_chars'].fillna(0)
        
        # Remove invalid rows (zero length or all missing features)
        valid_mask = (y > 0) & (X.sum(axis=1) > 0)
        X = X[valid_mask]
        y = y[valid_mask]
        df_temp = df_temp[valid_mask]
        
        if len(X) < self.thresholds['min_regression_samples']:
            adjusted_bias_report['reason_skipped'] = (
                f"Insufficient valid data: {len(X)} rows (need {self.thresholds['min_regression_samples']})"
            )
            logger.warning("Skipping adjusted bias analysis: insufficient valid data")
            return adjusted_bias_report
        
        try:
            # ====================================================================
            # STEP 5: STANDARDIZE FEATURES
            # ====================================================================
            # Put all features on the same scale for better regression performance
            # Example: comorbidity_score (0-20) and age (18-80) → both become (-2 to +2)
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # ====================================================================
            # STEP 6: FIT REGRESSION MODEL
            # ====================================================================
            logger.info(f"Fitting regression model with {len(available_features)} complexity features")
            model = LinearRegression()
            model.fit(X_scaled, y)
            
            # Calculate model performance (R-squared)
            # R² = proportion of variance explained by the model
            # 0.0 = model explains nothing, 1.0 = perfect predictions
            r2_score = model.score(X_scaled, y)
            logger.info(f"Regression model R-squared: {r2_score:.3f}")
            
            # ====================================================================
            # STEP 7: CALCULATE RESIDUALS
            # ====================================================================
            # Residuals = what the model CANNOT explain
            predictions = model.predict(X_scaled)
            df_temp['text_chars_residual'] = y.values - predictions
            
            # ====================================================================
            # STEP 8: ANALYZE RESIDUALS BY DEMOGRAPHICS
            # ====================================================================
            residual_stats = {}
            
            for demo_col in ['age_group', 'ethnicity_clean', 'gender']:
                if demo_col not in df_temp.columns:
                    continue
                
                # Calculate residual statistics for each demographic group
                residuals_by_group = df_temp.groupby(demo_col)['text_chars_residual'].agg(['mean', 'std', 'count'])
                
                # Calculate coefficient of variation for residuals
                # Use absolute values to avoid issues with near-zero means
                group_means = residuals_by_group['mean'].abs()
                
                # Calculate CV using median absolute deviation for robustness
                # This handles outliers better than standard deviation
                if len(group_means) > 0 and group_means.median() > 0:
                    # Use median absolute deviation scaled to std
                    mad = np.median(np.abs(group_means - group_means.median()))
                    cv = (mad / group_means.median()) * 100
                else:
                    cv = 0.0
                
                # Alternative: Use range-based measure if CV is extreme
                if cv > 100:
                    # Fallback to simpler range-based measure
                    residual_range = group_means.max() - group_means.min()
                    cv = (residual_range / group_means.mean()) * 100 if group_means.mean() > 0 else 0
                
                # ====================================================================
                # STEP 9: DETERMINE IF VARIATION REPRESENTS BIAS
                # ====================================================================
                # Compare residual CV to threshold (5%)
                is_biased = cv > self.thresholds['residual_cv_threshold']
                
                residual_stats[demo_col] = {
                    'residuals_by_group': residuals_by_group['mean'].to_dict(),
                    'residual_std_by_group': residuals_by_group['std'].to_dict(),
                    'sample_sizes': residuals_by_group['count'].to_dict(),
                    'cv': float(cv),
                    'interpretation': 'POTENTIAL_BIAS' if is_biased else 'LEGITIMATE_VARIATION',
                    'explanation': (
                        f"Residual CV of {cv:.2f}% indicates "
                        f"{'unexplained bias beyond clinical factors' if is_biased else 'variation is explained by clinical complexity'}. "
                        f"Mean absolute residuals range from {group_means.min():.0f} to {group_means.max():.0f} characters."
                    )
                }
            
            # ====================================================================
            # STEP 10: COMPILE RESULTS
            # ====================================================================
            adjusted_bias_report = {
                'analysis_performed': True,
                'model_features_used': available_features + ['age_numeric'],
                'model_r2_score': float(r2_score),
                'model_explanation': (
                    f"Model explains {r2_score*100:.1f}% of text length variation using "
                    f"clinical complexity factors. Remaining {(1-r2_score)*100:.1f}% may "
                    f"represent random variation or systematic bias."
                ),
                'residual_analysis': residual_stats,
                'overall_interpretation': self._interpret_adjusted_results(residual_stats, r2_score)
            }
            
            logger.info("Adjusted bias analysis completed successfully")
            
        except Exception as e:
            adjusted_bias_report['reason_skipped'] = f"Analysis failed: {str(e)}"
            logger.error(f"Adjusted bias analysis failed: {str(e)}")
        
        return adjusted_bias_report
    
    def _interpret_adjusted_results(self, residual_stats: Dict, r2_score: float) -> str:
        """
        Interpret adjusted bias analysis results to provide actionable insights
        
        Considers both model fit and residual patterns to determine whether
        observed raw bias represents legitimate clinical variation or systematic
        documentation bias requiring intervention.
        
        Args:
            residual_stats: Dictionary of residual statistics by demographic
            r2_score: Model R-squared value
            
        Returns:
            Human-readable interpretation string with recommendations
        """
        # Identify demographics flagged as potentially biased
        biased_demos = [
            demo for demo, stats in residual_stats.items()
            if stats.get('interpretation') == 'POTENTIAL_BIAS'
        ]
        
        # ---- Check Model Fit Quality ----
        # If R² < 0.3, model is weak and results are unreliable
        if r2_score < 0.3:
            return (
                "Model Warning: Low R-squared indicates clinical complexity features "
                "explain less than 30% of text length variation. This suggests either "
                "missing important complexity features or high random variation in "
                "documentation practices. Interpret adjusted bias results with caution."
            )
        
        # ---- Interpret Results ----
        if not biased_demos:
            # ✓ NO BIAS - Variation is legitimate
            return (
                "Positive Finding: After controlling for clinical complexity, no significant "
                "unexplained variation remains across demographic groups. The raw bias appears "
                "to reflect appropriate differences in clinical care complexity rather than "
                "systematic documentation bias. Recommendation: Focus on maintaining quality "
                "parity and monitoring for changes over time."
            )
        else:
            # ✗ BIAS DETECTED - Systematic problem
            return (
                f"Concern Identified: After controlling for clinical complexity, unexplained "
                f"variation persists for: {', '.join(biased_demos)}. This suggests potential "
                f"systematic documentation bias that is not explained by differences in patient "
                f"clinical needs. Recommendation: Conduct detailed audit of documentation "
                f"practices and implement bias mitigation strategies."
            )
    
    # ========================================================================
    # STAGE 3: QUALITY PARITY ANALYSIS
    # ========================================================================
    # Ensures all demographics receive complete essential documentation
    # regardless of text length differences.
    # ========================================================================
    
    def detect_documentation_quality_parity(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Stage 3: Check if all demographics receive equal quality documentation
        
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        WHY QUALITY PARITY MATTERS MORE THAN TEXT LENGTH:
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        
        Scenario 1: ACCEPTABLE
          - 65+ patient: 12,000 character note, all sections complete
          - 25 year old: 6,000 character note, all sections complete
          - Result: Different lengths OK if both have essentials ✓
        
        Scenario 2: PROBLEMATIC
          - White patient: 10,000 characters, diagnosis + meds + follow-up ✓
          - Black patient: 10,000 characters, diagnosis + meds, NO follow-up ✗
          - Result: DISPARITY even with same length! Quality not equal ✗
        
        The Key Principle:
          Every patient deserves complete documentation of:
          1. Diagnosis (what's wrong)
          2. Medications (what was prescribed)
          3. Follow-up (what to do next)
          
          Regardless of how long the note is!
        
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        
        What we check:
        1. documentation_completeness: Overall score (0.0 to 1.0)
        2. has_diagnosis: Is diagnosis documented? (True/False)
        3. has_medications: Are medications documented? (True/False)
        4. has_follow_up: Are follow-up instructions documented? (True/False)
        
        Quality parity is distinct from text length. Even if older patients have longer
        notes (which may be appropriate), all patients should have complete documentation
        of critical elements.
        
        Disparities here are concerning regardless of whether text length variation
        is legitimate.
        
        Args:
            df: Input DataFrame with features
            
        Returns:
            Dictionary with quality parity analysis results for each demographic
        """
        quality_parity_report = {}
        
        # ====================================================================
        # STEP 1: DEFINE QUALITY INDICATORS
        # ====================================================================
        # These represent critical documentation elements
        quality_metrics = {}
        
        if 'documentation_completeness' in df.columns:
            quality_metrics['documentation_completeness'] = 'Overall Documentation Completeness Score'
        
        # Check for boolean quality indicators
        for col in ['has_diagnosis', 'has_medications', 'has_follow_up']:
            if col in df.columns:
                quality_metrics[col] = col.replace('has_', '').replace('_', ' ').title() + ' Present'
        
        if not quality_metrics:
            logger.warning("No quality metrics found for parity analysis")
            return quality_parity_report
        
        # ====================================================================
        # STEP 2: ANALYZE QUALITY PARITY FOR EACH DEMOGRAPHIC
        # ====================================================================
        for demo_col in ['age_group', 'ethnicity_clean', 'gender']:
            if demo_col not in df.columns:
                continue
            
            quality_parity_report[demo_col] = {}
            
            for metric, description in quality_metrics.items():
                if metric not in df.columns:
                    continue
                
                # Calculate quality metric by demographic group
                quality_by_group = df.groupby(demo_col)[metric].mean()
                
                # Calculate coefficient of variation
                # Lower CV indicates better parity (more consistent quality)
                mean_quality = quality_by_group.mean()
                if mean_quality > 0:
                    cv = (quality_by_group.std() / mean_quality) * 100
                else:
                    cv = 0
                
                # Perform statistical test
                test_result = self._perform_statistical_test(df, metric, demo_col)
                
                # ====================================================================
                # STEP 3: DETERMINE ASSESSMENT
                # ====================================================================
                # CV under 10% indicates good parity (acceptable variation)
                # CV over 10% indicates disparity (concerning variation)
                assessment = 'PARITY' if cv < self.thresholds['quality_cv_max'] else 'DISPARITY'
                
                quality_parity_report[demo_col][metric] = {
                    'description': description,
                    'quality_by_group': quality_by_group.to_dict(),
                    'cv': float(cv),
                    'statistical_test': test_result,
                    'assessment': assessment,
                    'interpretation': (
                        f"{assessment}: {description} shows {cv:.1f}% variation across "
                        f"{demo_col.replace('_', ' ')} groups. "
                        f"{'This indicates consistent quality.' if assessment == 'PARITY' else 'This disparity warrants investigation.'}"
                    )
                }
        
        return quality_parity_report
    
    def calculate_bias_metrics(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate summary bias metrics using coefficient of variation
        
        These metrics provide a high-level overview of raw bias (before adjustment).
        Lower CV values indicate more consistent documentation across groups.
        
        Coefficient of Variation (CV) Interpretation:
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        - Under 5%: Minimal variation ✓
        - 5-15%: Moderate variation (monitor)
        - Over 15%: High variation (investigate) ⚠
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        
        Args:
            df: Input DataFrame with features
            
        Returns:
            Dictionary with summary bias metrics and alert flags
        """
        metrics = {}
        
        if 'text_chars' not in df.columns:
            logger.warning("text_chars column not found, cannot calculate bias metrics")
            return metrics
        
        # Calculate CV for each demographic
        for demo_col, threshold_key in [
            ('gender', 'gender_cv_max'),
            ('ethnicity_clean', 'ethnicity_cv_max'),
            ('age_group', 'age_cv_max')
        ]:
            if demo_col not in df.columns:
                continue
            
            demo_means = df.groupby(demo_col)['text_chars'].mean()
            
            if len(demo_means) > 1 and demo_means.mean() > 0:
                cv = (demo_means.std() / demo_means.mean()) * 100
                metrics[f'{demo_col.split("_")[0]}_cv'] = float(cv)
                metrics[f'{demo_col.split("_")[0]}_bias_alert'] = cv > self.thresholds[threshold_key]
        
        # Calculate overall bias score as average of CVs
        cv_values = [v for k, v in metrics.items() if k.endswith('_cv')]
        if cv_values:
            overall_score = float(np.mean(cv_values))
            metrics['overall_bias_score'] = overall_score
            metrics['overall_bias_alert'] = overall_score > self.thresholds['overall_bias_score_max']
        
        return metrics
    
    def interpret_bias_findings(self, bias_report: Dict) -> Dict[str, str]:
        """
        Generate human-readable interpretations of bias analysis results
        
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        PURPOSE: Translate Statistics into Action
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        
        This function converts technical statistical metrics into clear,
        actionable insights for stakeholders who need to understand:
        
        1. What was found?
        2. Is it problematic?
        3. What should we do about it?
        
        The interpretations combine results from all three stages:
        - Stage 1 (Raw Bias): Initial detection
        - Stage 2 (Adjusted Bias): Legitimacy determination
        - Stage 3 (Quality Parity): Essential completeness check
        
        Output Format:
        Each interpretation includes:
        - Clear verdict (LEGITIMATE, POTENTIAL_BIAS, SERIOUS_CONCERN)
        - Plain-English explanation
        - Specific recommended actions
        
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        
        Args:
            bias_report: Complete bias report dictionary
            
        Returns:
            Dictionary mapping finding categories to interpretation strings
        """
        interpretations = {}
        
        metrics = bias_report.get('summary_metrics', {})
        adjusted = bias_report.get('adjusted_bias_analysis', {})
        quality = bias_report.get('quality_parity_analysis', {})
        
        # ====================================================================
        # INTERPRET AGE VARIATION
        # ====================================================================
        age_cv = metrics.get('age_cv', 0)
        age_adjusted_cv = adjusted.get('residual_analysis', {}).get('age_group', {}).get('cv', None)
        
        if age_cv > 15:
            if adjusted.get('analysis_performed') and age_adjusted_cv is not None:
                if age_adjusted_cv < self.thresholds['residual_cv_threshold']:
                    # ✓ High raw CV but low adjusted CV = LEGITIMATE
                    interpretations['age_variation'] = (
                        f"Age Variation Analysis: Raw CV {age_cv:.1f}%, Adjusted CV {age_adjusted_cv:.1f}%\n\n"
                        "Verdict: LEGITIMATE CLINICAL VARIATION\n\n"
                        "The substantial variation in text length across age groups disappears after "
                        "controlling for clinical complexity factors. This indicates the differences "
                        "are appropriate and reflect legitimate variations in patient care needs.\n\n"
                        "Explanation: Older patients typically have more comorbidities, require more "
                        "medications, and need more detailed discharge instructions. The longer notes "
                        "for older patients are clinically justified.\n\n"
                        "Recommended Actions:\n"
                        "- No mitigation needed for text length differences\n"
                        "- Ensure ML models are trained to handle age-related complexity appropriately\n"
                        "- Monitor outcome fairness to ensure equivalent care quality\n"
                        "- Continue regular quality parity assessments"
                    )
                else:
                    # ✗ High raw CV and high adjusted CV = BIAS
                    interpretations['age_variation'] = (
                        f"Age Variation Analysis: Raw CV {age_cv:.1f}%, Adjusted CV {age_adjusted_cv:.1f}%\n\n"
                        "Verdict: POTENTIAL DOCUMENTATION BIAS\n\n"
                        "Significant variation persists across age groups even after controlling for "
                        "clinical complexity. This suggests systematic differences in documentation "
                        "practices that are not explained by patient care requirements.\n\n"
                        "Recommended Actions:\n"
                        "- IMMEDIATE: Conduct detailed audit of documentation by age group\n"
                        "- REQUIRED: Implement bias mitigation strategies\n"
                        "- INVESTIGATE: Review documentation training and institutional practices\n"
                        "- MONITOR: Establish continuous tracking of improvement"
                    )
            else:
                # Cannot determine due to missing adjusted analysis
                interpretations['age_variation'] = (
                    f"Age Variation Analysis: Raw CV {age_cv:.1f}%\n\n"
                    "Adjusted analysis could not be performed due to insufficient complexity features. "
                    "Cannot definitively determine if this variation is legitimate or represents bias. "
                    "Recommend adding clinical complexity features (comorbidity scores, treatment "
                    "intensity measures) to enable complete analysis."
                )
        else:
            # Low CV = No concern
            interpretations['age_variation'] = (
                f"Age Variation: {age_cv:.1f}% (within acceptable range)\n"
                "Documentation length is reasonably consistent across age groups."
            )
        
        # ====================================================================
        # INTERPRET GENDER VARIATION
        # ====================================================================
        gender_cv = metrics.get('gender_cv', 0)
        
        if gender_cv > self.thresholds['gender_cv_max']:
            interpretations['gender_variation'] = (
                f"Gender Variation Analysis: CV {gender_cv:.1f}%\n\n"
                "Verdict: LIKELY PROBLEMATIC\n\n"
                "Gender-based variation in documentation is concerning because clinical complexity "
                "does not typically vary significantly by gender for the same conditions. This "
                "pattern suggests potential implicit bias in documentation practices.\n\n"
                "Recommended Actions:\n"
                "- HIGH PRIORITY: Conduct audit comparing male and female patients with similar conditions\n"
                "- INVESTIGATE: Examine whether certain conditions are documented differently by gender\n"
                "- REQUIRED: Implement bias awareness training for clinical documentation\n"
                "- MITIGATE: Apply appropriate bias mitigation strategies"
            )
        else:
            interpretations['gender_variation'] = (
                f"Gender Variation: {gender_cv:.1f}% (within acceptable range)\n"
                "Documentation appears consistent across gender groups."
            )
        
        # ====================================================================
        # INTERPRET ETHNICITY VARIATION
        # ====================================================================
        ethnicity_cv = metrics.get('ethnicity_cv', 0)
        
        if ethnicity_cv > self.thresholds['ethnicity_cv_max']:
            interpretations['ethnicity_variation'] = (
                f"Ethnicity Variation Analysis: CV {ethnicity_cv:.1f}%\n\n"
                "Verdict: SERIOUS CONCERN - IMMEDIATE ACTION REQUIRED\n\n"
                "Ethnicity-based variation in documentation is highly concerning and may indicate "
                "systemic bias. This pattern can result from multiple factors:\n"
                "- Language barriers affecting documentation completeness\n"
                "- Implicit bias in clinical interactions and documentation\n"
                "- Differences in care access or clinical settings\n"
                "- Cultural factors affecting patient-provider communication\n\n"
                "Recommended Actions:\n"
                "- URGENT: Comprehensive institutional audit of documentation by ethnicity\n"
                "- REQUIRED: Review and strengthen DEI policies and training programs\n"
                "- INVESTIGATE: Assess language services and translation support adequacy\n"
                "- MITIGATE: Implement aggressive bias mitigation strategies immediately\n"
                "- MONITOR: Establish continuous bias monitoring with regular reporting"
            )
        else:
            interpretations['ethnicity_variation'] = (
                f"Ethnicity Variation: {ethnicity_cv:.1f}% (within acceptable range)\n"
                "Documentation appears consistent across ethnic groups."
            )
        
        # ====================================================================
        # INTERPRET QUALITY PARITY FINDINGS
        # ====================================================================
        if quality:
            quality_issues = []
            for demo_col, metrics_dict in quality.items():
                for metric_name, metric_data in metrics_dict.items():
                    if metric_data.get('assessment') == 'DISPARITY':
                        quality_issues.append(
                            f"{metric_data.get('description')} varies {metric_data.get('cv', 0):.1f}% "
                            f"across {demo_col.replace('_', ' ')} groups"
                        )
            
            if quality_issues:
                interpretations['quality_parity'] = (
                    "Quality Parity Analysis: DISPARITIES DETECTED\n\n"
                    "Critical quality metrics show concerning variation across demographic groups:\n\n" +
                    "\n".join(f"- {issue}" for issue in quality_issues) +
                    "\n\nImportance: Quality disparities are MORE concerning than text length variation "
                    "because they indicate some demographic groups may be receiving incomplete "
                    "documentation of critical care information (diagnoses, medications, follow-up).\n\n"
                    "Recommended Actions:\n"
                    "- URGENT: Address quality disparities with highest priority\n"
                    "- REQUIRED: Ensure all critical documentation sections are complete for all patients\n"
                    "- AUDIT: Review cases with missing critical information by demographic group\n"
                    "- STANDARDIZE: Implement structured documentation templates or checklists"
                )
            else:
                interpretations['quality_parity'] = (
                    "Quality Parity Analysis: PARITY MAINTAINED\n\n"
                    "All demographic groups receive consistent documentation quality with similar "
                    "presence of critical sections including diagnoses, medications, and follow-up "
                    "instructions. This is the most important finding - even if text length varies, "
                    "all patients are receiving complete essential information."
                )
        
        # ====================================================================
        # GENERATE OVERALL ASSESSMENT
        # ====================================================================
        if age_cv > 15 and gender_cv < self.thresholds['gender_cv_max'] and ethnicity_cv < self.thresholds['ethnicity_cv_max']:
            if age_adjusted_cv and age_adjusted_cv < self.thresholds['residual_cv_threshold']:
                interpretations['overall_assessment'] = (
                    "Overall Assessment: Age-Specific Variation (LEGITIMATE)\n\n"
                    "Summary of Findings:\n"
                    "- Only age shows significant text length variation\n"
                    "- Variation disappears after controlling for clinical complexity\n"
                    "- No gender or ethnicity bias detected\n"
                    "- Quality parity maintained across all groups\n\n"
                    "Conclusion: The detected variation represents appropriate clinical care. Older "
                    "patients legitimately require more detailed documentation due to greater clinical "
                    "complexity. This is not bias requiring mitigation.\n\n"
                    "Strategy:\n"
                    "- DO NOT attempt to artificially equalize text lengths\n"
                    "- DO maintain quality parity monitoring\n"
                    "- DO ensure ML models handle age-related complexity fairly\n"
                    "- DO continue regular bias assessments to detect changes"
                )
            else:
                interpretations['overall_assessment'] = (
                    "Overall Assessment: Age-Specific Bias Detected\n\n"
                    "Age shows variation that exceeds what clinical complexity explains, suggesting "
                    "systematic age-based documentation bias requiring intervention."
                )
        elif gender_cv > self.thresholds['gender_cv_max'] or ethnicity_cv > self.thresholds['ethnicity_cv_max']:
            interpretations['overall_assessment'] = (
                "Overall Assessment: Multi-Demographic Bias Detected\n\n"
                "SERIOUS CONCERN: Bias detected across multiple demographic dimensions.\n\n"
                "This pattern strongly indicates systematic documentation bias requiring immediate "
                "comprehensive organizational response including audit, training, and mitigation.\n\n"
                "Required Actions:\n"
                "- Immediate comprehensive organizational audit\n"
                "- Implementation of bias mitigation strategies\n"
                "- Review and update documentation training programs\n"
                "- Establish continuous monitoring and accountability systems\n"
                "- Consider institutional policy changes if patterns persist"
            )
        else:
            interpretations['overall_assessment'] = (
                "Overall Assessment: No Significant Bias Detected\n\n"
                "Documentation appears consistent across all demographic groups with variation "
                "within acceptable thresholds. Continue standard monitoring and maintain current "
                "documentation practices. Regular reassessment recommended."
            )
        
        return interpretations
    
    # ========================================================================
    # VISUALIZATION GENERATION
    # ========================================================================
    # Creates plots showing bias patterns and clinical complexity relationships
    # ========================================================================
    
    def create_bias_visualizations(self, df: pd.DataFrame):
        """
        Create comprehensive visualizations for bias patterns
        
        Generates plots showing:
        1. Documentation length distributions by demographics (raw bias)
        2. Clinical complexity patterns that may explain length differences
        3. Quality parity across groups (essential documentation completeness)
        
        Visualizations include sample sizes in labels for transparency.
        All plots saved to: logs/bias_plots/
        
        Args:
            df: Input DataFrame with features
        """
        plt.style.use('default')
        sns.set_palette("husl")
        
        plots_dir = os.path.join(self.output_path, 'bias_plots')
        
        # ====================================================================
        # PLOT 1: DOCUMENTATION LENGTH BY AGE (Shows raw differences)
        # ====================================================================
        if 'age_group' in df.columns and 'text_chars' in df.columns:
            fig, ax = plt.subplots(figsize=(12, 6))
            
            df.boxplot(column='text_chars', by='age_group', ax=ax)
            
            # Add sample sizes to labels for transparency
            age_counts = df['age_group'].value_counts()
            if hasattr(df['age_group'], 'cat'):
                labels = [f"{age}\n(n={age_counts.get(age, 0)})" for age in df['age_group'].cat.categories]
            else:
                # Get unique age groups and handle mixed types safely
                unique_ages = df['age_group'].dropna().unique()
                # Convert to strings for consistent sorting
                age_order = ['<18', '18-35', '35-50', '50-65', '65+']
                # Only use ages that exist in data
                labels = [f"{age}\n(n={age_counts.get(age, 0)})" 
                         for age in age_order if age in unique_ages]
            
            if labels:  # Only set labels if we have any
                ax.set_xticklabels(labels)
            
            ax.set_title('Documentation Length by Age Group', fontsize=14, fontweight='bold')
            ax.set_ylabel('Text Length (characters)', fontsize=12)
            ax.set_xlabel('Age Group', fontsize=12)
            plt.suptitle('')
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, 'text_length_by_age.png'), dpi=300, bbox_inches='tight')
            plt.close()
            logger.info("Created visualization: text_length_by_age.png")
        
        # ====================================================================
        # PLOT 2: TREATMENT INTENSITY (Shows clinical complexity explanation)
        # ====================================================================
        if 'treatment_intensity' in df.columns and 'age_group' in df.columns:
            fig, ax = plt.subplots(figsize=(10, 6))
            df.boxplot(column='treatment_intensity', by='age_group', ax=ax)
            ax.set_title('Treatment Intensity by Age Group\n(Clinical complexity indicator)', 
                        fontsize=12, fontweight='bold')
            ax.set_ylabel('Treatment Intensity Score', fontsize=11)
            ax.set_xlabel('Age Group', fontsize=11)
            plt.suptitle('')
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, 'treatment_intensity_by_age.png'), dpi=300, bbox_inches='tight')
            plt.close()
            logger.info("Created visualization: treatment_intensity_by_age.png")
        
        # ====================================================================
        # PLOT 3: ABNORMAL LABS (Shows another complexity indicator)
        # ====================================================================
        if 'abnormal_lab_count' in df.columns and 'age_group' in df.columns:
            fig, ax = plt.subplots(figsize=(10, 6))
            df.boxplot(column='abnormal_lab_count', by='age_group', ax=ax)
            ax.set_title('Abnormal Lab Count by Age Group\n(Clinical complexity indicator)', 
                        fontsize=12, fontweight='bold')
            ax.set_ylabel('Abnormal Lab Count', fontsize=11)
            ax.set_xlabel('Age Group', fontsize=11)
            plt.suptitle('')
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, 'abnormal_labs_by_age.png'), dpi=300, bbox_inches='tight')
            plt.close()
            logger.info("Created visualization: abnormal_labs_by_age.png")
        
        # ====================================================================
        # PLOT 4: DOCUMENTATION QUALITY PARITY (Most important check)
        # ====================================================================
        if 'documentation_completeness' in df.columns and 'ethnicity_clean' in df.columns:
            fig, ax = plt.subplots(figsize=(12, 6))
            completeness_means = df.groupby('ethnicity_clean')['documentation_completeness'].mean()
            completeness_means.plot(kind='bar', ax=ax, color='steelblue')
            ax.set_title('Documentation Completeness by Ethnicity\n(Quality parity assessment)', 
                        fontsize=12, fontweight='bold')
            ax.set_ylabel('Completeness Score (0-1)', fontsize=11)
            ax.set_xlabel('Ethnicity', fontsize=11)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
            ax.set_ylim([0, 1])
            ax.axhline(y=completeness_means.mean(), color='red', linestyle='--', 
                      label=f'Overall Mean: {completeness_means.mean():.3f}')
            ax.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, 'completeness_by_ethnicity.png'), dpi=300, bbox_inches='tight')
            plt.close()
            logger.info("Created visualization: completeness_by_ethnicity.png")
        
        logger.info(f"All bias visualizations saved to {plots_dir}/")
    
    # ========================================================================
    # PIPELINE ORCHESTRATION
    # ========================================================================
    # Coordinates all three stages and generates reports
    # ========================================================================
    
    def run_bias_detection_pipeline(self) -> Tuple[Dict, pd.DataFrame]:
        """
        Execute complete bias detection pipeline with all three stages
        
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        PIPELINE EXECUTION ORDER:
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        
        1. Load and Validate Data
           - Load mimic_features.csv
           - Validate sufficient sample size
           - Document demographic distributions
        
        2. Stage 1: Raw Bias Detection
           - Documentation bias (text length)
           - Clinical risk bias
           - Treatment complexity bias
           - Lab testing bias
           - Readability bias
           - Calculate summary metrics (CV)
        
        3. Stage 2: Adjusted Bias Analysis
           - Build regression model
           - Control for clinical complexity
           - Analyze residuals
           - Determine LEGITIMATE vs BIAS
        
        4. Stage 3: Quality Parity Analysis
           - Check completeness scores
           - Verify essential sections present
           - Identify disparities
        
        5. Generate Interpretations
           - Convert stats to plain English
           - Provide actionable recommendations
           - Assign verdicts (PASS/FAIL/CONCERN)
        
        6. Create Visualizations
           - Documentation length plots
           - Complexity indicator plots
           - Quality parity plots
        
        7. Save Reports
           - bias_report.json (detailed)
           - bias_summary.csv (executive summary)
           - bias_plots/ (visualizations)
        
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        
        Returns:
            Tuple containing:
            - Complete bias report dictionary with all analysis results
            - Summary DataFrame for quick review of key findings
        """
        logger.info("="*60)
        logger.info("COMPREHENSIVE BIAS DETECTION PIPELINE")
        logger.info("="*60)
        
        # ====================================================================
        # STEP 1: LOAD AND VALIDATE DATA
        # ====================================================================
        df = self.load_data()
        
        # Initialize bias report structure
        bias_report = {
            'timestamp': datetime.now().isoformat(),
            'total_records_analyzed': len(df),
            'total_features_analyzed': len(df.columns),
            'demographic_distribution': {},
            'bias_thresholds': self.thresholds,
            'analysis_version': 'comprehensive_v1.0'
        }
        
        # Document demographic distributions for transparency
        logger.info("Analyzing demographic distributions...")
        for col in ['gender', 'ethnicity_clean', 'age_group']:
            if col in df.columns:
                distribution = df[col].value_counts().to_dict()
                distribution = {str(k): int(v) for k, v in distribution.items()}
                bias_report['demographic_distribution'][col] = distribution
        
        # ====================================================================
        # STEP 2: STAGE 1 - RAW BIAS DETECTION
        # ====================================================================
        logger.info("Stage 1: Detecting raw bias patterns...")
        bias_report['documentation_bias'] = self.detect_documentation_bias(df)
        bias_report['clinical_risk_bias'] = self.detect_clinical_risk_bias(df)
        bias_report['treatment_complexity_bias'] = self.detect_treatment_complexity_bias(df)
        bias_report['lab_testing_bias'] = self.detect_lab_testing_bias(df)
        bias_report['readability_bias'] = self.detect_readability_bias(df)
        bias_report['summary_metrics'] = self.calculate_bias_metrics(df)
        
        # ====================================================================
        # STEP 3: STAGE 2 - ADJUSTED BIAS ANALYSIS
        # ====================================================================
        logger.info("Stage 2: Performing adjusted bias analysis with complexity controls...")
        bias_report['adjusted_bias_analysis'] = self.detect_adjusted_bias(df)
        
        # ====================================================================
        # STEP 4: STAGE 3 - QUALITY PARITY ASSESSMENT
        # ====================================================================
        logger.info("Stage 3: Assessing documentation quality parity...")
        bias_report['quality_parity_analysis'] = self.detect_documentation_quality_parity(df)
        
        # ====================================================================
        # STEP 5: GENERATE INTERPRETATIONS
        # ====================================================================
        logger.info("Stage 4: Generating actionable interpretations...")
        bias_report['interpretations'] = self.interpret_bias_findings(bias_report)
        
        # ====================================================================
        # STEP 6: CREATE VISUALIZATIONS
        # ====================================================================
        logger.info("Stage 5: Creating visualization suite...")
        self.create_bias_visualizations(df)
        
        # ====================================================================
        # STEP 7: SAVE REPORTS
        # ====================================================================
        
        # Convert numpy types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, pd.Series):
                return obj.to_dict()
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            return obj
        
        bias_report = convert_numpy(bias_report)
        
        # Save complete bias report (JSON)
        report_path = os.path.join(self.output_path, 'bias_report.json')
        with open(report_path, 'w') as f:
            json.dump(bias_report, f, indent=2, default=str)
        logger.info(f"Complete bias report saved to {report_path}")
        
        # Create and save summary (CSV)
        summary_df = self.create_bias_summary(bias_report)
        summary_path = os.path.join(self.output_path, 'bias_summary.csv')
        summary_df.to_csv(summary_path, index=False)
        logger.info(f"Summary report saved to {summary_path}")
        
        logger.info("="*60)
        logger.info("BIAS DETECTION PIPELINE COMPLETE")
        logger.info("="*60)
        
        # ====================================================================
        # STEP 8: PRINT INTERPRETATIONS TO CONSOLE
        # ====================================================================
        # Print interpretations for immediate review
        if 'interpretations' in bias_report:
            print("\n" + "="*70)
            print("BIAS ANALYSIS INTERPRETATIONS")
            print("="*70)
            for key, interpretation in bias_report['interpretations'].items():
                print(f"\n{interpretation}\n")
                print("-"*70)
        
        return bias_report, summary_df
    
    def create_bias_summary(self, report: Dict) -> pd.DataFrame:
        """
        Create concise summary table from comprehensive bias report
        
        Extracts key findings into a structured table format for quick review
        and executive reporting. Includes raw metrics, adjusted metrics, and
        quality parity assessments.
        
        Args:
            report: Complete bias report dictionary
            
        Returns:
            DataFrame with summary of all key bias metrics and assessments
        """
        summary_data = {
            'Analysis_Stage': [],
            'Metric': [],
            'Value': [],
            'Alert': [],
            'Interpretation': []
        }
        
        # ====================================================================
        # ADD RAW BIAS METRICS
        # ====================================================================
        if 'summary_metrics' in report:
            for metric, value in report['summary_metrics'].items():
                if not metric.endswith('_alert'):
                    summary_data['Analysis_Stage'].append('Raw Bias')
                    summary_data['Metric'].append(metric)
                    summary_data['Value'].append(f"{value:.2f}%" if isinstance(value, float) else value)
                    
                    alert_key = f"{metric.split('_')[0]}_bias_alert"
                    alert_status = report['summary_metrics'].get(alert_key, False)
                    summary_data['Alert'].append('YES' if alert_status else 'NO')
                    summary_data['Interpretation'].append('See detailed interpretations section')
        
        # ====================================================================
        # ADD ADJUSTED BIAS METRICS
        # ====================================================================
        if 'adjusted_bias_analysis' in report:
            adjusted = report['adjusted_bias_analysis']
            if adjusted.get('analysis_performed'):
                # Add model performance
                summary_data['Analysis_Stage'].append('Adjusted Bias')
                summary_data['Metric'].append('model_r2_score')
                summary_data['Value'].append(f"{adjusted['model_r2_score']:.3f}")
                summary_data['Alert'].append('INFO')
                summary_data['Interpretation'].append(adjusted.get('model_explanation', ''))
                
                # Add residual CVs
                for demo, stats in adjusted.get('residual_analysis', {}).items():
                    summary_data['Analysis_Stage'].append('Adjusted Bias')
                    summary_data['Metric'].append(f"{demo}_residual_cv")
                    summary_data['Value'].append(f"{stats['cv']:.2f}%")
                    summary_data['Alert'].append('YES' if stats['interpretation'] == 'POTENTIAL_BIAS' else 'NO')
                    summary_data['Interpretation'].append(stats.get('explanation', ''))
        
        # ====================================================================
        # ADD QUALITY PARITY METRICS
        # ====================================================================
        if 'quality_parity_analysis' in report:
            for demo, metrics in report['quality_parity_analysis'].items():
                for metric_name, metric_data in metrics.items():
                    if metric_name == 'documentation_completeness':
                        summary_data['Analysis_Stage'].append('Quality Parity')
                        summary_data['Metric'].append(f"{demo}_quality_cv")
                        summary_data['Value'].append(f"{metric_data['cv']:.2f}%")
                        summary_data['Alert'].append('YES' if metric_data['assessment'] == 'DISPARITY' else 'NO')
                        summary_data['Interpretation'].append(metric_data.get('interpretation', ''))
        
        return pd.DataFrame(summary_data)


# ============================================================================
# MAIN EXECUTION BLOCK
# ============================================================================
# This runs when you execute from the project root:
#
#   python data-pipeline/scripts/bias_detection.py
#
# IMPORTANT: Always run this command from the project root directory!
#
# What happens:
# 1. Loads pipeline_config.json for paths and thresholds
# 2. Initializes MIMICBiasDetector class
# 3. Executes complete three-stage bias detection pipeline
# 4. Prints interpretations to console
# 5. Saves detailed reports
#
# Expected runtime: 2-5 minutes for ~9,600 records
#
# Output files:
# - logs/bias_report.json (comprehensive analysis)
# - logs/bias_summary.csv (executive summary)
# - logs/bias_plots/*.png (visualizations)
# ============================================================================

if __name__ == "__main__":
    """
    Main Execution Block
    ====================
    This runs when you execute from the project root:
    
      python data-pipeline/scripts/bias_detection.py
    
    IMPORTANT: Always run this command from the project root directory!
    
    What happens:
    1. Loads pipeline_config.json for paths and thresholds
    2. Initializes MIMICBiasDetector class
    3. Executes complete three-stage bias detection pipeline
    4. Prints interpretations to console
    5. Saves detailed reports
    
    Expected runtime: 2-5 minutes for ~9,600 records
    """
    import json
    
    # Load pipeline configuration
    config_path = os.path.join(os.path.dirname(__file__), '../configs/pipeline_config.json')
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Initialize bias detector
    detector = MIMICBiasDetector(
        input_path=config['pipeline_config']['output_path'],
        output_path=config['pipeline_config']['logs_path'],
        config=config
    )
    
    # Execute bias detection pipeline
    report, summary = detector.run_bias_detection_pipeline()
    
    # ========================================================================
    # Print Summary to Console
    # ========================================================================
    print("\n" + "="*70)
    print("BIAS DETECTION RESULTS SUMMARY")
    print("="*70)
    print(f"Records Analyzed: {report['total_records_analyzed']}")
    print(f"Features Analyzed: {report['total_features_analyzed']}")
    
    if 'summary_metrics' in report:
        print("\nRaw Bias Metrics:")
        for key in ['overall_bias_score', 'gender_cv', 'ethnicity_cv', 'age_cv']:
            if key in report['summary_metrics']:
                value = report['summary_metrics'][key]
                alert_key = f"{key.split('_')[0]}_bias_alert"
                alert = report['summary_metrics'].get(alert_key, False)
                print(f"  {key}: {value:.2f}% {'[ALERT]' if alert else '[OK]'}")
    
    if 'adjusted_bias_analysis' in report:
        adjusted = report['adjusted_bias_analysis']
        if adjusted.get('analysis_performed'):
            print("\nAdjusted Bias Analysis:")
            print(f"  Model R-squared: {adjusted['model_r2_score']:.3f}")
            for demo, stats in adjusted.get('residual_analysis', {}).items():
                print(f"  {demo} residual CV: {stats['cv']:.2f}% [{stats['interpretation']}]")
    
    print("\nOutput Files:")
    print(f"  Bias Report: {config['pipeline_config']['logs_path']}/bias_report.json")
    print(f"  Summary Table: {config['pipeline_config']['logs_path']}/bias_summary.csv")
    print(f"  Visualizations: {config['pipeline_config']['logs_path']}/bias_plots/")
    print("="*70)
    
    print("\nDetailed Summary Table:")
    print(summary.to_string(index=False))