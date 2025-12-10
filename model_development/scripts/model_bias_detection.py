#!/usr/bin/env python3
"""
Model Bias Detection Using Slicing Techniques
Evaluates model performance across demographic groups to detect bias
"""

import os
import sys
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import json

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.training.model_validation import ModelValidator
from src.utils.logging_config import get_logger
from src.utils.error_handling import ErrorHandler, safe_execute

logger = get_logger(__name__)

try:
    from fairlearn.metrics import (
        demographic_parity_difference,
        equalized_odds_difference,
        MetricFrame
    )
    FAIRLEARN_AVAILABLE = True
except ImportError:
    # Only log warning once at module level, not on every import
    FAIRLEARN_AVAILABLE = False


class ModelBiasDetector:
    """
    Model bias detection using slicing techniques across demographic groups
    """
    
    def __init__(self, bias_threshold: float = 0.1):
        """
        Initialize bias detector
        
        Args:
            bias_threshold: Threshold for flagging bias (default: 0.1 = 10% difference)
        """
        self.bias_threshold = bias_threshold
        self.validator = ModelValidator()
        self.error_handler = ErrorHandler(logger)
    
    @safe_execute("detect_bias", logger, ErrorHandler(logger))
    def detect_bias(
        self,
        df: pd.DataFrame,
        prediction_column: str = 'gemini_summary',
        reference_column: str = 'cleaned_text',
        demographic_columns: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Detect bias across demographic slices
        
        Args:
            df: DataFrame with predictions, references, and demographics
            prediction_column: Column with model predictions
            reference_column: Column with reference summaries
            demographic_columns: List of demographic columns to slice by
                              (default: ['gender', 'ethnicity_clean', 'age_group'])
            
        Returns:
            Dictionary with bias detection results
        """
        if demographic_columns is None:
            demographic_columns = ['gender', 'ethnicity_clean', 'age_group']
        
        # Filter to valid predictions
        valid_df = df[
            df[prediction_column].notna() &
            df[reference_column].notna() &
            (df[prediction_column].astype(str).str.strip() != '') &
            (df[reference_column].astype(str).str.strip() != '')
        ].copy()
        
        if len(valid_df) == 0:
            raise ValueError("No valid predictions found for bias detection")
        
        logger.info(f"Analyzing bias for {len(valid_df)} samples")
        
        bias_report = {
            'total_samples': len(valid_df),
            'demographic_slices': {},
            'overall_metrics': {},
            'bias_alerts': []
        }
        
        # Calculate overall metrics
        predictions = valid_df[prediction_column].astype(str).tolist()
        references = valid_df[reference_column].astype(str).tolist()
        overall_metrics = self.validator.validate_model(predictions, references)
        bias_report['overall_metrics'] = overall_metrics
        
        # Analyze each demographic slice
        for demo_col in demographic_columns:
            if demo_col not in valid_df.columns:
                logger.warning(f"Demographic column '{demo_col}' not found, skipping")
                continue
            
            slice_results = self._analyze_slice(
                valid_df, demo_col, prediction_column, reference_column
            )
            bias_report['demographic_slices'][demo_col] = slice_results
            
            # Check for bias alerts
            alerts = self._check_bias_alerts(slice_results, demo_col)
            bias_report['bias_alerts'].extend(alerts)
        
        # Calculate overall bias score
        bias_report['overall_bias_score'] = self._calculate_overall_bias_score(
            bias_report['demographic_slices']
        )
        
        logger.info(f"Bias detection completed. Found {len(bias_report['bias_alerts'])} alerts")
        return bias_report
    
    def _analyze_slice(
        self,
        df: pd.DataFrame,
        demographic_column: str,
        prediction_column: str,
        reference_column: str
    ) -> Dict[str, Any]:
        """
        Analyze model performance for a specific demographic slice
        
        Args:
            df: DataFrame with data
            demographic_column: Column to slice by
            prediction_column: Column with predictions
            reference_column: Column with references
            
        Returns:
            Dictionary with slice analysis results
        """
        slice_results = {
            'groups': {},
            'group_counts': {},
            'metrics_by_group': {}
        }
        
        # Get unique groups
        groups = df[demographic_column].dropna().unique()
        
        for group in groups:
            group_df = df[df[demographic_column] == group]
            group_count = len(group_df)
            
            if group_count < 5:  # Skip groups with too few samples
                logger.warning(f"Skipping {group} in {demographic_column}: only {group_count} samples")
                continue
            
            slice_results['group_counts'][str(group)] = group_count
            
            # Calculate metrics for this group
            predictions = group_df[prediction_column].astype(str).tolist()
            references = group_df[reference_column].astype(str).tolist()
            
            group_metrics = self.validator.validate_model(predictions, references)
            slice_results['metrics_by_group'][str(group)] = group_metrics
            
            # Store detailed results
            slice_results['groups'][str(group)] = {
                'count': group_count,
                'rouge1_f': group_metrics.get('rouge1_f', 0.0),
                'rouge2_f': group_metrics.get('rouge2_f', 0.0),
                'rougeL_f': group_metrics.get('rougeL_f', 0.0),
                'bleu': group_metrics.get('bleu', 0.0),
                'overall_score': group_metrics.get('overall_score', 0.0)
            }
        
        # Calculate disparities
        if len(slice_results['groups']) > 1:
            disparities = self._calculate_disparities(slice_results['groups'])
            slice_results['disparities'] = disparities
        
        return slice_results
    
    def _calculate_disparities(self, groups: Dict[str, Dict]) -> Dict[str, float]:
        """
        Calculate performance disparities between groups
        
        Args:
            groups: Dictionary with group metrics
            
        Returns:
            Dictionary with disparity metrics
        """
        disparities = {}
        
        # Get all scores for each metric
        metrics = ['rouge1_f', 'rouge2_f', 'rougeL_f', 'bleu', 'overall_score']
        
        for metric in metrics:
            scores = [group_data.get(metric, 0.0) for group_data in groups.values()]
            if len(scores) > 1 and max(scores) > 0:
                # Calculate coefficient of variation
                mean_score = np.mean(scores)
                std_score = np.std(scores)
                cv = (std_score / mean_score) * 100 if mean_score > 0 else 0
                
                # Calculate max difference
                max_diff = max(scores) - min(scores)
                max_diff_pct = (max_diff / max(scores)) * 100 if max(scores) > 0 else 0
                
                disparities[f'{metric}_cv'] = float(cv)
                disparities[f'{metric}_max_diff'] = float(max_diff)
                disparities[f'{metric}_max_diff_pct'] = float(max_diff_pct)
        
        return disparities
    
    def _check_bias_alerts(
        self,
        slice_results: Dict[str, Any],
        demographic_column: str
    ) -> List[Dict[str, Any]]:
        """
        Check for bias alerts based on disparities
        
        Args:
            slice_results: Results from slice analysis
            demographic_column: Name of demographic column
            
        Returns:
            List of bias alerts
        """
        alerts = []
        
        if 'disparities' not in slice_results:
            return alerts
        
        disparities = slice_results['disparities']
        
        # Check overall score disparity
        if 'overall_score_max_diff_pct' in disparities:
            diff_pct = disparities['overall_score_max_diff_pct']
            if diff_pct > (self.bias_threshold * 100):
                alerts.append({
                    'demographic': demographic_column,
                    'metric': 'overall_score',
                    'disparity_pct': diff_pct,
                    'threshold': self.bias_threshold * 100,
                    'severity': 'high' if diff_pct > 20 else 'medium',
                    'message': f"Performance disparity of {diff_pct:.1f}% detected across {demographic_column} groups"
                })
        
        # Check ROUGE-L disparity
        if 'rougeL_f_max_diff_pct' in disparities:
            diff_pct = disparities['rougeL_f_max_diff_pct']
            if diff_pct > (self.bias_threshold * 100):
                alerts.append({
                    'demographic': demographic_column,
                    'metric': 'rougeL_f',
                    'disparity_pct': diff_pct,
                    'threshold': self.bias_threshold * 100,
                    'severity': 'high' if diff_pct > 20 else 'medium',
                    'message': f"ROUGE-L disparity of {diff_pct:.1f}% detected across {demographic_column} groups"
                })
        
        return alerts
    
    def _calculate_overall_bias_score(self, slices: Dict[str, Dict]) -> float:
        """
        Calculate overall bias score from all slices
        
        Args:
            slices: Dictionary with slice results
            
        Returns:
            Overall bias score (0-1, higher = more bias)
        """
        all_cvs = []
        
        for demo_col, slice_data in slices.items():
            if 'disparities' in slice_data:
                disparities = slice_data['disparities']
                if 'overall_score_cv' in disparities:
                    all_cvs.append(disparities['overall_score_cv'])
        
        if not all_cvs:
            return 0.0
        
        # Normalize CV to 0-1 scale (assuming max CV of 50%)
        normalized_cvs = [min(cv / 50.0, 1.0) for cv in all_cvs]
        return float(np.mean(normalized_cvs))
    
    def save_bias_report(self, bias_report: Dict[str, Any], output_path: str):
        """
        Save bias detection report to JSON
        
        Args:
            bias_report: Bias detection results
            output_path: Path to save report
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(bias_report, f, indent=2, default=str)
        
        logger.info(f"Saved bias report to {output_path}")

