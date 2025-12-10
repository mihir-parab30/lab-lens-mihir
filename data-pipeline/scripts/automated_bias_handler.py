"""
Intelligent Bias Mitigation Pipeline for MIMIC-III Data
Author: Lab Lens Team
Description: Smart bias mitigation using adjusted bias analysis results

This module provides intelligent bias mitigation that:
1. Reads adjusted bias analysis from detection pipeline
2. Determines if mitigation is appropriate (legitimate vs problematic variation)
3. Applies appropriate strategy based on bias type and severity
4. Validates mitigation effectiveness

Key Principle: Only mitigate problematic bias, not legitimate clinical variation.
"""

import pandas as pd
import numpy as np
import json
import os
import sys
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime
from pathlib import Path
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


class IntelligentBiasHandler:
    """
    Intelligent bias mitigation system that distinguishes between
    legitimate clinical variation and problematic documentation bias.
    
    Decision Framework:
    - If adjusted CV under 5 percent: Skip mitigation (variation is legitimate)
    - If adjusted CV 5 to 20 percent: Apply feature normalization
    - If adjusted CV over 20 percent: Apply normalization and flag for review
    - If quality disparity exists: Always apply quality-focused mitigation
    
    Mitigation never involves oversampling or balancing for continuous variables
    like text length, as these approaches are statistically inappropriate.
    """
    
    def __init__(self, input_path: str = 'data/processed', output_path: str = 'logs', config: Dict = None):
        """
        Initialize intelligent bias handler
        
        Args:
            input_path: Path to processed data directory
            output_path: Path to save mitigation reports
            config: Configuration dictionary with bias thresholds
        """
        self.input_path = input_path
        self.output_path = output_path
        self.error_handler = ErrorHandler(logger)
        
        try:
            os.makedirs(output_path, exist_ok=True)
            logger.info(f"Initialized intelligent bias handler with input: {input_path}, output: {output_path}")
        except Exception as e:
            raise self.error_handler.handle_file_error("directory_creation", output_path, e)
        
        # Load configuration with comprehensive defaults
        default_thresholds = {
            'gender_cv_max': 5.0,
            'ethnicity_cv_max': 10.0,
            'age_cv_max': 20.0,
            'residual_cv_threshold': 5.0,
            'residual_cv_moderate': 20.0,
            'quality_cv_max': 10.0,
            'overall_bias_score_max': 10.0,
            'min_sample_size': 30,
            'significance_threshold': 0.05
        }
        
        if config and 'bias_detection_config' in config:
            config_thresholds = config['bias_detection_config'].get('alert_thresholds', {})
            self.thresholds = {**default_thresholds, **config_thresholds}
            self.mitigation_config = config['bias_detection_config']
        else:
            self.thresholds = default_thresholds
            self.mitigation_config = {
                'auto_mitigation': True,
                'mitigation_enabled': True
            }
        
        logger.info(f"Using bias mitigation thresholds: {self.thresholds}")
    
    def load_bias_report(self, filename: str = 'bias_report.json') -> Dict[str, Any]:
        """
        Load comprehensive bias detection report from detection pipeline
        
        This report should contain:
        - Raw bias metrics
        - Adjusted bias analysis results
        - Quality parity analysis
        - Interpretations
        
        Args:
            filename: Name of bias report JSON file
            
        Returns:
            Dictionary with complete bias detection results
            
        Raises:
            FileNotFoundError: If bias report does not exist
        """
        filepath = os.path.join(self.output_path, filename)
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(
                f"Bias report not found at {filepath}. "
                f"Please run bias_detection.py first to generate the report."
            )
        
        logger.info(f"Loading bias report from {filepath}")
        with open(filepath, 'r') as f:
            bias_report = json.load(f)
        
        # Validate report structure
        required_sections = ['summary_metrics', 'adjusted_bias_analysis']
        missing_sections = [s for s in required_sections if s not in bias_report]
        
        if missing_sections:
            logger.warning(
                f"Bias report missing sections: {missing_sections}. "
                f"Mitigation may be limited."
            )
        
        return bias_report
    
    def load_features_data(self, filename: str = 'mimic_features.csv') -> pd.DataFrame:
        """
        Load feature-engineered data for mitigation
        
        Args:
            filename: Name of features CSV file
            
        Returns:
            DataFrame with features
            
        Raises:
            FileNotFoundError: If features file does not exist
        """
        filepath = os.path.join(self.input_path, filename)
        
        validate_file_path(filepath, logger, must_exist=True)
        
        logger.info(f"Loading features data from {filepath}")
        df = pd.read_csv(filepath)
        
        # Validate required columns exist
        required_cols = ['hadm_id', 'subject_id', 'text_chars']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            raise ValueError(f"Required columns missing from features data: {missing_cols}")
        
        logger.info(f"Loaded {len(df)} records with {len(df.columns)} features")
        
        return df
    
    def determine_mitigation_strategy(self, bias_report: Dict[str, Any]) -> Dict[str, Any]:
        """
        Intelligently determine appropriate mitigation strategy based on adjusted analysis
        
        Decision Logic:
        1. Check if adjusted bias analysis was performed
        2. If adjusted CV under threshold: No mitigation needed (legitimate variation)
        3. If adjusted CV moderate: Apply feature normalization
        4. If adjusted CV high: Apply normalization and flag for institutional review
        5. If quality disparity: Apply quality-focused mitigation regardless
        
        Args:
            bias_report: Complete bias detection report
            
        Returns:
            Dictionary with mitigation strategy and reasoning
        """
        strategy = {
            'action': 'none',
            'methods': [],
            'reasoning': '',
            'requires_review': False,
            'demographic_targets': []
        }
        
        # Extract adjusted bias analysis
        adjusted = bias_report.get('adjusted_bias_analysis', {})
        
        if not adjusted.get('analysis_performed', False):
            logger.warning("Adjusted bias analysis not available, using conservative approach")
            strategy['action'] = 'conservative'
            strategy['reasoning'] = (
                "Adjusted bias analysis unavailable. Applying conservative mitigation "
                "as precaution. Consider adding clinical complexity features for better analysis."
            )
            strategy['methods'] = ['feature_normalization']
            return strategy
        
        # Check residual analysis for each demographic
        residual_analysis = adjusted.get('residual_analysis', {})
        
        for demo_col, demo_stats in residual_analysis.items():
            residual_cv = demo_stats.get('cv', 0)
            interpretation = demo_stats.get('interpretation', 'LEGITIMATE_VARIATION')
            
            if interpretation == 'LEGITIMATE_VARIATION':
                # Variation explained by clinical complexity - no mitigation needed
                logger.info(
                    f"{demo_col}: Residual CV {residual_cv:.2f}% indicates legitimate variation. "
                    f"No mitigation required."
                )
                continue
            
            # Problematic bias detected - determine severity
            if residual_cv < self.thresholds['residual_cv_threshold']:
                # Under threshold despite being flagged - likely edge case
                logger.info(
                    f"{demo_col}: Residual CV {residual_cv:.2f}% near threshold. "
                    f"Monitoring recommended but mitigation optional."
                )
                
            elif residual_cv < self.thresholds['residual_cv_moderate']:
                # Moderate bias - apply feature normalization
                logger.warning(
                    f"{demo_col}: Residual CV {residual_cv:.2f}% indicates moderate bias. "
                    f"Applying feature normalization."
                )
                strategy['action'] = 'moderate'
                strategy['methods'].append('feature_normalization')
                strategy['demographic_targets'].append(demo_col)
                strategy['reasoning'] += (
                    f"Moderate unexplained bias detected for {demo_col} (CV: {residual_cv:.1f}%). "
                )
                
            else:
                # High bias - apply normalization and flag for review
                logger.error(
                    f"{demo_col}: Residual CV {residual_cv:.2f}% indicates severe bias. "
                    f"Applying mitigation and flagging for institutional review."
                )
                strategy['action'] = 'severe'
                strategy['methods'].append('feature_normalization')
                strategy['demographic_targets'].append(demo_col)
                strategy['requires_review'] = True
                strategy['reasoning'] += (
                    f"Severe unexplained bias detected for {demo_col} (CV: {residual_cv:.1f}%). "
                    f"Institutional review required. "
                )
        
        # Check quality parity
        quality_analysis = bias_report.get('quality_parity_analysis', {})
        
        for demo_col, quality_metrics in quality_analysis.items():
            for metric_name, metric_data in quality_metrics.items():
                if metric_data.get('assessment') == 'DISPARITY':
                    logger.warning(
                        f"Quality disparity detected for {demo_col}: {metric_name}. "
                        f"Adding quality-focused mitigation."
                    )
                    if 'quality_monitoring' not in strategy['methods']:
                        strategy['methods'].append('quality_monitoring')
                    strategy['reasoning'] += (
                        f"Quality disparity in {metric_name} for {demo_col}. "
                    )
        
        # Set default reasoning if none provided
        if not strategy['reasoning']:
            strategy['reasoning'] = (
                "No significant problematic bias detected. All variation appears to be "
                "legitimate clinical variation or within acceptable thresholds."
            )
        
        logger.info(f"Mitigation strategy determined: {strategy['action']}")
        
        return strategy
    
    def normalize_features_by_demographics(self, 
                                          df: pd.DataFrame, 
                                          demo_cols: List[str]) -> pd.DataFrame:
        """
        Normalize features to remove demographic effects while preserving clinical information
        
        This method adjusts features to remove systematic demographic differences
        while maintaining the relationship between clinical complexity and feature values.
        
        Approach:
        1. Calculate mean feature value for each demographic group
        2. Calculate overall mean across all groups
        3. Adjust each record: new_value = original_value - group_mean + overall_mean
        
        This centers all demographic groups at the same mean while preserving
        within-group variation and clinical patterns.
        
        Args:
            df: Input DataFrame
            demo_cols: List of demographic columns to normalize across
            
        Returns:
            DataFrame with normalized features
        """
        df_normalized = df.copy()
        
        # Features to normalize (typically continuous measures affected by bias)
        features_to_normalize = ['text_chars', 'text_tokens', 'word_count']
        available_features = [f for f in features_to_normalize if f in df.columns]
        
        if not available_features:
            logger.warning("No features available for normalization")
            return df_normalized
        
        logger.info(f"Normalizing {len(available_features)} features across {len(demo_cols)} demographics")
        
        for feature in available_features:
            for demo_col in demo_cols:
                if demo_col not in df.columns:
                    continue
                
                # Calculate group means
                group_means = df.groupby(demo_col)[feature].transform('mean')
                overall_mean = df[feature].mean()
                
                # Normalize: adjust to center all groups at overall mean
                # This removes demographic effect while preserving within-group patterns
                df_normalized[feature] = df[feature] - group_means + overall_mean
                
                # Log normalization results
                original_cv = (df.groupby(demo_col)[feature].mean().std() / 
                              df.groupby(demo_col)[feature].mean().mean()) * 100
                normalized_cv = (df_normalized.groupby(demo_col)[feature].mean().std() / 
                                df_normalized.groupby(demo_col)[feature].mean().mean()) * 100
                
                logger.info(
                    f"Normalized {feature} by {demo_col}: "
                    f"CV reduced from {original_cv:.2f}% to {normalized_cv:.2f}%"
                )
        
        # Add normalization metadata
        df_normalized['features_normalized'] = True
        df_normalized['normalization_demographics'] = ','.join(demo_cols)
        df_normalized['normalization_timestamp'] = datetime.now().isoformat()
        
        return df_normalized
    
    def add_quality_monitoring_flags(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add flags for quality monitoring to ensure completeness across demographics
        
        Creates binary flags indicating whether critical documentation sections
        are present. These flags can be used for ongoing quality monitoring.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with quality monitoring flags added
        """
        df_flagged = df.copy()
        
        # Define critical sections for quality monitoring
        critical_sections = {
            'has_diagnosis': 'discharge_diagnosis',
            'has_medications': 'discharge_medications',
            'has_follow_up': 'follow_up'
        }
        
        # Create flags for each critical section
        for flag_name, section_col in critical_sections.items():
            if section_col in df.columns:
                # Flag is True if section exists and is not empty
                df_flagged[flag_name] = (
                    (~df[section_col].isna()) & 
                    (df[section_col].astype(str).str.len() > 10)
                ).astype(int)
                
                # Log quality statistics
                completeness_rate = df_flagged[flag_name].mean() * 100
                logger.info(f"Quality flag {flag_name}: {completeness_rate:.1f}% complete")
        
        # Calculate overall quality score
        flag_cols = [f for f in critical_sections.keys() if f in df_flagged.columns]
        if flag_cols:
            df_flagged['quality_monitoring_score'] = df_flagged[flag_cols].mean(axis=1)
        
        return df_flagged
    
    def apply_mitigation(self, 
                        df: pd.DataFrame, 
                        strategy: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Apply determined mitigation strategy to data
        
        Args:
            df: Input DataFrame
            strategy: Mitigation strategy dictionary
            
        Returns:
            Tuple of (mitigated DataFrame, mitigation details)
        """
        if strategy['action'] == 'none':
            logger.info("No mitigation applied - bias within acceptable limits or legitimate")
            return df, {'methods_applied': [], 'changes_made': []}
        
        mitigated_df = df.copy()
        mitigation_details = {
            'methods_applied': [],
            'changes_made': [],
            'features_modified': []
        }
        
        # Apply feature normalization if recommended
        if 'feature_normalization' in strategy['methods']:
            logger.info("Applying feature normalization to remove demographic effects")
            
            # Get demographic targets for normalization
            demo_targets = strategy.get('demographic_targets', ['age_group'])
            
            # Normalize features
            mitigated_df = self.normalize_features_by_demographics(
                mitigated_df, 
                demo_targets
            )
            
            mitigation_details['methods_applied'].append('feature_normalization')
            mitigation_details['changes_made'].append(
                f"Normalized features across {', '.join(demo_targets)}"
            )
            mitigation_details['features_modified'].extend(
                ['text_chars', 'text_tokens', 'word_count']
            )
        
        # Apply quality monitoring if recommended
        if 'quality_monitoring' in strategy['methods']:
            logger.info("Adding quality monitoring flags")
            
            mitigated_df = self.add_quality_monitoring_flags(mitigated_df)
            
            mitigation_details['methods_applied'].append('quality_monitoring')
            mitigation_details['changes_made'].append(
                "Added quality monitoring flags for critical sections"
            )
        
        # Add mitigation metadata
        mitigated_df['bias_mitigation_applied'] = True
        mitigated_df['mitigation_strategy'] = strategy['action']
        mitigated_df['mitigation_timestamp'] = datetime.now().isoformat()
        
        logger.info(f"Mitigation complete: {len(mitigation_details['methods_applied'])} methods applied")
        
        return mitigated_df, mitigation_details
    
    def calculate_post_mitigation_metrics(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate bias metrics after mitigation to verify effectiveness
        
        Recalculates coefficient of variation for text length across demographics
        to measure improvement from mitigation.
        
        Args:
            df: Mitigated DataFrame
            
        Returns:
            Dictionary with post-mitigation bias metrics
        """
        metrics = {}
        
        if 'text_chars' not in df.columns:
            logger.warning("text_chars column not found, cannot calculate post-mitigation metrics")
            return metrics
        
        # Calculate CV for each demographic
        for demo_col, metric_name in [
            ('gender', 'gender_cv'),
            ('ethnicity_clean', 'ethnicity_cv'),
            ('age_group', 'age_cv')
        ]:
            if demo_col not in df.columns:
                continue
            
            demo_means = df.groupby(demo_col)['text_chars'].mean()
            
            if len(demo_means) > 1 and demo_means.mean() > 0:
                cv = (demo_means.std() / demo_means.mean()) * 100
                metrics[metric_name] = float(cv)
        
        # Calculate overall bias score
        cv_values = [v for k, v in metrics.items() if k.endswith('_cv')]
        if cv_values:
            metrics['overall_bias_score'] = float(np.mean(cv_values))
        
        return metrics
    
    def generate_mitigation_report(self, 
                                   bias_report: Dict[str, Any],
                                   original_metrics: Dict[str, float],
                                   post_metrics: Dict[str, float],
                                   strategy: Dict[str, Any],
                                   mitigation_details: Dict[str, Any],
                                   mitigation_applied: bool) -> Dict[str, Any]:
        """
        Generate comprehensive mitigation report with analysis and recommendations
        
        Args:
            bias_report: Original bias detection report
            original_metrics: Bias metrics before mitigation
            post_metrics: Bias metrics after mitigation
            strategy: Mitigation strategy used
            mitigation_details: Details of mitigation applied
            mitigation_applied: Whether mitigation was actually applied
            
        Returns:
            Complete mitigation report dictionary
        """
        report = {
            'timestamp': datetime.now().isoformat(),
            'mitigation_applied': mitigation_applied,
            'strategy': strategy,
            'mitigation_details': mitigation_details,
            'before_mitigation': original_metrics,
            'after_mitigation': post_metrics if mitigation_applied else original_metrics,
            'improvement': {},
            'effectiveness_assessment': {},
            'recommendations': []
        }
        
        # Calculate improvement if mitigation was applied
        if mitigation_applied and post_metrics:
            for metric_name in ['overall_bias_score', 'gender_cv', 'ethnicity_cv', 'age_cv']:
                if metric_name in original_metrics and metric_name in post_metrics:
                    original_value = original_metrics[metric_name]
                    new_value = post_metrics[metric_name]
                    
                    if original_value > 0:
                        improvement_pct = ((original_value - new_value) / original_value) * 100
                    else:
                        improvement_pct = 0
                    
                    report['improvement'][metric_name] = {
                        'original': original_value,
                        'new': new_value,
                        'improvement_percentage': improvement_pct,
                        'absolute_reduction': original_value - new_value
                    }
        
        # Assess mitigation effectiveness
        if mitigation_applied:
            effectiveness = self._assess_mitigation_effectiveness(
                original_metrics,
                post_metrics,
                strategy
            )
            report['effectiveness_assessment'] = effectiveness
        
        # Generate recommendations based on results
        report['recommendations'] = self._generate_recommendations(
            bias_report,
            strategy,
            mitigation_applied,
            report.get('effectiveness_assessment', {})
        )
        
        return report
    
    def _assess_mitigation_effectiveness(self,
                                        original_metrics: Dict[str, float],
                                        post_metrics: Dict[str, float],
                                        strategy: Dict[str, Any]) -> Dict[str, Any]:
        """
        Assess how effective the mitigation was
        
        Args:
            original_metrics: Metrics before mitigation
            post_metrics: Metrics after mitigation
            strategy: Strategy that was applied
            
        Returns:
            Assessment dictionary
        """
        assessment = {
            'overall_effectiveness': 'unknown',
            'metrics_improved': [],
            'metrics_unchanged': [],
            'metrics_worsened': [],
            'verdict': ''
        }
        
        for metric_name in ['overall_bias_score', 'age_cv', 'gender_cv', 'ethnicity_cv']:
            if metric_name not in original_metrics or metric_name not in post_metrics:
                continue
            
            original = original_metrics[metric_name]
            new = post_metrics[metric_name]
            
            if new < original * 0.9:  # Improved by at least 10 percent
                assessment['metrics_improved'].append(metric_name)
            elif new > original * 1.1:  # Worsened by more than 10 percent
                assessment['metrics_worsened'].append(metric_name)
            else:
                assessment['metrics_unchanged'].append(metric_name)
        
        # Determine overall effectiveness
        if len(assessment['metrics_improved']) > len(assessment['metrics_worsened']):
            assessment['overall_effectiveness'] = 'effective'
            assessment['verdict'] = (
                "Mitigation was effective. Bias metrics improved across most dimensions "
                "without introducing new disparities."
            )
        elif len(assessment['metrics_worsened']) > 0:
            assessment['overall_effectiveness'] = 'partially_effective'
            assessment['verdict'] = (
                "Mitigation had mixed results. Some metrics improved but others worsened. "
                "Consider adjusting mitigation approach."
            )
        else:
            assessment['overall_effectiveness'] = 'minimal_impact'
            assessment['verdict'] = (
                "Mitigation had minimal impact on bias metrics. This may indicate the "
                "variation is inherent to the data structure rather than correctable bias."
            )
        
        return assessment
    
    def _generate_recommendations(self,
                                 bias_report: Dict[str, Any],
                                 strategy: Dict[str, Any],
                                 mitigation_applied: bool,
                                 effectiveness: Dict[str, Any]) -> List[str]:
        """
        Generate actionable recommendations based on mitigation results
        
        Args:
            bias_report: Original bias detection report
            strategy: Strategy that was used
            mitigation_applied: Whether mitigation was applied
            effectiveness: Effectiveness assessment
            
        Returns:
            List of recommendation strings
        """
        recommendations = []
        
        if not mitigation_applied:
            recommendations.append(
                "No mitigation applied. Detected variation appears to be legitimate "
                "clinical variation rather than problematic bias."
            )
            recommendations.append(
                "Continue regular monitoring to ensure variation remains appropriate."
            )
            recommendations.append(
                "Ensure ML models are trained to handle clinical complexity fairly."
            )
            return recommendations
        
        # Recommendations based on strategy
        if strategy.get('requires_review', False):
            recommendations.append(
                "CRITICAL: Severe bias detected requiring institutional review. "
                "Convene stakeholder meeting to address systematic documentation practices."
            )
            recommendations.append(
                "Conduct detailed audit of documentation procedures by demographic group."
            )
        
        # Recommendations based on effectiveness
        if effectiveness.get('overall_effectiveness') == 'effective':
            recommendations.append(
                "Mitigation was successful. Use mitigated dataset for ML model training."
            )
            recommendations.append(
                "Continue monitoring bias metrics with regular re-assessment."
            )
        elif effectiveness.get('overall_effectiveness') == 'partially_effective':
            recommendations.append(
                "Mitigation had mixed results. Consider additional approaches such as "
                "adversarial debiasing during model training."
            )
            recommendations.append(
                "Review mitigation parameters and consider more aggressive normalization."
            )
        else:
            recommendations.append(
                "Mitigation had minimal impact. This suggests the variation may be "
                "inherent to clinical practice patterns rather than correctable through "
                "feature engineering alone."
            )
            recommendations.append(
                "Focus on fairness constraints during model training rather than "
                "additional data preprocessing."
            )
        
        # Quality-related recommendations
        quality_analysis = bias_report.get('quality_parity_analysis', {})
        has_quality_issues = any(
            any(m.get('assessment') == 'DISPARITY' for m in metrics.values())
            for metrics in quality_analysis.values()
        )
        
        if has_quality_issues:
            recommendations.append(
                "Quality disparities detected. Implement standardized documentation "
                "templates or checklists to ensure completeness across all demographics."
            )
        
        return recommendations
    
    def run_mitigation_pipeline(self) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Execute complete intelligent bias mitigation pipeline
        
        Pipeline Steps:
        1. Load bias detection report with adjusted analysis
        2. Load features data
        3. Determine appropriate mitigation strategy based on adjusted bias
        4. Apply mitigation if warranted
        5. Calculate post-mitigation metrics
        6. Generate comprehensive report
        7. Save results
        
        Returns:
            Tuple of (mitigated DataFrame, mitigation report)
        """
        logger.info("="*60)
        logger.info("INTELLIGENT BIAS MITIGATION PIPELINE")
        logger.info("="*60)
        
        # Load bias detection report
        bias_report = self.load_bias_report()
        
        # Load features data
        df = self.load_features_data()
        original_size = len(df)
        
        # Extract original metrics
        original_metrics = bias_report.get('summary_metrics', {})
        
        # Determine intelligent mitigation strategy based on adjusted analysis
        logger.info("Analyzing adjusted bias results to determine mitigation strategy...")
        strategy = self.determine_mitigation_strategy(bias_report)
        
        logger.info(f"Strategy determined: {strategy['action']}")
        logger.info(f"Reasoning: {strategy['reasoning']}")
        
        # Apply mitigation based on strategy
        mitigation_applied = strategy['action'] != 'none'
        
        if mitigation_applied and self.mitigation_config.get('mitigation_enabled', True):
            logger.info("Applying intelligent bias mitigation...")
            mitigated_df, mitigation_details = self.apply_mitigation(df, strategy)
            
            # Calculate post-mitigation metrics
            logger.info("Calculating post-mitigation bias metrics...")
            post_metrics = self.calculate_post_mitigation_metrics(mitigated_df)
        else:
            logger.info("Mitigation not required or disabled")
            mitigated_df = df
            mitigation_details = {'methods_applied': [], 'changes_made': []}
            post_metrics = original_metrics
        
        # Generate comprehensive report
        logger.info("Generating mitigation report...")
        mitigation_report = self.generate_mitigation_report(
            bias_report,
            original_metrics,
            post_metrics,
            strategy,
            mitigation_details,
            mitigation_applied
        )
        
        # Save mitigated dataset if mitigation was applied
        if mitigation_applied:
            output_file = os.path.join(self.input_path, 'mimic_features_mitigated.csv')
            mitigated_df.to_csv(output_file, index=False)
            logger.info(f"Saved mitigated dataset: {output_file}")
            mitigation_report['mitigated_dataset_path'] = output_file
        
        # Save mitigation report
        report_file = os.path.join(self.output_path, 'bias_mitigation_report.json')
        with open(report_file, 'w') as f:
            json.dump(mitigation_report, f, indent=2, default=str)
        logger.info(f"Saved mitigation report: {report_file}")
        
        logger.info("="*60)
        logger.info("INTELLIGENT BIAS MITIGATION COMPLETE")
        logger.info("="*60)
        
        return mitigated_df, mitigation_report


if __name__ == "__main__":
    import json
    
    # Load pipeline configuration
    config_path = os.path.join(os.path.dirname(__file__), '../configs/pipeline_config.json')
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Initialize intelligent bias handler
    handler = IntelligentBiasHandler(
        input_path=config['pipeline_config']['output_path'],
        output_path=config['pipeline_config']['logs_path'],
        config=config
    )
    
    # Run intelligent mitigation pipeline
    mitigated_df, report = handler.run_mitigation_pipeline()
    
    # Print comprehensive results
    print("\n" + "="*70)
    print("INTELLIGENT BIAS MITIGATION RESULTS")
    print("="*70)
    print(f"Mitigation Applied: {report['mitigation_applied']}")
    print(f"Strategy: {report['strategy']['action']}")
    
    if report['mitigation_applied']:
        print("\nMitigation Details:")
        print(f"  Methods Applied: {', '.join(report['mitigation_details']['methods_applied'])}")
        print(f"  Changes Made:")
        for change in report['mitigation_details']['changes_made']:
            print(f"    - {change}")
        
        print("\nBias Score Changes:")
        if 'improvement' in report and 'overall_bias_score' in report['improvement']:
            imp = report['improvement']['overall_bias_score']
            print(f"  Before: {imp['original']:.2f}%")
            print(f"  After:  {imp['new']:.2f}%")
            print(f"  Improvement: {imp['improvement_percentage']:.1f}%")
        
        print("\nEffectiveness Assessment:")
        if 'effectiveness_assessment' in report:
            eff = report['effectiveness_assessment']
            print(f"  Overall: {eff['overall_effectiveness']}")
            print(f"  Verdict: {eff['verdict']}")
            if eff['metrics_improved']:
                print(f"  Improved: {', '.join(eff['metrics_improved'])}")
            if eff['metrics_worsened']:
                print(f"  Worsened: {', '.join(eff['metrics_worsened'])}")
    else:
        print("\nReasoning for No Mitigation:")
        print(f"  {report['strategy']['reasoning']}")
    
    print("\nRecommendations:")
    for i, rec in enumerate(report['recommendations'], 1):
        print(f"  {i}. {rec}")
    
    print("\nOutput Files:")
    print(f"  Mitigation Report: {config['pipeline_config']['logs_path']}/bias_mitigation_report.json")
    if report['mitigation_applied']:
        print(f"  Mitigated Dataset: {report.get('mitigated_dataset_path', 'N/A')}")
    print("="*70)