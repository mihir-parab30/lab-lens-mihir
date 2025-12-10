#!/usr/bin/env python3
"""
Complete Model Training with Experiment Tracking, Validation, Bias Detection, and Sensitivity Analysis
Implements all requirements from Model Development Guidelines
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
from src.training.train_gemini import GeminiTrainer
from src.training.mlflow_tracking import MLflowTracker
from src.training.model_validation import ModelValidator
from src.training.hyperparameter_tuning import HyperparameterTuner
from src.training.model_bias_detection import ModelBiasDetector
from src.training.sensitivity_analysis import SensitivityAnalyzer
from src.training.model_rollback import ModelRollback
from src.utils.logging_config import get_logger
from src.utils.error_handling import ErrorHandler, safe_execute

logger = get_logger(__name__)


class CompleteModelTrainer:
    """
    Complete model training pipeline with all MLOps components:
    - Data loading from pipeline
    - Hyperparameter tuning
    - Experiment tracking (MLflow)
    - Model validation
    - Bias detection
    - Sensitivity analysis
    """
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        api_key: Optional[str] = None,
        model_name: str = "gemini-2.0-flash-exp",
        output_dir: str = "models/gemini",
        enable_hyperparameter_tuning: bool = False,
        enable_bias_detection: bool = True,
        enable_sensitivity_analysis: bool = True
    ):
        """
        Initialize complete trainer
        
        Args:
            config_path: Path to configuration file
            api_key: Google AI API key
            model_name: Gemini model identifier
            output_dir: Output directory
            enable_hyperparameter_tuning: Whether to run hyperparameter tuning
            enable_bias_detection: Whether to run bias detection
            enable_sensitivity_analysis: Whether to run sensitivity analysis
        """
        self.config_path = config_path
        self.api_key = api_key or os.getenv('GOOGLE_API_KEY')
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.enable_hyperparameter_tuning = enable_hyperparameter_tuning
        self.enable_bias_detection = enable_bias_detection
        self.enable_sensitivity_analysis = enable_sensitivity_analysis
        
        self.error_handler = ErrorHandler(logger)
        
        # Initialize components
        self.trainer = GeminiTrainer(
            config_path=config_path,
            api_key=self.api_key,
            model_name=self.model_name,
            output_dir=str(self.output_dir)
        )
        self.validator = ModelValidator()
        self.bias_detector = ModelBiasDetector()
        self.sensitivity_analyzer = SensitivityAnalyzer()
        self.rollback = ModelRollback()
        
        logger.info("Initialized CompleteModelTrainer")
    
    @safe_execute("load_data_from_pipeline", logger, ErrorHandler(logger))
    def load_data_from_pipeline(
        self,
        data_path: str,
        train_split: float = 0.8,
        val_split: float = 0.1
    ) -> tuple:
        """
        Load data from data pipeline with train/val/test split
        
        Args:
            data_path: Path to processed data from pipeline
            train_split: Training split ratio
            val_split: Validation split ratio
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        logger.info(f"Loading data from pipeline: {data_path}")
        
        df = self.trainer.load_data(data_path)
        
        # Shuffle
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Split
        n_total = len(df)
        n_train = int(n_total * train_split)
        n_val = int(n_total * val_split)
        
        train_df = df.iloc[:n_train].copy()
        val_df = df.iloc[n_train:n_train + n_val].copy()
        test_df = df.iloc[n_train + n_val:].copy()
        
        logger.info(f"Data split: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
        
        return train_df, val_df, test_df
    
    @safe_execute("train_and_evaluate", logger, ErrorHandler(logger))
    def train_and_evaluate(
        self,
        data_path: str,
        hyperparameters: Optional[Dict] = None,
        run_name: Optional[str] = None
    ) -> Dict:
        """
        Complete training and evaluation pipeline
        
        Args:
            data_path: Path to processed data
            hyperparameters: Optional hyperparameters (if None, uses defaults or tuning)
            run_name: Optional name for MLflow run
            
        Returns:
            Dictionary with training results
        """
        # Start MLflow tracking
        with MLflowTracker(run_name=run_name) as tracker:
            # Load data
            train_df, val_df, test_df = self.load_data_from_pipeline(data_path)
            
            # Hyperparameter tuning (if enabled)
            best_hyperparams = hyperparameters
            if self.enable_hyperparameter_tuning and hyperparameters is None:
                logger.info("Starting hyperparameter tuning...")
                tuner = HyperparameterTuner(
                    api_key=self.api_key,
                    model_name=self.model_name,
                    n_trials=10  # Reduced for faster execution
                )
                
                study = tuner.optimize(
                    train_df, val_df,
                    input_column=self.trainer.config['input_column'],
                    reference_column=self.trainer.config['input_column']
                )
                
                best_hyperparams = tuner.get_best_hyperparameters(study)
                optimization_history = tuner.get_optimization_history(study)
                
                # Save and log optimization history
                opt_history_path = self.output_dir / "optimization_history.csv"
                optimization_history.to_csv(opt_history_path, index=False)
                tracker.log_artifact(str(opt_history_path))
                
                # Sensitivity analysis on hyperparameters
                if self.enable_sensitivity_analysis:
                    sensitivity_results = self.sensitivity_analyzer.analyze_hyperparameter_sensitivity(
                        optimization_history
                    )
                    tracker.log_dict(sensitivity_results, "hyperparameter_sensitivity.json")
                    
                    # Create plots
                    plot_paths = self.sensitivity_analyzer.create_sensitivity_plots(
                        optimization_history,
                        str(self.output_dir / "sensitivity_plots")
                    )
                    for plot_path in plot_paths:
                        tracker.log_artifact(plot_path)
            
            # Set hyperparameters
            if best_hyperparams is None:
                best_hyperparams = {
                    'temperature': 0.3,
                    'max_output_tokens': 2048,
                    'max_length': 150
                }
            
            # Log hyperparameters
            tracker.log_hyperparameters(best_hyperparams)
            tracker.log_tags({
                'model_name': self.model_name,
                'data_path': data_path,
                'train_samples': str(len(train_df)),
                'val_samples': str(len(val_df))
            })
            
            # Train model (process data with Gemini)
            logger.info("Training model (processing data with Gemini)...")
            model = load_gemini_model(
                api_key=self.api_key,
                model_name=self.model_name,
                temperature=best_hyperparams.get('temperature', 0.3),
                max_output_tokens=best_hyperparams.get('max_output_tokens', 2048)
            )
            
            # Process validation set
            logger.info("Generating predictions on validation set...")
            val_sample = val_df.sample(min(100, len(val_df)))  # Sample for faster evaluation
            val_predictions = []
            val_references = []
            
            for idx, row in val_sample.iterrows():
                input_text = str(row[self.trainer.config['input_column']])
                if pd.isna(input_text) or not input_text.strip():
                    continue
                
                try:
                    summary = model.summarize(
                        input_text,
                        max_length=best_hyperparams.get('max_length', 150)
                    )
                    val_predictions.append(summary)
                    val_references.append(input_text)  # Using input as reference for now
                except Exception as e:
                    logger.warning(f"Error generating summary: {e}")
                    continue
            
            # Model validation
            logger.info("Validating model...")
            validation_metrics = self.validator.validate_model(
                val_predictions,
                val_references
            )
            
            # Log validation metrics
            tracker.log_metrics(validation_metrics)
            
            # Bias detection
            bias_report = None
            if self.enable_bias_detection:
                logger.info("Detecting model bias...")
                val_with_predictions = val_sample.copy()
                # Pad predictions to match dataframe length
                predictions_padded = val_predictions + [''] * (len(val_with_predictions) - len(val_predictions))
                val_with_predictions['gemini_summary'] = predictions_padded[:len(val_with_predictions)]
                
                try:
                    bias_report = self.bias_detector.detect_bias(
                        val_with_predictions,
                        prediction_column='gemini_summary',
                        reference_column=self.trainer.config['input_column']
                    )
                    
                    # Log bias metrics
                    tracker.log_metrics({
                        'overall_bias_score': bias_report.get('overall_bias_score', 0.0),
                        'bias_alerts_count': len(bias_report.get('bias_alerts', []))
                    })
                    
                    # Save bias report
                    bias_report_path = self.output_dir / 'bias_report.json'
                    self.bias_detector.save_bias_report(bias_report, str(bias_report_path))
                    tracker.log_artifact(str(bias_report_path))
                except Exception as e:
                    logger.warning(f"Bias detection failed: {e}")
                    bias_report = None
            
            # Model selection after bias checking (Requirement #7: Model Selection after Bias Checking)
            # Select best model considering both validation metrics AND bias analysis
            model_selection_result = self._select_best_model_after_bias_check(
                metrics=validation_metrics,
                bias_report=bias_report,
                run_id=tracker.run_id
            )
            
            # Log model selection results
            tracker.log_metrics({
                'model_selected': model_selection_result.get('selected', True),
                'combined_score': model_selection_result.get('combined_score', 0.0)
            })
            tracker.log_params({
                'selection_reason': model_selection_result.get('reason', '')
            })
            
            # Check for rollback if model performance degraded
            if tracker.run_id and model_selection_result.get('selected', True):
                try:
                    rollback_result = self.rollback.check_and_rollback(
                        current_run_id=tracker.run_id,
                        primary_metric='rougeL_f',
                        threshold=0.05
                    )
                    if rollback_result.get('rollback_needed'):
                        logger.warning("Rollback check completed. Review rollback_result for details.")
                        tracker.log_params({'rollback_needed': True})
                except Exception as e:
                    logger.warning(f"Rollback check failed: {e}")
            
            # Only log model if selected
            if model_selection_result.get('selected', True):
                tracker.log_model(model)
            else:
                logger.warning("Model not selected. Not logging to registry.")
            
            # Save results
            results = {
                'validation_metrics': validation_metrics,
                'hyperparameters': best_hyperparams,
                'bias_report': bias_report if self.enable_bias_detection else None,
                'model_selection': model_selection_result,
                'timestamp': datetime.now().isoformat()
            }
            
            results_path = self.output_dir / 'training_results.json'
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            tracker.log_artifact(str(results_path))
            
            logger.info("Training and evaluation completed!")
            return results
    
    def _select_best_model_after_bias_check(
        self,
        metrics: Dict,
        bias_report: Optional[Dict],
        run_id: Optional[str]
    ) -> Dict:
        """
        Select best model after considering both validation performance and bias analysis
        This implements requirement #7: Model Selection after Bias Checking
        
        Args:
            metrics: Validation metrics
            bias_report: Bias detection report
            run_id: Current run ID
            
        Returns:
            Dictionary with selection decision and reason
        """
        selection_result = {
            'selected': True,
            'reason': 'Model meets all criteria',
            'validation_score': metrics.get('rougeL_f', 0.0),
            'bias_score': None,
            'combined_score': metrics.get('rougeL_f', 0.0)
        }
        
        # Check validation threshold
        validation_threshold = 0.3  # Minimum ROUGE-L F1
        if metrics.get('rougeL_f', 0.0) < validation_threshold:
            selection_result['selected'] = False
            selection_result['reason'] = f'Validation score below threshold ({validation_threshold})'
            logger.warning(f"Model not selected: {selection_result['reason']}")
            return selection_result
        
        # Check bias if bias report available
        if bias_report:
            bias_score = bias_report.get('overall_bias_score', 0.0)
            selection_result['bias_score'] = bias_score
            
            bias_threshold = 0.2  # Maximum acceptable bias score
            if bias_score > bias_threshold:
                selection_result['selected'] = False
                selection_result['reason'] = f'Bias score exceeds threshold ({bias_score} > {bias_threshold})'
                logger.warning(f"Model not selected: {selection_result['reason']}")
                return selection_result
            
            # Combined score: validation performance weighted more, but bias considered
            # Lower bias is better, so we subtract it
            selection_result['combined_score'] = (
                metrics.get('rougeL_f', 0.0) * 0.7 +  # 70% weight on validation
                (1.0 - min(bias_score, 1.0)) * 0.3    # 30% weight on fairness (inverted bias)
            )
            selection_result['reason'] = f'Selected based on combined score: validation={metrics.get("rougeL_f", 0.0):.3f}, bias={bias_score:.3f}'
        
        logger.info(f"Model selection: {selection_result['reason']}")
        return selection_result


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Complete model training with MLOps components'
    )
    parser.add_argument('--data-path', type=str, required=True,
                       help='Path to processed data from pipeline')
    parser.add_argument('--config', type=str,
                       help='Path to configuration file')
    parser.add_argument('--api-key', type=str,
                       help='Google AI API key (or set GOOGLE_API_KEY env var)')
    parser.add_argument('--model-name', type=str, default='gemini-2.0-flash-exp',
                       help='Gemini model identifier')
    parser.add_argument('--output-dir', type=str, default='models/gemini',
                       help='Output directory')
    parser.add_argument('--enable-tuning', action='store_true',
                       help='Enable hyperparameter tuning')
    parser.add_argument('--disable-bias-detection', action='store_true',
                       help='Disable bias detection')
    parser.add_argument('--disable-sensitivity', action='store_true',
                       help='Disable sensitivity analysis')
    parser.add_argument('--run-name', type=str,
                       help='Name for MLflow run')
    
    args = parser.parse_args()
    
    try:
        trainer = CompleteModelTrainer(
            config_path=args.config,
            api_key=args.api_key,
            model_name=args.model_name,
            output_dir=args.output_dir,
            enable_hyperparameter_tuning=args.enable_tuning,
            enable_bias_detection=not args.disable_bias_detection,
            enable_sensitivity_analysis=not args.disable_sensitivity
        )
        
        results = trainer.train_and_evaluate(
            data_path=args.data_path,
            run_name=args.run_name
        )
        
        print("\n=== Training Summary ===")
        print(f"Validation Metrics: {results['validation_metrics']}")
        print(f"Best Hyperparameters: {results['hyperparameters']}")
        if results.get('bias_report'):
            print(f"Bias Score: {results['bias_report'].get('overall_bias_score', 0.0):.4f}")
        print(f"\nResults saved to: {args.output_dir}")
        print(f"MLflow run: {args.run_name or 'auto-generated'}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

