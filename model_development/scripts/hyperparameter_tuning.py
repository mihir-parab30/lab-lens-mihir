#!/usr/bin/env python3
"""
Hyperparameter Tuning for Gemini Model
Uses Optuna for Bayesian optimization of temperature, max_output_tokens, and prompt parameters
"""

import os
import sys
import optuna
from typing import Dict, Any, Optional, Callable
import pandas as pd

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.training.gemini_model import GeminiSummarizer
from src.training.model_validation import ModelValidator
from src.utils.logging_config import get_logger
from src.utils.error_handling import ErrorHandler, safe_execute

logger = get_logger(__name__)


class HyperparameterTuner:
    """
    Hyperparameter tuning using Optuna for Gemini model optimization
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gemini-2.0-flash-exp",
        n_trials: int = 20,
        study_name: Optional[str] = None
    ):
        """
        Initialize hyperparameter tuner
        
        Args:
            api_key: Google AI API key
            model_name: Gemini model identifier
            n_trials: Number of optimization trials
            study_name: Name for Optuna study
        """
        self.api_key = api_key
        self.model_name = model_name
        self.n_trials = n_trials
        self.study_name = study_name or "gemini_hyperparameter_tuning"
        self.error_handler = ErrorHandler(logger)
        self.validator = ModelValidator()
    
    @safe_execute("objective_function", logger, ErrorHandler(logger))
    def objective(
        self,
        trial: optuna.Trial,
        train_data: pd.DataFrame,
        val_data: pd.DataFrame,
        input_column: str = 'cleaned_text',
        reference_column: str = 'cleaned_text',
        sample_size: int = 50
    ) -> float:
        """
        Objective function for Optuna optimization
        
        Args:
            trial: Optuna trial object
            train_data: Training data
            val_data: Validation data
            input_column: Column with input text
            reference_column: Column with reference summaries
            sample_size: Number of samples to evaluate per trial
            
        Returns:
            Validation score (ROUGE-L F1)
        """
        # Suggest hyperparameters
        temperature = trial.suggest_float('temperature', 0.1, 1.0, step=0.1)
        max_output_tokens = trial.suggest_int('max_output_tokens', 100, 500, step=50)
        max_length = trial.suggest_int('max_length', 50, 200, step=25)
        
        logger.info(f"Trial {trial.number}: temperature={temperature}, "
                   f"max_output_tokens={max_output_tokens}, max_length={max_length}")
        
        try:
            # Initialize model with suggested hyperparameters
            model = GeminiSummarizer(
                api_key=self.api_key,
                model_name=self.model_name,
                temperature=temperature,
                max_output_tokens=max_output_tokens
            )
            
            # Sample validation data for faster evaluation
            val_sample = val_data.sample(min(sample_size, len(val_data))).copy()
            
            # Generate predictions
            predictions = []
            references = []
            
            for idx, row in val_sample.iterrows():
                input_text = str(row[input_column])
                if pd.isna(input_text) or not input_text.strip():
                    continue
                
                try:
                    summary = model.summarize(input_text, max_length=max_length)
                    predictions.append(summary)
                    references.append(str(row[reference_column]))
                except Exception as e:
                    logger.warning(f"Error generating summary for sample {idx}: {e}")
                    continue
            
            if len(predictions) == 0:
                logger.warning("No valid predictions generated")
                return 0.0
            
            # Calculate validation metrics
            metrics = self.validator.validate_model(predictions, references)
            
            # Return ROUGE-L F1 as the objective (to maximize)
            score = metrics.get('rougeL_f', 0.0)
            
            # Log intermediate values for pruning
            trial.set_user_attr('rouge1_f', metrics.get('rouge1_f', 0.0))
            trial.set_user_attr('rouge2_f', metrics.get('rouge2_f', 0.0))
            trial.set_user_attr('bleu', metrics.get('bleu', 0.0))
            
            logger.info(f"Trial {trial.number} score: {score:.4f}")
            return score
            
        except Exception as e:
            logger.error(f"Error in trial {trial.number}: {e}")
            return 0.0
    
    @safe_execute("optimize", logger, ErrorHandler(logger))
    def optimize(
        self,
        train_data: pd.DataFrame,
        val_data: pd.DataFrame,
        input_column: str = 'cleaned_text',
        reference_column: str = 'cleaned_text',
        sample_size: int = 50,
        direction: str = 'maximize'
    ) -> optuna.Study:
        """
        Run hyperparameter optimization
        
        Args:
            train_data: Training data
            val_data: Validation data
            input_column: Column with input text
            reference_column: Column with reference summaries
            sample_size: Number of samples to evaluate per trial
            direction: Optimization direction ('maximize' or 'minimize')
            
        Returns:
            Optuna study with optimization results
        """
        logger.info(f"Starting hyperparameter optimization with {self.n_trials} trials")
        
        # Create study
        study = optuna.create_study(
            study_name=self.study_name,
            direction=direction,
            sampler=optuna.samplers.TPESampler(seed=42)
        )
        
        # Create objective function with data
        objective_func = lambda trial: self.objective(
            trial, train_data, val_data, input_column, reference_column, sample_size
        )
        
        # Run optimization
        study.optimize(objective_func, n_trials=self.n_trials, show_progress_bar=True)
        
        logger.info(f"Optimization completed!")
        logger.info(f"Best trial: {study.best_trial.number}")
        logger.info(f"Best score: {study.best_value:.4f}")
        logger.info(f"Best params: {study.best_params}")
        
        return study
    
    def get_best_hyperparameters(self, study: optuna.Study) -> Dict[str, Any]:
        """
        Extract best hyperparameters from study
        
        Args:
            study: Optuna study object
            
        Returns:
            Dictionary with best hyperparameters
        """
        return study.best_params
    
    def get_optimization_history(self, study: optuna.Study) -> pd.DataFrame:
        """
        Get optimization history as DataFrame
        
        Args:
            study: Optuna study object
            
        Returns:
            DataFrame with trial history
        """
        trials_data = []
        for trial in study.trials:
            trial_data = {
                'trial_number': trial.number,
                'value': trial.value,
                'temperature': trial.params.get('temperature'),
                'max_output_tokens': trial.params.get('max_output_tokens'),
                'max_length': trial.params.get('max_length'),
                'state': trial.state.name
            }
            trials_data.append(trial_data)
        
        return pd.DataFrame(trials_data)






