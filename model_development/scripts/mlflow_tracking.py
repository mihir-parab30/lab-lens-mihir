#!/usr/bin/env python3
"""
MLflow Integration for Experiment Tracking
Tracks hyperparameters, metrics, model versions, and artifacts
"""

import os
import sys
import mlflow
import mlflow.sklearn
from typing import Dict, Any, Optional
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class MLflowTracker:
    """
    MLflow experiment tracking for Gemini model development
    """
    
    def __init__(
        self,
        experiment_name: str = "gemini-medical-summarization",
        tracking_uri: Optional[str] = None,
        run_name: Optional[str] = None
    ):
        """
        Initialize MLflow tracker
        
        Args:
            experiment_name: Name of MLflow experiment
            tracking_uri: MLflow tracking URI (default: local ./mlruns)
            run_name: Name for this run (default: timestamp-based)
        """
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri or "./mlruns"
        
        # Set tracking URI
        mlflow.set_tracking_uri(self.tracking_uri)
        
        # Create or get experiment
        try:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                experiment_id = mlflow.create_experiment(experiment_name)
                logger.info(f"Created new experiment: {experiment_name} (ID: {experiment_id})")
            else:
                logger.info(f"Using existing experiment: {experiment_name} (ID: {experiment.experiment_id})")
        except Exception as e:
            logger.warning(f"Could not set up experiment: {e}. Using default.")
        
        mlflow.set_experiment(experiment_name)
        
        # Start run
        self.run_name = run_name or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.active_run = mlflow.start_run(run_name=self.run_name)
        logger.info(f"Started MLflow run: {self.run_name}")
    
    def log_hyperparameters(self, hyperparams: Dict[str, Any]):
        """
        Log hyperparameters
        
        Args:
            hyperparams: Dictionary of hyperparameter names and values
        """
        try:
            mlflow.log_params(hyperparams)
            logger.info(f"Logged {len(hyperparams)} hyperparameters")
        except Exception as e:
            logger.error(f"Error logging hyperparameters: {e}")
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """
        Log metrics
        
        Args:
            metrics: Dictionary of metric names and values
            step: Optional step number for time-series metrics
        """
        try:
            if step is not None:
                for metric_name, metric_value in metrics.items():
                    mlflow.log_metric(metric_name, metric_value, step=step)
            else:
                mlflow.log_metrics(metrics)
            logger.info(f"Logged {len(metrics)} metrics")
        except Exception as e:
            logger.error(f"Error logging metrics: {e}")
    
    def log_model(self, model, artifact_path: str = "model"):
        """
        Log model artifact
        
        Args:
            model: Model object to log
            artifact_path: Path within run artifacts
        """
        try:
            # For Gemini API models, we log the configuration instead
            mlflow.log_dict(
                {
                    "model_type": "gemini-api",
                    "model_name": getattr(model, 'model_name', 'unknown'),
                    "temperature": getattr(model, 'temperature', None),
                    "max_output_tokens": getattr(model, 'max_output_tokens', None)
                },
                artifact_path=f"{artifact_path}/model_config.json"
            )
            logger.info(f"Logged model configuration to {artifact_path}")
        except Exception as e:
            logger.error(f"Error logging model: {e}")
    
    def log_artifacts(self, local_dir: str, artifact_path: Optional[str] = None):
        """
        Log directory of artifacts
        
        Args:
            local_dir: Local directory to log
            artifact_path: Optional path within run artifacts
        """
        try:
            mlflow.log_artifacts(local_dir, artifact_path)
            logger.info(f"Logged artifacts from {local_dir}")
        except Exception as e:
            logger.error(f"Error logging artifacts: {e}")
    
    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None):
        """
        Log single artifact file
        
        Args:
            local_path: Local file path
            artifact_path: Optional path within run artifacts
        """
        try:
            mlflow.log_artifact(local_path, artifact_path)
            logger.info(f"Logged artifact: {local_path}")
        except Exception as e:
            logger.error(f"Error logging artifact: {e}")
    
    def log_tags(self, tags: Dict[str, str]):
        """
        Log tags for run
        
        Args:
            tags: Dictionary of tag names and values
        """
        try:
            mlflow.set_tags(tags)
            logger.info(f"Logged {len(tags)} tags")
        except Exception as e:
            logger.error(f"Error logging tags: {e}")
    
    def end_run(self, status: str = "FINISHED"):
        """
        End the current MLflow run
        
        Args:
            status: Run status (FINISHED, FAILED, KILLED)
        """
        try:
            mlflow.end_run(status=status)
            logger.info(f"Ended MLflow run with status: {status}")
        except Exception as e:
            logger.error(f"Error ending run: {e}")
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        status = "FINISHED" if exc_type is None else "FAILED"
        self.end_run(status=status)
        return False


def get_best_run(experiment_name: str, metric: str = "rouge_l_f", ascending: bool = False):
    """
    Get the best run from an experiment based on a metric
    
    Args:
        experiment_name: Name of experiment
        metric: Metric name to optimize
        ascending: If True, lower is better; if False, higher is better
        
    Returns:
        Best run object
    """
    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            logger.error(f"Experiment {experiment_name} not found")
            return None
        
        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=[f"metrics.{metric} {'ASC' if ascending else 'DESC'}"]
        )
        
        if len(runs) == 0:
            logger.warning("No runs found in experiment")
            return None
        
        best_run = runs.iloc[0]
        logger.info(f"Best run: {best_run['run_id']} with {metric}={best_run[f'metrics.{metric}']}")
        return best_run
    except Exception as e:
        logger.error(f"Error getting best run: {e}")
        return None






