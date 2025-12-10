#!/usr/bin/env python3
"""
Model Rollback Mechanism
Implements rollback functionality to revert to previous model if new model performs worse
"""

import os
import sys
import json
from typing import Dict, Optional, List, Tuple
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import mlflow
from src.utils.logging_config import get_logger
from src.utils.error_handling import ErrorHandler, safe_execute

logger = get_logger(__name__)


class ModelRollback:
    """
    Rollback mechanism for model deployment
    Reverts to previous model if new model performs worse
    """
    
    def __init__(
        self,
        tracking_uri: Optional[str] = None,
        experiment_name: str = "gemini-medical-summarization"
    ):
        """
        Initialize rollback manager
        
        Args:
            tracking_uri: MLflow tracking URI
            experiment_name: MLflow experiment name
        """
        self.tracking_uri = tracking_uri or "./mlruns"
        self.experiment_name = experiment_name
        mlflow.set_tracking_uri(self.tracking_uri)
        self.error_handler = ErrorHandler(logger)
    
    @safe_execute("get_latest_models", logger, ErrorHandler(logger))
    def get_latest_models(self, n: int = 2) -> List[Dict]:
        """
        Get the latest N model versions from registry
        
        Args:
            n: Number of models to retrieve
            
        Returns:
            List of model dictionaries with metrics
        """
        try:
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            if not experiment:
                logger.warning(f"Experiment {self.experiment_name} not found")
                return []
            
            # Get all runs
            runs = mlflow.search_runs(
                experiment_ids=[experiment.experiment_id],
                order_by=["start_time DESC"],
                max_results=n
            )
            
            models = []
            for _, run in runs.iterrows():
                model_info = {
                    'run_id': run['run_id'],
                    'start_time': run['start_time'],
                    'metrics': {},
                    'params': {},
                    'status': run.get('status', 'UNKNOWN')
                }
                
                # Get metrics
                try:
                    client = mlflow.tracking.MlflowClient()
                    metrics = client.get_run(run['run_id']).data.metrics
                    model_info['metrics'] = dict(metrics)
                except:
                    pass
                
                # Get params
                try:
                    params = client.get_run(run['run_id']).data.params
                    model_info['params'] = dict(params)
                except:
                    pass
                
                models.append(model_info)
            
            return models
            
        except Exception as e:
            logger.error(f"Error retrieving models: {e}")
            return []
    
    def compare_models(
        self,
        current_metrics: Dict[str, float],
        previous_metrics: Dict[str, float],
        primary_metric: str = "rougeL_f",
        threshold: float = 0.05
    ) -> Tuple[bool, Dict]:
        """
        Compare current model with previous model
        
        Args:
            current_metrics: Metrics from current model
            previous_metrics: Metrics from previous model
            primary_metric: Primary metric to compare (default: rougeL_f)
            threshold: Minimum improvement threshold (default: 5%)
            
        Returns:
            Tuple of (should_rollback, comparison_details)
        """
        comparison = {
            'current_metric': current_metrics.get(primary_metric, 0.0),
            'previous_metric': previous_metrics.get(primary_metric, 0.0),
            'improvement': 0.0,
            'should_rollback': False,
            'all_metrics': {}
        }
        
        # Calculate improvement
        if previous_metrics.get(primary_metric, 0.0) > 0:
            improvement = (
                (current_metrics.get(primary_metric, 0.0) - 
                 previous_metrics.get(primary_metric, 0.0)) /
                previous_metrics.get(primary_metric, 0.0)
            ) * 100
            comparison['improvement'] = improvement
            
            # Rollback if performance degraded significantly
            if improvement < -threshold * 100:  # More than threshold% worse
                comparison['should_rollback'] = True
                logger.warning(
                    f"Model performance degraded: {improvement:.2f}% worse. "
                    f"Rollback recommended."
                )
        
        # Compare all metrics
        all_metrics = {}
        for metric_name in set(list(current_metrics.keys()) + list(previous_metrics.keys())):
            current_val = current_metrics.get(metric_name, 0.0)
            previous_val = previous_metrics.get(metric_name, 0.0)
            all_metrics[metric_name] = {
                'current': current_val,
                'previous': previous_val,
                'change': current_val - previous_val
            }
        comparison['all_metrics'] = all_metrics
        
        return comparison['should_rollback'], comparison
    
    @safe_execute("check_and_rollback", logger, ErrorHandler(logger))
    def check_and_rollback(
        self,
        current_run_id: str,
        primary_metric: str = "rougeL_f",
        threshold: float = 0.05
    ) -> Dict:
        """
        Check if rollback is needed and perform rollback if necessary
        
        Args:
            current_run_id: Current model run ID
            primary_metric: Primary metric for comparison
            threshold: Minimum improvement threshold
            
        Returns:
            Dictionary with rollback decision and details
        """
        logger.info("Checking if rollback is needed...")
        
        # Get latest models
        models = self.get_latest_models(n=2)
        
        if len(models) < 2:
            logger.info("Not enough models for comparison. Skipping rollback check.")
            return {
                'rollback_needed': False,
                'reason': 'insufficient_models',
                'models_compared': len(models)
            }
        
        current_model = models[0]  # Latest
        previous_model = models[1]  # Previous
        
        # Get metrics
        current_metrics = current_model.get('metrics', {})
        previous_metrics = previous_model.get('metrics', {})
        
        if not current_metrics or not previous_metrics:
            logger.warning("Missing metrics for comparison. Skipping rollback check.")
            return {
                'rollback_needed': False,
                'reason': 'missing_metrics',
                'current_metrics': current_metrics,
                'previous_metrics': previous_metrics
            }
        
        # Compare models
        should_rollback, comparison = self.compare_models(
            current_metrics,
            previous_metrics,
            primary_metric=primary_metric,
            threshold=threshold
        )
        
        result = {
            'rollback_needed': should_rollback,
            'current_run_id': current_run_id,
            'previous_run_id': previous_model['run_id'],
            'comparison': comparison,
            'timestamp': datetime.now().isoformat()
        }
        
        if should_rollback:
            logger.warning("=" * 60)
            logger.warning("ROLLBACK RECOMMENDED")
            logger.warning("=" * 60)
            logger.warning(f"Current model performance: {comparison['current_metric']:.4f}")
            logger.warning(f"Previous model performance: {comparison['previous_metric']:.4f}")
            logger.warning(f"Performance change: {comparison['improvement']:.2f}%")
            logger.warning("=" * 60)
            
            # Perform rollback
            rollback_result = self.perform_rollback(previous_model['run_id'])
            result['rollback_performed'] = rollback_result
        else:
            logger.info("No rollback needed. Model performance is acceptable.")
            result['rollback_performed'] = False
        
        return result
    
    @safe_execute("perform_rollback", logger, ErrorHandler(logger))
    def perform_rollback(self, target_run_id: str) -> Dict:
        """
        Perform rollback to a previous model version
        
        Args:
            target_run_id: Run ID of model to rollback to
            
        Returns:
            Dictionary with rollback details
        """
        logger.info(f"Performing rollback to run_id: {target_run_id}")
        
        try:
            client = mlflow.tracking.MlflowClient()
            run = client.get_run(target_run_id)
            
            # Get model artifacts
            artifacts = client.list_artifacts(target_run_id)
            
            rollback_info = {
                'target_run_id': target_run_id,
                'rollback_time': datetime.now().isoformat(),
                'model_uri': f"runs:/{target_run_id}/model",
                'artifacts': [a.path for a in artifacts],
                'metrics': dict(run.data.metrics),
                'params': dict(run.data.params)
            }
            
            # In production, you would:
            # 1. Update model registry to mark this as production
            # 2. Deploy this model version
            # 3. Update deployment configuration
            # 4. Send notifications
            
            logger.info("Rollback completed successfully")
            logger.info(f"Rolled back to model: {target_run_id}")
            
            # Save rollback log
            rollback_log_path = Path("logs/rollback_log.json")
            rollback_log_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Load existing log or create new
            if rollback_log_path.exists():
                with open(rollback_log_path, 'r') as f:
                    rollback_log = json.load(f)
            else:
                rollback_log = {'rollbacks': []}
            
            rollback_log['rollbacks'].append(rollback_info)
            
            with open(rollback_log_path, 'w') as f:
                json.dump(rollback_log, f, indent=2)
            
            return rollback_info
            
        except Exception as e:
            logger.error(f"Error performing rollback: {e}")
            raise
    
    def get_rollback_history(self) -> List[Dict]:
        """
        Get history of rollbacks
        
        Returns:
            List of rollback records
        """
        rollback_log_path = Path("logs/rollback_log.json")
        
        if not rollback_log_path.exists():
            return []
        
        try:
            with open(rollback_log_path, 'r') as f:
                rollback_log = json.load(f)
            return rollback_log.get('rollbacks', [])
        except Exception as e:
            logger.error(f"Error reading rollback history: {e}")
            return []


def main():
    """Example usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Model rollback mechanism')
    parser.add_argument('--run-id', type=str, required=True,
                       help='Current model run ID to check')
    parser.add_argument('--primary-metric', type=str, default='rougeL_f',
                       help='Primary metric for comparison')
    parser.add_argument('--threshold', type=float, default=0.05,
                       help='Performance degradation threshold')
    
    args = parser.parse_args()
    
    rollback = ModelRollback()
    result = rollback.check_and_rollback(
        current_run_id=args.run_id,
        primary_metric=args.primary_metric,
        threshold=args.threshold
    )
    
    print("\n" + "=" * 60)
    print("ROLLBACK CHECK RESULTS")
    print("=" * 60)
    print(f"Rollback Needed: {result['rollback_needed']}")
    if result.get('comparison'):
        comp = result['comparison']
        print(f"Current Metric: {comp['current_metric']:.4f}")
        print(f"Previous Metric: {comp['previous_metric']:.4f}")
        print(f"Improvement: {comp['improvement']:.2f}%")
    print("=" * 60)


if __name__ == '__main__':
    main()




