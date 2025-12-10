"""
Optimized Lab Lens MIMIC-III Processing Pipeline DAG
Author: Lab Lens Team
Description: Docker-compatible Airflow DAG with runtime path resolution

This DAG runs in Docker containers and resolves all paths at runtime
to ensure compatibility across different environments (Mac/Windows/Linux).

Pipeline Flow:
check_data → preprocess → validate → engineer_features → 
detect_bias → mitigate_bias → generate_summary
"""

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.trigger_rule import TriggerRule
from datetime import datetime, timedelta
import sys
import os
import json
import time
import pandas as pd
from pathlib import Path
from typing import Dict, Any

# DAG default arguments
default_args = {
    'owner': 'lab-lens-team',
    'depends_on_past': False,
    'email': ['team@lablens.com'],
    'email_on_failure': False,  # Disabled to avoid SMTP errors
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=2),
    'start_date': datetime(2025, 11, 1),
    'execution_timeout': timedelta(minutes=30)
}


def get_project_root() -> Path:
    """
    Find the lab-lens project root directory
    Works in both local and Docker environments
    
    Returns:
        Path object pointing to project root
    """
    # In Docker, we're mounted at /opt/airflow
    # Check if we're in Docker
    if Path('/opt/airflow').exists():
        return Path('/opt/airflow')
    
    # Local environment - use traditional search
    dag_file = Path(__file__).resolve()
    current = dag_file.parent
    
    # Strategy 1: Look for marker file
    for _ in range(5):
        if (current / '.lab-lens-root').exists():
            return current
        if current.parent == current:
            break
        current = current.parent
    
    # Strategy 2: Airflow DAG location
    if dag_file.parent.name == 'dags' and dag_file.parent.parent.name == 'airflow':
        return dag_file.parent.parent.parent
    
    # Fallback
    return Path('/opt/airflow')


def validate_task_input(file_path: Path, min_records: int = 100) -> None:
    """
    Validate that input file exists and meets minimum quality requirements
    
    Args:
        file_path: Path to input CSV file
        min_records: Minimum number of records required
        
    Raises:
        FileNotFoundError: If input file does not exist
        ValueError: If data quality is insufficient
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Required input file missing: {file_path}")
    
    # Quick validation without loading entire file
    df = pd.read_csv(file_path, nrows=min_records)
    
    if len(df) < min_records:
        raise ValueError(
            f"Insufficient data in {file_path}: {len(df)} records "
            f"(minimum required: {min_records})"
        )


def track_performance(task_name: str, start_time: float, context: Dict) -> None:
    """
    Track and push performance metrics for monitoring
    
    Args:
        task_name: Name of the task
        start_time: Task start timestamp
        context: Airflow task context
    """
    duration = time.time() - start_time
    context['ti'].xcom_push(key=f'{task_name}_duration', value=duration)
    context['ti'].xcom_push(key=f'{task_name}_timestamp', value=datetime.now().isoformat())
    print(f"Task {task_name} completed in {duration:.2f} seconds")


def check_data_availability(**context):
    """
    Task 0: Verify raw MIMIC-III data availability and quality
    """
    start_time = time.time()
    
    # Get project root and construct paths at runtime
    project_root = get_project_root()
    data_file = project_root / 'data-pipeline' / 'data' / 'raw' / 'mimic_discharge_labs.csv'
    
    if not data_file.exists():
        raise FileNotFoundError(
            f"Raw data not found: {data_file}\n"
            f"Please ensure MIMIC-III data is available in data-pipeline/data/raw/"
        )
    
    # Validate file quality
    file_size_bytes = data_file.stat().st_size
    file_size_mb = file_size_bytes / (1024 * 1024)
    
    if file_size_mb < 1:
        raise ValueError(f"Data file too small ({file_size_mb:.2f} MB), may be corrupted")
    
    # Quick schema check
    df_sample = pd.read_csv(data_file, nrows=100)
    required_columns = ['hadm_id', 'subject_id', 'cleaned_text']
    missing_cols = [col for col in required_columns if col not in df_sample.columns]
    
    if missing_cols:
        raise ValueError(f"Required columns missing from data: {missing_cols}")
    
    print(f"Data validation successful:")
    print(f"  File: {data_file}")
    print(f"  Size: {file_size_mb:.2f} MB")
    print(f"  Columns: {len(df_sample.columns)}")
    
    # Push metrics to XCom
    context['ti'].xcom_push(key='data_size_mb', value=file_size_mb)
    context['ti'].xcom_push(key='data_path', value=str(data_file))
    
    track_performance('check_data', start_time, context)
    
    return {'status': 'data_ready', 'size_mb': file_size_mb}


def run_preprocessing_task(**context):
    """
    Task 1: Execute preprocessing pipeline using direct import
    """
    start_time = time.time()
    
    # Get paths at runtime
    project_root = get_project_root()
    input_path = project_root / 'data-pipeline' / 'data' / 'raw'
    output_path = project_root / 'data-pipeline' / 'data' / 'processed'
    scripts_path = project_root / 'data-pipeline' / 'scripts'
    
    # Validate input
    input_file = input_path / 'mimic_discharge_labs.csv'
    validate_task_input(input_file, min_records=100)
    
    # Add scripts to path and import
    sys.path.insert(0, str(scripts_path))
    sys.path.insert(0, str(project_root / 'src'))
    
    try:
        from preprocessing import MIMICPreprocessor
        
        # Initialize and run
        preprocessor = MIMICPreprocessor(
            input_path=str(input_path),
            output_path=str(output_path)
        )
        
        df_processed, report = preprocessor.run_preprocessing_pipeline()
        
        # Push metrics
        context['ti'].xcom_push(key='records_processed', value=report['final_records'])
        context['ti'].xcom_push(key='duplicates_removed', value=report['duplicates_removed'])
        context['ti'].xcom_push(key='avg_text_length', value=report['avg_text_length'])
        
        track_performance('preprocessing', start_time, context)
        
        print(f"Preprocessing complete: {report['final_records']} records processed")
        
        return {'status': 'success', 'records': report['final_records']}
        
    except Exception as e:
        print(f"Preprocessing error: {str(e)}")
        raise


def run_validation_task(**context):
    """
    Task 2: Execute validation pipeline using direct import
    """
    start_time = time.time()
    
    # Get paths at runtime
    project_root = get_project_root()
    processed_path = project_root / 'data-pipeline' / 'data' / 'processed'
    logs_path = project_root / 'data-pipeline' / 'logs'
    scripts_path = project_root / 'data-pipeline' / 'scripts'
    
    # Load config
    config_path = project_root / 'data-pipeline' / 'configs' / 'pipeline_config.json'
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Validate input
    input_file = processed_path / 'processed_discharge_summaries.csv'
    validate_task_input(input_file, min_records=100)
    
    # Import and run
    sys.path.insert(0, str(scripts_path))
    sys.path.insert(0, str(project_root / 'src'))
    
    try:
        from validation import MIMICDataValidator
        
        # Initialize and run
        validator = MIMICDataValidator(
            input_path=str(processed_path),
            output_path=str(logs_path),
            config=config
        )
        
        report, summary = validator.run_validation_pipeline()
        
        validation_score = report['overall_score']
        threshold = config.get('validation_config', {}).get('validation_score_threshold', 80)
        
        # Push metrics
        context['ti'].xcom_push(key='validation_score', value=validation_score)
        context['ti'].xcom_push(key='validation_threshold', value=threshold)
        context['ti'].xcom_push(key='validation_passed', value=validation_score >= threshold)
        
        track_performance('validation', start_time, context)
        
        print(f"Validation Score: {validation_score:.2f}% (threshold: {threshold}%)")
        
        if validation_score < 70:
            raise ValueError(
                f"Validation score critically low: {validation_score:.2f}%. "
                f"Data quality must be improved."
            )
        
        return {'status': 'success', 'score': validation_score}
        
    except Exception as e:
        print(f"Validation error: {str(e)}")
        raise


def run_feature_engineering_task(**context):
    """
    Task 3: Execute feature engineering pipeline using direct import
    """
    start_time = time.time()
    
    # Get paths at runtime
    project_root = get_project_root()
    processed_path = project_root / 'data-pipeline' / 'data' / 'processed'
    logs_path = project_root / 'data-pipeline' / 'logs'
    scripts_path = project_root / 'data-pipeline' / 'scripts'
    
    # Validate input
    input_file = processed_path / 'processed_discharge_summaries.csv'
    validate_task_input(input_file, min_records=100)
    
    # Import and run
    sys.path.insert(0, str(scripts_path))
    sys.path.insert(0, str(project_root / 'src'))
    
    try:
        from feature_engineering import engineer_features, setup_logger
        
        # Setup logger
        log_path = logs_path / 'feature_engineering.log'
        logger = setup_logger(log_path)
        
        # Load and process data
        df = pd.read_csv(input_file, low_memory=False)
        initial_records = len(df)
        
        df_features = engineer_features(df, logger=logger, with_sections=False)
        
        # Save features
        output_file = processed_path / 'mimic_features.csv'
        df_features.to_csv(output_file, index=False)
        
        # Calculate metrics
        features_created = len(df_features.columns)
        records_per_second = initial_records / (time.time() - start_time)
        
        # Push metrics
        context['ti'].xcom_push(key='features_created', value=features_created)
        context['ti'].xcom_push(key='feature_eng_records', value=len(df_features))
        context['ti'].xcom_push(key='processing_rate', value=records_per_second)
        
        track_performance('feature_engineering', start_time, context)
        
        print(f"Feature engineering complete: {len(df_features)} records, {features_created} features")
        
        return {'status': 'success', 'features': features_created}
        
    except Exception as e:
        print(f"Feature engineering error: {str(e)}")
        raise


def run_bias_detection_task(**context):
    """
    Task 4: Execute comprehensive bias detection using direct import
    """
    start_time = time.time()
    
    # Get paths at runtime
    project_root = get_project_root()
    processed_path = project_root / 'data-pipeline' / 'data' / 'processed'
    logs_path = project_root / 'data-pipeline' / 'logs'
    scripts_path = project_root / 'data-pipeline' / 'scripts'
    
    # Load config
    config_path = project_root / 'data-pipeline' / 'configs' / 'pipeline_config.json'
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Validate input
    input_file = processed_path / 'mimic_features.csv'
    validate_task_input(input_file, min_records=100)
    
    # Import and run
    sys.path.insert(0, str(scripts_path))
    sys.path.insert(0, str(project_root / 'src'))
    
    try:
        from bias_detection import MIMICBiasDetector
        
        # Initialize and run
        detector = MIMICBiasDetector(
            input_path=str(processed_path),
            output_path=str(logs_path),
            config=config
        )
        
        report, summary = detector.run_bias_detection_pipeline()
        
        # Extract metrics
        summary_metrics = report.get('summary_metrics', {})
        adjusted_analysis = report.get('adjusted_bias_analysis', {})
        
        bias_score_raw = summary_metrics.get('overall_bias_score', 0)
        age_cv_raw = summary_metrics.get('age_cv', 0)
        gender_cv = summary_metrics.get('gender_cv', 0)
        ethnicity_cv = summary_metrics.get('ethnicity_cv', 0)
        
        # Get adjusted metrics
        age_cv_adjusted = 0
        bias_interpretation = 'UNKNOWN'
        
        if adjusted_analysis.get('analysis_performed'):
            residual_analysis = adjusted_analysis.get('residual_analysis', {})
            age_residuals = residual_analysis.get('age_group', {})
            age_cv_adjusted = age_residuals.get('cv', 0)
            bias_interpretation = age_residuals.get('interpretation', 'UNKNOWN')
        
        # Push metrics
        context['ti'].xcom_push(key='bias_score_raw', value=bias_score_raw)
        context['ti'].xcom_push(key='age_cv_raw', value=age_cv_raw)
        context['ti'].xcom_push(key='age_cv_adjusted', value=age_cv_adjusted)
        context['ti'].xcom_push(key='gender_cv', value=gender_cv)
        context['ti'].xcom_push(key='ethnicity_cv', value=ethnicity_cv)
        context['ti'].xcom_push(key='bias_interpretation', value=bias_interpretation)
        
        track_performance('bias_detection', start_time, context)
        
        print(f"Bias detection complete:")
        print(f"  Raw bias: {bias_score_raw:.2f}%")
        print(f"  Age CV (raw): {age_cv_raw:.2f}%")
        print(f"  Age CV (adjusted): {age_cv_adjusted:.2f}%")
        print(f"  Interpretation: {bias_interpretation}")
        
        return {'status': 'success', 'bias_score': bias_score_raw}
        
    except Exception as e:
        print(f"Bias detection error: {str(e)}")
        raise


def run_bias_mitigation_task(**context):
    """
    Task 5: Execute intelligent bias mitigation using direct import
    """
    start_time = time.time()
    
    # Get paths at runtime
    project_root = get_project_root()
    processed_path = project_root / 'data-pipeline' / 'data' / 'processed'
    logs_path = project_root / 'data-pipeline' / 'logs'
    scripts_path = project_root / 'data-pipeline' / 'scripts'
    
    # Load config
    config_path = project_root / 'data-pipeline' / 'configs' / 'pipeline_config.json'
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Validate inputs
    features_file = processed_path / 'mimic_features.csv'
    bias_report_file = logs_path / 'bias_report.json'
    
    validate_task_input(features_file, min_records=100)
    
    if not bias_report_file.exists():
        raise FileNotFoundError(f"Bias report not found: {bias_report_file}")
    
    # Import and run
    sys.path.insert(0, str(scripts_path))
    sys.path.insert(0, str(project_root / 'src'))
    
    try:
        from automated_bias_handler import IntelligentBiasHandler
        
        # Initialize and run
        handler = IntelligentBiasHandler(
            input_path=str(processed_path),
            output_path=str(logs_path),
            config=config
        )
        
        mitigated_df, report = handler.run_mitigation_pipeline()
        
        # Extract metrics
        mitigation_applied = report.get('mitigation_applied', False)
        strategy = report.get('strategy', {})
        
        after_mitigation = report.get('after_mitigation', {})
        bias_score_after = after_mitigation.get('overall_bias_score', 0)
        
        # Calculate improvement
        bias_improvement = 0
        if 'improvement' in report and 'overall_bias_score' in report['improvement']:
            bias_improvement = report['improvement']['overall_bias_score'].get('improvement_percentage', 0)
        
        # Push metrics
        context['ti'].xcom_push(key='mitigation_applied', value=mitigation_applied)
        context['ti'].xcom_push(key='mitigation_strategy', value=strategy.get('action', 'none'))
        context['ti'].xcom_push(key='bias_score_after', value=bias_score_after)
        context['ti'].xcom_push(key='bias_improvement_pct', value=bias_improvement)
        context['ti'].xcom_push(key='requires_review', value=strategy.get('requires_review', False))
        
        track_performance('bias_mitigation', start_time, context)
        
        print(f"Bias mitigation complete:")
        print(f"  Mitigation applied: {mitigation_applied}")
        print(f"  Strategy: {strategy.get('action', 'none')}")
        print(f"  Bias after: {bias_score_after:.2f}%")
        
        return {'status': 'success', 'bias_after': bias_score_after}
        
    except Exception as e:
        print(f"Bias mitigation error: {str(e)}")
        raise


def generate_pipeline_summary(**context):
    """
    Task 6: Generate comprehensive pipeline execution summary
    """
    start_time = time.time()
    ti = context['ti']
    
    # Get project root
    project_root = get_project_root()
    logs_path = project_root / 'data-pipeline' / 'logs'
    
    # Pull all metrics from XCom
    metrics = {
        'data_size_mb': ti.xcom_pull(task_ids='check_data', key='data_size_mb') or 0,
        'records_processed': ti.xcom_pull(task_ids='preprocess_data', key='records_processed') or 0,
        'duplicates_removed': ti.xcom_pull(task_ids='preprocess_data', key='duplicates_removed') or 0,
        'validation_score': ti.xcom_pull(task_ids='validate_data', key='validation_score') or 0,
        'features_created': ti.xcom_pull(task_ids='engineer_features', key='features_created') or 0,
        'bias_score_raw': ti.xcom_pull(task_ids='detect_bias', key='bias_score_raw') or 0,
        'age_cv_raw': ti.xcom_pull(task_ids='detect_bias', key='age_cv_raw') or 0,
        'age_cv_adjusted': ti.xcom_pull(task_ids='detect_bias', key='age_cv_adjusted') or 0,
        'gender_cv': ti.xcom_pull(task_ids='detect_bias', key='gender_cv') or 0,
        'ethnicity_cv': ti.xcom_pull(task_ids='detect_bias', key='ethnicity_cv') or 0,
        'bias_interpretation': ti.xcom_pull(task_ids='detect_bias', key='bias_interpretation') or 'UNKNOWN',
        'mitigation_applied': ti.xcom_pull(task_ids='mitigate_bias', key='mitigation_applied') or False,
        'mitigation_strategy': ti.xcom_pull(task_ids='mitigate_bias', key='mitigation_strategy') or 'none',
        'bias_score_after': ti.xcom_pull(task_ids='mitigate_bias', key='bias_score_after') or 0,
        'bias_improvement_pct': ti.xcom_pull(task_ids='mitigate_bias', key='bias_improvement_pct') or 0,
        'requires_review': ti.xcom_pull(task_ids='mitigate_bias', key='requires_review') or False
    }
    
    # Build summary
    summary = {
        'pipeline_metadata': {
            'execution_time': datetime.now().isoformat(),
            'dag_run_id': context['dag_run'].run_id,
            'status': 'SUCCESS'
        },
        'data_metrics': {
            'raw_data_size_mb': metrics['data_size_mb'],
            'records_processed': metrics['records_processed'],
            'duplicates_removed': metrics['duplicates_removed'],
            'features_created': metrics['features_created']
        },
        'quality_assessment': {
            'validation_score': metrics['validation_score'],
            'validation_status': 'PASS' if metrics['validation_score'] >= 80 else 'FAIL',
            'data_quality_level': (
                'EXCELLENT' if metrics['validation_score'] >= 90 else
                'GOOD' if metrics['validation_score'] >= 80 else
                'NEEDS_IMPROVEMENT'
            )
        },
        'bias_analysis': {
            'raw_bias_score': metrics['bias_score_raw'],
            'age_cv_adjusted': metrics['age_cv_adjusted'],
            'interpretation': metrics['bias_interpretation'],
            'mitigation_applied': metrics['mitigation_applied'],
            'bias_score_after': metrics['bias_score_after'],
            'improvement_percentage': metrics['bias_improvement_pct'],
            'final_bias_status': (
                'ACCEPTABLE' if metrics['bias_score_after'] <= 10 else
                'MODERATE' if metrics['bias_score_after'] <= 15 else
                'HIGH'
            ),
            'requires_institutional_review': metrics['requires_review']
        },
        'recommendations': []
    }
    
    # Generate recommendations
    if metrics['validation_score'] < 80:
        summary['recommendations'].append(
            "Data quality below threshold. Review validation report before model training."
        )
    
    if metrics['age_cv_adjusted'] > 20:
        summary['recommendations'].append(
            "CRITICAL: Severe bias detected. Institutional review required."
        )
    
    if metrics['mitigation_applied'] and metrics['bias_improvement_pct'] > 50:
        summary['recommendations'].append(
            f"Mitigation effective ({metrics['bias_improvement_pct']:.1f}% improvement). Use mitigated dataset for ML."
        )
    
    if metrics['validation_score'] >= 80 and metrics['bias_score_after'] <= 10:
        summary['recommendations'].append(
            "Pipeline successful. Data ready for ML model training."
        )
    
    # Save summary
    summary_path = logs_path / 'airflow_pipeline_summary.json'
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    track_performance('generate_summary', start_time, context)
    
    # Print summary
    print("\n" + "="*70)
    print("PIPELINE SUMMARY")
    print("="*70)
    print(f"Records: {metrics['records_processed']}")
    print(f"Features: {metrics['features_created']}")
    print(f"Validation: {metrics['validation_score']:.2f}% [{summary['quality_assessment']['validation_status']}]")
    print(f"Bias (raw): {metrics['bias_score_raw']:.2f}%")
    print(f"Bias (adjusted CV): {metrics['age_cv_adjusted']:.2f}%")
    print(f"Bias (after mitigation): {metrics['bias_score_after']:.2f}%")
    print(f"Status: {summary['bias_analysis']['final_bias_status']}")
    print("="*70)
    
    return summary


# Define the DAG
dag = DAG(
    'lab_lens_mimic_pipeline',
    default_args=default_args,
    description='Optimized bias-aware MLOps pipeline for MIMIC-III (Docker-compatible)',
    schedule_interval=None,
    catchup=False,
    max_active_runs=1,
    tags=['healthcare', 'mlops', 'bias-detection', 'fairness', 'mimic-iii']
)

# Define tasks
check_data = PythonOperator(
    task_id='check_data',
    python_callable=check_data_availability,
    dag=dag
)

preprocess = PythonOperator(
    task_id='preprocess_data',
    python_callable=run_preprocessing_task,
    dag=dag
)

validate = PythonOperator(
    task_id='validate_data',
    python_callable=run_validation_task,
    dag=dag
)

engineer_features = PythonOperator(
    task_id='engineer_features',
    python_callable=run_feature_engineering_task,
    dag=dag
)

detect_bias = PythonOperator(
    task_id='detect_bias',
    python_callable=run_bias_detection_task,
    dag=dag
)

mitigate_bias = PythonOperator(
    task_id='mitigate_bias',
    python_callable=run_bias_mitigation_task,
    dag=dag
)

generate_summary = PythonOperator(
    task_id='generate_summary',
    python_callable=generate_pipeline_summary,
    dag=dag,
    trigger_rule=TriggerRule.ALL_DONE
)

# Define task dependencies
check_data >> preprocess >> validate >> engineer_features >> detect_bias >> mitigate_bias >> generate_summary
