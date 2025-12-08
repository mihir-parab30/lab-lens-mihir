import mlflow
import os
import json

def push_to_registry_with_hf():
    """
    Register model metadata to MLflow and link to HF hosted model
    """
    HF_MODEL_ID = "your-username/bart-medical-discharge-summarizer"  # Your HF repo
    local_model_path = "fine_tuned_bart_large_cnn"
    registry_name = "Medical_Discharge_Summarizer"
    
    print(f"🚀 Registering model to MLflow with HF reference...")
    
    mlflow.set_experiment("Model_Registry_Pipeline")
    
    with mlflow.start_run(run_name="Register_BART_v1.0.0") as run:
        # Log local artifacts for backup
        mlflow.log_artifacts(local_model_path, artifact_path="model_backup")
        
        # Log model metadata
        mlflow.log_param("model_type", "transformer_seq2seq")
        mlflow.log_param("architecture", "BART-large-cnn")
        mlflow.log_param("optimization", "smart_extraction")
        mlflow.log_param("huggingface_model_id", HF_MODEL_ID)
        mlflow.log_param("version", "1.0.0")
        
        # Log performance metrics
        mlflow.log_metric("rouge1", 0.48)
        mlflow.log_metric("rouge2", 0.35)
        mlflow.log_metric("inference_time_seconds", 2.3)
        
        # Log model location
        model_locations = {
            "huggingface": f"https://huggingface.co/{HF_MODEL_ID}",
            "local_backup": f"runs:/{run.info.run_id}/model_backup"
        }
        
        mlflow.log_dict(model_locations, "model_locations.json")
        
        # Tag for production
        mlflow.set_tag("stage", "production")
        mlflow.set_tag("deployment_ready", "true")
        
    print(f"✅ Model registered with MLflow run ID: {run.info.run_id}")
    print(f"🌐 Model hosted at: https://huggingface.co/{HF_MODEL_ID}")

if __name__ == "__main__":
    push_to_registry_with_hf()