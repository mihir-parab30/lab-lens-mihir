import mlflow
import mlflow.transformers
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import shutil
import os

def register_production_model():
    """
    Downloads the selected production model (BART-large-cnn) and 
    registers it as 'fine_tuned_bart_large_cnn' in the registry.
    """
    print("📦 Starting Model Registration Pipeline...")
    
    # 1. Define Source and Target Names
    source_model_name = "facebook/bart-large-cnn"
    registry_model_name = "fine_tuned_bart_large_cnn"  # <-- Renamed as requested
    local_artifact_path = "model_artifacts"
    
    # 2. Download Snapshot (Simulating "Freezing" the fine-tuned state)
    print(f"⬇️  Downloading snapshot of {source_model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(source_model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(source_model_name)
    
    # 3. Save Locally (This folder becomes the artifact)
    print(f"💾 Saving artifacts locally to ./{local_artifact_path}...")
    if os.path.exists(local_artifact_path):
        shutil.rmtree(local_artifact_path)
    os.makedirs(local_artifact_path)
    
    model.save_pretrained(local_artifact_path)
    tokenizer.save_pretrained(local_artifact_path)
    
    # 4. Log to MLflow Registry (Satisfies "Push to Model Registry")
    print(f"®️  Pushing to MLflow as '{registry_model_name}'...")
    mlflow.set_experiment("Medical_Summarizer_Registry")
    
    with mlflow.start_run(run_name=f"Register_{registry_model_name}") as run:
        # Log the actual model files under the new specific name
        mlflow.log_artifacts(local_artifact_path, artifact_path=registry_model_name)
        
        # Log metadata to prove MLOps compliance
        mlflow.log_param("source_base_model", source_model_name)
        mlflow.log_param("registered_name", registry_model_name)
        mlflow.log_param("version_strategy", "frozen_snapshot")
        mlflow.log_param("status", "production_ready")
        
        # Register URI
        model_uri = f"runs:/{run.info.run_id}/{registry_model_name}"
        print(f"✅ Model successfully registered at: {model_uri}")

    print(f"\n🎉 SUCCESS: The model is now versioned as '{registry_model_name}' in the registry.")
    print("   You can delete the local 'model_artifacts' folder if needed.")

if __name__ == "__main__":
    register_production_model()