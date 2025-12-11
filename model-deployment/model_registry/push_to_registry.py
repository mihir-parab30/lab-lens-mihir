# prepare_and_push_hf.py
from huggingface_hub import HfApi, create_repo
import os
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

def push_to_huggingface():
    # Your HF credentials
    HF_USERNAME = os.environ.get("HF_USERNAME")
    HF_TOKEN = os.environ.get("HF_TOKEN")
    if not HF_TOKEN:
        raise ValueError("Please set HF_TOKEN environment variable")
        
    repo_id = f"{HF_USERNAME}/bart-medical-discharge-summarizer"
    
    api = HfApi()
    
    # Create repo if doesn't exist
    try:
        create_repo(
            repo_id=repo_id,
            token=HF_TOKEN,
            private=False,
            repo_type="model"
        )
        print(f"✅ Created repository: {repo_id}")
    except:
        print(f"📦 Repository already exists: {repo_id}")
    
    # Upload your local folder
    api.upload_folder(
        folder_path="./fine_tuned_bart_large_cnn",
        repo_id=repo_id,
        token=HF_TOKEN,
        commit_message="Upload fine-tuned BART v1.0.0"
    )
    
    print(f"🎉 Model uploaded to: https://huggingface.co/{repo_id}")
    return repo_id

if __name__ == "__main__":
    repo_id = push_to_huggingface()