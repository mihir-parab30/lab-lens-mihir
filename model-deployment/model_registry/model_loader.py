# model_loader.py
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import os

class ModelLoader:
    """Flexible model loading from multiple sources"""
    
    def __init__(self, source="huggingface"):
        self.source = source
        self.model = None
        self.tokenizer = None
        
    def load_model(self):
        if self.source == "huggingface":
            model_id = "your-username/bart-medical-discharge-summarizer"
            print(f"📥 Loading model from Hugging Face: {model_id}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_id)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
            
        elif self.source == "local":
            model_path = "./fine_tuned_bart_large_cnn"
            print(f"📥 Loading model from local: {model_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
            
        print("✅ Model loaded successfully")
        return self.model, self.tokenizer