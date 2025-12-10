# model-development/scripts/rag_medical_summarizer.py
"""
RAG-based Medical Discharge Summarization
Combines retrieval with your fine-tuned LLaMA model
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from sklearn.neighbors import NearestNeighbors
from sentence_transformers import SentenceTransformer
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import pickle
import json

class MedicalRAGSummarizer:
    """
    Retrieval-Augmented Generation for Medical Summarization
    """
    
    def __init__(self, model_path: str, knowledge_base_path: str = None):
        """
        Initialize RAG system
        
        Args:
            model_path: Path to your fine-tuned LLaMA model
            knowledge_base_path: Path to medical simplification database
        """
        print("Initializing RAG Medical Summarizer...")
        
        # 1. Load embedding model for retrieval
        print("Loading embedding model...")
        self.embedder = SentenceTransformer('pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb')
        
        # 2. Load your undertrained LLaMA (we'll make it useful!)
        print("Loading LLaMA model...")
        self.model_path = model_path
        self.load_llama_model()
        
        # 3. Initialize knowledge base
        print("Building knowledge base...")
        self.build_knowledge_base()
        
        # 4. Create FAISS index for fast retrieval
        print("Creating vector index...")
        self.create_vector_index()
    
    def load_llama_model(self):
        """Load your existing fine-tuned model"""
        base_model = "unsloth/llama-3.2-3b-instruct"
        
        # Load base model
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.float16,
            device_map="cpu",
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Apply LoRA adapter
        self.model = PeftModel.from_pretrained(self.model, self.model_path)
    
    def build_knowledge_base(self):
        """
        Build knowledge base of medical term simplifications
        In production, this would be a curated database
        """
        self.knowledge_base = [
            # Diagnoses
            {
                "medical": "myocardial infarction",
                "simple": "heart attack - your heart muscle didn't get enough blood",
                "context": "requires immediate treatment and lifestyle changes"
            },
            {
                "medical": "cerebrovascular accident CVA",
                "simple": "stroke - blood flow to part of your brain was blocked",
                "context": "may affect movement, speech, or thinking"
            },
            {
                "medical": "pneumonia",
                "simple": "lung infection - your lungs filled with fluid from infection",
                "context": "treated with antibiotics and rest"
            },
            {
                "medical": "sepsis bacteremia",
                "simple": "serious blood infection that spread through your body",
                "context": "requires strong antibiotics and close monitoring"
            },
            {
                "medical": "acute renal failure",
                "simple": "kidney failure - your kidneys stopped working properly",
                "context": "may need dialysis or special medications"
            },
            {
                "medical": "COPD exacerbation",
                "simple": "breathing problems got worse - your lung disease flared up",
                "context": "needs inhalers and possibly oxygen"
            },
            {
                "medical": "diabetic ketoacidosis DKA",
                "simple": "dangerous high blood sugar - your diabetes became severe",
                "context": "requires insulin and careful monitoring"
            },
            {
                "medical": "gastrointestinal bleeding",
                "simple": "bleeding in your stomach or intestines",
                "context": "may need endoscopy and blood transfusion"
            },
            {
                "medical": "atrial fibrillation",
                "simple": "irregular heartbeat - your heart rhythm is off",
                "context": "managed with blood thinners and heart medications"
            },
            {
                "medical": "deep vein thrombosis DVT",
                "simple": "blood clot in your leg vein",
                "context": "treated with blood thinners to prevent complications"
            },
            
            # Procedures
            {
                "medical": "cardiac catheterization",
                "simple": "procedure to look at your heart arteries with a small camera",
                "context": "helps doctors see blockages"
            },
            {
                "medical": "percutaneous coronary intervention PCI",
                "simple": "opening blocked heart arteries with a tiny balloon and stent",
                "context": "improves blood flow to heart"
            },
            {
                "medical": "bronchoscopy",
                "simple": "looking into your lungs with a small camera",
                "context": "helps diagnose lung problems"
            },
            {
                "medical": "endoscopy EGD",
                "simple": "looking into your stomach with a small camera",
                "context": "checks for ulcers or bleeding"
            },
            {
                "medical": "tracheostomy",
                "simple": "breathing tube placed through your neck",
                "context": "helps when you can't breathe on your own"
            },
            {
                "medical": "PEG tube placement",
                "simple": "feeding tube placed in your stomach",
                "context": "provides nutrition when you can't eat normally"
            },
            
            # Medications categories
            {
                "medical": "anticoagulants warfarin heparin",
                "simple": "blood thinners to prevent clots",
                "context": "requires regular monitoring"
            },
            {
                "medical": "antibiotics vancomycin ceftriaxone",
                "simple": "medicines to fight infection",
                "context": "take complete course as directed"
            },
            {
                "medical": "diuretics furosemide lasix",
                "simple": "water pills to reduce fluid buildup",
                "context": "helps with swelling and breathing"
            },
            {
                "medical": "beta blockers metoprolol carvedilol",
                "simple": "heart medications to control rate and pressure",
                "context": "helps protect your heart"
            },
            {
                "medical": "proton pump inhibitors omeprazole",
                "simple": "stomach acid reducers",
                "context": "protects stomach lining"
            },
            
            # Common medical terms
            {
                "medical": "bilateral",
                "simple": "on both sides",
                "context": "affects both left and right"
            },
            {
                "medical": "acute",
                "simple": "sudden or severe",
                "context": "happened quickly"
            },
            {
                "medical": "chronic",
                "simple": "long-term ongoing",
                "context": "been happening for a while"
            },
            {
                "medical": "exacerbation",
                "simple": "got worse",
                "context": "condition flared up"
            },
            {
                "medical": "edema",
                "simple": "swelling from fluid",
                "context": "body holding extra water"
            },
            {
                "medical": "hypoxia",
                "simple": "not enough oxygen",
                "context": "low oxygen levels"
            },
            {
                "medical": "tachycardia",
                "simple": "fast heart rate",
                "context": "heart beating too quickly"
            },
            {
                "medical": "bradycardia",
                "simple": "slow heart rate",
                "context": "heart beating too slowly"
            },
            {
                "medical": "hypertension",
                "simple": "high blood pressure",
                "context": "needs medication to control"
            },
            {
                "medical": "hypotension",
                "simple": "low blood pressure",
                "context": "may cause dizziness"
            }
        ]
        
        # Create embeddings for all entries
        self.kb_embeddings = []
        for entry in self.knowledge_base:
            # Embed the medical term + context
            text = f"{entry['medical']} {entry['context']}"
            embedding = self.embedder.encode(text)
            self.kb_embeddings.append(embedding)
        
        self.kb_embeddings = np.array(self.kb_embeddings)
    
    def create_vector_index(self):
        """Create FAISS index for fast similarity search"""
        dimension = self.kb_embeddings.shape[1]
        
        # Create FAISS index
        self.index = NearestNeighbors(n_neighbors=5, metric='cosine')
        self.index.fit(self.kb_embeddings)
    
    def retrieve_relevant_simplifications(self, text: str, k: int = 5) -> List[Dict]:
        """
        Retrieve relevant medical simplifications
        
        Args:
            text: Medical text to simplify
            k: Number of relevant entries to retrieve
        
        Returns:
            List of relevant simplification entries
        """
        # Embed the query text
        query_embedding = self.embedder.encode(text)
        
        # Search for similar entries
        distances, indices = self.index.kneighbors(query_embedding.reshape(1, -1), n_neighbors=k)

        
        # Get relevant entries
        relevant_entries = []
        for idx in indices[0]:
            if idx < len(self.knowledge_base):
                relevant_entries.append(self.knowledge_base[idx])
        
        return relevant_entries
    
    def extract_key_information(self, discharge_text: str) -> Dict:
        """Extract structured information from discharge summary"""
        info = {
            'diagnosis': '',
            'procedures': [],
            'disposition': '',
            'medications': []
        }
        
        # Simple extraction (in production, use NER models)
        lines = discharge_text.lower().split('\n')
        
        for line in lines:
            if 'diagnosis:' in line:
                info['diagnosis'] = line.split('diagnosis:')[1].strip()[:100]
            elif 'discharge disposition:' in line:
                info['disposition'] = line.split('disposition:')[1].strip()[:50]
            elif 'procedure:' in line or 'operation:' in line:
                info['procedures'].append(line.strip()[:100])
        
        return info
    
    def generate_rag_summary(self, discharge_text: str) -> str:
        """
        Generate summary using RAG approach
        
        Args:
            discharge_text: Full discharge summary
        
        Returns:
            Patient-friendly summary
        """
        # 1. Extract key medical information
        key_info = self.extract_key_information(discharge_text)
        
        # 2. Retrieve relevant simplifications
        relevant_context = self.retrieve_relevant_simplifications(
            discharge_text[:1000],  # Use first 1000 chars for retrieval
            k=5
        )
        
        # 3. Build context for generation
        context_text = "\n".join([
            f"- {entry['medical']} means {entry['simple']}"
            for entry in relevant_context
        ])
        
        # 4. Create enhanced prompt with retrieved context
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a medical assistant. Use the provided medical term definitions to explain the discharge summary in simple language.

Medical Term Definitions:
{context_text}

Instructions:
1. Use simple words from the definitions provided
2. Keep it under 100 words
3. Focus on: what happened, current condition, and next steps
4. Do NOT list medications<|eot_id|><|start_header_id|>user<|end_header_id|>

Discharge Summary:
Diagnosis: {key_info['diagnosis'][:200]}
Disposition: {key_info['disposition']}

Full Text: {discharge_text[:500]}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

Based on the medical definitions provided, here's what happened: """
        
        # 5. Generate with your model
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=150,
                temperature=0.3,  # Low for consistency
                top_p=0.9,
                repetition_penalty=1.2,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(
            outputs[0][len(inputs.input_ids[0]):], 
            skip_special_tokens=True
        )
        
        # 6. Post-process to ensure quality
        response = self.post_process_summary(response, relevant_context)
        
        return response
    
    def post_process_summary(self, summary: str, context: List[Dict]) -> str:
        """Clean up and enhance generated summary"""
        
        # Replace any remaining medical terms
        for entry in context:
            medical_terms = entry['medical'].split()
            for term in medical_terms:
                if len(term) > 4 and term in summary.lower():
                    summary = summary.replace(term, entry['simple'].split('-')[0].strip())
        
        # Ensure proper ending
        if not summary.strip().endswith('.'):
            summary = summary.strip() + '.'
        
        # Remove any medication lists that slipped through
        lines = summary.split('.')
        cleaned_lines = [line for line in lines if 'mg' not in line.lower()]
        summary = '. '.join(cleaned_lines)
        
        return summary.strip()
    
    def save_index(self, path: str):
        """Save the RAG system for deployment"""
        # Save FAISS index
        faiss.write_index(self.index, f"{path}/faiss_index.bin")
        
        # Save knowledge base
        with open(f"{path}/knowledge_base.pkl", 'wb') as f:
            pickle.dump(self.knowledge_base, f)
        
        print(f"RAG system saved to {path}")
    
    def evaluate_with_rag(self, test_df: pd.DataFrame, sample_size: int = 10):
        """Evaluate performance with RAG enhancement"""
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'])
        
        scores = []
        
        for idx in range(min(sample_size, len(test_df))):
            row = test_df.iloc[idx]
            
            # Generate RAG-enhanced summary
            rag_summary = self.generate_rag_summary(row['cleaned_text'])
            
            # If you have reference summaries, compute ROUGE
            # For now, we'll just collect the summaries
            print(f"\nSample {idx+1}:")
            print(f"Diagnosis: {row.get('discharge_diagnosis', 'N/A')[:50]}")
            print(f"RAG Summary: {rag_summary}")
            
            scores.append({
                'hadm_id': row.get('hadm_id', idx),
                'rag_summary': rag_summary
            })
        
        return pd.DataFrame(scores)


# Usage example
if __name__ == "__main__":
    # Initialize RAG system with your model
    rag_summarizer = MedicalRAGSummarizer(
        model_path="../saved_models/content/saved_model"
    )
    
    # Test with your discharge summary
    test_text = """
    Name: Unit No: Admission Date: Discharge Date: Date of Birth: Sex: M Service: ADDENDUM: The patient is a 72-year-old male with a known history of 3-vessel coronary artery disease who was transferred from an outside hospital for coronary artery bypass grafting after a positive exercise treadmill test and a catheterization which showed 3-vessel disease and an ejection fraction of 25%. Please refer to the previously dictated Discharge Summary for the period covering through HOSPITAL COURSE: The patient's discharged was delayed from through due to a lack of bed availability at the facility to which he was to be transferred. The patient was ultimately transferred from the on The only notable event during the period through was a syncopal episode the patient had while attempting to void in the bathroom. The syncopal episode coincided with a pause of several seconds in the patient's cardiac rhythm strip. The patient suffered no injury during the fall. The patient did state that he suffered similar syncopal episodes about once per week while at home, and they would usually occur when he had an over-full bladder and was attempting to void. The patient returned to a normal sinus rhythm following the episode. His mental status remained at baseline. The patient was already expected to undergo a workup for a pacemaker, and this event will obviously factor into the decision to proceed with placement. M.D. Dictated By: MEDQUIST36 D: 11:09 T: 03:24 JOB#:
    """
    
    # Generate RAG-enhanced summary
    summary = rag_summarizer.generate_rag_summary(test_text)
    print(f"RAG-Enhanced Summary: {summary}")
    
    # Save for deployment
    rag_summarizer.save_index("../models/rag_system")