# model-development/scripts/hybrid_biobart_rag.py
"""
Hybrid Medical Summarization System
Combines BioBART for medical extraction with RAG for simplification
"""

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
import numpy as np
import pandas as pd
import re
from typing import Dict, List, Tuple
import pickle
import os
from google import genai
from google.genai.errors import APIError

class HybridMedicalSummarizer:
    """
    Production-ready medical summarization combining:
    1. BioBART for accurate medical information extraction
    2. RAG for medical term simplification
    3. Post-processing for patient-friendly output
    """
    # In HybridMedicalSummarizer class

    def _call_external_llm_api(self, prompt: str) -> str:
        """Handles the secure call to the Gemini API."""
        
        # 1. Initialize client. It automatically picks up GEMINI_API_KEY
        try:
            client = genai.Client()
        except Exception as e:
            # Check if the key is missing or invalid
            if "GEMINI_API_KEY" not in os.environ:
                raise EnvironmentError("GEMINI_API_KEY environment variable is not set. Please set it before running.")
            else:
                raise e

        # 2. Define the system instruction to set the model's persona
        system_instruction = "You are a medical patient advocate writing a discharge summary. Your output MUST follow the user-defined structure exactly as requested in the prompt."

        # 3. Call the API
        try:
            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=[prompt],
                config={
                    "system_instruction": system_instruction,
                    "temperature": 0.2,  # Low temperature for factual, structured output
                }
            )
            return response.text.strip()
            
        except APIError as e:
            print(f"Gemini API Error: {e}")
            # Allows for the script to continue to the fallback method
            raise APIError(f"Gemini API call failed: {e}")
    
    def __init__(self, use_gpu: bool = False):
        """Initialize the hybrid system"""
        
        print("Initializing Hybrid Medical Summarization System...")
        
        # Set device
        self.device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # 1. Load BioBART for medical summarization
        print("Loading BioBART model...")
        self.biobart_tokenizer = AutoTokenizer.from_pretrained("GanjinZero/biobart-base")
        self.biobart_model = AutoModelForSeq2SeqLM.from_pretrained("GanjinZero/biobart-base")
        self.biobart_model.to(self.device)

        #1.1 Testing BAT-large-cnn
        print("Loading BART-large-cnn for comparison...")
        self.bart_tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
        self.bart_model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")
        self.bart_model.to(self.device)
        
        # 2. Load embedding model for RAG
        print("Loading BioBERT embedder...")
        self.embedder = SentenceTransformer('dmis-lab/biobert-base-cased-v1.2')
        
        # 3. Build medical simplification knowledge base
        print("Building simplification knowledge base...")
        self.build_simplification_kb()
        
        # 4. Create vector index for RAG
        print("Creating retrieval index...")
        self.create_retrieval_index()
        
        print("✅ System ready!")
    
    def build_simplification_kb(self):
        """Build comprehensive medical term simplification database"""
        
        self.medical_kb = [
            # Critical conditions
            {"medical": "myocardial infarction MI", "simple": "heart attack", "context": "blocked blood flow to heart"},
            {"medical": "cerebrovascular accident CVA stroke", "simple": "stroke", "context": "blood flow problem in brain"},
            {"medical": "acute coronary syndrome", "simple": "heart emergency", "context": "reduced blood to heart"},
            {"medical": "pulmonary embolism PE", "simple": "blood clot in lungs", "context": "blocks lung blood flow"},
            {"medical": "sepsis bacteremia", "simple": "severe blood infection", "context": "infection spread through body"},
            
            # Common diagnoses
            {"medical": "pneumonia", "simple": "lung infection", "context": "fluid in lungs from infection"},
            {"medical": "congestive heart failure CHF", "simple": "weak heart pumping", "context": "heart can't pump enough blood"},
            {"medical": "chronic obstructive pulmonary disease COPD", "simple": "breathing disease", "context": "damaged lungs make breathing hard"},
            {"medical": "diabetes mellitus", "simple": "high blood sugar", "context": "body can't control sugar levels"},
            {"medical": "hypertension HTN", "simple": "high blood pressure", "context": "blood pushes too hard on vessels"},
            {"medical": "atrial fibrillation afib", "simple": "irregular heartbeat", "context": "heart rhythm is off"},
            {"medical": "deep vein thrombosis DVT", "simple": "leg blood clot", "context": "clot in deep leg vein"},
            {"medical": "urinary tract infection UTI", "simple": "bladder infection", "context": "bacteria in urinary system"},
            {"medical": "acute kidney injury AKI renal failure", "simple": "kidney failure", "context": "kidneys stop filtering blood"},
            {"medical": "cirrhosis", "simple": "liver scarring", "context": "damaged liver from disease"},
            
            # Hemorrhages/Bleeding
            {"medical": "intracranial hemorrhage", "simple": "brain bleeding", "context": "bleeding inside skull"},
            {"medical": "brainstem hemorrhage", "simple": "bleeding in brain stem", "context": "bleeding in brain's control center"},
            {"medical": "gastrointestinal bleed GI bleed", "simple": "stomach or intestine bleeding", "context": "bleeding in digestive system"},
            {"medical": "subdural hematoma", "simple": "blood collection near brain", "context": "blood between brain and skull"},
            
            # Procedures
            {"medical": "coronary artery bypass graft CABG", "simple": "heart bypass surgery", "context": "reroute blood around blocked arteries"},
            {"medical": "percutaneous coronary intervention PCI", "simple": "opening heart arteries", "context": "balloon and stent to open blockage"},
            {"medical": "cardiac catheterization", "simple": "heart artery test", "context": "camera looks at heart vessels"},
            {"medical": "endoscopy EGD", "simple": "stomach camera test", "context": "camera looks inside stomach"},
            {"medical": "bronchoscopy", "simple": "lung camera test", "context": "camera looks in airways"},
            {"medical": "tracheostomy", "simple": "breathing tube in neck", "context": "hole in neck for breathing"},
            {"medical": "percutaneous endoscopic gastrostomy PEG", "simple": "feeding tube in stomach", "context": "tube through skin for feeding"},
            {"medical": "dialysis hemodialysis", "simple": "blood cleaning machine", "context": "machine filters blood for failed kidneys"},
            
            # Symptoms/Signs
            {"medical": "dyspnea", "simple": "trouble breathing", "context": "shortness of breath"},
            {"medical": "syncope", "simple": "fainting", "context": "brief loss of consciousness"},
            {"medical": "angina chest pain", "simple": "chest pain", "context": "heart not getting enough oxygen"},
            {"medical": "edema", "simple": "swelling", "context": "fluid buildup in body"},
            {"medical": "hypoxia", "simple": "low oxygen", "context": "not enough oxygen in blood"},
            {"medical": "tachycardia", "simple": "fast heartbeat", "context": "heart beating too fast"},
            {"medical": "bradycardia", "simple": "slow heartbeat", "context": "heart beating too slow"},
            {"medical": "hypotension", "simple": "low blood pressure", "context": "blood pressure too low"},
            
            # Anatomical terms
            {"medical": "bilateral", "simple": "both sides", "context": "left and right"},
            {"medical": "unilateral", "simple": "one side", "context": "only left or right"},
            {"medical": "anterior", "simple": "front", "context": "front part"},
            {"medical": "posterior", "simple": "back", "context": "back part"},
            {"medical": "proximal", "simple": "near center", "context": "close to body center"},
            {"medical": "distal", "simple": "far from center", "context": "away from body center"},
            
            # Dispositions
            {"medical": "discharge home", "simple": "going home", "context": "well enough to leave hospital"},
            {"medical": "extended care facility ECF SNF", "simple": "nursing facility", "context": "place for ongoing care"},
            {"medical": "rehabilitation rehab", "simple": "recovery center", "context": "place to regain strength"},
            {"medical": "hospice", "simple": "comfort care", "context": "care focused on comfort"},
            
            # Medications (categories)
            {"medical": "anticoagulants blood thinners", "simple": "blood thinners", "context": "prevents blood clots"},
            {"medical": "antibiotics", "simple": "infection medicine", "context": "kills bacteria"},
            {"medical": "analgesics pain medication", "simple": "pain medicine", "context": "reduces pain"},
            {"medical": "antihypertensives", "simple": "blood pressure medicine", "context": "lowers blood pressure"},
            {"medical": "diuretics water pills", "simple": "water pills", "context": "removes extra fluid"},
            {"medical": "insulin", "simple": "diabetes medicine", "context": "controls blood sugar"},
        ]
        
        # Create embeddings for retrieval
        self.kb_embeddings = []
        for entry in self.medical_kb:
            text = f"{entry['medical']} {entry['context']}"
            embedding = self.embedder.encode(text)
            self.kb_embeddings.append(embedding)
        self.kb_embeddings = np.array(self.kb_embeddings)
    
    def create_retrieval_index(self):
        """Create sklearn NearestNeighbors index for fast retrieval"""
        self.retriever = NearestNeighbors(
            n_neighbors=5,
            metric='cosine',
            algorithm='brute'  # For small dataset
        )
        self.retriever.fit(self.kb_embeddings)
    
    def extract_key_sections(self, discharge_text: str) -> Dict[str, str]:
        """Extract structured information from discharge summary"""
        
        sections = {
            'diagnosis': '',
            'procedures': '',
            'disposition': '',
            'course': '',
            'medications': ''
        }
        
        text_lower = discharge_text.lower()
        
        # Extract diagnosis
        diagnosis_patterns = [
            r'discharge diagnosis[:\s]+([^\n]+)',
            r'final diagnosis[:\s]+([^\n]+)',
            r'primary diagnosis[:\s]+([^\n]+)',
            r'diagnosis[:\s]+([^\n]+)'
        ]
        for pattern in diagnosis_patterns:
            match = re.search(pattern, text_lower)
            if match:
                sections['diagnosis'] = match.group(1).strip()
                break
        
        # Extract disposition
        disposition_patterns = [
            r'discharge disposition[:\s]+([^\n]+)',
            r'disposition[:\s]+([^\n]+)',
            r'discharge to[:\s]+([^\n]+)'
        ]
        for pattern in disposition_patterns:
            match = re.search(pattern, text_lower)
            if match:
                sections['disposition'] = match.group(1).strip()
                break
        
        # Extract procedures
        procedure_patterns = [
            r'procedures?[:\s]+([^\n]+)',
            r'operations?[:\s]+([^\n]+)',
            r'surgical procedures?[:\s]+([^\n]+)'
        ]
        for pattern in procedure_patterns:
            match = re.search(pattern, text_lower)
            if match:
                sections['procedures'] = match.group(1).strip()
                break
        
        # Extract hospital course (first 500 chars if found)
        course_pattern = r'hospital course[:\s]+(.{,500})'
        course_match = re.search(course_pattern, text_lower, re.DOTALL)
        if course_match:
            sections['course'] = course_match.group(1).strip()
        
        return sections
    
    def smart_extract_for_summary(self, text: str, max_tokens: int = 1024) -> str:
        """
        Intelligently extract most relevant content for summarization.
        Handles both structured and unstructured discharge summaries.
        """
        
        # First, try structured extraction (your existing method)
        sections = self.extract_key_sections(text)
        
        # Build initial text from structured sections if found
        extracted_parts = []
        
        # Priority 1: Diagnosis (most important)
        if sections['diagnosis']:
            extracted_parts.append(f"Discharge Diagnosis: {sections['diagnosis']}")
        
        # Priority 2: Hospital course (what happened)
        if sections['course']:
            # Take first 300 words of course
            course_words = sections['course'].split()[:300]
            extracted_parts.append(f"Hospital Course: {' '.join(course_words)}")
        
        # Priority 3: Procedures
        if sections['procedures']:
            extracted_parts.append(f"Procedures: {sections['procedures']}")
        
        # Priority 4: Disposition
        if sections['disposition']:
            extracted_parts.append(f"Discharge Disposition: {sections['disposition']}")
        
        # If we got structured content, use it
        if extracted_parts:
            structured_text = '\n'.join(extracted_parts)
            
            # Check token count
            if len(structured_text.split()) < max_tokens * 0.7:  # If under 70% limit
                # Add more content from original text if space allows
                remaining_words = max_tokens - len(structured_text.split())
                
                # Look for other important content not captured
                text_lower = text.lower()
                
                # Important keywords to look for
                important_patterns = [
                    (r'admitted (?:for|with|due to)[^.]+\.', 50),  # Admission reason
                    (r'found to have[^.]+\.', 50),  # Findings
                    (r'diagnosed with[^.]+\.', 50),  # Diagnosis
                    (r'underwent[^.]+\.', 50),  # Procedures
                    (r'will need[^.]+\.', 50),  # Future care
                    (r'unable to[^.]+\.', 30),  # Limitations
                    (r'follow[- ]?up[^.]+\.', 50),  # Follow-up
                ]
                
                for pattern, weight in important_patterns:
                    if remaining_words <= 0:
                        break
                    matches = re.findall(pattern, text_lower)
                    for match in matches[:2]:  # Max 2 matches per pattern
                        if match not in structured_text.lower():
                            extracted_parts.append(match.capitalize())
                            remaining_words -= len(match.split())
            
            return '\n'.join(extracted_parts)
    
        # FALLBACK: No clear sections found - extract intelligently from unstructured text
        else:
            return self.extract_from_unstructured(text, max_tokens)

    def extract_from_unstructured(self, text: str, max_tokens: int = 1024) -> str:
        """
        Extract key information from unstructured discharge text.
        Used when clear sections aren't found.
        """
        
        sentences = text.split('. ')
        scored_sentences = []
        
        # Keywords that indicate important medical information
        high_priority_keywords = [
            'diagnosis', 'diagnosed', 'discharge', 'admitted', 
            'hemorrhage', 'bleeding', 'infection', 'failure',
            'pneumonia', 'stroke', 'attack', 'cancer'
        ]
        
        medium_priority_keywords = [
            'treated', 'underwent', 'procedure', 'surgery',
            'medication', 'prescribed', 'will need', 'requires',
            'transfer', 'facility', 'home', 'follow'
        ]
        
        low_priority_keywords = [
            'stable', 'improved', 'history', 'presented',
            'noted', 'showed', 'revealed', 'consistent'
        ]
        
        # Score each sentence
        for sent in sentences:
            sent_lower = sent.lower()
            score = 0
            
            # High priority
            for keyword in high_priority_keywords:
                if keyword in sent_lower:
                    score += 3
            
            # Medium priority
            for keyword in medium_priority_keywords:
                if keyword in sent_lower:
                    score += 2
            
            # Low priority
            for keyword in low_priority_keywords:
                if keyword in sent_lower:
                    score += 1
            
            # Bonus for being in first or last 20% of document
            position = sentences.index(sent) / len(sentences)
            if position < 0.2 or position > 0.8:
                score += 1
            
            scored_sentences.append((score, sent))
        
        # Sort by score
        scored_sentences.sort(reverse=True, key=lambda x: x[0])
        
        # Build output up to token limit
        output_sentences = []
        word_count = 0
        
        for score, sent in scored_sentences:
            sent_words = len(sent.split())
            if word_count + sent_words < max_tokens:
                output_sentences.append(sent)
                word_count += sent_words
            else:
                break
        
        # Re-order chronologically (maintain narrative flow)
        final_sentences = []
        for sent in sentences:
            if sent in output_sentences:
                final_sentences.append(sent)
        
        return '. '.join(final_sentences)
    
    def summarize_with_bart(self, text: str, max_length: int = 150) -> str:
        """Generate summary using BART-large-cnn for comparison"""
        
        # Use smart extraction
        input_text = self.smart_extract_for_summary(text, max_tokens=1024)
        
        print(f"   Extracted {len(input_text.split())} words for BART")
        
        # Tokenize
        inputs = self.bart_tokenizer(
            input_text,
            max_length=1024,
            truncation=True,
            return_tensors="pt",
            padding=True
        ).to(self.device)
        
        # Generate summary
        with torch.no_grad():
            summary_ids = self.bart_model.generate(
                inputs.input_ids,
                max_length=max_length,
                min_length=30,
                length_penalty=2.0,
                num_beams=4,
                early_stopping=True,
                no_repeat_ngram_size=3,
            )
        
        # Decode
        summary = self.bart_tokenizer.decode(
            summary_ids[0],
            skip_special_tokens=True
        )
        
        return summary
    
    def summarize_with_biobart(self, text: str, max_length: int = 150) -> str:
        """Generate medical summary using BioBART"""
        
        # Prepare input - focus on key medical information
        input_text = self.smart_extract_for_summary(text, max_tokens=1024)
        
        print(f"   Extracted {len(input_text.split())} words for BioBART")


        # Add instruction prefix for better results
        prompted_text = input_text
        
        # Tokenize
        inputs = self.biobart_tokenizer(
            prompted_text,
            max_length=1024,
            truncation=True,
            return_tensors="pt",
            padding=True
        ).to(self.device)
        
        # Generate summary
        with torch.no_grad():
            summary_ids = self.biobart_model.generate(
                inputs.input_ids,
                max_length=max_length,
                min_length=30,
                length_penalty=2.0,
                num_beams=4,
                early_stopping=True,
                no_repeat_ngram_size=3,
            )
        
        # Decode
        summary = self.biobart_tokenizer.decode(
            summary_ids[0],
            skip_special_tokens=True
        )

        # Clean up
        summary = summary.replace("Summarize this discharge summary:", "").strip()
        summary = re.sub(r'[^\x00-\x7F]+', ' ', summary)  # Remove non-ASCII
        summary = ' '.join(summary.split())  # Fix spacing
        
        return summary
    
    def retrieve_simplifications(self, text: str, k: int = 5) -> List[Dict]:
        """Retrieve relevant medical term simplifications"""
        
        # Encode query
        query_embedding = self.embedder.encode(text.lower())
        
        # Find nearest neighbors
        distances, indices = self.retriever.kneighbors(
            query_embedding.reshape(1, -1),
            n_neighbors=min(k, len(self.medical_kb))
        )
        
        # Get relevant entries
        relevant = []
        for idx in indices[0]:
            if idx < len(self.medical_kb):
                relevant.append(self.medical_kb[idx])
        
        return relevant
    
    def simplify_medical_terms(self, text: str, simplifications: List[Dict]) -> str:
        """Replace medical terms with simple explanations"""
        
        simplified = text
        
        # Sort by length (longer terms first to avoid partial replacements)
        sorted_terms = sorted(simplifications, 
                            key=lambda x: len(x['medical']), 
                            reverse=True)
        
        for entry in sorted_terms:
            # Try to find and replace medical terms
            medical_terms = entry['medical'].split()
            for term in medical_terms:
                if len(term) > 3:  # Skip short words
                    # Case-insensitive replacement
                    pattern = re.compile(re.escape(term), re.IGNORECASE)
                    if pattern.search(simplified):
                        simplified = pattern.sub(entry['simple'], simplified)
                        break  # Only replace once per entry
        
        return simplified

    def _construct_llm_refinement_prompt(self, sections: Dict, simplified_biobart: str) -> str:
        """Creates the detailed instruction prompt for the external LLM."""
        
        diagnosis_clean = self.simplify_medical_terms(sections['diagnosis'], self.medical_kb).strip().capitalize()
        disposition_clean = self.simplify_medical_terms(sections['disposition'], self.medical_kb).strip().capitalize()
        procedures_clean = self.simplify_medical_terms(sections['procedures'], self.medical_kb).strip()
        
        # Use Markdown for emphasis, the LLM will usually preserve this structure
        prompt = f"""
        You are a medical patient advocate. Your task is to rewrite the provided clinical summary into a clear, empathetic, and structured summary for the patient.

        **Instructions & Constraints:**
        1.  **Target Audience:** 6th-grade reading level. Use simple, non-medical language.
        2.  **Tone:** Empathetic, supportive, and informative.
        3.  **Required Structure:** The final output MUST be organized under the following three EXACT BOLDED HEADINGS. Do not add any other text outside this structure.
            - **What Happened** (Describe the main diagnosis and the cause simply)
            - **What Was Done** (Summarize key procedures and treatments during the stay)
            - **What is Next** (Explain the discharge destination and next steps for recovery)
        4.  **Do not include any other text or introductions.**

        **CLINICAL FACTS TO USE:**
        - Primary Diagnosis: {diagnosis_clean}
        - Key Procedures: {procedures_clean if procedures_clean else 'None listed'}
        - Discharge Destination: {disposition_clean}

        **DETAILED HOSPITAL COURSE SUMMARY (from BioBART/RAG):**
        {simplified_biobart}

        **FINAL OUTPUT MUST FOLLOW THE REQUIRED STRUCTURE.**
        """
        return prompt

    def create_patient_summary(self, sections: Dict, biobart_summary: str, 
                                simplifications: List[Dict]) -> str:
        """Main method: Generates patient-friendly summary using LLM for refinement."""
        
        # 1. Simplify the BioBART summary using RAG (still necessary)
        simplified_biobart = self.simplify_medical_terms(biobart_summary, simplifications)
        
        # 2. Construct the detailed prompt
        prompt = self._construct_llm_refinement_prompt(sections, simplified_biobart)
        
        # 3. Call the external LLM with robust error handling
        try:
            patient_summary = self._call_external_llm_api(prompt) 
            
            # Simple structural check
            if "What Happened" in patient_summary and "What is Next" in patient_summary:
                return patient_summary
            else:
                raise ValueError("LLM returned an invalid structure.")
                
        except (APIError, EnvironmentError, ValueError) as e:
            print(f"⚠️ LLM processing failed. Falling back to rule-based summary: {e}")
            # Fallback to the original rule-based method
            return self._create_patient_summary_fallback(sections, simplified_biobart, simplifications)
        

    
    def create_patient_summary_fallback(self, sections: Dict, biobart_summary: str, 
                             simplifications: List[Dict]) -> str:
        """Create final patient-friendly summary"""
        
        # Start with structured format
        summary_parts = []
        
        # 1. What happened (diagnosis)
        if sections['diagnosis']:
            diagnosis = sections['diagnosis']
            # Simplify diagnosis
            for entry in simplifications:
                if any(term in diagnosis.lower() for term in entry['medical'].split()):
                    diagnosis = entry['simple']
                    break
            summary_parts.append(f"**What happened:** You had {diagnosis}.")
        
        # 2. What was done (procedures)
        if sections['procedures']:
            procedures = self.simplify_medical_terms(sections['procedures'], simplifications)
            summary_parts.append(f"**Treatment:** You had {procedures}.")
        
        # 3. Where you're going
        if sections['disposition']:
            disposition = sections['disposition']
            if 'home' in disposition.lower():
                summary_parts.append("**Going home:** You're well enough to go home.")
            elif 'extended' in disposition.lower() or 'snf' in disposition.lower():
                summary_parts.append("**Next step:** You'll go to a care facility to continue recovering.")
            elif 'rehab' in disposition.lower():
                summary_parts.append("**Next step:** You'll go to rehabilitation to regain strength.")
        
        # 4. Add key points from BioBART summary if not redundant
        if biobart_summary and len(summary_parts) < 3:
            # Simplify BioBART output
            simple_biobart = self.simplify_medical_terms(biobart_summary, simplifications)
            # Add if it contains new information
            if not any(part in simple_biobart for part in summary_parts):
                summary_parts.append(f"**Additional info:** {simple_biobart[:100]}")
        
        # Combine parts
        final_summary = " ".join(summary_parts)
        
        # Ensure reasonable length
        if len(final_summary) < 50:
            final_summary += " Please ask your doctor for more details about your condition."
        
        return final_summary
    
    
    def generate_summary(self, discharge_text: str) -> Dict[str, str]:
        """
        Main method: Generate patient-friendly summary using hybrid approach
        NOW WITH BART COMPARISON
        """
        
        print("\n" + "="*50)
        print("HYBRID SUMMARIZATION PIPELINE")
        print("="*50)
        
        # Step 1: Extract key sections
        print("1. Extracting key sections...")
        sections = self.extract_key_sections(discharge_text)
        
        # Step 2a: Generate medical summary with BioBART
        print("\n2a. Generating BioBART summary...")
        biobart_summary = self.summarize_with_biobart(discharge_text)
        print(f"   BioBART Output: {biobart_summary}")
        
        # Step 2b: Generate summary with BART-large-cnn for comparison
        print("\n2b. Generating BART-large-cnn summary...")
        bart_summary = self.summarize_with_bart(discharge_text)
        print(f"   BART Output: {bart_summary}")
        
        # Step 3: Retrieve relevant simplifications
        print("\n3. Retrieving medical simplifications...")
        combined_text = f"{sections['diagnosis']} {sections['procedures']} {biobart_summary}"
        simplifications = self.retrieve_simplifications(combined_text, k=10)
        print(f"   Found {len(simplifications)} relevant terms")
        
        # Step 4a: Create patient summary using BioBART
        print("\n4a. Creating patient summary (BioBART → Gemini)...")
        patient_summary_biobart = self.create_patient_summary(
            sections, biobart_summary, simplifications
        )
        
        # Step 4b: Create patient summary using BART
        print("4b. Creating patient summary (BART → Gemini)...")
        patient_summary_bart = self.create_patient_summary(
            sections, bart_summary, simplifications
        )
        
        # Step 5: Create simplified versions
        print("\n5. Creating RAG-simplified versions...")
        simplified_biobart = self.simplify_medical_terms(biobart_summary, simplifications)
        simplified_bart = self.simplify_medical_terms(bart_summary, simplifications)
        
        print("="*50)
        
        return {
            'patient_summary_biobart': patient_summary_biobart,
            'patient_summary_bart': patient_summary_bart,
            'biobart_original': biobart_summary,
            'bart_original': bart_summary,
            'biobart_simplified': simplified_biobart,
            'bart_simplified': simplified_bart,
            'diagnosis': sections['diagnosis'],
            'disposition': sections['disposition']
        }

    
    def evaluate_on_test_set(self, test_df: pd.DataFrame, sample_size: int = 10):
        """Evaluate the hybrid approach on test data"""
        
        results = []
        
        for idx in range(min(sample_size, len(test_df))):
            row = test_df.iloc[idx]
            
            # Generate summary
            summaries = self.generate_summary(row['cleaned_text'])
            
            results.append({
                'hadm_id': row.get('hadm_id', idx),
                'original_diagnosis': row.get('discharge_diagnosis', ''),
                'patient_summary': summaries['patient_summary'],
                'biobart_summary': summaries['biobart_original']
            })
            
            print(f"\n{'='*50}")
            print(f"Sample {idx+1}")
            print(f"Diagnosis: {summaries['diagnosis']}")
            print(f"Patient Summary: {summaries['patient_summary']}")
        
        return pd.DataFrame(results)
    
    def save_system(self, path: str):
        """Save the system for deployment"""
        os.makedirs(path, exist_ok=True)
        
        # Save knowledge base
        with open(f"{path}/medical_kb.pkl", 'wb') as f:
            pickle.dump(self.medical_kb, f)
        
        # Save embeddings
        np.save(f"{path}/kb_embeddings.npy", self.kb_embeddings)
        
        print(f"✅ System saved to {path}")


# Usage example
if __name__ == "__main__":
    
    # Initialize hybrid system
    print("Initializing Hybrid Medical Summarizer...")
    summarizer = HybridMedicalSummarizer(use_gpu=False)
    
    # Test with your brain hemorrhage case
    test_text = """
    Admission Date: [**] Discharge Date: [**]

    Service: NEUROLOGY

    Chief Complaint:
    Found unresponsive on floor.

    History of Present Illness:
    Patient is 67 yo M with history of schizophrenia, dementia, anxiety, 
    found lying face-down unresponsive. O2 sat in 40s, pupils fixed.

    Hospital Course:
    Patient admitted to NeuroICU for management of brainstem hemorrhage. 
    Blood pressure maintained <160. Patient remained minimally responsive 
    for first 2 weeks. Day 14 showed improvement with eye opening. 
    Developed ventilator-associated pneumonia, treated with vancomycin 
    and cefepime 8-day course. Required tracheostomy and PEG placement 
    for long-term management.

    Discharge Diagnosis:
    Brainstem hemorrhage involving R midbrain, bilateral pons, and 
    4th ventricle, likely hypertensive etiology.

    Discharge Medications:
    [List medications...]

    Discharge Disposition:
    Extended Care Facility

    Discharge Instructions:
    Patient suffered major brain hemorrhage resulting in inability to 
    move, eat or speak. Requires ongoing medical care and therapy.
    """
    
    # Generate summary
    print("\nProcessing discharge summary...")
    results = summarizer.generate_summary(test_text)
    
    print("\n" + "="*60)
    print("COMPARISON RESULTS")
    print("="*60)
    
    print("\n📊 BIOBART PATH:")
    print("-"*50)
    print("Raw Output:", results['biobart_original'][:200])
    print("\nRAG Simplified:", results['biobart_simplified'][:200])
    print("\nFinal Patient Summary:")
    print(results['patient_summary_biobart'])
    
    print("\n" + "="*60)
    
    print("\n📊 BART-LARGE-CNN PATH:")
    print("-"*50)
    print("Raw Output:", results['bart_original'][:200])
    print("\nRAG Simplified:", results['bart_simplified'][:200])
    print("\nFinal Patient Summary:")
    print(results['patient_summary_bart'])
    
    print("\n" + "="*60)
    print("Smart extraction successfully handled text length!")
    print(f"Original text: {len(test_text.split())} words")
    print("="*60)
    
    # Save the system
    summarizer.save_system("../models/hybrid_system")