---
language: en
tags:
- medical
- summarization
- bart
- discharge-summaries
license: apache-2.0
metrics:
- rouge
widget:
- text: "Discharge Diagnosis: Myocardial infarction. Hospital Course: Patient admitted with chest pain..."
---

# Fine-tuned BART for Medical Discharge Summarization

## Model Description
Fine-tuned BART-large-CNN on MIMIC III dataset optimized for medical discharge summary simplification with smart extraction preprocessing.

## Training Configuration
- Base Model: facebook/bart-large-cnn
- Optimization: Smart extraction (150 tokens max)
- Pipeline: Smart Extract → BART → RAG → Gemini

