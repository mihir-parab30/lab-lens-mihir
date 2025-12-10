#!/usr/bin/env python3
"""
Model Validation with ROUGE and BLEU Metrics
Validates summarization model performance on hold-out dataset
"""

import os
import sys
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.utils.logging_config import get_logger
from src.utils.error_handling import ErrorHandler, safe_execute

logger = get_logger(__name__)

try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except ImportError:
    # rouge-score is optional - no warning needed
    ROUGE_AVAILABLE = False

try:
    import nltk
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
    BLEU_AVAILABLE = True
except ImportError:
    # nltk is optional - no warning needed
    BLEU_AVAILABLE = False

try:
    from sacrebleu import BLEU
    SACREBLEU_AVAILABLE = True
except ImportError:
    SACREBLEU_AVAILABLE = False


class ModelValidator:
    """
    Model validation with ROUGE and BLEU metrics for summarization
    """
    
    def __init__(self):
        """Initialize validator with metric calculators"""
        self.error_handler = ErrorHandler(logger)
        
        if ROUGE_AVAILABLE:
            self.rouge_scorer = rouge_scorer.RougeScorer(
                ['rouge1', 'rouge2', 'rougeL', 'rougeLsum'],
                use_stemmer=True
            )
        else:
            self.rouge_scorer = None
        
        if BLEU_AVAILABLE:
            self.smoothing = SmoothingFunction().method1
        else:
            self.smoothing = None
        
        if SACREBLEU_AVAILABLE:
            self.sacrebleu = BLEU()
        else:
            self.sacrebleu = None
    
    @safe_execute("calculate_rouge", logger, ErrorHandler(logger))
    def calculate_rouge(
        self,
        predictions: List[str],
        references: List[str]
    ) -> Dict[str, float]:
        """
        Calculate ROUGE scores (ROUGE-1, ROUGE-2, ROUGE-L, ROUGE-Lsum)
        
        Args:
            predictions: List of predicted summaries
            references: List of reference summaries
            
        Returns:
            Dictionary with ROUGE scores
        """
        if not ROUGE_AVAILABLE:
            raise ImportError("rouge-score package required. Install with: pip install rouge-score")
        
        if len(predictions) != len(references):
            raise ValueError(f"Mismatch: {len(predictions)} predictions vs {len(references)} references")
        
        rouge_scores = {
            'rouge1': {'precision': [], 'recall': [], 'fmeasure': []},
            'rouge2': {'precision': [], 'recall': [], 'fmeasure': []},
            'rougeL': {'precision': [], 'recall': [], 'fmeasure': []},
            'rougeLsum': {'precision': [], 'recall': [], 'fmeasure': []}
        }
        
        for pred, ref in zip(predictions, references):
            if pd.isna(pred) or pd.isna(ref) or not pred.strip() or not ref.strip():
                continue
            
            scores = self.rouge_scorer.score(ref, pred)
            
            for rouge_type in rouge_scores.keys():
                rouge_scores[rouge_type]['precision'].append(scores[rouge_type].precision)
                rouge_scores[rouge_type]['recall'].append(scores[rouge_type].recall)
                rouge_scores[rouge_type]['fmeasure'].append(scores[rouge_type].fmeasure)
        
        # Calculate averages
        results = {}
        for rouge_type in rouge_scores.keys():
            if rouge_scores[rouge_type]['fmeasure']:
                results[f'{rouge_type}_precision'] = np.mean(rouge_scores[rouge_type]['precision'])
                results[f'{rouge_type}_recall'] = np.mean(rouge_scores[rouge_type]['recall'])
                results[f'{rouge_type}_f'] = np.mean(rouge_scores[rouge_type]['fmeasure'])
            else:
                results[f'{rouge_type}_precision'] = 0.0
                results[f'{rouge_type}_recall'] = 0.0
                results[f'{rouge_type}_f'] = 0.0
        
        logger.info(f"Calculated ROUGE scores for {len(predictions)} samples")
        return results
    
    @safe_execute("calculate_bleu", logger, ErrorHandler(logger))
    def calculate_bleu(
        self,
        predictions: List[str],
        references: List[str]
    ) -> Dict[str, float]:
        """
        Calculate BLEU scores
        
        Args:
            predictions: List of predicted summaries
            references: List of reference summaries
            
        Returns:
            Dictionary with BLEU scores
        """
        if not BLEU_AVAILABLE:
            raise ImportError("nltk package required. Install with: pip install nltk")
        
        if len(predictions) != len(references):
            raise ValueError(f"Mismatch: {len(predictions)} predictions vs {len(references)} references")
        
        bleu_scores = []
        
        for pred, ref in zip(predictions, references):
            if pd.isna(pred) or pd.isna(ref) or not pred.strip() or not ref.strip():
                continue
            
            # Tokenize
            pred_tokens = nltk.word_tokenize(pred.lower())
            ref_tokens = nltk.word_tokenize(ref.lower())
            
            # Calculate BLEU
            try:
                bleu = sentence_bleu([ref_tokens], pred_tokens, smoothing_function=self.smoothing)
                bleu_scores.append(bleu)
            except Exception as e:
                logger.warning(f"Error calculating BLEU: {e}")
                continue
        
        results = {
            'bleu': np.mean(bleu_scores) if bleu_scores else 0.0,
            'bleu_std': np.std(bleu_scores) if bleu_scores else 0.0
        }
        
        # Also calculate sacreBLEU if available
        if SACREBLEU_AVAILABLE and bleu_scores:
            try:
                # SacreBLEU expects list of references
                sacrebleu_score = self.sacrebleu.corpus_score(
                    predictions,
                    [[ref] for ref in references]
                )
                results['sacrebleu'] = sacrebleu_score.score / 100.0  # Convert to 0-1 scale
            except Exception as e:
                logger.warning(f"Error calculating sacreBLEU: {e}")
        
        logger.info(f"Calculated BLEU scores for {len(predictions)} samples")
        return results
    
    @safe_execute("validate_model", logger, ErrorHandler(logger))
    def validate_model(
        self,
        predictions: List[str],
        references: List[str],
        include_rouge: bool = True,
        include_bleu: bool = True
    ) -> Dict[str, float]:
        """
        Comprehensive model validation with ROUGE and BLEU metrics
        
        Args:
            predictions: List of predicted summaries
            references: List of reference summaries
            include_rouge: Whether to calculate ROUGE metrics
            include_bleu: Whether to calculate BLEU metrics
            
        Returns:
            Dictionary with all validation metrics
        """
        results = {}
        
        if include_rouge:
            rouge_results = self.calculate_rouge(predictions, references)
            results.update(rouge_results)
        
        if include_bleu:
            bleu_results = self.calculate_bleu(predictions, references)
            results.update(bleu_results)
        
        # Calculate overall score (weighted average of ROUGE-L F1 and BLEU)
        if 'rougeL_f' in results and 'bleu' in results:
            results['overall_score'] = 0.7 * results['rougeL_f'] + 0.3 * results['bleu']
        elif 'rougeL_f' in results:
            results['overall_score'] = results['rougeL_f']
        elif 'bleu' in results:
            results['overall_score'] = results['bleu']
        else:
            results['overall_score'] = 0.0
        
        logger.info(f"Model validation completed. Overall score: {results.get('overall_score', 0.0):.4f}")
        return results
    
    @safe_execute("validate_from_dataframe", logger, ErrorHandler(logger))
    def validate_from_dataframe(
        self,
        df: pd.DataFrame,
        prediction_column: str = 'gemini_summary',
        reference_column: str = 'cleaned_text',  # Or a ground truth summary column if available
        include_rouge: bool = True,
        include_bleu: bool = True
    ) -> Dict[str, float]:
        """
        Validate model from DataFrame
        
        Args:
            df: DataFrame with predictions and references
            prediction_column: Column name with model predictions
            reference_column: Column name with reference summaries
            include_rouge: Whether to calculate ROUGE metrics
            include_bleu: Whether to calculate BLEU metrics
            
        Returns:
            Dictionary with validation metrics
        """
        if prediction_column not in df.columns:
            raise ValueError(f"Prediction column '{prediction_column}' not found")
        if reference_column not in df.columns:
            raise ValueError(f"Reference column '{reference_column}' not found")
        
        predictions = df[prediction_column].astype(str).tolist()
        references = df[reference_column].astype(str).tolist()
        
        # Filter out empty predictions/references
        valid_pairs = [
            (p, r) for p, r in zip(predictions, references)
            if pd.notna(p) and pd.notna(r) and p.strip() and r.strip()
        ]
        
        if len(valid_pairs) == 0:
            raise ValueError("No valid prediction-reference pairs found")
        
        predictions_clean, references_clean = zip(*valid_pairs)
        
        logger.info(f"Validating {len(predictions_clean)} samples from DataFrame")
        return self.validate_model(
            list(predictions_clean),
            list(references_clean),
            include_rouge=include_rouge,
            include_bleu=include_bleu
        )

