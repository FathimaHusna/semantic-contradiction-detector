"""
Semantic Contradiction Detector
Assignment - Part 2
"""

from typing import List, Tuple, Dict, Any
from dataclasses import dataclass
import numpy as np
import re
import ast
import warnings
from sentence_transformers import CrossEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

@dataclass
class ContradictionResult:
    has_contradiction: bool
    confidence: float
    contradicting_pairs: List[Tuple[str, str]]
    explanation: str

class SemanticContradictionDetector:
    """
    Detects semantic contradictions within a single document.
    """
    
    def __init__(self, model_name: str = "default"):
        """
        Initialize the detector with specified model.
        
        Args:
            model_name: Model identifier or path
        """
        # Use the "Small" DeBERTa model for speed and accuracy
        target_model = "cross-encoder/nli-deberta-v3-small" if model_name == "default" else model_name
        
        print(f"Loading model: {target_model}...")
        self.model = CrossEncoder(target_model, device='cpu')
        
        # --- ROBUST AUTO-CALIBRATION ---
        # Dynamically find which index corresponds to 'Contradiction'
        print("Calibrating label mapping...")
        calibration_data = [
            ("The door is open.", "The door is closed."),       # Hard Contradiction
            ("The cat is sleeping.", "The animal is resting."), # Hard Entailment
        ]
        
        scores = self.model.predict(calibration_data)
        
        # Index 0 of scores corresponds to pair 0 (Contradiction)
        self.contradiction_id = np.argmax(scores[0])
        # Index 1 of scores corresponds to pair 1 (Entailment)
        self.entailment_id = np.argmax(scores[1])
        
        print(f"Calibration Complete. Contradiction Index: {self.contradiction_id}")
        
        if self.contradiction_id == self.entailment_id:
            # Fallback for standard MNLI mapping if calibration fails
            self.contradiction_id = 2 

    def _softmax(self, x):
        """Helper to calculate softmax."""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()
    
    def preprocess(self, text: str) -> List[str]:
        """
        Preprocess text into analyzable units.
        
        Args:
            text: Raw review text
            
        Returns:
            List of sentences or semantic units
        """
        text = text.strip()
        # Robust splitting by punctuation
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        cleaned = []
        for s in sentences:
            s = s.strip()
            if len(s) < 5: continue
            
            # --- CRITICAL FIX: CLEAN DISCOURSE MARKERS ---
            # Remove "However", "But" to reveal raw logic
            s = re.sub(r'^(However|But|Although|Yet),?\s*', '', s, flags=re.IGNORECASE)
            s = re.sub(r'\s*,?\s*(though|however)\.?$', '.', s, flags=re.IGNORECASE)
            
            cleaned.append(s)
        return cleaned
    
    def extract_claims(self, sentences: List[str]) -> List[Dict[str, Any]]:
        """
        Extract factual claims from sentences.
        
        Args:
            sentences: Preprocessed sentences
            
        Returns:
            List of claim dictionaries with metadata
        """
        # Wrap sentences in dicts as required by signature
        return [{"id": i, "text": s} for i, s in enumerate(sentences)]
    
    def check_contradiction(self, claim_a: Dict, claim_b: Dict) -> Tuple[bool, float]:
        """
        Check if two claims contradict each other.
        
        Args:
            claim_a: First claim
            claim_b: Second claim
            
        Returns:
            Tuple of (is_contradiction, confidence)
        """
        text_a = claim_a['text']
        text_b = claim_b['text']
        
        # --- BIDIRECTIONAL CHECK ---
        # Check A->B and B->A because NLI is directional
        inputs = [(text_a, text_b), (text_b, text_a)]
        
        scores = self.model.predict(inputs)
        probs_0 = self._softmax(scores[0])
        probs_1 = self._softmax(scores[1])
        
        score_0 = probs_0[self.contradiction_id]
        score_1 = probs_1[self.contradiction_id]
        
        # Take the strongest signal
        max_score = max(score_0, score_1)
        
        # Threshold 0.80 filters out "soft" contradictions (like "mediocre" vs "hear noise")
        return (max_score > 0.80), float(max_score)
    
    def analyze(self, text: str) -> ContradictionResult:
        """
        Main analysis pipeline.
        
        Args:
            text: Review text to analyze
            
        Returns:
            ContradictionResult with findings
        """
        # 1. Preprocess
        sentences = self.preprocess(text)
        if len(sentences) < 2:
            return ContradictionResult(False, 0.0, [], "Insufficient text to analyze.")
            
        # 2. Extract Claims
        claims = self.extract_claims(sentences)
        
        contradictions = []
        max_confidence = 0.0
        
        # 3. Pairwise Comparison
        for i in range(len(claims)):
            for j in range(i + 1, len(claims)):
                # Skip identical strings
                if claims[i]['text'] == claims[j]['text']:
                    continue
                    
                is_contra, conf = self.check_contradiction(claims[i], claims[j])
                
                if is_contra:
                    contradictions.append((claims[i]['text'], claims[j]['text']))
                    max_confidence = max(max_confidence, conf)
        
        has_contradiction = len(contradictions) > 0
        explanation = f"Found {len(contradictions)} logical inconsistencies." if has_contradiction else "Text is internally consistent."
        
        # Confidence logic:
        # If contradiction found: confidence = max_contradiction_score
        # If consistent: confidence = (1.0 - max_contradiction_score)
        final_conf = max_confidence if has_contradiction else (1.0 - max_confidence)
        
        return ContradictionResult(
            has_contradiction=has_contradiction,
            confidence=round(final_conf, 3),
            contradicting_pairs=contradictions,
            explanation=explanation
        )


def evaluate(detector: SemanticContradictionDetector, 
             test_data: List[Dict]) -> Dict[str, float]:
    """
    Evaluate detector performance.
    
    Args:
        detector: Initialized detector
        test_data: List of test samples with ground truth
        
    Returns:
        Dictionary of metrics (accuracy, precision, recall, f1)
    """
    y_true = []
    y_pred = []
    
    print("\n--- Detailed Evaluation ---")
    for item in test_data:
        text = item['text']
        expected = item['has_contradiction']
        
        result = detector.analyze(text)
        
        y_true.append(expected)
        y_pred.append(result.has_contradiction)
        
        status = "CORRECT" if expected == result.has_contradiction else "WRONG"
        print(f"ID {item.get('id', '?')}: {status} | Pred: {result.has_contradiction} | True: {expected}")
        
        if expected != result.has_contradiction:
             print(f"  > Fail Context: {text[:60]}...")
        elif result.has_contradiction:
             # Print first detected pair for verification
             print(f"  > Detected: {result.contradicting_pairs[0]}")

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0)
    }


if __name__ == "__main__":
    # Initialize detector
    detector = SemanticContradictionDetector()
    
    # Run on sample data from 'dataset.txt'
    try:
        with open('dataset.txt', 'r') as f:
            content = f.read()
            # Safely evaluate the string list representation into a Python list
            SAMPLE_REVIEWS = ast.literal_eval(content)
            print(f"Loaded {len(SAMPLE_REVIEWS)} samples from dataset.txt")
            
            # Print first sample for verification
            print(f"First sample text: {SAMPLE_REVIEWS[0]['text'][:50]}...")
            
    except FileNotFoundError:
        print("Error: 'dataset.txt' not found. Please ensure the file exists.")
        SAMPLE_REVIEWS = []
    except Exception as e:
        print(f"Error parsing dataset: {e}")
        SAMPLE_REVIEWS = []

    if SAMPLE_REVIEWS:
        # Run Analysis loop
        print("\n--- Running Analysis ---")
        for review in SAMPLE_REVIEWS:
            result = detector.analyze(review["text"])
            print(f"Review {review['id']}: {result.has_contradiction} (Conf: {result.confidence})")
        
        # Evaluate
        metrics = evaluate(detector, SAMPLE_REVIEWS)
        print(f"\nFinal Metrics: {metrics}")