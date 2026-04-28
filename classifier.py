"""
Main Classification Pipeline - v5

Integrates:
- Semantic similarity (SBERT)
- LLM reasoning (Ollama)
- Intent detection
- Source detection (watermarks)
- Network analysis (MinHash/LSH) ← NEW
- Proper explanations
"""
from typing import Optional, List
from dataclasses import dataclass

from config import IntentLevel
from preprocessing import preprocess_text
from knowledge_base import get_knowledge_base
from decision_engine import get_decision_engine, SimilarityScores
from network_analysis import get_claim_network, compute_network_risk, add_to_network  # NEW

@dataclass 
class ClassificationResult:
    """Complete result of classification pipeline"""
    # Input info
    original_text: str
    detected_language: str
    translated_text: str
    cleaned_text: str
    
    # Classification
    label: str
    confidence: float
    explanation: str
    warning: str
    
    # Similarity scores
    fact_similarity: float
    misinfo_similarity: float
    matched_facts: List[str]
    matched_misinfo: List[str]
    
    # Intent
    claim_intent: str
    is_negation: bool
    
    # LLM
    llm_used: bool
    llm_reasoning: Optional[str]
    
    # Source detection
    source_detected: Optional[str] = None
    
    # Network analysis (NEW)
    network_risk_score: float = 0.0
    similar_claims_count: int = 0
    misinfo_neighbors: int = 0
    correct_neighbors: int = 0
    network_explanation: str = ""

class MisinformationClassifier:
    """Main classifier with LLM intelligence, source detection, and network analysis."""
    
    def __init__(self):
        self.kb = get_knowledge_base()
        self.engine = get_decision_engine()
        self.network = get_claim_network()  # NEW
    
    def classify(self, text: str, add_to_network: bool = True) -> ClassificationResult:
        """
        Main classification method.
        
        Args:
            text: Input text to classify
            add_to_network: Whether to add this claim to network after classification
        """
        
        # Step 1: Preprocessing
        preprocessed = preprocess_text(text)
        
        # Step 2: Network risk analysis (NEW - check BEFORE main classification)
        network_risk = compute_network_risk(preprocessed.cleaned)
        
        # Step 3: Encode claim
        query_emb = self.kb.encode_text(preprocessed.cleaned)
        
        # Step 4: Find similar facts and misinformation
        fact_results = self.kb.find_similar_facts(query_emb, top_k=10)
        misinfo_results = self.kb.find_similar_misinfo(query_emb, top_k=10)
        
        # Extract matches
        fact_matches = [r['entry'].text for r in fact_results]
        misinfo_matches = [r['entry'].text for r in misinfo_results]
        
        best_fact_score = fact_results[0]['score'] if fact_results else 0.0
        best_misinfo_score = misinfo_results[0]['score'] if misinfo_results else 0.0
        
        # Build semantic scores (NEW - include network risk)
        semantic_scores = SimilarityScores(
            fact_score=best_fact_score,
            fact_matches=fact_matches,
            fact_source=fact_results[0]['entry'].source if fact_results else None,
            fact_intent=None,
            misinfo_score=best_misinfo_score,
            misinfo_matches=misinfo_matches,
            misinfo_source=misinfo_results[0]['entry'].source if misinfo_results else None,
            misinfo_intent=None
        )
        
        # Step 5: Decision engine with LLM (NEW - pass network risk)
        decision = self.engine.classify(
            claim_text=preprocessed.cleaned,
            semantic_scores=semantic_scores,
            network_risk=network_risk  # NEW parameter
        )
        
        # Step 6: Add to network for future lookups (NEW)
        if add_to_network:
            try:
                from network_analysis import add_to_network as add_claim
                add_claim(
                    text=text,
                    cleaned_text=preprocessed.cleaned,
                    label=decision.label
                )
            except Exception as e:
                print(f"Warning: Could not add to network: {e}")
        
        # Build result
        return ClassificationResult(
            original_text=text,
            detected_language=preprocessed.detected_language,
            translated_text=preprocessed.translated,
            cleaned_text=preprocessed.cleaned,
            label=decision.label,
            confidence=decision.confidence,
            explanation=decision.explanation,
            warning=decision.warning_message,
            fact_similarity=best_fact_score,
            misinfo_similarity=best_misinfo_score,
            matched_facts=decision.matched_facts,
            matched_misinfo=misinfo_matches[:3],
            claim_intent=decision.claim_intent.name,
            is_negation=decision.is_negation,
            llm_used=decision.llm_used,
            llm_reasoning=decision.llm_reasoning,
            source_detected=getattr(decision, 'source_detected', None),
            # Network analysis results (NEW)
            network_risk_score=network_risk.get('risk_score', 0.0),
            similar_claims_count=network_risk.get('similar_claims_count', 0),
            misinfo_neighbors=network_risk.get('misinfo_neighbors', 0),
            correct_neighbors=network_risk.get('correct_neighbors', 0),
            network_explanation=network_risk.get('explanation', '')
        )


def classify_claim(text: str) -> ClassificationResult:
    """Quick classification"""
    classifier = MisinformationClassifier()
    return classifier.classify(text)