"""
Decision Engine - With Network Risk Integration

Now considers:
1. LLM analysis (if available)
2. Semantic similarity
3. Content analysis rules
4. Network risk score (NEW)
"""
from typing import Tuple, Optional, List, Dict
from dataclasses import dataclass
import re

from config import LABELS, IntentLevel

@dataclass
class SimilarityScores:
    fact_score: float
    fact_matches: List[str]
    fact_source: Optional[str]
    fact_intent: Optional[IntentLevel]
    misinfo_score: float
    misinfo_matches: List[str]
    misinfo_source: Optional[str]
    misinfo_intent: Optional[IntentLevel]

@dataclass
class DecisionResult:
    label: str
    confidence: float
    explanation: str
    warning_message: str
    claim_intent: IntentLevel
    is_negation: bool
    llm_used: bool
    llm_reasoning: Optional[str]
    matched_facts: List[str]
    source_detected: Optional[str] = None
    network_influenced: bool = False  # NEW

class DecisionEngine:
    """Decision engine with proper general logic and network risk integration."""
    
    # Network risk thresholds (NEW)
    NETWORK_HIGH_RISK = 0.70      # Strong signal from network
    NETWORK_MEDIUM_RISK = 0.50    # Moderate signal
    NETWORK_CONFIDENCE_BOOST = 0.10  # How much to boost confidence
    
    def __init__(self):
        self.llm_verifier = None
        self._init_llm()
    
    def _init_llm(self):
        try:
            from llm_verifier import get_llm_verifier
            self.llm_verifier = get_llm_verifier()
        except Exception as e:
            print(f"LLM not available: {e}")
    
    def _detect_intent(self, text: str) -> IntentLevel:
        """Detect the intent level of the claim"""
        text_lower = text.lower()
        
        if any(w in text_lower for w in ['cure', 'cures', 'cured']):
            return IntentLevel.CURE
        elif any(w in text_lower for w in ['treat', 'treatment', 'treats']):
            return IntentLevel.TREATMENT
        elif any(w in text_lower for w in ['prevent', 'prevents', 'protect', 'protects']):
            return IntentLevel.PREVENTION
        elif any(w in text_lower for w in ['symptom', 'symptoms', 'fever', 'cough']):
            return IntentLevel.SYMPTOM_MANAGEMENT
        return IntentLevel.GENERAL_WELLNESS
    
    def _has_negation(self, text: str) -> bool:
        """Check if text contains negation of medical claims"""
        text_lower = text.lower()
        patterns = [
            r"can'?t\s+(cure|treat|heal|prevent)",
            r"cannot\s+(cure|treat|heal|prevent)",
            r"does\s*n'?t\s+(cure|treat|heal|prevent)",
            r"not\s+(a\s+)?(cure|treatment)",
            r"no\s+(cure|treatment|evidence)",
        ]
        return any(re.search(p, text_lower) for p in patterns)
    
    def _apply_network_risk(
        self, 
        base_label: str, 
        base_confidence: float,
        network_risk: Dict
    ) -> Tuple[str, float, str]:
        """
        Adjust classification based on network risk score.
        
        Returns: (adjusted_label, adjusted_confidence, network_explanation)
        """
        risk_score = network_risk.get('risk_score', 0.0)
        misinfo_neighbors = network_risk.get('misinfo_neighbors', 0)
        correct_neighbors = network_risk.get('correct_neighbors', 0)
        similar_count = network_risk.get('similar_claims_count', 0)
        
        # No similar claims found - no adjustment
        if similar_count == 0:
            return base_label, base_confidence, ""
        
        network_explanation = ""
        adjusted_label = base_label
        adjusted_confidence = base_confidence
        
        # HIGH network risk - strong misinfo signal
        if risk_score >= self.NETWORK_HIGH_RISK:
            network_explanation = f"Network signal: {misinfo_neighbors} similar misinformation claims found"
            
            # If base classification is uncertain, push towards MISINFO
            if 'UNVERIFIED' in base_label:
                adjusted_label = LABELS['likely_misinfo']
                adjusted_confidence = min(0.85, base_confidence + 0.15)
            
            # If already MISINFO, boost confidence
            elif 'MISINFO' in base_label:
                adjusted_confidence = min(0.95, base_confidence + self.NETWORK_CONFIDENCE_BOOST)
            
            # If classified as CORRECT but network says MISINFO - flag conflict
            elif 'CORRECT' in base_label and base_confidence < 0.80:
                network_explanation += " (CONFLICT: semantic=correct, network=misinfo)"
                # Don't change label, but note the conflict
        
        # HIGH correct neighbor count - positive signal
        elif correct_neighbors > misinfo_neighbors and correct_neighbors >= 2:
            network_explanation = f"Network signal: {correct_neighbors} similar verified claims found"
            
            # If uncertain, push towards CORRECT
            if 'UNVERIFIED' in base_label:
                adjusted_label = LABELS['likely_correct']
                adjusted_confidence = min(0.80, base_confidence + 0.10)
            
            # If already CORRECT, boost confidence
            elif 'CORRECT' in base_label:
                adjusted_confidence = min(0.95, base_confidence + self.NETWORK_CONFIDENCE_BOOST)
        
        # MEDIUM network risk - moderate signal
        elif risk_score >= self.NETWORK_MEDIUM_RISK:
            network_explanation = f"Network signal: mixed ({misinfo_neighbors} misinfo, {correct_neighbors} correct)"
            # Only slight confidence adjustment, no label change
            if 'MISINFO' in base_label:
                adjusted_confidence = min(0.90, base_confidence + 0.05)
        
        return adjusted_label, adjusted_confidence, network_explanation
    
    def classify(
        self, 
        claim_text: str, 
        semantic_scores: SimilarityScores,
        network_risk: Dict = None  # NEW parameter
    ) -> DecisionResult:
        """Main classification - uses LLM first, then semantic similarity, then network risk"""
        
        if network_risk is None:
            network_risk = {}
        
        intent = self._detect_intent(claim_text)
        is_negation = self._has_negation(claim_text)
        
        # === TRY LLM VERIFIER FIRST ===
        if self.llm_verifier:
            try:
                from llm_verifier import verify_with_llm, VerificationResult
                
                llm_result = verify_with_llm(claim_text, semantic_scores.fact_matches)
                
                if llm_result.result != VerificationResult.UNCERTAIN:
                    is_correct = llm_result.result == VerificationResult.CORRECT
                    
                    base_label = LABELS['correct'] if is_correct else LABELS['misinfo']
                    base_confidence = llm_result.confidence
                    
                    # Apply network risk adjustment (NEW)
                    final_label, final_confidence, network_exp = self._apply_network_risk(
                        base_label, base_confidence, network_risk
                    )
                    
                    explanation = llm_result.reasoning
                    if network_exp:
                        explanation += f" | {network_exp}"
                    
                    return DecisionResult(
                        label=final_label,
                        confidence=final_confidence,
                        explanation=explanation,
                        warning_message=f"{'✓ VERIFIED' if is_correct else '⚠️ MISINFORMATION'}\n{explanation}",
                        claim_intent=intent,
                        is_negation=is_negation,
                        llm_used=llm_result.llm_available,
                        llm_reasoning=llm_result.reasoning,
                        matched_facts=semantic_scores.fact_matches[:3],
                        source_detected=llm_result.source_detected,
                        network_influenced=bool(network_exp)
                    )
            except Exception as e:
                print(f"LLM error: {e}")
        
        # === FALLBACK TO SEMANTIC SIMILARITY ===
        fact_score = semantic_scores.fact_score
        misinfo_score = semantic_scores.misinfo_score
        
        # High similarity to facts
        if fact_score >= 0.70 and fact_score > misinfo_score:
            match_info = semantic_scores.fact_matches[0][:80] if semantic_scores.fact_matches else ""
            base_explanation = f"Matches verified information ({fact_score:.0%} similarity). Similar to: '{match_info}...'"
            
            base_label = LABELS['correct'] if fact_score >= 0.80 else LABELS['likely_correct']
            base_confidence = fact_score
            
            # Apply network risk (NEW)
            final_label, final_confidence, network_exp = self._apply_network_risk(
                base_label, base_confidence, network_risk
            )
            
            explanation = base_explanation
            if network_exp:
                explanation += f" | {network_exp}"
            
            return DecisionResult(
                label=final_label,
                confidence=final_confidence,
                explanation=explanation,
                warning_message=f"✓ {'VERIFIED' if fact_score >= 0.80 else 'LIKELY ACCURATE'}\n{explanation}",
                claim_intent=intent,
                is_negation=is_negation,
                llm_used=False,
                llm_reasoning=None,
                matched_facts=semantic_scores.fact_matches[:3],
                network_influenced=bool(network_exp)
            )
        
        # High similarity to misinformation
        if misinfo_score >= 0.70 and misinfo_score > fact_score:
            # Check if it's negating the misinformation
            if is_negation:
                explanation = "This appears to correctly deny a known false claim."
                return DecisionResult(
                    label=LABELS['likely_correct'],
                    confidence=0.75,
                    explanation=explanation,
                    warning_message=f"✓ LIKELY ACCURATE\n{explanation}",
                    claim_intent=intent,
                    is_negation=True,
                    llm_used=False,
                    llm_reasoning=None,
                    matched_facts=semantic_scores.fact_matches[:3],
                    network_influenced=False
                )
            
            match_info = semantic_scores.misinfo_matches[0][:80] if semantic_scores.misinfo_matches else ""
            base_explanation = f"Matches known misinformation ({misinfo_score:.0%} similarity). Similar to: '{match_info}...'"
            
            base_label = LABELS['misinfo'] if misinfo_score >= 0.80 else LABELS['likely_misinfo']
            base_confidence = misinfo_score
            
            # Apply network risk - will likely boost confidence (NEW)
            final_label, final_confidence, network_exp = self._apply_network_risk(
                base_label, base_confidence, network_risk
            )
            
            explanation = base_explanation
            if network_exp:
                explanation += f" | {network_exp}"
            
            return DecisionResult(
                label=final_label,
                confidence=final_confidence,
                explanation=explanation,
                warning_message=f"⚠️ {'MISINFORMATION' if misinfo_score >= 0.80 else 'LIKELY MISINFORMATION'}\n{explanation}",
                claim_intent=intent,
                is_negation=False,
                llm_used=False,
                llm_reasoning=None,
                matched_facts=semantic_scores.fact_matches[:3],
                network_influenced=bool(network_exp)
            )
        
        # Medium similarity - this is where network risk helps most (NEW)
        if fact_score >= 0.55:
            base_explanation = f"Partially matches verified information ({fact_score:.0%}). Verify with official sources."
            base_label = LABELS['likely_correct']
            base_confidence = 0.60
            
            final_label, final_confidence, network_exp = self._apply_network_risk(
                base_label, base_confidence, network_risk
            )
            
            explanation = base_explanation
            if network_exp:
                explanation += f" | {network_exp}"
            
            return DecisionResult(
                label=final_label,
                confidence=final_confidence,
                explanation=explanation,
                warning_message=f"Likely accurate. {explanation}",
                claim_intent=intent,
                is_negation=is_negation,
                llm_used=False,
                llm_reasoning=None,
                matched_facts=semantic_scores.fact_matches[:3],
                network_influenced=bool(network_exp)
            )
        
        if misinfo_score >= 0.55 and not is_negation:
            base_explanation = f"Has similarity to known misinformation ({misinfo_score:.0%}). Please verify."
            base_label = LABELS['likely_misinfo']
            base_confidence = 0.60
            
            final_label, final_confidence, network_exp = self._apply_network_risk(
                base_label, base_confidence, network_risk
            )
            
            explanation = base_explanation
            if network_exp:
                explanation += f" | {network_exp}"
            
            return DecisionResult(
                label=final_label,
                confidence=final_confidence,
                explanation=explanation,
                warning_message=f"⚠️ Possibly inaccurate. {explanation}",
                claim_intent=intent,
                is_negation=False,
                llm_used=False,
                llm_reasoning=None,
                matched_facts=semantic_scores.fact_matches[:3],
                network_influenced=bool(network_exp)
            )
        
        # Cannot determine - network risk can help here (NEW)
        base_explanation = "Could not verify. Please check WHO (who.int) or CDC (cdc.gov) for accurate information."
        base_label = LABELS['unverified']
        base_confidence = 0.40
        
        final_label, final_confidence, network_exp = self._apply_network_risk(
            base_label, base_confidence, network_risk
        )
        
        explanation = base_explanation
        if network_exp:
            explanation += f" | {network_exp}"
        
        return DecisionResult(
            label=final_label,
            confidence=final_confidence,
            explanation=explanation,
            warning_message=f"? UNVERIFIED\n{explanation}",
            claim_intent=intent,
            is_negation=is_negation,
            llm_used=False,
            llm_reasoning=None,
            matched_facts=semantic_scores.fact_matches[:3],
            network_influenced=bool(network_exp)
        )


_engine = None

def get_decision_engine() -> DecisionEngine:
    global _engine
    if _engine is None:
        _engine = DecisionEngine()
    return _engine