"""
Intent Detection Module - Updated with Negation Awareness

Detects the intent level of health claims to prevent:
- Symptom relief facts from validating cure claims
- General wellness facts from validating treatment claims
- FALSE POSITIVES from negation claims (e.g., "X cannot cure COVID")

This is a RULE-BASED system, not ML - fully explainable.
"""
import re
from typing import Tuple, List, Optional
from dataclasses import dataclass
from config import IntentLevel, INTENT_HIERARCHY

# Negation patterns that indicate a claim is DEBUNKING rather than ASSERTING
NEGATION_PATTERNS = [
    r'\bcannot\b', r'\bcan\'t\b', r'\bcant\b',
    r'\bdoes\s+not\b', r'\bdo\s+not\b', r'\bdoesn\'t\b', r'\bdon\'t\b',
    r'\bwill\s+not\b', r'\bwon\'t\b', r'\bwont\b',
    r'\bnot\s+\w+\b', r'\bno\s+evidence\b', r'\bno\s+proof\b',
    r'\bnever\b', r'\bnone\b',
    r'\bis\s+not\b', r'\bare\s+not\b', r'\bisn\'t\b', r'\baren\'t\b',
    r'\bhas\s+not\b', r'\bhave\s+not\b', r'\bhasn\'t\b', r'\bhaven\'t\b',
    r'\bfalse\s+that\b', r'\bnot\s+true\b', r'\buntrue\b',
    r'\bdisproven\b', r'\bdebunked\b', r'\bmyth\b',
    r'\bno\s+cure\b', r'\bno\s+treatment\b',
]

@dataclass
class IntentResult:
    """Result of intent detection"""
    level: IntentLevel
    confidence: float
    matched_patterns: List[str]
    explanation: str
    is_negation: bool = False  # NEW: indicates if claim contains negation

class IntentDetector:
    """
    Rule-based intent detection for health claims.
    
    Intent Hierarchy (lowest to highest):
    1. GENERAL_WELLNESS - "supports immune function"
    2. SYMPTOM_MANAGEMENT - "reduces fever", "soothes cough"
    3. PREVENTION - "prevents infection", "reduces risk"
    4. TREATMENT - "treats COVID", "medication for"
    5. CURE - "cures COVID", "eliminates virus"
    
    Key Rules:
    - Lower-level evidence CANNOT validate higher-level claims
    - Negation claims (e.g., "X cannot cure") are detected separately
    """
    
    def __init__(self):
        # Patterns ordered by priority (check CURE first)
        self.patterns = {
            IntentLevel.CURE: [
                r'\bcures?\b',
                r'\bcured\b',
                r'\bcuring\b',
                r'\beliminate[sd]?\b',
                r'\beradicate[sd]?\b',
                r'\bkills?\s+(the\s+)?virus\b',
                r'\bdestroy[s]?\s+(the\s+)?(virus|covid)\b',
                r'\b100\s*%\s*(effective|protection)\b',
                r'\bguaranteed\s+(cure|protection)\b',
                r'\bcomplete(ly)?\s+(cure|eliminat|eradicat)\b',
                r'\btotal\s+immunit[y]\b',
                r'\bpermanent(ly)?\s+(cure|immun)\b',
                r'\bget\s+rid\s+of\s+(covid|virus|coronavirus)\b',
                r'\bmake[s]?\s+(you\s+)?immune\b',
            ],
            IntentLevel.TREATMENT: [
                r'\btreat[s]?\b',
                r'\btreatment\b',
                r'\btreating\b',
                r'\bmedication\s+for\s+(covid|coronavirus)\b',
                r'\bdrug\s+for\s+(covid|coronavirus)\b',
                r'\bprescribed\s+for\s+(covid|coronavirus)\b',
                r'\btherapy\s+for\s+(covid|coronavirus)\b',
                r'\bfda\s+approved\s+for\s+(covid|coronavirus)\b',
                r'\bantiviral\b',
                r'\bpaxlovid\b',
                r'\bremdesivir\b',
            ],
            IntentLevel.PREVENTION: [
                r'\bprevent[s]?\b',
                r'\bprevention\b',
                r'\bpreventing\b',
                r'\bprotect[s]?\s+(against|from)\b',
                r'\breduce[sd]?\s+(the\s+)?risk\b',
                r'\blower[s]?\s+(the\s+)?(risk|chance)\b',
                r'\bstop[s]?\s+(the\s+)?spread\b',
                r'\bblock[s]?\s+(the\s+)?(virus|infection)\b',
                r'\bimmuniz\w+\b',
                r'\bvaccinat\w+\b',
                r'\bbooster\b',
            ],
            IntentLevel.SYMPTOM_MANAGEMENT: [
                r'\bsymptom[s]?\b',
                r'\brelieve[sd]?\b',
                r'\brelief\b',
                r'\breduce[sd]?\s+(fever|inflammation|pain|cough)\b',
                r'\bsoothe[sd]?\b',
                r'\bease[sd]?\b',
                r'\balleviate[sd]?\b',
                r'\bhelp[s]?\s+with\s+(fever|cough|pain|symptom)\b',
                r'\bmake[s]?\s+(you\s+)?feel\s+better\b',
                r'\banti-?inflammator\w+\b',
                r'\bpain\s+relief\b',
                r'\bfever\s+reducer\b',
            ],
            IntentLevel.GENERAL_WELLNESS: [
                r'\bsupport[s]?\s+(immune|health)\b',
                r'\bboost[s]?\s+(immune|immunity)\b',
                r'\bhealthy\b',
                r'\bwellness\b',
                r'\bnutritious\b',
                r'\bbeneficial\b',
                r'\bgood\s+for\s+(you|health)\b',
            ]
        }
        
        # Approved treatments/cures (very limited list)
        self.approved_cures = []  # Nothing truly "cures" COVID
        self.approved_treatments = [
            'paxlovid', 'remdesivir', 'molnupiravir', 
            'dexamethasone', 'antiviral', 'monoclonal antibod'
        ]
    
    def _contains_negation(self, text: str) -> bool:
        """Check if text contains negation patterns"""
        text_lower = text.lower()
        for pattern in NEGATION_PATTERNS:
            if re.search(pattern, text_lower):
                return True
        return False
    
    def _is_negated_claim(self, text: str, intent_patterns: List[str]) -> bool:
        """
        Check if the intent keyword is negated in the claim.
        
        Example: "Vitamin C cannot cure COVID" - "cure" is negated
        """
        text_lower = text.lower()
        
        # Check for common negation + intent patterns
        negated_patterns = [
            r'\b(cannot|can\'t|cant|does\s+not|doesn\'t|will\s+not|won\'t)\s+\w*\s*(cure|treat|prevent|heal)',
            r'\bno\s+(cure|treatment|prevention)\b',
            r'\b(not|never)\s+\w*\s*(cure|treat|prevent|heal)',
            r'\b(cure|treat|prevent|heal)\w*\s+\w*\s*(not|never|no)\b',
        ]
        
        for pattern in negated_patterns:
            if re.search(pattern, text_lower):
                return True
        
        return False
    
    def detect_intent(self, text: str) -> IntentResult:
        """
        Detect the intent level of a claim.
        
        Returns IntentResult with level, confidence, and explanation.
        Now includes negation detection.
        """
        text_lower = text.lower()
        matched = []
        
        # Check for negation first
        is_negation = self._contains_negation(text)
        is_negated_cure = self._is_negated_claim(text, self.patterns[IntentLevel.CURE])
        
        # Check patterns in priority order (highest intent first)
        for level in [IntentLevel.CURE, IntentLevel.TREATMENT, 
                      IntentLevel.PREVENTION, IntentLevel.SYMPTOM_MANAGEMENT,
                      IntentLevel.GENERAL_WELLNESS]:
            
            for pattern in self.patterns[level]:
                if re.search(pattern, text_lower):
                    matched.append((level, pattern))
        
        if not matched:
            return IntentResult(
                level=IntentLevel.GENERAL_WELLNESS,
                confidence=0.5,
                matched_patterns=[],
                explanation="No specific intent patterns detected, defaulting to general wellness",
                is_negation=is_negation
            )
        
        # Highest matched level wins
        highest_level = matched[0][0]
        level_patterns = [p for l, p in matched if l == highest_level]
        
        # If claim contains negation AND matches CURE patterns,
        # it's likely debunking a cure claim, not making one
        # In this case, we still detect CURE level but flag it as negation
        # The decision engine will handle this appropriately
        
        # Confidence based on number of matching patterns
        confidence = min(0.95, 0.6 + 0.1 * len(level_patterns))
        
        # Adjust explanation for negation
        if is_negation and highest_level in [IntentLevel.CURE, IntentLevel.TREATMENT]:
            explanation = f"Detected NEGATED {highest_level.name} claim (debunking): {level_patterns[:2]}"
        else:
            explanation = f"Detected {highest_level.name} intent based on: {level_patterns[:3]}"
        
        return IntentResult(
            level=highest_level,
            confidence=confidence,
            matched_patterns=level_patterns,
            explanation=explanation,
            is_negation=is_negation
        )
    
    def can_validate(self, evidence_intent: IntentLevel, claim_intent: IntentLevel) -> bool:
        """
        Check if evidence at one intent level can validate a claim at another level.
        
        Key Rule: Evidence can only validate claims at same or LOWER intent levels.
        Example: SYMPTOM_MANAGEMENT evidence CANNOT validate CURE claims.
        """
        allowed_claims = INTENT_HIERARCHY.get(evidence_intent, [])
        return claim_intent in allowed_claims
    
    def check_intent_mismatch(
        self, 
        claim_text: str, 
        matched_evidence_text: str,
        matched_evidence_intent: Optional[IntentLevel] = None
    ) -> Tuple[bool, str]:
        """
        Check if there's an intent mismatch between claim and evidence.
        
        Returns: (has_mismatch, explanation)
        """
        claim_result = self.detect_intent(claim_text)
        
        if matched_evidence_intent is None:
            evidence_result = self.detect_intent(matched_evidence_text)
            evidence_intent = evidence_result.level
        else:
            evidence_intent = matched_evidence_intent
        
        can_support = self.can_validate(evidence_intent, claim_result.level)
        
        if not can_support:
            return True, (
                f"Intent mismatch: Claim is {claim_result.level.name} level, "
                f"but evidence is only {evidence_intent.name} level. "
                f"{evidence_intent.name} evidence cannot validate {claim_result.level.name} claims."
            )
        
        return False, "Intent levels are compatible"
    
    def is_cure_claim(self, text: str) -> bool:
        """Quick check if text contains cure-level claims"""
        result = self.detect_intent(text)
        # Only return True for POSITIVE cure claims, not negated ones
        return result.level == IntentLevel.CURE and not result.is_negation
    
    def is_approved_treatment(self, text: str) -> bool:
        """Check if claim mentions an approved treatment"""
        text_lower = text.lower()
        return any(t in text_lower for t in self.approved_treatments)


# Singleton instance
_detector = None

def get_intent_detector() -> IntentDetector:
    global _detector
    if _detector is None:
        _detector = IntentDetector()
    return _detector

def detect_claim_intent(text: str) -> IntentResult:
    """Convenience function"""
    return get_intent_detector().detect_intent(text)

def check_intent_mismatch(claim: str, evidence: str) -> Tuple[bool, str]:
    """Convenience function"""
    return get_intent_detector().check_intent_mismatch(claim, evidence)