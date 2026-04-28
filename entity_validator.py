"""
Entity Validation Module

Purpose:
Validate specific factual claims by extracting entities
and checking them against Wikidata knowledge.

Example:
- Claim: "Pfizer vaccine is 95% effective"
- Extract: Entity="Pfizer vaccine", Attribute="efficacy"  
- Validate: Check against Wikidata facts

Why this matters:
Semantic similarity alone can't verify specific numbers or dates.
Entity validation adds a second verification layer.

Design Choice: Offline Wikidata
- No dependency on live SPARQL endpoint
- Faster response times
- Reproducible for demos
- Pre-curated for COVID-19 domain
"""
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import re

# spaCy for NER (lightweight model)
try:
    import spacy
    NLP_AVAILABLE = True
except ImportError:
    NLP_AVAILABLE = False
    print("⚠️ spaCy not available. Entity validation disabled.")

from knowledge_base import get_knowledge_base

@dataclass
class ExtractedEntity:
    """Entity extracted from text"""
    text: str
    label: str  # PERSON, ORG, DATE, etc.
    start: int
    end: int

@dataclass
class ValidationResult:
    """Result of entity validation"""
    entity: str
    found_in_wikidata: bool
    wikidata_facts: Optional[Dict]
    conflicts: List[str]  # Any contradictions found
    supports: List[str]   # Any supporting facts found

class EntityValidator:
    """
    Extracts entities from claims and validates against Wikidata.
    
    Workflow:
    1. Use spaCy NER to extract entities
    2. Lookup entities in curated Wikidata cache
    3. Compare claim attributes with known facts
    4. Return validation results
    """
    
    def __init__(self):
        self.kb = get_knowledge_base()
        self.nlp = None
        
        if NLP_AVAILABLE:
            self._load_spacy()
        
        # COVID-specific entity patterns (regex fallback)
        self.covid_patterns = {
            'vaccine': r'\b(pfizer|moderna|astrazeneca|johnson|j&j|covaxin|sputnik|sinovac)\b',
            'drug': r'\b(hydroxychloroquine|ivermectin|remdesivir|dexamethasone)\b',
            'organization': r'\b(who|cdc|fda|nih|nhs|ema)\b',
            'disease': r'\b(covid[-\s]?19|coronavirus|sars[-\s]?cov[-\s]?2)\b',
            'number': r'\b(\d+(?:\.\d+)?%?)\b'  # Percentages and numbers
        }
    
    def _load_spacy(self):
        """Load spaCy model"""
        try:
            self.nlp = spacy.load("en_core_web_sm")
            print("✓ spaCy NER model loaded")
        except OSError:
            print("⚠️ spaCy model not found. Run: python -m spacy download en_core_web_sm")
            self.nlp = None
    
    def extract_entities(self, text: str) -> List[ExtractedEntity]:
        """
        Extract named entities from text.
        Uses spaCy if available, falls back to regex patterns.
        """
        entities = []
        text_lower = text.lower()
        
        # Method 1: spaCy NER
        if self.nlp:
            doc = self.nlp(text)
            for ent in doc.ents:
                entities.append(ExtractedEntity(
                    text=ent.text,
                    label=ent.label_,
                    start=ent.start_char,
                    end=ent.end_char
                ))
        
        # Method 2: COVID-specific patterns (always run for domain coverage)
        for entity_type, pattern in self.covid_patterns.items():
            for match in re.finditer(pattern, text_lower):
                # Avoid duplicates from spaCy
                overlap = any(
                    e.start <= match.start() < e.end or 
                    e.start < match.end() <= e.end
                    for e in entities
                )
                if not overlap:
                    entities.append(ExtractedEntity(
                        text=match.group(),
                        label=entity_type.upper(),
                        start=match.start(),
                        end=match.end()
                    ))
        
        return entities
    
    def validate_entity(self, entity: ExtractedEntity, claim_text: str) -> ValidationResult:
        """
        Validate a single entity against Wikidata cache.
        
        Returns validation result with:
        - Whether entity was found
        - Relevant Wikidata facts
        - Any conflicts or supporting evidence
        """
        # Normalize entity name for lookup
        normalized = entity.text.lower().strip()
        
        # Try direct lookup
        wiki_entry = self.kb.lookup_entity(normalized)
        
        # Try common variations
        if not wiki_entry:
            variations = [
                normalized.replace('-', ' '),
                normalized.replace(' ', '-'),
                f"{normalized} vaccine",
                f"covid-19 {normalized}"
            ]
            for var in variations:
                wiki_entry = self.kb.lookup_entity(var)
                if wiki_entry:
                    break
        
        if not wiki_entry:
            return ValidationResult(
                entity=entity.text,
                found_in_wikidata=False,
                wikidata_facts=None,
                conflicts=[],
                supports=[]
            )
        
        # Check for conflicts and supporting facts
        conflicts = []
        supports = []
        
        facts = wiki_entry.get('facts', {})
        claim_lower = claim_text.lower()
        
        # Check specific fact types
        if 'covid_status' in facts:
            status = facts['covid_status'].lower()
            if 'not recommended' in status or 'not approved' in status:
                if 'cure' in claim_lower or 'treat' in claim_lower or 'effective' in claim_lower:
                    conflicts.append(f"{entity.text}: {facts['covid_status']}")
        
        if 'does_not' in facts:
            negative_fact = facts['does_not'].lower()
            if negative_fact in claim_lower:
                conflicts.append(f"Claim contradicts: {entity.text} does not {facts['does_not']}")
        
        if 'efficacy' in facts:
            supports.append(f"Known efficacy: {facts['efficacy']}")
        
        if 'approved_by' in facts:
            supports.append(f"Approved by: {', '.join(facts['approved_by'])}")
        
        return ValidationResult(
            entity=entity.text,
            found_in_wikidata=True,
            wikidata_facts=facts,
            conflicts=conflicts,
            supports=supports
        )
    
    def validate_claim(self, claim_text: str) -> Dict:
        """
        Full entity validation pipeline for a claim.
        
        Returns dict with:
        - entities_found: List of extracted entities
        - validations: Results for each entity
        - has_conflicts: Boolean
        - has_support: Boolean
        - summary: Human-readable summary
        """
        # Extract entities
        entities = self.extract_entities(claim_text)
        
        if not entities:
            return {
                'entities_found': [],
                'validations': [],
                'has_conflicts': False,
                'has_support': False,
                'summary': "No verifiable entities found in claim"
            }
        
        # Validate each entity
        validations = []
        all_conflicts = []
        all_supports = []
        
        for entity in entities:
            result = self.validate_entity(entity, claim_text)
            validations.append(result)
            all_conflicts.extend(result.conflicts)
            all_supports.extend(result.supports)
        
        # Generate summary
        if all_conflicts:
            summary = f"⚠️ Potential conflicts: {'; '.join(all_conflicts)}"
        elif all_supports:
            summary = f"✓ Supporting facts: {'; '.join(all_supports)}"
        else:
            summary = f"Found {len(entities)} entities, no direct conflicts or support"
        
        return {
            'entities_found': [e.text for e in entities],
            'validations': validations,
            'has_conflicts': len(all_conflicts) > 0,
            'has_support': len(all_supports) > 0,
            'summary': summary
        }


# Singleton
_validator = None

def get_entity_validator() -> EntityValidator:
    """Get or create entity validator"""
    global _validator
    if _validator is None:
        _validator = EntityValidator()
    return _validator

def validate_claim_entities(claim_text: str) -> Dict:
    """Convenience function for entity validation"""
    return get_entity_validator().validate_claim(claim_text)