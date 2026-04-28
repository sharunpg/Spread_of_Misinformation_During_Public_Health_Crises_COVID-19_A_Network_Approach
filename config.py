"""
Configuration Module - v2

Changes:
- Removed TESTING mode and tester credentials
- Added FACT_GATHERING mode
- Updated for automatic scraping workflow
"""
import os
from enum import Enum
from datetime import datetime

# Base paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
CACHE_DIR = os.path.join(BASE_DIR, "cache")
LOGS_DIR = os.path.join(BASE_DIR, "logs")

# Create directories
for d in [DATA_DIR, CACHE_DIR, LOGS_DIR]:
    os.makedirs(d, exist_ok=True)

# File paths
VERIFIED_FACTS_PATH = os.path.join(DATA_DIR, "verified_facts.json")
KNOWN_MISINFO_PATH = os.path.join(DATA_DIR, "known_misinformation.json")
WIKIDATA_ENTITIES_PATH = os.path.join(DATA_DIR, "wikidata_entities.json")
UNVERIFIED_CLAIMS_PATH = os.path.join(DATA_DIR, "unverified_claims.json")
PROCESSED_CLAIMS_PATH = os.path.join(DATA_DIR, "processed_claims.csv")
CLAIM_GRAPH_PATH = os.path.join(CACHE_DIR, "claim_graph.pkl")
INGESTION_LOG_PATH = os.path.join(LOGS_DIR, "ingestion_log.jsonl")
STAGED_CLAIMS_PATH = os.path.join(DATA_DIR, "staged_claims.json")

# Model settings
SBERT_MODEL = "all-MiniLM-L6-v2"
SPACY_MODEL = "en_core_web_sm"

# Similarity thresholds
THRESHOLDS = {
    "high_confidence": 0.75,
    "medium_confidence": 0.60,
    "low_confidence": 0.45,
    "duplicate_detection": 0.90,
    "conflict_detection": 0.80,
    "exact_match": 0.90,  # For exact fact matching
}

# Decision weights
WEIGHTS = {
    "semantic_similarity": 0.50,
    "entity_validation": 0.30,
    "network_risk": 0.20
}

# Classification labels
LABELS = {
    "correct": "CORRECT INFORMATION",
    "misinfo": "MISINFORMATION",
    "likely_correct": "LIKELY CORRECT",
    "likely_misinfo": "LIKELY MISINFORMATION",
    "unverified": "UNVERIFIED"
}

# ============== SYSTEM MODES ==============

class SystemMode(Enum):
    VERIFY = "verify"           # Normal verification mode
    FACT_GATHERING = "gather"   # Scrape and gather facts mode

# ============== INTENT LEVELS ==============

class IntentLevel(Enum):
    GENERAL_WELLNESS = 1
    SYMPTOM_MANAGEMENT = 2
    PREVENTION = 3
    TREATMENT = 4
    CURE = 5

# Intent hierarchy - what can validate what
INTENT_HIERARCHY = {
    IntentLevel.GENERAL_WELLNESS: [IntentLevel.GENERAL_WELLNESS],
    IntentLevel.SYMPTOM_MANAGEMENT: [IntentLevel.GENERAL_WELLNESS, IntentLevel.SYMPTOM_MANAGEMENT],
    IntentLevel.PREVENTION: [IntentLevel.GENERAL_WELLNESS, IntentLevel.SYMPTOM_MANAGEMENT, IntentLevel.PREVENTION],
    IntentLevel.TREATMENT: [IntentLevel.GENERAL_WELLNESS, IntentLevel.SYMPTOM_MANAGEMENT, IntentLevel.PREVENTION, IntentLevel.TREATMENT],
    IntentLevel.CURE: [IntentLevel.CURE]  # CURE can only be validated by CURE-level evidence
}

# ============== SOURCE METADATA ==============

KNOWLEDGE_BASE_VERSION = "2.0.0"
KNOWLEDGE_BASE_LAST_UPDATE = datetime.now().strftime("%Y-%m-%d")

# Approved sources for fact gathering
APPROVED_SOURCES = {
    "WHO": {
        "name": "World Health Organization",
        "url": "https://www.who.int",
        "myth_busters": "https://www.who.int/emergencies/diseases/novel-coronavirus-2019/advice-for-public/myth-busters",
        "trust_level": "highest"
    },
    "CDC": {
        "name": "Centers for Disease Control and Prevention",
        "url": "https://www.cdc.gov",
        "facts_page": "https://www.cdc.gov/coronavirus/2019-ncov/vaccines/facts.html",
        "trust_level": "highest"
    },
    "NHS": {
        "name": "National Health Service (UK)",
        "url": "https://www.nhs.uk",
        "covid_page": "https://www.nhs.uk/conditions/coronavirus-covid-19/",
        "trust_level": "highest"
    },
    "NIH": {
        "name": "National Institutes of Health",
        "url": "https://www.nih.gov",
        "trust_level": "high"
    }
}

# Scraping settings
SCRAPING_CONFIG = {
    "request_timeout": 15,
    "delay_between_requests": 1,  # Be nice to servers
    "max_claims_per_source": 100,
    "min_claim_length": 20,
    "max_claim_length": 500,
}