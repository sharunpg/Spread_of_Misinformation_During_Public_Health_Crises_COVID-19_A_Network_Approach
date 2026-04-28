"""
Two-Tier Knowledge Base Module - Expanded

Tier 1: Government Health Guidance (WHO, CDC, NHS, NIH)
        - Curated verified facts
        - Known misinformation from myth busters
        
Tier 2: Wikidata Entities
        - Structured facts for entity validation
        - Pre-downloaded, offline-first
"""
import json
import os
from typing import Dict, List, Optional
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer, util
import torch

from config import (
    VERIFIED_FACTS_PATH, KNOWN_MISINFO_PATH, 
    WIKIDATA_ENTITIES_PATH, SBERT_MODEL, CACHE_DIR
)

@dataclass
class KnowledgeEntry:
    """Single knowledge base entry with metadata"""
    text: str
    source: str
    category: str
    embedding: Optional[torch.Tensor] = None

class KnowledgeBase:
    """
    Manages verified facts and known misinformation.
    Embeddings are computed once and cached for fast lookup.
    """
    
    def __init__(self):
        self.model = SentenceTransformer(SBERT_MODEL)
        self.verified_facts: List[KnowledgeEntry] = []
        self.known_misinfo: List[KnowledgeEntry] = []
        self.wikidata_entities: Dict[str, Dict] = {}
        
        self._load_knowledge_bases()
        self._compute_embeddings()
    
    def _load_knowledge_bases(self):
        """Load from JSON files or create defaults"""
        # Load verified facts
        if os.path.exists(VERIFIED_FACTS_PATH):
            with open(VERIFIED_FACTS_PATH, 'r') as f:
                data = json.load(f)
                for item in data:
                    self.verified_facts.append(KnowledgeEntry(
                        text=item['text'],
                        source=item['source'],
                        category=item.get('category', 'general')
                    ))
        else:
            self._create_default_facts()
        
        # Load known misinformation
        if os.path.exists(KNOWN_MISINFO_PATH):
            with open(KNOWN_MISINFO_PATH, 'r') as f:
                data = json.load(f)
                for item in data:
                    self.known_misinfo.append(KnowledgeEntry(
                        text=item['text'],
                        source=item['source'],
                        category=item.get('category', 'myth')
                    ))
        else:
            self._create_default_misinfo()
        
        # Load Wikidata entities
        if os.path.exists(WIKIDATA_ENTITIES_PATH):
            with open(WIKIDATA_ENTITIES_PATH, 'r') as f:
                self.wikidata_entities = json.load(f)
        else:
            self._create_default_wikidata()
        
        print(f"✓ Loaded {len(self.verified_facts)} verified facts")
        print(f"✓ Loaded {len(self.known_misinfo)} known misinformation patterns")
        print(f"✓ Loaded {len(self.wikidata_entities)} Wikidata entities")
    
    def _compute_embeddings(self):
        """Pre-compute embeddings for fast similarity search"""
        cache_path = os.path.join(CACHE_DIR, "kb_embeddings.pt")
        
        # Check if cache is stale (different number of entries)
        cache_valid = False
        if os.path.exists(cache_path):
            try:
                cached = torch.load(cache_path)
                if (cached['facts'].shape[0] == len(self.verified_facts) and 
                    cached['misinfo'].shape[0] == len(self.known_misinfo)):
                    self._fact_embeddings = cached['facts']
                    self._misinfo_embeddings = cached['misinfo']
                    cache_valid = True
                    print("✓ Loaded cached embeddings")
            except:
                pass
        
        if not cache_valid:
            print("Computing embeddings (one-time)...")
            fact_texts = [e.text for e in self.verified_facts]
            misinfo_texts = [e.text for e in self.known_misinfo]
            
            self._fact_embeddings = self.model.encode(
                fact_texts, convert_to_tensor=True, show_progress_bar=True
            )
            self._misinfo_embeddings = self.model.encode(
                misinfo_texts, convert_to_tensor=True, show_progress_bar=True
            )
            
            torch.save({
                'facts': self._fact_embeddings,
                'misinfo': self._misinfo_embeddings
            }, cache_path)
            print("✓ Embeddings computed and cached")
    
    def find_similar_facts(self, query_embedding: torch.Tensor, top_k: int = 3):
        """Find most similar verified facts"""
        scores = util.cos_sim(query_embedding, self._fact_embeddings)[0]
        top_indices = scores.argsort(descending=True)[:top_k]
        
        results = []
        for idx in top_indices:
            results.append({
                'entry': self.verified_facts[idx],
                'score': scores[idx].item()
            })
        return results
    
    def find_similar_misinfo(self, query_embedding: torch.Tensor, top_k: int = 3):
        """Find most similar known misinformation"""
        scores = util.cos_sim(query_embedding, self._misinfo_embeddings)[0]
        top_indices = scores.argsort(descending=True)[:top_k]
        
        results = []
        for idx in top_indices:
            results.append({
                'entry': self.known_misinfo[idx],
                'score': scores[idx].item()
            })
        return results
    
    def lookup_entity(self, entity_name: str) -> Optional[Dict]:
        """Lookup entity in Wikidata cache"""
        normalized = entity_name.lower().strip()
        return self.wikidata_entities.get(normalized)
    
    def encode_text(self, text: str) -> torch.Tensor:
        """Encode text to embedding"""
        return self.model.encode([text], convert_to_tensor=True)
    
    def _create_default_facts(self):
        """Create expanded verified facts from WHO/CDC guidance"""
        default_facts = [
            # === VACCINES ===
            {"text": "COVID-19 vaccines are safe and effective at preventing severe illness", "source": "WHO", "category": "vaccines"},
            {"text": "COVID-19 vaccines reduce hospitalization and death rates significantly", "source": "CDC", "category": "vaccines"},
            {"text": "mRNA vaccines do not alter human DNA", "source": "CDC", "category": "vaccines"},
            {"text": "Vaccines undergo rigorous clinical trials before approval", "source": "WHO", "category": "vaccines"},
            {"text": "Mild side effects like fatigue and soreness are normal after vaccination", "source": "NHS", "category": "vaccines"},
            {"text": "COVID-19 vaccines do not contain microchips", "source": "WHO", "category": "vaccines"},
            {"text": "Vaccines help build immunity without causing the disease", "source": "CDC", "category": "vaccines"},
            {"text": "Booster doses help maintain protection against COVID-19", "source": "WHO", "category": "vaccines"},
            
            # === TRANSMISSION ===
            {"text": "COVID-19 spreads primarily through respiratory droplets", "source": "WHO", "category": "transmission"},
            {"text": "The virus can spread through airborne transmission in poorly ventilated spaces", "source": "CDC", "category": "transmission"},
            {"text": "COVID-19 does not spread through 5G networks", "source": "WHO", "category": "transmission"},
            {"text": "The coronavirus is not transmitted through mosquito bites", "source": "WHO", "category": "transmission"},
            
            # === PREVENTION ===
            {"text": "Wearing masks reduces transmission of COVID-19", "source": "CDC", "category": "prevention"},
            {"text": "Social distancing helps prevent spread of coronavirus", "source": "WHO", "category": "prevention"},
            {"text": "Proper hand hygiene reduces risk of infection", "source": "CDC", "category": "prevention"},
            {"text": "Well-ventilated spaces reduce risk of airborne transmission", "source": "WHO", "category": "prevention"},
            {"text": "Regular handwashing with soap and water helps prevent COVID-19", "source": "WHO", "category": "prevention"},
            {"text": "Covering coughs and sneezes helps prevent spread of respiratory illnesses", "source": "CDC", "category": "prevention"},
            
            # === SANITIZERS (WHO Myth Busters) ===
            {"text": "Alcohol-based hand sanitizers are safe and effective when used correctly", "source": "WHO", "category": "prevention"},
            {"text": "Touching a communal bottle of hand sanitizer will not give you COVID-19", "source": "WHO", "category": "prevention"},
            {"text": "Hand sanitizers with at least 60% alcohol are effective against COVID-19", "source": "CDC", "category": "prevention"},
            {"text": "Using alcohol-based sanitizers is safe for adults and children", "source": "WHO", "category": "prevention"},
            {"text": "Hand sanitizer bottles are safe to share", "source": "WHO", "category": "prevention"},
            
            # === DISEASE & SYMPTOMS ===
            {"text": "COVID-19 is caused by the SARS-CoV-2 virus", "source": "WHO", "category": "disease"},
            {"text": "COVID-19 can affect people of all ages", "source": "NHS", "category": "disease"},
            {"text": "COVID-19 symptoms include fever, cough, and difficulty breathing", "source": "CDC", "category": "symptoms"},
            {"text": "Loss of taste or smell can be a symptom of COVID-19", "source": "NHS", "category": "symptoms"},
            {"text": "Most people recover from COVID-19 without special treatment", "source": "WHO", "category": "disease"},
            {"text": "Some people experience long-term symptoms after COVID-19 infection", "source": "NHS", "category": "disease"},
            
            # === TREATMENT ===
            {"text": "There is no proven home remedy that cures COVID-19", "source": "WHO", "category": "treatment"},
            {"text": "Antibiotics do not work against viral infections like COVID-19", "source": "WHO", "category": "treatment"},
            {"text": "Isolation helps prevent spread to others when infected", "source": "CDC", "category": "treatment"},
            {"text": "Drinking water and staying hydrated supports overall health", "source": "NHS", "category": "treatment"},
            {"text": "Rest and fluids help recovery from mild COVID-19", "source": "CDC", "category": "treatment"},
            
            # === SUPPLEMENTS & ALTERNATIVE REMEDIES (Negation facts) ===
            {"text": "Vitamin and mineral supplements cannot cure COVID-19", "source": "WHO", "category": "supplements"},
            {"text": "Vitamins do not cure coronavirus", "source": "WHO", "category": "supplements"},
            {"text": "Vitamin C does not cure COVID-19", "source": "WHO", "category": "supplements"},
            {"text": "Vitamin D supplements do not prevent COVID-19 infection", "source": "WHO", "category": "supplements"},
            {"text": "Zinc supplements do not cure coronavirus", "source": "WHO", "category": "supplements"},
            {"text": "No supplement can cure or prevent COVID-19", "source": "WHO", "category": "supplements"},
            {"text": "Herbal remedies do not cure COVID-19", "source": "WHO", "category": "supplements"},
            {"text": "Turmeric does not cure COVID-19", "source": "WHO", "category": "supplements"},
            {"text": "Ginger does not cure coronavirus", "source": "WHO", "category": "supplements"},
            
            # === WHO MYTH BUSTERS - Debunked Claims (stated as facts) ===
            {"text": "Cold weather does not kill the coronavirus", "source": "WHO", "category": "myths_debunked"},
            {"text": "Hot baths do not prevent COVID-19 infection", "source": "WHO", "category": "myths_debunked"},
            {"text": "COVID-19 cannot be transmitted through houseflies", "source": "WHO", "category": "myths_debunked"},
            {"text": "Ultraviolet lamps should not be used to disinfect hands or skin", "source": "WHO", "category": "myths_debunked"},
            {"text": "Thermal scanners cannot detect COVID-19 in people without fever", "source": "WHO", "category": "myths_debunked"},
            {"text": "Spraying alcohol or chlorine on the body does not kill viruses inside", "source": "WHO", "category": "myths_debunked"},
            {"text": "Pneumonia vaccines do not protect against COVID-19", "source": "WHO", "category": "myths_debunked"},
            {"text": "Regularly rinsing nose with saline does not prevent COVID-19", "source": "WHO", "category": "myths_debunked"},
            {"text": "Eating garlic does not prevent COVID-19 infection", "source": "WHO", "category": "myths_debunked"},
            {"text": "People of all ages can be infected by COVID-19", "source": "WHO", "category": "myths_debunked"},
            {"text": "Drinking alcohol does not protect against COVID-19", "source": "WHO", "category": "myths_debunked"},
            {"text": "Adding pepper to food does not prevent or cure COVID-19", "source": "WHO", "category": "myths_debunked"},
            {"text": "Hydroxychloroquine is not a proven treatment for COVID-19", "source": "WHO", "category": "myths_debunked"},
            {"text": "Ivermectin is not recommended for COVID-19 treatment outside clinical trials", "source": "WHO", "category": "myths_debunked"},
        ]
        
        for item in default_facts:
            self.verified_facts.append(KnowledgeEntry(**item))
        
        with open(VERIFIED_FACTS_PATH, 'w') as f:
            json.dump(default_facts, f, indent=2)
        print("✓ Created default verified facts")
    
    def _create_default_misinfo(self):
        """Create default misinformation patterns from WHO myth busters"""
        default_misinfo = [
            # Conspiracy theories
            {"text": "COVID-19 is a hoax created by governments", "source": "WHO_MYTHBUSTER", "category": "conspiracy"},
            {"text": "5G networks spread coronavirus", "source": "WHO_MYTHBUSTER", "category": "conspiracy"},
            {"text": "COVID-19 vaccines contain microchips for tracking", "source": "WHO_MYTHBUSTER", "category": "conspiracy"},
            {"text": "Bill Gates created COVID-19 for population control", "source": "WHO_MYTHBUSTER", "category": "conspiracy"},
            {"text": "The pandemic was planned by world governments", "source": "WHO_MYTHBUSTER", "category": "conspiracy"},
            
            # False treatments and cures
            {"text": "Drinking bleach or disinfectant cures COVID-19", "source": "WHO_MYTHBUSTER", "category": "false_cure"},
            {"text": "Drinking hot water kills coronavirus", "source": "WHO_MYTHBUSTER", "category": "false_cure"},
            {"text": "Garlic prevents COVID-19 infection", "source": "WHO_MYTHBUSTER", "category": "false_cure"},
            {"text": "Garlic cures COVID-19", "source": "WHO_MYTHBUSTER", "category": "false_cure"},
            {"text": "Antibiotics can treat COVID-19", "source": "WHO_MYTHBUSTER", "category": "false_cure"},
            {"text": "Hydroxychloroquine is a proven cure for COVID-19", "source": "CDC_MYTHBUSTER", "category": "false_cure"},
            {"text": "Ivermectin cures COVID-19", "source": "CDC_MYTHBUSTER", "category": "false_cure"},
            {"text": "Turmeric cures COVID-19", "source": "WHO_MYTHBUSTER", "category": "false_cure"},
            {"text": "Ginger cures coronavirus", "source": "WHO_MYTHBUSTER", "category": "false_cure"},
            {"text": "Vitamin C cures COVID-19", "source": "WHO_MYTHBUSTER", "category": "false_cure"},
            {"text": "Essential oils cure COVID-19", "source": "WHO_MYTHBUSTER", "category": "false_cure"},
            {"text": "Colloidal silver cures coronavirus", "source": "CDC_MYTHBUSTER", "category": "false_cure"},
            {"text": "MMS miracle mineral solution cures COVID", "source": "FDA_WARNING", "category": "false_cure"},
            {"text": "Drinking cow urine prevents COVID-19", "source": "WHO_MYTHBUSTER", "category": "false_cure"},
            {"text": "Cocaine kills coronavirus", "source": "WHO_MYTHBUSTER", "category": "false_cure"},
            
            # Vaccine misinformation
            {"text": "COVID-19 vaccines change your DNA", "source": "CDC_MYTHBUSTER", "category": "vaccine_myth"},
            {"text": "COVID-19 vaccines cause infertility", "source": "WHO_MYTHBUSTER", "category": "vaccine_myth"},
            {"text": "Natural immunity is better than vaccine immunity", "source": "CDC_MYTHBUSTER", "category": "vaccine_myth"},
            {"text": "Vaccines give you COVID-19", "source": "WHO_MYTHBUSTER", "category": "vaccine_myth"},
            {"text": "COVID vaccines contain fetal tissue", "source": "CDC_MYTHBUSTER", "category": "vaccine_myth"},
            {"text": "mRNA vaccines are gene therapy", "source": "CDC_MYTHBUSTER", "category": "vaccine_myth"},
            {"text": "Vaccines cause autism", "source": "CDC_MYTHBUSTER", "category": "vaccine_myth"},
            
            # False claims about the virus
            {"text": "COVID-19 only affects old people", "source": "WHO_MYTHBUSTER", "category": "false_claim"},
            {"text": "Children cannot get COVID-19", "source": "WHO_MYTHBUSTER", "category": "false_claim"},
            {"text": "Summer heat kills coronavirus", "source": "WHO_MYTHBUSTER", "category": "false_claim"},
            {"text": "Cold weather kills coronavirus", "source": "WHO_MYTHBUSTER", "category": "false_claim"},
            {"text": "Masks cause oxygen deficiency", "source": "CDC_MYTHBUSTER", "category": "false_claim"},
            {"text": "Masks cause carbon dioxide poisoning", "source": "CDC_MYTHBUSTER", "category": "false_claim"},
            {"text": "Masks are dangerous for children", "source": "WHO_MYTHBUSTER", "category": "false_claim"},
            {"text": "You can tell if someone has COVID by looking at them", "source": "WHO_MYTHBUSTER", "category": "false_claim"},
            {"text": "If you can hold your breath for 10 seconds you dont have COVID", "source": "WHO_MYTHBUSTER", "category": "false_claim"},
            
            # Transmission myths
            {"text": "COVID-19 spreads through mosquito bites", "source": "WHO_MYTHBUSTER", "category": "false_transmission"},
            {"text": "Houseflies transmit coronavirus", "source": "WHO_MYTHBUSTER", "category": "false_transmission"},
            {"text": "5G towers spread COVID-19", "source": "WHO_MYTHBUSTER", "category": "false_transmission"},
            {"text": "Mobile phone networks spread the virus", "source": "WHO_MYTHBUSTER", "category": "false_transmission"},
            {"text": "Packages from China spread COVID-19", "source": "WHO_MYTHBUSTER", "category": "false_transmission"},
            
            # Sanitizer misinformation
            {"text": "Hand sanitizers are dangerous", "source": "WHO_MYTHBUSTER", "category": "false_claim"},
            {"text": "Alcohol sanitizers spread COVID", "source": "WHO_MYTHBUSTER", "category": "false_claim"},
            {"text": "Touching hand sanitizer bottles gives you COVID", "source": "WHO_MYTHBUSTER", "category": "false_claim"},
        ]
        
        for item in default_misinfo:
            self.known_misinfo.append(KnowledgeEntry(**item))
        
        with open(KNOWN_MISINFO_PATH, 'w') as f:
            json.dump(default_misinfo, f, indent=2)
        print("✓ Created default misinformation patterns")
    
    def _create_default_wikidata(self):
        """Create curated Wikidata entity facts"""
        default_entities = {
            "covid-19": {
                "qid": "Q84263196",
                "label": "COVID-19",
                "description": "infectious disease caused by SARS-CoV-2",
                "facts": {
                    "caused_by": "SARS-CoV-2",
                    "instance_of": "infectious disease",
                    "first_outbreak": "December 2019",
                    "location_first_detected": "Wuhan, China"
                }
            },
            "sars-cov-2": {
                "qid": "Q82069695",
                "label": "SARS-CoV-2",
                "description": "strain of coronavirus that causes COVID-19",
                "facts": {
                    "instance_of": "virus",
                    "causes": "COVID-19",
                    "discovered": "2019"
                }
            },
            "mrna vaccine": {
                "qid": "Q98270496",
                "label": "mRNA vaccine",
                "description": "vaccine using messenger RNA",
                "facts": {
                    "mechanism": "instructs cells to produce spike protein",
                    "does_not": "alter DNA",
                    "does_not_contain": "live virus"
                }
            },
            "pfizer vaccine": {
                "qid": "Q100120146",
                "label": "Pfizer-BioNTech COVID-19 vaccine",
                "facts": {
                    "type": "mRNA vaccine",
                    "approved_by": ["FDA", "EMA", "WHO"],
                    "efficacy": "approximately 95%"
                }
            },
            "moderna vaccine": {
                "qid": "Q100119755",
                "label": "Moderna COVID-19 vaccine",
                "facts": {
                    "type": "mRNA vaccine",
                    "approved_by": ["FDA", "EMA", "WHO"],
                    "efficacy": "approximately 94%"
                }
            },
            "hydroxychloroquine": {
                "qid": "Q421094",
                "label": "Hydroxychloroquine",
                "facts": {
                    "approved_use": "malaria, lupus, rheumatoid arthritis",
                    "covid_status": "not recommended for COVID-19 treatment",
                    "fda_warning": "risk of heart problems"
                }
            },
            "ivermectin": {
                "qid": "Q422212",
                "label": "Ivermectin",
                "facts": {
                    "approved_use": "parasitic infections",
                    "covid_status": "not approved for COVID-19",
                    "who_recommendation": "only in clinical trials"
                }
            },
            "hand sanitizer": {
                "qid": "Q24931725",
                "label": "Hand sanitizer",
                "facts": {
                    "effective_against": "many viruses including coronaviruses",
                    "recommended_concentration": "at least 60% alcohol",
                    "safety": "safe for regular use"
                }
            }
        }
        
        self.wikidata_entities = default_entities
        
        with open(WIKIDATA_ENTITIES_PATH, 'w') as f:
            json.dump(default_entities, f, indent=2)
        print("✓ Created default Wikidata entities")


# Singleton instance
_kb_instance = None

def get_knowledge_base() -> KnowledgeBase:
    """Get or create knowledge base singleton"""
    global _kb_instance
    if _kb_instance is None:
        _kb_instance = KnowledgeBase()
    return _kb_instance

def refresh_knowledge_base() -> KnowledgeBase:
    """Force refresh of knowledge base (use after adding new facts)"""
    global _kb_instance
    _kb_instance = None
    # Delete cached embeddings to force recomputation
    cache_path = os.path.join(CACHE_DIR, "kb_embeddings.pt")
    if os.path.exists(cache_path):
        os.remove(cache_path)
    return get_knowledge_base()