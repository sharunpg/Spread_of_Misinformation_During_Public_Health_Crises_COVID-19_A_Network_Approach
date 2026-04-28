"""
Knowledge Ingestion Module - Task 2 Implementation

Safe, offline knowledge base expansion from trusted sources.
NO live web scraping - only pre-downloaded documents.

Supported formats:
- PDF documents
- HTML pages (saved locally)
- CSV/JSON files

Sources limited to: WHO, CDC, NHS, NIH
"""
import json
import os
import re
import csv
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path

from config import (
    DATA_DIR, INGESTION_LOG_PATH, VERIFIED_FACTS_PATH,
    KNOWN_MISINFO_PATH, APPROVED_SOURCES, THRESHOLDS,
    IntentLevel
)
from intent_detector import detect_claim_intent

# Optional imports for document processing
try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False

try:
    import pdfplumber
    PDF_AVAILABLE = True
except ImportError:
    try:
        from PyPDF2 import PdfReader
        PDF_AVAILABLE = True
    except ImportError:
        PDF_AVAILABLE = False

@dataclass
class SourceMetadata:
    """Metadata for an ingested source"""
    source_name: str  # WHO, CDC, NHS, NIH
    source_url: str
    download_date: str
    file_path: str
    document_title: str
    version: Optional[str] = None

@dataclass
class ExtractedClaim:
    """A claim extracted from a source document"""
    text: str
    source: str
    source_url: str
    download_date: str
    category: str
    intent_level: str
    claim_type: str  # "fact" or "myth"
    original_context: Optional[str] = None

@dataclass 
class ConflictReport:
    """Report of a potential conflict"""
    new_claim: str
    existing_claim: str
    similarity: float
    conflict_type: str
    recommendation: str

class KnowledgeIngestionPipeline:
    """
    Offline ingestion pipeline for expanding knowledge base.
    
    Workflow:
    1. Human downloads official documents to sources/ directory
    2. Human creates metadata.json for each source
    3. Run ingestion script
    4. Review flagged conflicts
    5. Approve additions
    
    This is NOT automated - requires human oversight at multiple steps.
    """
    
    SOURCES_DIR = os.path.join(DATA_DIR, "sources")
    
    def __init__(self):
        os.makedirs(self.SOURCES_DIR, exist_ok=True)
        self._create_source_directories()
        
        # COVID-relevant keywords for filtering
        self.covid_keywords = [
            'covid', 'coronavirus', 'sars-cov-2', 'pandemic',
            'vaccine', 'vaccination', 'mask', 'social distancing',
            'quarantine', 'isolation', 'symptom', 'transmission',
            'infection', 'virus', 'outbreak', 'immunity'
        ]
        
        # Category detection patterns
        self.category_patterns = {
            'vaccine': ['vaccine', 'vaccination', 'immunization', 'dose', 'booster', 'mrna', 'pfizer', 'moderna'],
            'transmission': ['spread', 'transmit', 'airborne', 'droplet', 'contagious', 'contact'],
            'symptoms': ['symptom', 'fever', 'cough', 'fatigue', 'taste', 'smell', 'breathing'],
            'treatment': ['treat', 'medication', 'drug', 'therapy', 'hospital', 'oxygen'],
            'prevention': ['prevent', 'protect', 'mask', 'distancing', 'hygiene', 'wash'],
            'myths': ['myth', 'false', 'misinformation', 'not true', 'no evidence', 'debunk']
        }
    
    def _create_source_directories(self):
        """Create directory structure for sources"""
        for source in APPROVED_SOURCES.keys():
            source_dir = os.path.join(self.SOURCES_DIR, source.lower())
            os.makedirs(source_dir, exist_ok=True)
    
    def create_metadata_template(self, source_name: str) -> str:
        """Create a metadata.json template for a source directory"""
        if source_name.upper() not in APPROVED_SOURCES:
            raise ValueError(f"Source {source_name} not in approved sources")
        
        source_dir = os.path.join(self.SOURCES_DIR, source_name.lower())
        metadata_path = os.path.join(source_dir, "metadata.json")
        
        template = {
            "source_name": source_name.upper(),
            "source_url": APPROVED_SOURCES[source_name.upper()]["url"],
            "download_date": datetime.now().strftime("%Y-%m-%d"),
            "documents": [
                {
                    "filename": "example.pdf",
                    "title": "Document Title",
                    "url": "https://example.com/document.pdf",
                    "type": "fact"  # or "myth" for myth-buster docs
                }
            ],
            "notes": "Add any relevant notes here"
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(template, f, indent=2)
        
        return metadata_path
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF file"""
        if not PDF_AVAILABLE:
            raise ImportError("PDF processing requires pdfplumber or PyPDF2")
        
        text = ""
        try:
            # Try pdfplumber first (better extraction)
            import pdfplumber
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
        except:
            # Fallback to PyPDF2
            from PyPDF2 import PdfReader
            reader = PdfReader(pdf_path)
            for page in reader.pages:
                text += page.extract_text() + "\n"
        
        return text
    
    def extract_text_from_html(self, html_path: str) -> str:
        """Extract text from HTML file"""
        if not BS4_AVAILABLE:
            raise ImportError("HTML processing requires beautifulsoup4")
        
        with open(html_path, 'r', encoding='utf-8') as f:
            soup = BeautifulSoup(f.read(), 'html.parser')
        
        # Remove script and style elements
        for element in soup(['script', 'style', 'nav', 'footer', 'header']):
            element.decompose()
        
        return soup.get_text(separator='\n')
    
    def extract_text_from_csv(self, csv_path: str) -> List[Dict]:
        """Extract claims from CSV file"""
        claims = []
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                claims.append(row)
        return claims
    
    def split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        # Simple sentence splitting
        text = re.sub(r'\n+', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        
        # Split on sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        # Filter out very short or very long sentences
        sentences = [
            s.strip() for s in sentences 
            if 20 < len(s.strip()) < 500
        ]
        
        return sentences
    
    def is_covid_relevant(self, text: str) -> bool:
        """Check if text is COVID-related"""
        text_lower = text.lower()
        return any(kw in text_lower for kw in self.covid_keywords)
    
    def is_factual_statement(self, text: str) -> bool:
        """Check if text appears to be a factual statement"""
        text_lower = text.lower()
        
        # Exclude questions
        if '?' in text:
            return False
        
        # Exclude navigation/boilerplate
        boilerplate = ['click here', 'learn more', 'subscribe', 'follow us', 'copyright']
        if any(bp in text_lower for bp in boilerplate):
            return False
        
        # Exclude too many numbers (likely statistics/tables)
        digit_ratio = sum(c.isdigit() for c in text) / max(len(text), 1)
        if digit_ratio > 0.3:
            return False
        
        return True
    
    def detect_category(self, text: str) -> str:
        """Detect category of a claim"""
        text_lower = text.lower()
        
        for category, keywords in self.category_patterns.items():
            if any(kw in text_lower for kw in keywords):
                return category
        
        return 'general'
    
    def normalize_claim(self, text: str) -> str:
        """Normalize claim text"""
        # Basic cleaning
        text = text.lower().strip()
        text = re.sub(r'\s+', ' ', text)
        
        # Remove citations
        text = re.sub(r'\[\d+\]', '', text)
        text = re.sub(r'\(\d{4}\)', '', text)
        
        # Standardize terminology
        replacements = {
            'covid-19': 'covid',
            'coronavirus disease': 'covid',
            'sars-cov-2': 'covid virus',
            'novel coronavirus': 'covid',
        }
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        return text.strip()
    
    def extract_claims_from_source(
        self, 
        source_name: str
    ) -> Tuple[List[ExtractedClaim], List[str]]:
        """
        Extract claims from a source directory.
        
        Returns: (extracted_claims, errors)
        """
        source_dir = os.path.join(self.SOURCES_DIR, source_name.lower())
        metadata_path = os.path.join(source_dir, "metadata.json")
        
        if not os.path.exists(metadata_path):
            return [], [f"No metadata.json found in {source_dir}"]
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        claims = []
        errors = []
        
        for doc_info in metadata.get('documents', []):
            filename = doc_info['filename']
            filepath = os.path.join(source_dir, filename)
            
            if not os.path.exists(filepath):
                errors.append(f"File not found: {filepath}")
                continue
            
            try:
                # Extract text based on file type
                if filename.endswith('.pdf'):
                    text = self.extract_text_from_pdf(filepath)
                elif filename.endswith(('.html', '.htm')):
                    text = self.extract_text_from_html(filepath)
                elif filename.endswith('.csv'):
                    # CSV is handled differently
                    csv_claims = self.extract_text_from_csv(filepath)
                    for row in csv_claims:
                        claim_text = row.get('claim', row.get('text', ''))
                        if claim_text and self.is_covid_relevant(claim_text):
                            intent = detect_claim_intent(claim_text)
                            claims.append(ExtractedClaim(
                                text=self.normalize_claim(claim_text),
                                source=metadata['source_name'],
                                source_url=doc_info.get('url', metadata['source_url']),
                                download_date=metadata['download_date'],
                                category=row.get('category', self.detect_category(claim_text)),
                                intent_level=intent.level.name,
                                claim_type=doc_info.get('type', 'fact')
                            ))
                    continue
                elif filename.endswith('.txt'):
                    with open(filepath, 'r', encoding='utf-8') as f:
                        text = f.read()
                else:
                    errors.append(f"Unsupported file type: {filename}")
                    continue
                
                # Split into sentences and extract claims
                sentences = self.split_into_sentences(text)
                
                for sentence in sentences:
                    if self.is_covid_relevant(sentence) and self.is_factual_statement(sentence):
                        normalized = self.normalize_claim(sentence)
                        intent = detect_claim_intent(normalized)
                        
                        claims.append(ExtractedClaim(
                            text=normalized,
                            source=metadata['source_name'],
                            source_url=doc_info.get('url', metadata['source_url']),
                            download_date=metadata['download_date'],
                            category=self.detect_category(normalized),
                            intent_level=intent.level.name,
                            claim_type=doc_info.get('type', 'fact'),
                            original_context=sentence
                        ))
                        
            except Exception as e:
                errors.append(f"Error processing {filename}: {str(e)}")
        
        return claims, errors
    
    def detect_conflicts(
        self, 
        new_claims: List[ExtractedClaim],
        existing_kb: Dict
    ) -> List[ConflictReport]:
        """
        Detect potential conflicts between new claims and existing KB.
        """
        # Import here to avoid circular dependency
        from knowledge_base import get_knowledge_base
        kb = get_knowledge_base()
        
        conflicts = []
        
        for claim in new_claims:
            # Encode new claim
            claim_emb = kb.encode_text(claim.text)
            
            # Check against verified facts
            fact_results = kb.find_similar_facts(claim_emb, top_k=1)
            if fact_results:
                best = fact_results[0]
                if best['score'] > THRESHOLDS['conflict_detection']:
                    # High similarity - check for type mismatch
                    if claim.claim_type == 'myth' and best['score'] > 0.85:
                        conflicts.append(ConflictReport(
                            new_claim=claim.text,
                            existing_claim=best['entry'].text,
                            similarity=best['score'],
                            conflict_type="New myth similar to existing fact",
                            recommendation="Review manually - may be paraphrase or contradiction"
                        ))
            
            # Check against known misinformation
            misinfo_results = kb.find_similar_misinfo(claim_emb, top_k=1)
            if misinfo_results:
                best = misinfo_results[0]
                if best['score'] > THRESHOLDS['conflict_detection']:
                    if claim.claim_type == 'fact' and best['score'] > 0.85:
                        conflicts.append(ConflictReport(
                            new_claim=claim.text,
                            existing_claim=best['entry'].text,
                            similarity=best['score'],
                            conflict_type="New fact similar to existing misinformation",
                            recommendation="CRITICAL: Review immediately"
                        ))
        
        return conflicts
    
    def run_ingestion(
        self, 
        source_name: str,
        dry_run: bool = True
    ) -> Dict:
        """
        Run ingestion pipeline for a source.
        
        Args:
            source_name: Name of source (WHO, CDC, NHS, NIH)
            dry_run: If True, don't actually add to KB
        
        Returns:
            Report of ingestion results
        """
        print(f"Starting ingestion for {source_name}...")
        
        # Extract claims
        claims, errors = self.extract_claims_from_source(source_name)
        
        print(f"Extracted {len(claims)} claims, {len(errors)} errors")
        
        if errors:
            print("Errors:")
            for e in errors:
                print(f"  - {e}")
        
        # Detect conflicts
        conflicts = []
        if claims:
            print("Checking for conflicts...")
            conflicts = self.detect_conflicts(claims, {})
            print(f"Found {len(conflicts)} potential conflicts")
        
        # Log the ingestion
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "source": source_name,
            "claims_extracted": len(claims),
            "errors": len(errors),
            "conflicts": len(conflicts),
            "dry_run": dry_run
        }
        
        with open(INGESTION_LOG_PATH, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
        
        # If not dry run, add claims (excluding conflicts)
        added = 0
        if not dry_run:
            conflict_texts = {c.new_claim for c in conflicts}
            for claim in claims:
                if claim.text not in conflict_texts:
                    if self._add_claim_to_kb(claim):
                        added += 1
        
        return {
            "source": source_name,
            "claims_extracted": len(claims),
            "conflicts_found": len(conflicts),
            "conflicts": [asdict(c) for c in conflicts],
            "errors": errors,
            "claims_added": added if not dry_run else 0,
            "dry_run": dry_run,
            "claims_preview": [asdict(c) for c in claims[:10]]  # First 10 for review
        }
    
    def _add_claim_to_kb(self, claim: ExtractedClaim) -> bool:
        """Add a single claim to appropriate knowledge base"""
        try:
            if claim.claim_type == 'myth':
                target_path = KNOWN_MISINFO_PATH
            else:
                target_path = VERIFIED_FACTS_PATH
            
            # Load existing
            if os.path.exists(target_path):
                with open(target_path, 'r') as f:
                    data = json.load(f)
            else:
                data = []
            
            # Check duplicate
            if any(e.get('text', '').lower() == claim.text.lower() for e in data):
                return False
            
            # Add new entry
            data.append({
                "text": claim.text,
                "source": claim.source,
                "source_url": claim.source_url,
                "category": claim.category,
                "intent_level": claim.intent_level,
                "ingested_at": datetime.now().isoformat(),
                "download_date": claim.download_date
            })
            
            with open(target_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            return True
            
        except Exception as e:
            print(f"Error adding claim: {e}")
            return False


def create_sample_source_structure():
    """Create sample source structure for demonstration"""
    pipeline = KnowledgeIngestionPipeline()
    
    # Create metadata templates
    for source in ['WHO', 'CDC', 'NHS']:
        try:
            path = pipeline.create_metadata_template(source)
            print(f"Created template: {path}")
        except Exception as e:
            print(f"Error creating {source} template: {e}")
    
    # Create sample CSV for easy data entry
    sample_csv_path = os.path.join(pipeline.SOURCES_DIR, "who", "sample_facts.csv")
    with open(sample_csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['claim', 'category', 'type'])
        writer.writerow(['covid vaccines are safe and effective', 'vaccine', 'fact'])
        writer.writerow(['5g networks do not spread coronavirus', 'myths', 'fact'])
    
    print(f"Created sample CSV: {sample_csv_path}")
    print("\nTo use:")
    print("1. Download official documents to data/sources/<source>/")
    print("2. Update metadata.json with document info")
    print("3. Run: pipeline.run_ingestion('WHO', dry_run=True)")


if __name__ == "__main__":
    create_sample_source_structure()