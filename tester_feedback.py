"""
Tester Feedback Module - Task 1 Implementation

Handles controlled learning in TESTING MODE only:
- Collects human validation on classifications
- Adds approved claims to knowledge base
- Logs all tester actions for audit
- Prevents unauthorized additions

Key Principle: The SBERT model is NEVER modified.
We only add new reference points to compare against.
"""
import json
import os
from datetime import datetime
from typing import Optional, Dict, Any
from dataclasses import dataclass, asdict
from enum import Enum

from config import (
    SystemMode, CURRENT_MODE, TESTER_CREDENTIALS,
    TESTER_LOG_PATH, THRESHOLDS, VERIFIED_FACTS_PATH,
    KNOWN_MISINFO_PATH, UNVERIFIED_CLAIMS_PATH,
    IntentLevel
)
from intent_detector import detect_claim_intent

class FeedbackAction(Enum):
    APPROVED = "approved"
    REJECTED = "rejected"
    SKIPPED = "skipped"

@dataclass
class TesterSession:
    """Tracks a tester's session"""
    tester_id: str
    session_start: str
    claims_reviewed: int = 0
    claims_approved: int = 0
    claims_rejected: int = 0

@dataclass
class FeedbackEntry:
    """Single feedback log entry"""
    timestamp: str
    tester_id: str
    claim_text: str
    cleaned_text: str
    predicted_label: str
    predicted_confidence: float
    detected_intent: str
    feedback_action: str
    tester_note: Optional[str]
    was_stored: bool
    storage_location: Optional[str]

class TesterFeedbackManager:
    """
    Manages tester feedback and controlled knowledge base updates.
    
    Workflow:
    1. Tester authenticates
    2. System classifies claim normally
    3. Tester validates: Yes/No
    4. If Yes + confidence >= threshold: Add to KB
    5. All actions logged
    """
    
    def __init__(self):
        self.current_session: Optional[TesterSession] = None
        self.is_authenticated = False
        self._ensure_log_file()
    
    def _ensure_log_file(self):
        """Ensure log file exists"""
        if not os.path.exists(TESTER_LOG_PATH):
            with open(TESTER_LOG_PATH, 'w') as f:
                pass  # Create empty file
    
    def authenticate(self, tester_id: str, password: str) -> bool:
        """
        Authenticate a tester.
        Returns True if successful.
        """
        if tester_id in TESTER_CREDENTIALS:
            if TESTER_CREDENTIALS[tester_id] == password:
                self.current_session = TesterSession(
                    tester_id=tester_id,
                    session_start=datetime.now().isoformat()
                )
                self.is_authenticated = True
                self._log_action({
                    "type": "session_start",
                    "tester_id": tester_id,
                    "timestamp": datetime.now().isoformat()
                })
                return True
        return False
    
    def logout(self):
        """End tester session"""
        if self.current_session:
            self._log_action({
                "type": "session_end",
                "tester_id": self.current_session.tester_id,
                "timestamp": datetime.now().isoformat(),
                "claims_reviewed": self.current_session.claims_reviewed,
                "claims_approved": self.current_session.claims_approved,
                "claims_rejected": self.current_session.claims_rejected
            })
        self.current_session = None
        self.is_authenticated = False
    
    def _log_action(self, entry: Dict[str, Any]):
        """Append entry to log file (JSONL format)"""
        with open(TESTER_LOG_PATH, 'a') as f:
            f.write(json.dumps(entry) + '\n')
    
    def process_feedback(
        self,
        claim_text: str,
        cleaned_text: str,
        predicted_label: str,
        predicted_confidence: float,
        feedback: FeedbackAction,
        tester_note: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process tester feedback on a classification.
        
        Args:
            claim_text: Original claim
            cleaned_text: Preprocessed claim
            predicted_label: System's classification
            predicted_confidence: System's confidence
            feedback: Tester's validation (APPROVED/REJECTED/SKIPPED)
            tester_note: Optional note from tester
        
        Returns:
            Dict with action taken and result
        """
        if not self.is_authenticated:
            return {
                "success": False,
                "error": "Not authenticated. Please login first."
            }
        
        # Detect intent for storage categorization
        intent_result = detect_claim_intent(cleaned_text)
        
        # Update session stats
        self.current_session.claims_reviewed += 1
        
        was_stored = False
        storage_location = None
        
        if feedback == FeedbackAction.APPROVED:
            self.current_session.claims_approved += 1
            
            # Check confidence threshold
            if predicted_confidence >= THRESHOLDS["min_confidence_for_storage"]:
                # Determine storage location based on label
                if "MISINFO" in predicted_label.upper():
                    storage_location = "known_misinformation"
                    was_stored = self._add_to_misinfo(
                        cleaned_text, 
                        predicted_label,
                        predicted_confidence,
                        intent_result.level
                    )
                elif "CORRECT" in predicted_label.upper():
                    storage_location = "verified_facts"
                    was_stored = self._add_to_facts(
                        cleaned_text,
                        predicted_label,
                        predicted_confidence,
                        intent_result.level
                    )
                else:
                    storage_location = "unverified_claims"
                    was_stored = self._add_to_unverified(
                        cleaned_text,
                        predicted_label,
                        predicted_confidence,
                        intent_result.level
                    )
            else:
                storage_location = None
                was_stored = False
        
        elif feedback == FeedbackAction.REJECTED:
            self.current_session.claims_rejected += 1
        
        # Log the feedback
        log_entry = FeedbackEntry(
            timestamp=datetime.now().isoformat(),
            tester_id=self.current_session.tester_id,
            claim_text=claim_text,
            cleaned_text=cleaned_text,
            predicted_label=predicted_label,
            predicted_confidence=predicted_confidence,
            detected_intent=intent_result.level.name,
            feedback_action=feedback.value,
            tester_note=tester_note,
            was_stored=was_stored,
            storage_location=storage_location
        )
        self._log_action(asdict(log_entry))
        
        return {
            "success": True,
            "action": feedback.value,
            "was_stored": was_stored,
            "storage_location": storage_location,
            "message": self._generate_message(feedback, was_stored, storage_location, predicted_confidence)
        }
    
    def _add_to_facts(
        self, 
        text: str, 
        label: str, 
        confidence: float,
        intent_level: IntentLevel
    ) -> bool:
        """Add claim to verified facts"""
        return self._add_to_knowledge_base(
            VERIFIED_FACTS_PATH,
            text,
            "TESTER_VALIDATED",
            "tester_validated",
            intent_level,
            confidence
        )
    
    def _add_to_misinfo(
        self,
        text: str,
        label: str,
        confidence: float,
        intent_level: IntentLevel
    ) -> bool:
        """Add claim to known misinformation"""
        return self._add_to_knowledge_base(
            KNOWN_MISINFO_PATH,
            text,
            "TESTER_VALIDATED",
            "tester_identified",
            intent_level,
            confidence
        )
    
    def _add_to_unverified(
        self,
        text: str,
        label: str,
        confidence: float,
        intent_level: IntentLevel
    ) -> bool:
        """Add claim to unverified claims pool"""
        return self._add_to_knowledge_base(
            UNVERIFIED_CLAIMS_PATH,
            text,
            "TESTER_FLAGGED",
            "needs_review",
            intent_level,
            confidence
        )
    
    def _add_to_knowledge_base(
        self,
        file_path: str,
        text: str,
        source: str,
        category: str,
        intent_level: IntentLevel,
        confidence: float
    ) -> bool:
        """Generic function to add entry to a knowledge base file"""
        try:
            # Load existing data
            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    data = json.load(f)
            else:
                data = []
            
            # Check for duplicates (simple text match)
            existing_texts = [entry.get('text', '').lower() for entry in data]
            if text.lower() in existing_texts:
                return False  # Already exists
            
            # Create new entry
            new_entry = {
                "text": text,
                "source": source,
                "category": category,
                "intent_level": intent_level.name,
                "added_by": self.current_session.tester_id,
                "added_at": datetime.now().isoformat(),
                "original_confidence": confidence
            }
            
            data.append(new_entry)
            
            # Save back
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            return True
            
        except Exception as e:
            print(f"Error adding to knowledge base: {e}")
            return False
    
    def _generate_message(
        self,
        feedback: FeedbackAction,
        was_stored: bool,
        storage_location: Optional[str],
        confidence: float
    ) -> str:
        """Generate user-friendly feedback message"""
        if feedback == FeedbackAction.REJECTED:
            return "Claim rejected. No changes made to knowledge base."
        
        if feedback == FeedbackAction.SKIPPED:
            return "Claim skipped."
        
        if feedback == FeedbackAction.APPROVED:
            if was_stored:
                return f"✓ Claim approved and added to {storage_location}."
            elif confidence < THRESHOLDS["min_confidence_for_storage"]:
                return (
                    f"Claim approved but NOT stored. "
                    f"Confidence ({confidence:.0%}) below threshold "
                    f"({THRESHOLDS['min_confidence_for_storage']:.0%})."
                )
            else:
                return "Claim approved but could not be stored (may be duplicate)."
        
        return "Unknown action."
    
    def get_session_stats(self) -> Optional[Dict]:
        """Get current session statistics"""
        if not self.current_session:
            return None
        return {
            "tester_id": self.current_session.tester_id,
            "session_start": self.current_session.session_start,
            "claims_reviewed": self.current_session.claims_reviewed,
            "claims_approved": self.current_session.claims_approved,
            "claims_rejected": self.current_session.claims_rejected
        }
    
    def get_recent_logs(self, n: int = 10) -> list:
        """Get recent log entries"""
        entries = []
        try:
            with open(TESTER_LOG_PATH, 'r') as f:
                for line in f:
                    if line.strip():
                        entries.append(json.loads(line))
            return entries[-n:]
        except Exception:
            return []


# Singleton
_manager = None

def get_feedback_manager() -> TesterFeedbackManager:
    global _manager
    if _manager is None:
        _manager = TesterFeedbackManager()
    return _manager