"""
Network Analysis Module - Lightweight Claim Clustering

Purpose:
Detect when a new claim is similar to previously seen misinformation.
This provides a "network signal" without requiring full social graph analysis.

Key Insight:
Misinformation often spreads through repetition with slight variations.
If a claim is semantically similar to many known misinfo claims,
it's more likely to be misinformation itself.

Approach:
1. MinHash + LSH for fast near-duplicate detection
2. Maintain a "claim graph" where edges = similarity
3. Compute "neighborhood risk score" based on nearby claims

Why this is appropriate for B.Tech:
- No need for actual Twitter social graph
- Works on claim text alone
- Lightweight computation
- Provides defensible "network approach" for project title
"""
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import pickle
import os

from datasketch import MinHash, MinHashLSH

from config import CLAIM_GRAPH_PATH, THRESHOLDS

@dataclass
class ClaimNode:
    """Represents a claim in our network"""
    claim_id: str
    text: str
    cleaned_text: str
    label: Optional[str] = None  # MISINFO, CORRECT, UNVERIFIED
    minhash: Optional[MinHash] = None
    neighbors: List[str] = field(default_factory=list)

class ClaimNetwork:
    """
    Maintains a network of processed claims.
    
    Key features:
    1. Near-duplicate detection using MinHash LSH
    2. Similarity-based clustering
    3. Neighborhood risk scoring
    
    The "network" is formed by semantic similarity between claims,
    not by social connections between users.
    """
    
    def __init__(self, num_perm: int = 128):
        """
        Args:
            num_perm: Number of permutations for MinHash (higher = more accurate)
        """
        self.num_perm = num_perm
        self.claims: Dict[str, ClaimNode] = {}
        self.lsh = MinHashLSH(threshold=0.5, num_perm=num_perm)
        self.claim_counter = 0
        
        self._load_if_exists()
    
    def _load_if_exists(self):
        """Load existing claim network from disk"""
        if os.path.exists(CLAIM_GRAPH_PATH):
            try:
                with open(CLAIM_GRAPH_PATH, 'rb') as f:
                    data = pickle.load(f)
                    self.claims = data['claims']
                    self.claim_counter = data['counter']
                    # Rebuild LSH index
                    for claim_id, node in self.claims.items():
                        if node.minhash:
                            try:
                                self.lsh.insert(claim_id, node.minhash)
                            except ValueError:
                                pass  # Already exists
                print(f"✓ Loaded claim network ({len(self.claims)} claims)")
            except Exception as e:
                print(f"⚠️ Could not load claim network: {e}")
    
    def _save(self):
        """Persist claim network to disk"""
        try:
            with open(CLAIM_GRAPH_PATH, 'wb') as f:
                pickle.dump({
                    'claims': self.claims,
                    'counter': self.claim_counter
                }, f)
        except Exception as e:
            print(f"⚠️ Could not save claim network: {e}")
    
    def _create_minhash(self, text: str) -> MinHash:
        """Create MinHash signature for text"""
        m = MinHash(num_perm=self.num_perm)
        # Create shingles (3-word sequences)
        words = text.lower().split()
        for i in range(len(words) - 2):
            shingle = ' '.join(words[i:i+3])
            m.update(shingle.encode('utf-8'))
        # Also add individual words for short texts
        for word in words:
            m.update(word.encode('utf-8'))
        return m
    
    def find_near_duplicates(self, text: str) -> List[Tuple[str, ClaimNode]]:
        """
        Find claims similar to input text using LSH.
        Returns list of (claim_id, node) tuples.
        """
        minhash = self._create_minhash(text)
        similar_ids = self.lsh.query(minhash)
        
        results = []
        for claim_id in similar_ids:
            if claim_id in self.claims:
                results.append((claim_id, self.claims[claim_id]))
        
        return results
    
    def add_claim(self, text: str, cleaned_text: str, label: str) -> str:
        """
        Add a new claim to the network.
        Returns the claim_id.
        """
        # Generate ID
        self.claim_counter += 1
        claim_id = f"claim_{self.claim_counter}"
        
        # Create MinHash
        minhash = self._create_minhash(cleaned_text)
        
        # Find neighbors (similar claims)
        similar = self.find_near_duplicates(cleaned_text)
        neighbor_ids = [cid for cid, _ in similar]
        
        # Create node
        node = ClaimNode(
            claim_id=claim_id,
            text=text,
            cleaned_text=cleaned_text,
            label=label,
            minhash=minhash,
            neighbors=neighbor_ids
        )
        
        # Add to network
        self.claims[claim_id] = node
        try:
            self.lsh.insert(claim_id, minhash)
        except ValueError:
            pass  # Duplicate hash
        
        # Update neighbors' neighbor lists
        for neighbor_id in neighbor_ids:
            if neighbor_id in self.claims:
                self.claims[neighbor_id].neighbors.append(claim_id)
        
        self._save()
        return claim_id
    
    def compute_neighborhood_risk(self, text: str) -> Dict:
        """
        Compute risk score based on neighboring claims.
        
        Risk Score Logic:
        - Find similar claims in the network
        - Count how many are labeled as misinformation
        - Higher ratio = higher risk
        
        Returns:
            {
                'risk_score': float (0-1),
                'similar_claims_count': int,
                'misinfo_neighbors': int,
                'correct_neighbors': int,
                'explanation': str
            }
        """
        similar = self.find_near_duplicates(text)
        
        if not similar:
            return {
                'risk_score': 0.0,
                'similar_claims_count': 0,
                'misinfo_neighbors': 0,
                'correct_neighbors': 0,
                'explanation': "No similar claims in network"
            }
        
        # Count labels in neighborhood
        misinfo_count = 0
        correct_count = 0
        unverified_count = 0
        
        for _, node in similar:
            if node.label:
                if 'MISINFO' in node.label.upper():
                    misinfo_count += 1
                elif 'CORRECT' in node.label.upper():
                    correct_count += 1
                else:
                    unverified_count += 1
        
        total = len(similar)
        
        # Calculate risk score
        # Higher if more misinformation neighbors
        # Lower if more correct neighbors
        if total > 0:
            risk_score = (misinfo_count - correct_count * 0.5) / total
            risk_score = max(0, min(1, (risk_score + 1) / 2))  # Normalize to 0-1
        else:
            risk_score = 0.0
        
        # Generate explanation
        if misinfo_count > correct_count:
            explanation = f"Claim is similar to {misinfo_count} known misinformation claims"
        elif correct_count > misinfo_count:
            explanation = f"Claim is similar to {correct_count} verified correct claims"
        else:
            explanation = f"Mixed signals from {total} similar claims"
        
        return {
            'risk_score': risk_score,
            'similar_claims_count': total,
            'misinfo_neighbors': misinfo_count,
            'correct_neighbors': correct_count,
            'explanation': explanation
        }
    
    def get_network_stats(self) -> Dict:
        """Get statistics about the claim network"""
        labels = defaultdict(int)
        for node in self.claims.values():
            labels[node.label or 'UNKNOWN'] += 1
        
        total_edges = sum(len(n.neighbors) for n in self.claims.values()) // 2
        
        return {
            'total_claims': len(self.claims),
            'total_edges': total_edges,
            'label_distribution': dict(labels),
            'avg_neighbors': total_edges * 2 / max(len(self.claims), 1)
        }


# Singleton
_network = None

def get_claim_network() -> ClaimNetwork:
    """Get or create claim network singleton"""
    global _network
    if _network is None:
        _network = ClaimNetwork()
    return _network

def compute_network_risk(text: str) -> Dict:
    """Convenience function for network risk computation"""
    return get_claim_network().compute_neighborhood_risk(text)

def add_to_network(text: str, cleaned_text: str, label: str) -> str:
    """Convenience function to add claim to network"""
    return get_claim_network().add_claim(text, cleaned_text, label)