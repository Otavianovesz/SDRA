"""
Ensemble Voter Module - Hierarchical Voting for 8GB RAM
========================================================

Implements intelligent voting for data extraction:

1. Hierarchical Priority: Native Text > Regex > Paddle > VLM > Gemini (fallback)
2. Regex Veto: Valid CNPJ/CPF/Date checksums override OCR
3. ConfidenceScorer: Penalizes non-ASCII in numeric fields
4. REVIEW_REQUIRED: High divergence flags for human review
5. RapidFuzz for fast similarity (required)
6. Gemini AI fallback: For high divergence or low confidence cases

Gold Rule: Deterministic validators (checksums) ALWAYS win over OCR.
Silver Rule: If local OCR agrees, don't call Gemini (save cost).
Bronze Rule: If divergence high, Gemini arbitrates (Oracle).
"""

import re
import logging
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import Counter
import difflib

logger = logging.getLogger(__name__)

# Require rapidfuzz for production
try:
    from rapidfuzz import fuzz, process
    RAPIDFUZZ_AVAILABLE = True
except ImportError:
    RAPIDFUZZ_AVAILABLE = False
    logger.warning("rapidfuzz not installed - using slow difflib fallback")

# Lazy import for Gemini voter (cloud AI)
GEMINI_AVAILABLE = False
GeminiVoter = None
get_gemini_voter = None

def _ensure_gemini():
    """Lazy import Gemini voter."""
    global GEMINI_AVAILABLE, GeminiVoter, get_gemini_voter
    if GeminiVoter is not None:
        return GEMINI_AVAILABLE
    try:
        from voters.gemini_voter import GeminiVoter as _GV, get_gemini_voter as _get_gv
        GeminiVoter = _GV
        get_gemini_voter = _get_gv
        GEMINI_AVAILABLE = True
    except ImportError:
        GEMINI_AVAILABLE = False
        logger.debug("Gemini voter not available")
    return GEMINI_AVAILABLE

# CNPJ/CPF regex patterns for veto detection
CNPJ_PATTERN = re.compile(r'\d{2}\.?\d{3}\.?\d{3}/?\d{4}-?\d{2}')
CPF_PATTERN = re.compile(r'\d{3}\.?\d{3}\.?\d{3}-?\d{2}')
DATE_PATTERN = re.compile(r'\d{2}/\d{2}/\d{4}')


@dataclass
class VoteCandidate:
    """A candidate value from a specific extraction source."""
    value: str
    confidence: float
    source: str
    weight: float = 1.0
    has_valid_checksum: bool = False  # If True, this candidate has veto power


@dataclass
class VoteResult:
    """Result of ensemble voting."""
    value: str
    confidence: float
    source: str
    needs_review: bool = False  # High divergence detected
    all_candidates: List[VoteCandidate] = field(default_factory=list)


class ConfidenceScorer:
    """
    Scores extraction confidence with penalties.
    
    Penalties for:
    - Non-ASCII characters in numeric fields
    - Values that differ too much from consensus
    - Suspiciously short/long values
    """
    
    # Non-ASCII characters in numeric fields (common OCR errors)
    NON_NUMERIC_PENALTY = 0.3
    
    # Length deviation penalty
    LENGTH_PENALTY = 0.2
    
    @staticmethod
    def score_numeric_field(value: str, expected_pattern: str = r'[\d,.\s]+') -> float:
        """
        Score a numeric field extraction.
        
        Penalizes non-ASCII and non-numeric characters.
        """
        if not value:
            return 0.0
        
        # Count valid characters
        numeric_chars = sum(1 for c in value if c.isdigit() or c in ',.R$ ')
        total_chars = len(value)
        
        if total_chars == 0:
            return 0.0
        
        # Base score from numeric ratio
        base_score = numeric_chars / total_chars
        
        # Penalize non-ASCII
        non_ascii = sum(1 for c in value if ord(c) > 127)
        if non_ascii > 0:
            base_score -= ConfidenceScorer.NON_NUMERIC_PENALTY
        
        return max(0.0, min(1.0, base_score))
    
    @staticmethod
    def score_date_field(value: str) -> float:
        """Score a date field extraction."""
        if not value:
            return 0.0
        
        # Check if matches expected pattern
        if DATE_PATTERN.search(value):
            return 1.0
        
        # Partial match - has numbers but wrong format
        if re.search(r'\d+', value):
            return 0.5
        
        return 0.0
    
    @staticmethod
    def score_cnpj_field(value: str) -> float:
        """Score a CNPJ field extraction."""
        if not value:
            return 0.0
        
        # Check format
        if CNPJ_PATTERN.search(value):
            return 1.0
        
        # Just digits, might need formatting
        digits = ''.join(c for c in value if c.isdigit())
        if len(digits) == 14:
            return 0.8
        
        return 0.3


class EnsembleVoter:
    """
    Hierarchical voting engine for combining extractions.
    
    Key principles:
    1. Deterministic data (checksums) VETO probabilistic data (OCR)
    2. Native PDF text > OCR
    3. High divergence = flag for human review
    """
    
    # Source weights (higher = more trusted)
    SOURCE_WEIGHTS = {
        # Deterministic sources (highest trust)
        "regex_validated": 10.0,    # Regex with checksum validation
        "native_text": 5.0,         # PDF text layer
        "barcode": 4.0,             # Decoded barcode
        
        # High-quality OCR
        "paddle_ocr": 2.0,
        "surya": 1.8,
        
        # VLM (can hallucinate)
        "florence2": 1.5,
        "extract_hybrid": 1.5,
        
        # Cloud AI (Gemini - high trust when invoked)
        "gemini": 3.5,
        
        # Regex without validation
        "regex_specific": 1.2,
        "regex_fallback": 0.8,
        
        # Legacy/low-trust
        "gliner": 1.0,
        "filename": 0.9,
        "tesseract": 0.7,
        "qwen2_vlm": 0.5,  # Known to hallucinate
    }
    
    # Divergence threshold for REVIEW_REQUIRED flag
    DIVERGENCE_THRESHOLD = 0.4
    
    # Minimum confidence to skip Gemini fallback
    MIN_CONFIDENCE_NO_GEMINI = 0.85
    
    def __init__(self, use_cloud_ai: bool = False):
        self.scorer = ConfidenceScorer()
        self.use_cloud_ai = use_cloud_ai
        self._gemini_voter = None
    
    @property
    def gemini_voter(self):
        """Lazy load Gemini voter."""
        if not self.use_cloud_ai:
            return None
        if self._gemini_voter is None and _ensure_gemini():
            self._gemini_voter = get_gemini_voter()
        return self._gemini_voter
    
    def set_cloud_ai(self, enabled: bool):
        """Enable or disable cloud AI fallback."""
        self.use_cloud_ai = enabled
        if not enabled:
            self._gemini_voter = None
    
    def calculate_similarity(self, s1: str, s2: str) -> float:
        """Calculate similarity between two strings (0.0 to 1.0)."""
        if not s1 or not s2:
            return 0.0
        
        s1 = s1.upper().strip()
        s2 = s2.upper().strip()
        
        if s1 == s2:
            return 1.0
        
        if RAPIDFUZZ_AVAILABLE:
            return fuzz.ratio(s1, s2) / 100.0
        else:
            return difflib.SequenceMatcher(None, s1, s2).ratio()
    
    def validate_cnpj(self, cnpj: str) -> bool:
        """Validate CNPJ checksum (Modulo 11)."""
        digits = ''.join(c for c in cnpj if c.isdigit())
        if len(digits) != 14:
            return False
        
        # First check digit
        weights1 = [5, 4, 3, 2, 9, 8, 7, 6, 5, 4, 3, 2]
        sum1 = sum(int(d) * w for d, w in zip(digits[:12], weights1))
        dv1 = 11 - (sum1 % 11)
        dv1 = 0 if dv1 >= 10 else dv1
        
        if int(digits[12]) != dv1:
            return False
        
        # Second check digit
        weights2 = [6, 5, 4, 3, 2, 9, 8, 7, 6, 5, 4, 3, 2]
        sum2 = sum(int(d) * w for d, w in zip(digits[:13], weights2))
        dv2 = 11 - (sum2 % 11)
        dv2 = 0 if dv2 >= 10 else dv2
        
        return int(digits[13]) == dv2
    
    def validate_cpf(self, cpf: str) -> bool:
        """Validate CPF checksum (Modulo 11)."""
        digits = ''.join(c for c in cpf if c.isdigit())
        if len(digits) != 11:
            return False
        
        # First check digit
        sum1 = sum(int(d) * w for d, w in zip(digits[:9], range(10, 1, -1)))
        dv1 = 11 - (sum1 % 11)
        dv1 = 0 if dv1 >= 10 else dv1
        
        if int(digits[9]) != dv1:
            return False
        
        # Second check digit
        sum2 = sum(int(d) * w for d, w in zip(digits[:10], range(11, 1, -1)))
        dv2 = 11 - (sum2 % 11)
        dv2 = 0 if dv2 >= 10 else dv2
        
        return int(digits[10]) == dv2
    
    def apply_regex_veto(
        self, 
        candidates: List[VoteCandidate],
        field_type: str = "any"
    ) -> Optional[VoteCandidate]:
        """
        GOLD RULE: If a candidate has a valid checksum, it wins absolutely.
        
        Returns the vetoing candidate or None if no veto applies.
        """
        for candidate in candidates:
            value = candidate.value
            
            # Check for valid CNPJ
            if field_type in ("any", "cnpj"):
                cnpj_match = CNPJ_PATTERN.search(value)
                if cnpj_match and self.validate_cnpj(cnpj_match.group()):
                    logger.info(f"CNPJ VETO: {value} has valid checksum")
                    candidate.has_valid_checksum = True
                    candidate.confidence = 0.99
                    candidate.source = "regex_validated"
                    return candidate
            
            # Check for valid CPF
            if field_type in ("any", "cpf"):
                cpf_match = CPF_PATTERN.search(value)
                if cpf_match and self.validate_cpf(cpf_match.group()):
                    logger.info(f"CPF VETO: {value} has valid checksum")
                    candidate.has_valid_checksum = True
                    candidate.confidence = 0.99
                    candidate.source = "regex_validated"
                    return candidate
        
        return None
    
    def check_divergence(self, candidates: List[VoteCandidate]) -> bool:
        """
        Check if candidates have high divergence (need human review).
        
        Returns True if divergence exceeds threshold.
        """
        if len(candidates) < 2:
            return False
        
        # Compare all pairs
        values = [c.value for c in candidates if c.value]
        if len(values) < 2:
            return False
        
        # Calculate average similarity
        similarities = []
        for i, v1 in enumerate(values):
            for v2 in values[i+1:]:
                similarities.append(self.calculate_similarity(v1, v2))
        
        if not similarities:
            return False
        
        avg_similarity = sum(similarities) / len(similarities)
        
        # High divergence if average similarity is low
        return avg_similarity < (1.0 - self.DIVERGENCE_THRESHOLD)
    
    def prioritize_native_text(
        self, 
        candidates: List[VoteCandidate]
    ) -> Optional[VoteCandidate]:
        """
        Prioritize native PDF text over OCR if available.
        
        Returns the native text candidate if found and confident.
        """
        for candidate in candidates:
            if candidate.source == "native_text" and candidate.confidence >= 0.8:
                logger.debug(f"Prioritizing native text: {candidate.value}")
                return candidate
        return None
    
    def weighted_vote(
        self, 
        candidates: List[VoteCandidate],
        field_type: str = "any",
        file_path: Optional[str] = None
    ) -> VoteResult:
        """
        Select best candidate using hierarchical voting.
        
        Priority:
        1. Valid checksum (VETO power)
        2. Native PDF text
        3. Weighted cluster voting
        4. Gemini AI fallback (if enabled and divergence high)
        
        Args:
            candidates: List of extraction candidates
            field_type: Type of field (cnpj, cpf, date, amount, any)
            file_path: Optional path to source file for Gemini fallback
            
        Returns:
            VoteResult with best value and metadata
        """
        if not candidates:
            return VoteResult(value="", confidence=0.0, source="none")
        
        if len(candidates) == 1:
            return VoteResult(
                value=candidates[0].value,
                confidence=candidates[0].confidence,
                source=candidates[0].source,
                all_candidates=candidates
            )
        
        # Step 1: Check for regex veto (checksum validation)
        veto = self.apply_regex_veto(candidates, field_type)
        if veto:
            return VoteResult(
                value=veto.value,
                confidence=veto.confidence,
                source=veto.source,
                all_candidates=candidates
            )
        
        # Step 2: Prioritize native text
        native = self.prioritize_native_text(candidates)
        if native:
            return VoteResult(
                value=native.value,
                confidence=native.confidence,
                source=native.source,
                all_candidates=candidates
            )
        
        # Step 3: Check for high divergence
        needs_review = self.check_divergence(candidates)
        if needs_review:
            logger.warning("High divergence detected - flagging for review")
            
            # Try Gemini arbitration if enabled and file path provided
            if self.gemini_voter and file_path:
                gemini_result = self._invoke_gemini_arbitration(file_path, field_type)
                if gemini_result:
                    return gemini_result
        
        # Step 4: Cluster similar values
        clusters = self._cluster_candidates(candidates)
        
        # Step 5: Score clusters
        best_cluster = None
        max_score = -1.0
        
        for cluster in clusters:
            score = self._score_cluster(cluster, field_type)
            if score > max_score:
                max_score = score
                best_cluster = cluster
        
        if not best_cluster:
            return VoteResult(
                value="",
                confidence=0.0,
                source="none",
                needs_review=True,
                all_candidates=candidates
            )
        
        # Get best candidate from cluster
        best_candidate = max(
            best_cluster,
            key=lambda c: self.SOURCE_WEIGHTS.get(c.source, 1.0) * c.confidence
        )
        
        # Final confidence
        final_confidence = min(0.95, max_score / 3.0)
        
        return VoteResult(
            value=best_candidate.value,
            confidence=final_confidence,
            source=best_candidate.source,
            needs_review=needs_review,
            all_candidates=candidates
        )
    
    def _cluster_candidates(
        self, 
        candidates: List[VoteCandidate],
        threshold: float = 0.80
    ) -> List[List[VoteCandidate]]:
        """Group similar candidates into clusters."""
        clusters = []
        processed = set()
        
        for i, c1 in enumerate(candidates):
            if i in processed:
                continue
            
            cluster = [c1]
            processed.add(i)
            
            for j, c2 in enumerate(candidates):
                if j in processed:
                    continue
                
                if self.calculate_similarity(c1.value, c2.value) >= threshold:
                    cluster.append(c2)
                    processed.add(j)
            
            clusters.append(cluster)
        
        return clusters
    
    def _score_cluster(
        self, 
        cluster: List[VoteCandidate],
        field_type: str
    ) -> float:
        """Score a cluster based on weights, confidence, and consensus."""
        if not cluster:
            return 0.0
        
        score = 0.0
        for candidate in cluster:
            weight = self.SOURCE_WEIGHTS.get(candidate.source, 1.0)
            
            # Apply field-specific scoring
            if field_type == "amount":
                field_score = self.scorer.score_numeric_field(candidate.value)
            elif field_type == "date":
                field_score = self.scorer.score_date_field(candidate.value)
            elif field_type == "cnpj":
                field_score = self.scorer.score_cnpj_field(candidate.value)
            else:
                field_score = 1.0
            
            score += candidate.confidence * weight * field_score
        
        # Consensus bonus
        if len(cluster) > 1:
            score *= 1.0 + (0.1 * len(cluster))
        
        return score
    
    def rover_alignment(self, text_list: List[str]) -> str:
        """
        ROVER-style alignment for OCR error correction.
        
        Merges multiple OCR outputs by character voting.
        """
        if not text_list:
            return ""
        if len(text_list) == 1:
            return text_list[0]
        
        # Get base string (longest)
        text_list = [t for t in text_list if t]
        if not text_list:
            return ""
        
        base = max(text_list, key=len)
        
        # For now, return string with most alphanumeric characters
        best = max(text_list, key=lambda s: sum(c.isalnum() for c in s))
        return best
    
    def _invoke_gemini_arbitration(
        self, 
        file_path: str, 
        field_type: str
    ) -> Optional[VoteResult]:
        """
        Invoke Gemini AI to arbitrate when local OCR has high divergence.
        
        This is the "Oracle" fallback - only called when local methods disagree.
        
        Args:
            file_path: Path to the source document
            field_type: Type of field being extracted
            
        Returns:
            VoteResult from Gemini or None if failed
        """
        if not self.gemini_voter:
            return None
        
        try:
            from pathlib import Path
            if not Path(file_path).exists():
                logger.warning(f"File not found for Gemini arbitration: {file_path}")
                return None
            
            logger.info(f"Invoking Gemini arbitration for: {file_path}")
            result = self.gemini_voter.extract(file_path)
            
            if not result.success:
                logger.warning(f"Gemini extraction failed: {result.error}")
                return None
            
            # Extract the relevant field from Gemini result
            data = result.data
            value = None
            
            if field_type == "amount":
                value = data.get('amount')
                if value is not None:
                    value = str(value)
            elif field_type == "date":
                value = data.get('due_date') or data.get('emission_date')
            elif field_type in ("cnpj", "cpf"):
                value = data.get('supplier_doc')
            elif field_type == "supplier":
                value = data.get('supplier_name')
            else:
                # Return full result for general extraction
                # Pick the most relevant field
                value = (
                    data.get('supplier_name') or 
                    data.get('due_date') or 
                    str(data.get('amount', ''))
                )
            
            if not value:
                return None
            
            logger.info(f"Gemini arbitration result: {value} (confidence: {result.confidence})")
            
            return VoteResult(
                value=str(value),
                confidence=result.confidence,
                source="gemini",
                needs_review=result.confidence < 0.7,
                all_candidates=[VoteCandidate(
                    value=str(value),
                    confidence=result.confidence,
                    source="gemini",
                    weight=self.SOURCE_WEIGHTS.get("gemini", 3.5)
                )]
            )
            
        except Exception as e:
            logger.error(f"Gemini arbitration error: {e}")
            return None


# =============================================================================
# SINGLETON
# =============================================================================

_voter_instance: Optional[EnsembleVoter] = None


def get_ensemble_voter() -> EnsembleVoter:
    """Get or create the ensemble voter singleton."""
    global _voter_instance
    if _voter_instance is None:
        _voter_instance = EnsembleVoter()
    return _voter_instance
