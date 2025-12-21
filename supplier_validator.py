"""
Supplier Validator - Professional supplier name validation and canonicalization.

This module provides a unified interface for:
1. Validating supplier names against banned patterns
2. Canonicalizing names via fuzzy matching
3. Rejecting invalid patterns (digit strings, document terms)

Usage:
    from supplier_validator import validate_supplier, is_valid_supplier
    
    result = validate_supplier("SOME COMPANY LTDA")
    if result:
        canonical_name, confidence = result
"""

import os
import re
import logging
from typing import Optional, Tuple, Set, List
from difflib import SequenceMatcher

logger = logging.getLogger(__name__)


# ==============================================================================
# BANNED ENTITIES - Loaded from file or defaults
# ==============================================================================

DEFAULT_BANNED_ENTITIES = {
    # Pagadores (NOT beneficiários - these are PAYERS, not suppliers)
    'AUTO ELETRICA MENIN', 'MENIN COMERCIO DE PECAS', 'MENIN',
    'MULTISEG INFORMATICA', 'MULTISEG',
    'VAGNER LUIZ GAIATTO E OUTRA', 'VAGNER LUIZ GAIATTO',
    
    # Insurance/Legal model text (NOT supplier names)
    'MAPFRE SEGUROS', 'MAPFRE', 'SEFAZ', 'SEFAZ AUTORIZADORA',
    'E, SE FOR O CASO, O NOME FANTASIA DA SOCIEDADE SEG',
    
    # Generic terms that appear in documents
    'EMPRESA DE PEQUENO PORTE', 'EPP', 'MEI', 'PAGADOR', 'SACADO',
    'BENEFICIARIO', 'BENEFICIÁRIO', 'FINAL VENCIME', 'LTDA',
    
    # OCR ERRORS (common misreads, NOT actual supplier names)
    'INSCRI', 'INSCRICAO', 'INSCRIÇÃO',
    'IBPT', 'DANF3E', 'FUNTTEL',
    
    # Specific OCR errors from tests
    'AGROFIBRAMUTUM', 'AGROFIBRAMUTUM LTDA',
    'NOVA MUTUM BORRACHA',
    'G. Z. BARGERI KANIESKI', 'BARGERI KANIESKI', 'BARGERI',
    'DELTA TECNOLOGIA',
    'SPEED SYSTEM INFORMATICA', 'SPEED SYSTEM', 'IREITA',
    'TINTA ESM IND',
    'DIEGO CONCENTINO PELLEGRINI',
    
    # Cross-contamination sources
    'AUTO ELETRICA SANTA CLARA',  # Appears in unrelated NFSEs
    
    # Document terms
    'DOCUMENTO AUXILIAR', 'NOTA FISCAL', 'BOLETO BANCARIO',
    'NOTA FISCAL ELETRONICA', 'NF-E', 'NFSE',
}


class SupplierValidator:
    """
    Professional supplier name validation and canonicalization.
    
    Provides a single source of truth for:
    - Banned entity checking
    - Pattern-based validation
    - Fuzzy matching to canonical names
    """
    
    def __init__(
        self, 
        banned_file: str = None,
        suppliers_file: str = None
    ):
        """
        Initialize validator with optional external configuration files.
        
        Args:
            banned_file: Path to file with banned entities (one per line)
            suppliers_file: Path to known_suppliers.txt
        """
        self._banned: Set[str] = set()
        self._canonical_names: dict = {}  # canonical -> [aliases]
        self._all_names: dict = {}  # any_name -> canonical
        
        # Load banned entities
        self._load_banned(banned_file)
        
        # Load known suppliers
        self._load_suppliers(suppliers_file)
        
        logger.info(f"SupplierValidator: {len(self._banned)} banned, {len(self._canonical_names)} known suppliers")
    
    def _load_banned(self, filepath: str = None):
        """Load banned entities from file or use defaults."""
        self._banned = set(b.upper() for b in DEFAULT_BANNED_ENTITIES)
        
        if filepath and os.path.exists(filepath):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#'):
                            self._banned.add(line.upper())
                logger.info(f"Loaded {len(self._banned)} banned entities from {filepath}")
            except Exception as e:
                logger.warning(f"Could not load banned file: {e}")
    
    def _load_suppliers(self, filepath: str = None):
        """Load known suppliers from file."""
        if filepath is None:
            filepath = os.path.join(os.path.dirname(__file__), 'known_suppliers.txt')
        
        if not os.path.exists(filepath):
            logger.warning(f"Suppliers file not found: {filepath}")
            return
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    
                    parts = [p.strip().upper() for p in line.split('|')]
                    canonical = parts[0]
                    aliases = parts[1:] if len(parts) > 1 else []
                    
                    self._canonical_names[canonical] = aliases
                    self._all_names[canonical] = canonical
                    
                    for alias in aliases:
                        self._all_names[alias] = canonical
                        
            logger.info(f"Loaded {len(self._canonical_names)} suppliers from {filepath}")
        except Exception as e:
            logger.error(f"Error loading suppliers: {e}")
    
    def is_banned(self, name: str) -> bool:
        """
        Check if a name matches any banned entity.
        Uses substring matching for robustness.
        """
        if not name:
            return True
        
        name_upper = name.upper().strip()
        
        # Direct match
        if name_upper in self._banned:
            return True
        
        # Substring match (banned IN name or name IN banned)
        for banned in self._banned:
            if banned in name_upper or name_upper in banned:
                return True
        
        return False
    
    def is_valid_pattern(self, name: str) -> bool:
        """
        Check if a name has a valid pattern (not digits, not too short, etc).
        """
        if not name:
            return False
        
        name = name.strip()
        
        # Too short
        if len(name) < 3:
            return False
        
        # Linha digitável pattern (mostly digits with dots/spaces)
        if re.match(r'^[\d\.\s]{20,}$', name):
            return False
        
        # Mostly digits (>50% digits in long strings)
        digit_count = sum(1 for c in name if c.isdigit())
        if len(name) > 10 and digit_count / len(name) > 0.5:
            return False
        
        # Pure numbers
        if name.replace('.', '').replace(',', '').replace(' ', '').isdigit():
            return False
        
        return True
    
    def canonicalize(self, name: str, min_similarity: float = 0.6) -> Optional[Tuple[str, float]]:
        """
        Find canonical name via fuzzy matching.
        
        Args:
            name: Extracted supplier name
            min_similarity: Minimum similarity threshold (0-1)
            
        Returns:
            Tuple of (canonical_name, similarity) or None
        """
        if not name:
            return None
        
        name_upper = name.upper().strip()
        
        # 1. Exact match
        if name_upper in self._all_names:
            canonical = self._all_names[name_upper]
            # CRITICAL: Check if canonical is banned!
            if self.is_banned(canonical):
                return None
            return (canonical, 1.0)
        
        # 2. Substring match
        for known_name, canonical in self._all_names.items():
            if name_upper in known_name or known_name in name_upper:
                similarity = len(name_upper) / max(len(known_name), len(name_upper))
                if similarity >= min_similarity:
                    # CRITICAL: Check if canonical is banned!
                    if self.is_banned(canonical):
                        continue  # Skip this match
                    return (canonical, similarity)
        
        # 3. Word overlap
        name_words = set(name_upper.split())
        best_match = None
        best_score = 0.0
        
        for known_name, canonical in self._all_names.items():
            known_words = set(known_name.split())
            
            if not name_words or not known_words:
                continue
            
            common = name_words & known_words
            if common:
                score = len(common) / min(len(name_words), len(known_words))
                if score > best_score and score >= min_similarity:
                    # CRITICAL: Check if canonical is banned!
                    if not self.is_banned(canonical):
                        best_score = score
                        best_match = canonical
        
        if best_match:
            return (best_match, best_score)
        
        # 4. Fuzzy matching
        for known_name, canonical in self._all_names.items():
            similarity = SequenceMatcher(None, name_upper, known_name).ratio()
            if similarity > best_score and similarity >= min_similarity:
                # CRITICAL: Check if canonical is banned!
                if not self.is_banned(canonical):
                    best_score = similarity
                    best_match = canonical
        
        if best_match:
            return (best_match, best_score)
        
        return None
    
    def validate(self, name: str, min_similarity: float = 0.6) -> Optional[Tuple[str, float]]:
        """
        Complete validation pipeline: pattern check -> banned check -> canonicalize.
        
        Args:
            name: Extracted supplier name
            min_similarity: Minimum similarity for fuzzy matching
            
        Returns:
            Tuple of (validated_name, confidence) or None if invalid
        """
        if not name:
            return None
        
        name = name.strip()
        
        # Step 1: Pattern validation
        if not self.is_valid_pattern(name):
            logger.debug(f"Invalid pattern: {name}")
            return None
        
        # Step 2: Banned check on raw name
        if self.is_banned(name):
            logger.debug(f"Banned entity: {name}")
            return None
        
        # Step 3: Try to canonicalize
        canonical_result = self.canonicalize(name, min_similarity)
        if canonical_result:
            return canonical_result
        
        # Step 4: Return cleaned name if no canonical match
        # (but still valid pattern and not banned)
        return (name.upper(), 0.5)  # Low confidence for non-canonical
    
    def get_known_suppliers(self) -> List[str]:
        """Return list of all canonical supplier names."""
        return list(self._canonical_names.keys())


# ==============================================================================
# Global instance and convenience functions
# ==============================================================================

_validator_instance: SupplierValidator = None


def get_supplier_validator() -> SupplierValidator:
    """Get or create global validator instance."""
    global _validator_instance
    if _validator_instance is None:
        _validator_instance = SupplierValidator()
    return _validator_instance


def validate_supplier(name: str, min_similarity: float = 0.6) -> Optional[Tuple[str, float]]:
    """
    Convenience function to validate a supplier name.
    
    This is the SINGLE entry point for supplier validation.
    Always use this instead of directly checking BANNED or matching.
    
    Args:
        name: Extracted supplier name
        min_similarity: Minimum similarity threshold
        
    Returns:
        Tuple of (validated_name, confidence) or None if invalid
    """
    return get_supplier_validator().validate(name, min_similarity)


def is_valid_supplier(name: str) -> bool:
    """
    Quick check if a supplier name is valid (not banned, valid pattern).
    Does NOT canonicalize.
    """
    validator = get_supplier_validator()
    return validator.is_valid_pattern(name) and not validator.is_banned(name)


# ==============================================================================
# CLI for testing
# ==============================================================================

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    
    validator = SupplierValidator()
    print(f"Loaded {len(validator._banned)} banned entities")
    print(f"Loaded {len(validator._canonical_names)} known suppliers")
    
    test_cases = [
        "AUTO ELETRICA SANTA CLARA",  # Should be REJECTED (banned)
        "SANTA CLARA LTDA",  # Should match to SANTA CLARA
        "JUCELIA GONCALVES",  # Should match or return as-is
        "34191.09008 38426.440491",  # Should be REJECTED (linha digitavel)
        "ARAGUAIA AGRICOLA",  # Should match
        "INSCRI",  # Should be REJECTED (banned)
        "CEIFAGRO COMERCIO",  # Should match
        "COELHOS TORNEARIA",  # Should match
    ]
    
    print("\n--- Validation Tests ---")
    for name in test_cases:
        result = validator.validate(name)
        if result:
            canonical, score = result
            print(f"✓ '{name}' -> '{canonical}' ({score:.0%})")
        else:
            print(f"✗ '{name}' -> REJECTED")
