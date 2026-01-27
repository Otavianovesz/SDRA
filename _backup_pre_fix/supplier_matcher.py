"""
Supplier Fuzzy Matcher - Uses known suppliers database for fuzzy matching.
Helps map extracted supplier names to canonical names when exact match fails.
"""

import os
from difflib import SequenceMatcher
from typing import Optional, Dict, List, Tuple


class SupplierMatcher:
    """Matches extracted supplier names against known suppliers database."""
    
    def __init__(self, suppliers_file: str = None):
        """Initialize with suppliers database file."""
        self.canonical_names: Dict[str, List[str]] = {}  # canonical -> [aliases]
        self.all_names: Dict[str, str] = {}  # any_name -> canonical
        
        if suppliers_file is None:
            suppliers_file = os.path.join(
                os.path.dirname(__file__), 'known_suppliers.txt'
            )
        
        self._load_suppliers(suppliers_file)
    
    def _load_suppliers(self, filepath: str):
        """Load suppliers from file."""
        if not os.path.exists(filepath):
            return
            
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                    
                parts = [p.strip().upper() for p in line.split('|')]
                canonical = parts[0]
                aliases = parts[1:] if len(parts) > 1 else []
                
                self.canonical_names[canonical] = aliases
                self.all_names[canonical] = canonical
                
                for alias in aliases:
                    self.all_names[alias] = canonical
    
    def match(self, extracted_name: str, min_similarity: float = 0.6) -> Optional[Tuple[str, float]]:
        """
        Find best matching supplier for extracted name.
        
        Args:
            extracted_name: Name extracted from document
            min_similarity: Minimum similarity threshold (0-1)
            
        Returns:
            Tuple of (canonical_name, similarity_score) or None if no match
        """
        if not extracted_name:
            return None
            
        extracted_upper = extracted_name.upper().strip()
        
        # 1. Exact match
        if extracted_upper in self.all_names:
            return (self.all_names[extracted_upper], 1.0)
        
        # 2. Substring match (extracted is part of a known name)
        for known_name, canonical in self.all_names.items():
            if extracted_upper in known_name or known_name in extracted_upper:
                similarity = len(extracted_upper) / max(len(known_name), len(extracted_upper))
                if similarity >= min_similarity:
                    return (canonical, similarity)
        
        # 3. Word overlap match
        extracted_words = set(extracted_upper.split())
        best_match = None
        best_score = 0.0
        
        for known_name, canonical in self.all_names.items():
            known_words = set(known_name.split())
            
            if not extracted_words or not known_words:
                continue
                
            # Count common words
            common = extracted_words & known_words
            if common:
                # Score based on overlap percentage
                score = len(common) / min(len(extracted_words), len(known_words))
                if score > best_score and score >= min_similarity:
                    best_score = score
                    best_match = canonical
        
        if best_match:
            return (best_match, best_score)
        
        # 4. Fuzzy string matching using SequenceMatcher
        for known_name, canonical in self.all_names.items():
            similarity = SequenceMatcher(None, extracted_upper, known_name).ratio()
            if similarity > best_score and similarity >= min_similarity:
                best_score = similarity
                best_match = canonical
        
        if best_match:
            return (best_match, best_score)
            
        return None
    
    def get_known_suppliers(self) -> List[str]:
        """Return list of all canonical supplier names."""
        return list(self.canonical_names.keys())


# Global instance for easy import
_matcher_instance = None

def get_supplier_matcher() -> SupplierMatcher:
    """Get or create global supplier matcher instance."""
    global _matcher_instance
    if _matcher_instance is None:
        _matcher_instance = SupplierMatcher()
    return _matcher_instance


def match_supplier(extracted_name: str, min_similarity: float = 0.6) -> Optional[Tuple[str, float]]:
    """
    Convenience function to match a supplier name.
    
    Args:
        extracted_name: Name extracted from document
        min_similarity: Minimum similarity threshold
        
    Returns:
        Tuple of (canonical_name, similarity) or None
    """
    return get_supplier_matcher().match(extracted_name, min_similarity)


if __name__ == '__main__':
    # Test the matcher
    matcher = SupplierMatcher()
    print(f"Loaded {len(matcher.canonical_names)} suppliers")
    
    test_cases = [
        "ABDALLA TRUCK CENTER SERVICOS AUTOMOTIVOS EIRELI",
        "SANTA CLARA LTDA",
        "AUTO ELETRICA SANTA CLARA",
        "CADORE, BIDOIA E CIA LTDA",
        "AGRO BAGGIO MAQUINAS AGRICOLAS",
        "DEL MORO",
        "INSS",  # Should not match
        "CONSTRUCAO CIVIL",  # Should not match
    ]
    
    print("\nTest matches:")
    for name in test_cases:
        result = matcher.match(name)
        if result:
            canonical, score = result
            print(f"  '{name}' -> '{canonical}' ({score:.2%})")
        else:
            print(f"  '{name}' -> NO MATCH")
