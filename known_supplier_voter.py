
import os
import re
import difflib
import logging
from pathlib import Path
from typing import List, Optional, Set

# Logging
logger = logging.getLogger(__name__)

class KnownSupplierVoter:
    """
    Voter that uses a dictionary of known suppliers derived from filenames
    to correct and validate OCR results.
    """
    
    def __init__(self, data_dir: str = None):
        self.known_suppliers: Set[str] = set()
        if data_dir:
            self.load_from_directory(data_dir)
            
    def load_from_directory(self, directory: str):
        """Scans a directory for filenames in standard format and extracts suppliers."""
        path = Path(directory)
        if not path.exists():
            logger.warning(f"Directory not found: {directory}")
            return
            
        count = 0
        # Format: DATE_ENTITY_SUPPLIER_...
        # Ex: 01.11.2025_VG_AGRO BAGGIO_...
        for f in path.glob("*.pdf"):
            parts = f.name.split('_')
            if len(parts) >= 3:
                # Basic validation of date and entity to ensure it's our format
                if re.match(r'\d{2}\.\d{2}\.\d{4}', parts[0]) and len(parts[1]) == 2:
                    supplier = parts[2].strip().upper()
                    # Filter out obviously wrong ones or variable ones
                    if len(supplier) > 3 and not any(x in supplier for x in ["BOLETO", "NFE", "NFSE"]):
                        self.known_suppliers.add(supplier)
                        count += 1
                        
        logger.info(f"Loaded {len(self.known_suppliers)} unique suppliers from {count} files.")

    def find_match(self, text: str, min_score: float = 0.85) -> Optional[str]:
        """
        Fuzzy matches the input text against the known supplier list.
        Returns the simplified known name if a match is found.
        """
        if not text or not self.known_suppliers:
            return None
            
        text_upper = text.upper()
        
        # 1. Exact Match
        if text_upper in self.known_suppliers:
            return text_upper
            
        # 2. Substring Match (if text is longer, e.g. "AGRO BAGGIO MAQUINAS" vs "AGRO BAGGIO")
        # Or if known is longer "AGRO BAGGIO LTDA" vs "AGRO BAGGIO"
        # Prioritize exact start matches
        for known in self.known_suppliers:
            if text_upper.startswith(known) or known.startswith(text_upper):
                # Only if length diff isn't huge
                if abs(len(text_upper) - len(known)) < 10:
                    return known
        

        # 3. Fuzzy Match (difflib) for typos like "AGRO BAGGJO"
        # Lower threshold to 0.7 for short names (<5 chars) to catch things like "ASTAM" vs "G.Z..."
        threshold = 0.75 if len(text_upper) < 6 else min_score
        matches = difflib.get_close_matches(text_upper, self.known_suppliers, n=1, cutoff=threshold)
        if matches:
            return matches[0]
            
        return None

if __name__ == "__main__":
    # Test
    voter = KnownSupplierVoter(r"c:\Users\otavi\Documents\Projetos_programação\SDRA_2\11.2025_NOVEMBRO_1.547")
    print(f"Total Suppliers: {len(voter.known_suppliers)}")
    
    test_cases = [
        "AGRO BAGGJO",
        "ENERGISA",
        "ENERGISA MATO GROSSO",
        "O MONTAGNA",
        "O MONTAGNA E CIA"
    ]
    
    for t in test_cases:
        print(f"'{t}' -> '{voter.find_match(t)}'")
