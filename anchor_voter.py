
import fitz
import re
import logging
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class AnchorResult:
    value: str
    confidence: float
    source: str

class AnchorVoter:
    """
    Voter using geometric anchors (Label -> Value below/right) 
    for standardized documents (Boletos).
    """
    
    def __init__(self):
        pass
        
    def extract_boleto_fields(self, doc: fitz.Document) -> Dict[str, AnchorResult]:
        """
        Extracts Date and Amount from standard Boleto layout.
        Standard CNAB layout:
        - Vencimento: Top Right box
        - Valor Documento: Below Vencimento (usually) or near Agência
        """
        results = {}
        try:
            page = doc[0]
            words = page.get_text("words")
            
            # 1. Find Anchors
            venc_anchor = None
            valor_anchor = None
            
            # Iterate words to find anchors accurately
            for i, w in enumerate(words):
                text = w[4].upper()
                
                # Check Vencimento
                if "VENCIMENTO" in text:
                    venc_anchor = w
                
                # Check Valor
                if "VALOR" in text and i+2 < len(words):
                    # Check next words for "DO DOCUMENTO"
                    next_text = words[i+1][4].upper() + " " + words[i+2][4].upper()
                    if "DOCUMENTO" in next_text:
                        valor_anchor = words[i+2] # Anchor is the last word
            
            # 2. Extract relative to anchors
            if venc_anchor:
                # Look BELOW the anchor (x +/- 10, y + 5 to y + 40)
                x0, y0, x1, y1 = venc_anchor[:4]
                search_rect = fitz.Rect(x0 - 20, y1, x1 + 100, y1 + 35) # Wider search box
                
                # Extract words in rect
                extracted_date = self._extract_text_in_rect(page, search_rect)
                
                # Validate date format regex
                match = re.search(r'(\d{2}/\d{2}/\d{4})', extracted_date)
                if match:
                    results["due_anchor"] = AnchorResult(match.group(1), 0.98, "anchor_vencimento")
            
            if valor_anchor:
                # Look BELOW the anchor
                x0, y0, x1, y1 = valor_anchor[:4]
                search_rect = fitz.Rect(x0 - 20, y1, x1 + 80, y1 + 35)
                
                extracted_val = self._extract_text_in_rect(page, search_rect)
                
                # Validate money regex
                match = re.search(r'([\d\.]+,?\d*)', extracted_val)
                if match:
                    val_str = match.group(1)
                    # Simple cleanup
                    if any(c.isdigit() for c in val_str):
                        results["amount_anchor"] = AnchorResult(val_str, 0.95, "anchor_valor")
                        
        except Exception as e:
            logger.warning(f"AnchorVoter error: {e}")
            
        return results

    def _extract_text_in_rect(self, page, rect) -> str:
        """Extracts text strictly within a rectangle."""
        # Use clip to get text inside
        return page.get_text("text", clip=rect).strip()

if __name__ == "__main__":
    # Test
    # sys needs to be imported if running standalone
    import sys
    if len(sys.argv) > 1:
        doc = fitz.open(sys.argv[1])
        voter = AnchorVoter()
        res = voter.extract_boleto_fields(doc)
        print(res)
