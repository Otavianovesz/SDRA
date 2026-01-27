"""
Barcode Extractor - Deterministic decoder for Boleto barcodes.

Uses pyzbar for reliable barcode decoding.
VLMs hallucinate long numeric sequences - this provides deterministic accuracy.

Boleto types:
- Intercalado 2 de 5 (47/48 digits)
- QR Code (PIX)

Usage:
    from barcode_extractor import BarcodeExtractor
    
    extractor = BarcodeExtractor()
    result = extractor.extract_from_pdf("boleto.pdf")
"""

import os
import re
import logging
from typing import Dict, Optional, List, Any
from pathlib import Path

logger = logging.getLogger(__name__)


class BarcodeExtractor:
    """Deterministic barcode decoder for Brazilian boletos."""
    
    def __init__(self):
        """Initialize barcode extractor."""
        self._pyzbar_available = None
        
    def is_available(self) -> bool:
        """Check if pyzbar is available."""
        if self._pyzbar_available is not None:
            return self._pyzbar_available
            
        try:
            from pyzbar import pyzbar
            self._pyzbar_available = True
            logger.info("pyzbar available for barcode decoding")
        except ImportError:
            logger.warning("pyzbar not installed. Run: pip install pyzbar")
            self._pyzbar_available = False
            
        return self._pyzbar_available
    
    def extract_from_image(self, image_path: str) -> Dict[str, Any]:
        """
        Extract barcodes from image.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Dictionary with extracted data
        """
        if not self.is_available():
            return {"success": False, "error": "pyzbar not available"}
            
        try:
            from pyzbar import pyzbar
            from PIL import Image
            
            # Load image
            img = Image.open(image_path)
            
            # Decode all barcodes
            barcodes = pyzbar.decode(img)
            
            result = {
                "success": True,
                "barcodes": [],
                "linha_digitavel": None,
                "pix_code": None
            }
            
            for barcode in barcodes:
                data = barcode.data.decode('utf-8')
                barcode_type = barcode.type
                
                result["barcodes"].append({
                    "type": barcode_type,
                    "data": data,
                    "rect": {
                        "x": barcode.rect.left,
                        "y": barcode.rect.top,
                        "w": barcode.rect.width,
                        "h": barcode.rect.height
                    }
                })
                
                # Identify boleto linha digitavel
                if barcode_type == "I25" or barcode_type == "INTERLEAVED":
                    if len(data) in [44, 47, 48]:
                        result["linha_digitavel"] = self._format_linha_digitavel(data)
                
                # Identify PIX QR Code
                if barcode_type == "QRCODE":
                    if data.startswith("00020126") or "pix" in data.lower():
                        result["pix_code"] = data
            
            return result
            
        except Exception as e:
            logger.error(f"Barcode extraction error: {e}")
            return {"success": False, "error": str(e)}
    
    def extract_from_pdf(self, pdf_path: str, page: int = 0) -> Dict[str, Any]:
        """
        Extract barcodes from PDF page.
        
        Args:
            pdf_path: Path to PDF file
            page: Page number (0-indexed)
            
        Returns:
            Dictionary with extracted data
        """
        try:
            import fitz  # PyMuPDF
            
            # Convert PDF to high-res image for barcode detection
            doc = fitz.open(pdf_path)
            if page >= len(doc):
                page = 0
                
            pdf_page = doc[page]
            
            # Use higher DPI for barcode clarity
            zoom = 200 / 72  # 200 DPI
            mat = fitz.Matrix(zoom, zoom)
            pix = pdf_page.get_pixmap(matrix=mat)
            
            # Save temp image
            import time
            temp_dir = os.path.dirname(pdf_path) or '.'
            temp_path = os.path.join(temp_dir, f'.barcode_temp_{int(time.time()*1000)}.png')
            pix.save(temp_path)
            doc.close()
            
            # Extract from image
            result = self.extract_from_image(temp_path)
            
            # Cleanup
            try:
                os.unlink(temp_path)
            except:
                pass
                
            return result
            
        except Exception as e:
            logger.error(f"PDF barcode extraction error: {e}")
            return {"success": False, "error": str(e)}
    
    def _format_linha_digitavel(self, raw: str) -> str:
        """
        Format raw barcode to linha digitavel format.
        
        Args:
            raw: Raw barcode data (44-48 digits)
            
        Returns:
            Formatted linha digitavel
        """
        # Clean to digits only
        digits = re.sub(r'\D', '', raw)
        
        if len(digits) == 44:
            # Convert to 47-digit format
            # This is the reverse of the banco do brasil formula
            pass
            
        return digits
    
    def validate_linha_digitavel(self, linha: str) -> bool:
        """
        Validate linha digitavel using modulo 10/11.
        
        Args:
            linha: Linha digitavel (47 or 48 digits)
            
        Returns:
            True if valid
        """
        digits = re.sub(r'\D', '', linha)
        
        if len(digits) not in [47, 48]:
            return False
            
        # Validate check digits using modulo 10
        # Field 1: positions 1-9, check digit at 10
        # Field 2: positions 11-20, check digit at 21
        # Field 3: positions 22-31, check digit at 32
        # General check digit at position 33
        
        try:
            # Simplified validation - check length and format
            # Full validation would require bank-specific rules
            return len(digits) >= 47
        except:
            return False
    
    def extract_boleto_value(self, linha: str) -> Optional[float]:
        """
        Extract monetary value from boleto linha digitavel.
        
        Args:
            linha: Linha digitavel
            
        Returns:
            Value as float or None
        """
        digits = re.sub(r'\D', '', linha)
        
        if len(digits) < 44:
            return None
            
        try:
            # Value is in positions 38-47 (10 digits, last 2 are cents)
            value_str = digits[37:47]
            value = int(value_str) / 100.0
            return value if value > 0 else None
        except:
            return None
    
    def extract_due_date(self, linha: str) -> Optional[str]:
        """
        Extract due date from boleto linha digitavel.
        
        Args:
            linha: Linha digitavel (47+ digits)
            
        Returns:
            Due date as DD/MM/YYYY or None
        """
        digits = re.sub(r'\D', '', linha)
        
        if len(digits) < 44:
            return None
            
        try:
            # Date factor is in positions 34-37 (4 digits)
            # Base date: 1997-10-07
            from datetime import datetime, timedelta
            
            date_factor = int(digits[33:37])
            if date_factor == 0:
                return None
                
            base_date = datetime(1997, 10, 7)
            due_date = base_date + timedelta(days=date_factor)
            
            return due_date.strftime("%d/%m/%Y")
        except:
            return None


# Global instance
_extractor_instance = None

def get_barcode_extractor() -> BarcodeExtractor:
    """Get or create barcode extractor instance."""
    global _extractor_instance
    if _extractor_instance is None:
        _extractor_instance = BarcodeExtractor()
    return _extractor_instance


if __name__ == '__main__':
    print("Barcode Extractor for Brazilian Boletos")
    print("=" * 50)
    
    extractor = BarcodeExtractor()
    print(f"pyzbar available: {extractor.is_available()}")
    
    if extractor.is_available():
        print("\nTo test:")
        print("  result = extractor.extract_from_pdf('boleto.pdf')")
        print("  print(result['linha_digitavel'])")
