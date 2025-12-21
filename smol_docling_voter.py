"""
SmolDocling VLM Voter - Ultra-compact document understanding model.

SmolDocling has only ~256M parameters, making it ideal for CPU inference with 8GB RAM.
It's specialized for document OCR, layout analysis, and structured text extraction.

Model: ds4sd/SmolDocling-256M-preview
Requirements: transformers, torch, pillow

Usage:
    from smol_docling_voter import SmolDoclingVoter
    
    voter = SmolDoclingVoter()
    result = voter.extract_from_pdf("document.pdf")
"""

import os
import re
import json
import logging
from typing import Dict, Optional, Any, List
from pathlib import Path

logger = logging.getLogger(__name__)


class SmolDoclingVoter:
    """Ultra-compact VLM voter for document extraction using SmolDocling (256M params)."""
    
    MODEL_ID = "ds4sd/SmolDocling-256M-preview"
    
    def __init__(self):
        """Initialize SmolDocling voter."""
        self._model = None
        self._processor = None
        self._loaded = False
        self._available = None
        
    def is_available(self) -> bool:
        """Check if SmolDocling can be loaded."""
        if self._available is not None:
            return self._available
            
        try:
            import torch
            from transformers import AutoProcessor, AutoModelForVision2Seq
            self._available = True
            logger.info("SmolDocling dependencies available")
        except ImportError as e:
            logger.warning(f"SmolDocling not available: {e}")
            self._available = False
            
        return self._available
    
    def _load_model(self):
        """Load the SmolDocling model (lazy loading to save memory)."""
        if self._loaded:
            return True
            
        if not self.is_available():
            return False
            
        try:
            import torch
            from transformers import AutoProcessor, AutoModelForVision2Seq
            
            logger.info(f"Loading SmolDocling model: {self.MODEL_ID}")
            
            # Load processor and model
            self._processor = AutoProcessor.from_pretrained(self.MODEL_ID)
            self._model = AutoModelForVision2Seq.from_pretrained(
                self.MODEL_ID,
                torch_dtype=torch.float32,  # Use float32 for CPU
                device_map="cpu"  # Force CPU
            )
            
            self._loaded = True
            logger.info("SmolDocling model loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load SmolDocling: {e}")
            return False
    
    def extract_from_image(self, image_path: str) -> Optional[Dict[str, Any]]:
        """
        Extract document data from an image.
        
        Args:
            image_path: Path to the document image
            
        Returns:
            Dictionary with extracted fields or None
        """
        if not self._load_model():
            return None
            
        try:
            from PIL import Image
            import torch
            
            # Load image
            image = Image.open(image_path).convert("RGB")
            
            # Prepare prompt with image placeholder
            # SmolDocling expects <image> token in the prompt
            prompt = """<image>
Analyze this financial document and extract:
- Supplier name (Beneficiário)
- Due date (Vencimento) DD/MM/YYYY
- Amount (Valor) in R$
- Document number
- Document type
"""
            
            # Process with proper image input
            inputs = self._processor(
                text=prompt,
                images=[image],
                return_tensors="pt"
            )
            
            # Generate
            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=False
                )
            
            # Decode response
            response = self._processor.decode(outputs[0], skip_special_tokens=True)
            
            # Parse response to extract fields
            return self._parse_response(response)
            
        except Exception as e:
            logger.error(f"SmolDocling extraction error: {e}")
            return None
    
    def extract_text_from_image(self, image_path: str) -> Optional[str]:
        """
        Extract raw text from document image using OCR mode.
        
        Args:
            image_path: Path to the document image
            
        Returns:
            Extracted text or None
        """
        if not self._load_model():
            return None
            
        try:
            from PIL import Image
            import torch
            
            image = Image.open(image_path).convert("RGB")
            
            # Use simple OCR prompt
            prompt = "<OCR>"
            
            inputs = self._processor(
                text=prompt,
                images=image,
                return_tensors="pt"
            )
            
            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=False
                )
            
            response = self._processor.decode(outputs[0], skip_special_tokens=True)
            return response.replace(prompt, "").strip()
            
        except Exception as e:
            logger.error(f"SmolDocling OCR error: {e}")
            return None
    
    def extract_from_pdf(self, pdf_path: str, page: int = 0) -> Optional[Dict[str, Any]]:
        """
        Extract data from a PDF page.
        
        Args:
            pdf_path: Path to the PDF file
            page: Page number (0-indexed)
            
        Returns:
            Dictionary with extracted fields or None
        """
        temp_path = None
        try:
            import fitz  # PyMuPDF
            
            # Convert PDF page to image
            doc = fitz.open(pdf_path)
            if page >= len(doc):
                page = 0
                
            pdf_page = doc[page]
            
            # Render at 150 DPI
            zoom = 150 / 72
            mat = fitz.Matrix(zoom, zoom)
            pix = pdf_page.get_pixmap(matrix=mat)
            
            # Save to temp file in current directory (avoid Windows permission issues)
            import tempfile
            import time
            temp_dir = os.path.dirname(pdf_path) if os.path.dirname(pdf_path) else '.'
            temp_path = os.path.join(temp_dir, f'.smol_temp_{int(time.time()*1000)}.png')
            pix.save(temp_path)
            
            doc.close()
            
            # Extract from image
            result = self.extract_from_image(temp_path)
            
            return result
            
        except Exception as e:
            logger.error(f"SmolDocling PDF error: {e}")
            return None
        finally:
            # Clean up temp file
            if temp_path and os.path.exists(temp_path):
                try:
                    import time
                    time.sleep(0.1)  # Give Windows time to release the file
                    os.unlink(temp_path)
                except:
                    pass  # Ignore cleanup errors
    
    def _parse_response(self, response: str) -> Dict[str, Any]:
        """Parse SmolDocling response to extract structured fields."""
        result = {}
        
        # Try to find key-value pairs in response
        lines = response.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Look for patterns like "Supplier: ABC LTDA" or "1. Supplier: ABC"
            patterns = [
                (r'(?:supplier|benefici[aá]rio|emitente)[\s:]+(.+)', 'supplier_name'),
                (r'(?:due\s*date|vencimento|data\s*venc)[\s:]+(\d{2}[/\-\.]\d{2}[/\-\.]\d{4})', 'due_date'),
                (r'(?:amount|valor)[\s:]+R?\$?\s*([\d\.,]+)', 'amount'),
                (r'(?:document\s*number|n[uú]mero|n[fº])[\s:]+(\d+)', 'doc_number'),
                (r'(?:type|tipo)[\s:]+(\w+)', 'doc_type'),
            ]
            
            for pattern, field in patterns:
                match = re.search(pattern, line, re.IGNORECASE)
                if match and field not in result:
                    result[field] = match.group(1).strip()
        
        return result if result else None
    
    def unload_model(self):
        """Unload model from memory to free RAM."""
        if self._model is not None:
            del self._model
            self._model = None
        if self._processor is not None:
            del self._processor
            self._processor = None
        self._loaded = False
        
        # Force garbage collection
        import gc
        gc.collect()
        
        logger.info("SmolDocling model unloaded")


# Global instance
_voter_instance = None

def get_smol_docling_voter() -> SmolDoclingVoter:
    """Get or create SmolDocling voter instance."""
    global _voter_instance
    if _voter_instance is None:
        _voter_instance = SmolDoclingVoter()
    return _voter_instance


if __name__ == '__main__':
    print("SmolDocling VLM Voter")
    print("=" * 50)
    print(f"Model: {SmolDoclingVoter.MODEL_ID}")
    print(f"Parameters: ~256M (CPU-friendly)")
    print(f"RAM Required: ~2-3GB")
    print()
    
    voter = SmolDoclingVoter()
    print(f"Dependencies available: {voter.is_available()}")
    
    if voter.is_available():
        print("\nTo test:")
        print("  result = voter.extract_from_pdf('document.pdf')")
        print("  text = voter.extract_text_from_image('page.png')")
