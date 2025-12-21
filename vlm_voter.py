"""
Vision Language Model (VLM) Voter for PDF Extraction.

Uses multimodal models (Qwen2-VL, LLaVA) to visually understand PDF pages
and extract structured data. Serves as fallback when regex/GLiNER fail.

Requirements:
- Ollama installed (recommended): https://ollama.ai
- Or: transformers, torch with GPU

Usage:
    from vlm_voter import VLMVoter
    
    voter = VLMVoter()
    if voter.is_available():
        result = voter.extract_from_pdf("document.pdf")
"""

import os
import json
import logging
from typing import Dict, Optional, Any
from pathlib import Path

logger = logging.getLogger(__name__)


class VLMVoter:
    """Vision Language Model voter for PDF extraction."""
    
    def __init__(self, model: str = "qwen2.5vl:7b"):
        """
        Initialize VLM voter.
        
        Args:
            model: Ollama model name (qwen2.5vl:7b, llava:13b, etc.)
        """
        self.model = model
        self._ollama_available = None
        self._client = None
        
    def is_available(self) -> bool:
        """Check if Ollama is available."""
        if self._ollama_available is not None:
            return self._ollama_available
            
        try:
            import ollama
            # Test connection
            ollama.list()
            self._ollama_available = True
            self._client = ollama
            logger.info(f"Ollama available, using model: {self.model}")
        except ImportError:
            logger.warning("Ollama Python package not installed. Run: pip install ollama")
            self._ollama_available = False
        except Exception as e:
            logger.warning(f"Ollama not running: {e}")
            self._ollama_available = False
            
        return self._ollama_available
    
    def extract_from_image(self, image_path: str) -> Optional[Dict[str, Any]]:
        """
        Extract data from a document image.
        
        Args:
            image_path: Path to the image file (PNG, JPEG)
            
        Returns:
            Dictionary with extracted fields or None
        """
        if not self.is_available():
            return None
            
        try:
            prompt = """Analyze this financial document image and extract:
1. supplier_name: Company name of the supplier/beneficiary
2. due_date: Due date in format DD/MM/YYYY
3. amount: Total amount in R$ (Brazilian Real)
4. doc_number: Document/nota number
5. doc_type: Type (BOLETO, NFE, NFSE, FATURA, COMPROVANTE)

Return ONLY a JSON object with these fields. Example:
{"supplier_name": "EMPRESA LTDA", "due_date": "15/11/2025", "amount": "1.234,56", "doc_number": "12345", "doc_type": "BOLETO"}
"""
            
            response = self._client.chat(
                model=self.model,
                messages=[{
                    'role': 'user',
                    'content': prompt,
                    'images': [image_path]
                }]
            )
            
            content = response['message']['content']
            
            # Try to parse JSON from response
            try:
                # Find JSON in response
                start = content.find('{')
                end = content.rfind('}') + 1
                if start >= 0 and end > start:
                    json_str = content[start:end]
                    return json.loads(json_str)
            except json.JSONDecodeError:
                logger.warning(f"Could not parse VLM response as JSON: {content[:100]}")
                
        except Exception as e:
            logger.error(f"VLM extraction error: {e}")
            
        return None
    
    def extract_from_pdf(self, pdf_path: str, page: int = 0) -> Optional[Dict[str, Any]]:
        """
        Extract data from a PDF page.
        
        Converts the PDF page to an image first, then uses VLM.
        
        Args:
            pdf_path: Path to the PDF file
            page: Page number to extract (0-indexed)
            
        Returns:
            Dictionary with extracted fields or None
        """
        if not self.is_available():
            return None
            
        try:
            import fitz  # PyMuPDF
            
            # Convert PDF page to image
            doc = fitz.open(pdf_path)
            if page >= len(doc):
                page = 0
                
            pdf_page = doc[page]
            
            # Render at 150 DPI for good balance of quality and speed
            zoom = 150 / 72
            mat = fitz.Matrix(zoom, zoom)
            pix = pdf_page.get_pixmap(matrix=mat)
            
            # Save to temp file
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
                pix.save(f.name)
                temp_path = f.name
            
            doc.close()
            
            # Extract from image
            result = self.extract_from_image(temp_path)
            
            # Clean up temp file
            try:
                os.unlink(temp_path)
            except:
                pass
                
            return result
            
        except Exception as e:
            logger.error(f"PDF extraction error: {e}")
            return None


def get_vlm_voter() -> VLMVoter:
    """Get or create a VLM voter instance."""
    return VLMVoter()


if __name__ == '__main__':
    # Test the VLM voter
    voter = VLMVoter()
    
    print("VLM Voter Status")
    print("=" * 40)
    print(f"Ollama available: {voter.is_available()}")
    
    if voter.is_available():
        print(f"Model: {voter.model}")
        print("\nTo test extraction:")
        print("  result = voter.extract_from_pdf('document.pdf')")
    else:
        print("\nTo enable VLM extraction:")
        print("1. Install Ollama: https://ollama.ai")
        print("2. Run: ollama pull qwen2.5vl:7b")
        print("3. pip install ollama")
