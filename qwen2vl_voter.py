"""
Qwen2-VL-2B VLM Voter - Optimal for CPU + 8GB RAM.

Based on technical analysis:
- Qwen2-VL-2B-Instruct: 2.2B params, ~1.7GB Q4_K_M quantized
- Dynamic resolution: handles DANFE A4 density
- M-ROPE: understands 2D spatial layout
- Native Portuguese tokenization

Uses llama-cpp-python for CPU inference.
GGUF model required from: https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct-GGUF

Usage:
    from qwen2vl_voter import Qwen2VLVoter
    
    voter = Qwen2VLVoter()
    if voter.is_available():
        result = voter.extract_from_pdf("document.pdf")
"""

import os
import re
import json
import logging
from typing import Dict, Optional, Any
from pathlib import Path

logger = logging.getLogger(__name__)


class Qwen2VLVoter:
    """
    Qwen2-VL-2B voter for document extraction.
    
    Optimal for:
    - CPU-only inference (8GB RAM)
    - Brazilian fiscal documents (NFe, Boleto, Recibos)
    - Dynamic resolution for small text
    """
    
    # Default model path - user should download GGUF model
    MODEL_PATH = os.environ.get(
        "QWEN2VL_MODEL_PATH",
        "models/qwen2-vl-2b-instruct-q4_k_m.gguf"
    )
    
    # Required files for multimodal
    MMPROJ_PATH = os.environ.get(
        "QWEN2VL_MMPROJ_PATH",
        "models/qwen2-vl-2b-mmproj-f16.gguf"
    )
    
    def __init__(self, model_path: str = None, mmproj_path: str = None):
        """
        Initialize Qwen2-VL voter.
        
        Args:
            model_path: Path to GGUF model (Q4_K_M recommended)
            mmproj_path: Path to vision projector GGUF
        """
        self.model_path = model_path or self.MODEL_PATH
        self.mmproj_path = mmproj_path or self.MMPROJ_PATH
        self._llm = None
        self._available = None
        
    def is_available(self) -> bool:
        """Check if Qwen2-VL model files exist and llama-cpp-python works."""
        if self._available is not None:
            return self._available
            
        # Check llama-cpp-python
        try:
            from llama_cpp import Llama
            logger.info("llama-cpp-python available")
        except ImportError:
            logger.warning("llama-cpp-python not installed. Run: pip install llama-cpp-python")
            self._available = False
            return False
        
        # Check model files
        model_exists = os.path.exists(self.model_path)
        mmproj_exists = os.path.exists(self.mmproj_path)
        
        if not model_exists:
            logger.warning(f"Qwen2-VL model not found: {self.model_path}")
            logger.warning("Download from: https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct-GGUF")
            self._available = False
            return False
            
        if not mmproj_exists:
            logger.warning(f"Vision projector not found: {self.mmproj_path}")
            self._available = False
            return False
            
        self._available = True
        return True
    
    def _load_model(self) -> bool:
        """Lazy load the model when first needed."""
        if self._llm is not None:
            return True
            
        if not self.is_available():
            return False
            
        try:
            from llama_cpp import Llama
            
            logger.info(f"Loading Qwen2-VL model: {self.model_path}")
            
            # Load with multimodal support
            self._llm = Llama(
                model_path=self.model_path,
                n_ctx=4096,  # Context window for complex documents
                n_threads=os.cpu_count() or 4,
                n_gpu_layers=0,  # CPU only
                verbose=False
            )
            
            logger.info("Qwen2-VL model loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load Qwen2-VL: {e}")
            return False
    
    def extract_from_image(self, image_path: str) -> Optional[Dict[str, Any]]:
        """
        Extract data from document image using Qwen2-VL.
        
        Args:
            image_path: Path to document image (PNG/JPEG)
            
        Returns:
            Dictionary with extracted fields or None
        """
        if not self._load_model():
            return None
            
        try:
            import base64
            
            # Read and encode image
            with open(image_path, 'rb') as f:
                image_data = base64.b64encode(f.read()).decode('utf-8')
            
            # Optimized prompt for Brazilian fiscal documents
            prompt = """<|im_start|>system
Você é um especialista em documentos fiscais brasileiros.
<|im_end|>
<|im_start|>user
<image>
Analise este documento fiscal brasileiro e extraia os seguintes campos em JSON:
- fornecedor: Nome da empresa emitente/beneficiário
- data_vencimento: Data de vencimento (DD/MM/YYYY)
- valor: Valor total em R$
- numero_documento: Número da nota/boleto
- tipo: BOLETO, NFE, NFSE, FATURA ou COMPROVANTE

Responda APENAS com JSON válido, sem texto adicional.
<|im_end|>
<|im_start|>assistant
"""
            
            # Generate with grammar for JSON
            response = self._llm.create_chat_completion(
                messages=[
                    {"role": "user", "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_data}"}}
                    ]}
                ],
                max_tokens=256,
                temperature=0.1
            )
            
            # Parse response
            content = response['choices'][0]['message']['content']
            return self._parse_response(content)
            
        except Exception as e:
            logger.error(f"Qwen2-VL extraction error: {e}")
            return None
    
    def extract_from_pdf(self, pdf_path: str, page: int = 0) -> Optional[Dict[str, Any]]:
        """
        Extract data from PDF page.
        
        Args:
            pdf_path: Path to PDF file
            page: Page number (0-indexed)
            
        Returns:
            Dictionary with extracted fields or None
        """
        if not self.is_available():
            return None
            
        try:
            import fitz  # PyMuPDF
            
            # Convert to image at 150 DPI (balance of quality and memory)
            doc = fitz.open(pdf_path)
            if page >= len(doc):
                page = 0
                
            pdf_page = doc[page]
            zoom = 150 / 72
            mat = fitz.Matrix(zoom, zoom)
            pix = pdf_page.get_pixmap(matrix=mat)
            
            # Save temp image
            import tempfile
            import time
            temp_dir = os.path.dirname(pdf_path) or '.'
            temp_path = os.path.join(temp_dir, f'.qwen_temp_{int(time.time()*1000)}.png')
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
            logger.error(f"Qwen2-VL PDF error: {e}")
            return None
    
    def _parse_response(self, response: str) -> Optional[Dict[str, Any]]:
        """Parse model response to structured dict."""
        try:
            # Find JSON in response
            start = response.find('{')
            end = response.rfind('}') + 1
            if start >= 0 and end > start:
                json_str = response[start:end]
                data = json.loads(json_str)
                
                # Normalize field names
                result = {}
                if 'fornecedor' in data:
                    result['supplier_name'] = data['fornecedor']
                if 'data_vencimento' in data:
                    result['due_date'] = data['data_vencimento']
                if 'valor' in data:
                    result['amount'] = str(data['valor'])
                if 'numero_documento' in data:
                    result['doc_number'] = str(data['numero_documento'])
                if 'tipo' in data:
                    result['doc_type'] = data['tipo'].upper()
                    
                return result if result else None
                
        except json.JSONDecodeError:
            logger.warning(f"Could not parse JSON from: {response[:100]}")
            
        return None
    
    def unload_model(self):
        """Unload model to free memory."""
        if self._llm is not None:
            del self._llm
            self._llm = None
            
            import gc
            gc.collect()
            logger.info("Qwen2-VL model unloaded")


# Global instance
_voter_instance = None

def get_qwen2vl_voter() -> Qwen2VLVoter:
    """Get or create Qwen2-VL voter instance."""
    global _voter_instance
    if _voter_instance is None:
        _voter_instance = Qwen2VLVoter()
    return _voter_instance


def download_model_instructions():
    """Print instructions to download the model."""
    print("""
===============================================================================
QWEN2-VL-2B MODEL SETUP
===============================================================================

1. Create 'models' directory:
   mkdir models

2. Download GGUF model (~1.7GB):
   From: https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct-GGUF
   File: qwen2-vl-2b-instruct-q4_k_m.gguf
   
   Or via Hugging Face CLI:
   huggingface-cli download Qwen/Qwen2-VL-2B-Instruct-GGUF \\
       qwen2-vl-2b-instruct-q4_k_m.gguf --local-dir models

3. Download vision projector:
   File: mmproj-qwen2-vl-2b-f16.gguf
   
4. Set model paths:
   export QWEN2VL_MODEL_PATH=models/qwen2-vl-2b-instruct-q4_k_m.gguf
   export QWEN2VL_MMPROJ_PATH=models/mmproj-qwen2-vl-2b-f16.gguf

===============================================================================
""")


if __name__ == '__main__':
    print("Qwen2-VL-2B VLM Voter")
    print("=" * 50)
    print("Model: Qwen2-VL-2B-Instruct (Q4_K_M)")
    print("Parameters: 2.2B (quantized)")
    print("RAM Required: ~1.7-2.5GB")
    print("Optimal for: CPU + 8GB RAM")
    print()
    
    voter = Qwen2VLVoter()
    
    if voter.is_available():
        print("✓ Model available and ready")
    else:
        print("✗ Model not available")
        download_model_instructions()
