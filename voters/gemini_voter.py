"""
Gemini AI Voter - Cloud-Based Semantic OCR
===========================================

Uses Google's Gemini 1.5 models for:
- Semantic understanding of financial documents
- Extraction of structured data from complex layouts
- Fallback when local OCR fails or disagrees

Cost optimization:
- Only invoked when local OCR has low confidence or divergence
- Uses flash model by default (cheaper, faster)
- Falls back to pro model for complex cases

Part of Project Cyborg - SRDA Autonomous Treasury Agent
"""

import json
import logging
import time
import re
from pathlib import Path
from typing import Optional, Dict, Any, Union, List
from dataclasses import dataclass, field

# Lazy import
genai = None
_genai_configured = False

logger = logging.getLogger('srda.gemini')


def _ensure_genai():
    """Lazy import and configure google.generativeai."""
    global genai, _genai_configured
    
    if genai is not None and _genai_configured:
        return
    
    try:
        import google.generativeai as _genai
        genai = _genai
        
        # Configure with API key
        import config
        if config.GEMINI_API_KEY:
            genai.configure(api_key=config.GEMINI_API_KEY)
            _genai_configured = True
            logger.info("Gemini AI configured successfully")
        else:
            logger.warning(
                "GEMINI_API_KEY not set. Set environment variable or update config.py"
            )
            
    except ImportError:
        raise ImportError(
            "google-generativeai not installed. "
            "Install with: pip install google-generativeai"
        )


# Import config
try:
    import config
except ImportError:
    class config:
        GEMINI_API_KEY = ""
        GEMINI_MODEL_FLASH = "gemini-1.5-flash"
        GEMINI_MODEL_PRO = "gemini-1.5-pro"


# Strict system instruction for financial document extraction
SYSTEM_INSTRUCTION_PT = """Você é um motor de extração de dados financeiros (OCR Semântico de Alta Precisão).
Sua tarefa é extrair informações estruturadas de documentos financeiros brasileiros.

CAMPOS A EXTRAIR:
1. data_vencimento: Data de vencimento do documento (formato YYYY-MM-DD)
2. data_emissao: Data de emissão do documento (formato YYYY-MM-DD) 
3. valor_total: Valor total em decimal (use ponto como separador, ex: 1234.56)
4. cnpj_fornecedor: CNPJ do fornecedor/emissor (apenas 14 dígitos numéricos)
5. nome_fornecedor: Razão social ou nome fantasia do fornecedor
6. codigo_barras: Linha digitável do boleto (47-48 dígitos) se disponível
7. numero_documento: Número da nota fiscal ou boleto
8. tipo_documento: Tipo (NFE, NFSE, BOLETO, COMPROVANTE, DUPLICATA)

REGRAS OBRIGATÓRIAS:
1. Retorne APENAS um objeto JSON válido
2. NÃO use markdown, comentários ou explicações
3. Use null para campos não encontrados
4. Datas sempre em formato ISO: YYYY-MM-DD
5. Valores decimais com PONTO como separador (1234.56, nunca 1.234,56)
6. CNPJ apenas dígitos (14 caracteres)
7. Se houver múltiplos valores, escolha o VALOR TOTAL (não parcela)
8. Para boletos, procure a linha digitável (47-48 dígitos numéricos)

EXEMPLO DE RESPOSTA CORRETA:
{"data_vencimento": "2025-01-27", "data_emissao": "2025-01-20", "valor_total": 1547.89, "cnpj_fornecedor": "12345678000190", "nome_fornecedor": "EMPRESA EXEMPLO LTDA", "codigo_barras": "23793.38128 60000.000003 00000.000400 1 84340000010000", "numero_documento": "12345", "tipo_documento": "BOLETO"}"""


# Few-shot examples for better accuracy
FEW_SHOT_EXAMPLES = [
    {
        "instruction": "Extraia os dados deste boleto bancário",
        "response": '{"data_vencimento": "2025-02-15", "valor_total": 2500.00, "cnpj_fornecedor": "12345678000199", "nome_fornecedor": "AGRO INSUMOS LTDA", "codigo_barras": "23793381286000000000300000004001843400000250000"}'
    },
    {
        "instruction": "Extraia os dados desta nota fiscal",
        "response": '{"data_vencimento": null, "data_emissao": "2025-01-10", "valor_total": 15750.50, "cnpj_fornecedor": "98765432000188", "nome_fornecedor": "COMERCIO DE SEMENTES SA", "codigo_barras": null, "numero_documento": "000123456", "tipo_documento": "NFE"}'
    }
]


@dataclass
class GeminiExtractionResult:
    """Result from Gemini extraction."""
    success: bool
    data: Dict[str, Any]
    confidence: float
    model_used: str
    tokens_used: int = 0
    processing_time_ms: int = 0
    error: Optional[str] = None
    raw_response: Optional[str] = None


@dataclass
class GeminiStats:
    """Statistics for Gemini usage."""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_tokens: int = 0
    total_processing_time_ms: int = 0
    
    def add_result(self, result: GeminiExtractionResult):
        """Update stats with extraction result."""
        self.total_requests += 1
        if result.success:
            self.successful_requests += 1
        else:
            self.failed_requests += 1
        self.total_tokens += result.tokens_used
        self.total_processing_time_ms += result.processing_time_ms
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_requests == 0:
            return 0.0
        return self.successful_requests / self.total_requests
    
    @property
    def avg_processing_time_ms(self) -> float:
        """Average processing time per request."""
        if self.total_requests == 0:
            return 0.0
        return self.total_processing_time_ms / self.total_requests


class GeminiVoter:
    """
    Voter based on Google Gemini 1.5 models.
    
    Implements the "Oracle" tier of the extraction hierarchy.
    Only called when local OCR fails or has low confidence.
    
    Features:
    - Automatic model selection (flash → pro fallback)
    - Structured JSON extraction
    - Token counting for cost tracking
    - Retry with exponential backoff
    - File upload and cleanup
    
    Usage:
        voter = GeminiVoter()
        result = voter.extract("/path/to/document.pdf")
        if result.success:
            print(result.data)
    """
    
    def __init__(
        self,
        default_model: str = None,
        fallback_model: str = None,
        max_retries: int = 3,
        enable_fallback: bool = True
    ):
        self.default_model = default_model or config.GEMINI_MODEL_FLASH
        self.fallback_model = fallback_model or config.GEMINI_MODEL_PRO
        self.max_retries = max_retries
        self.enable_fallback = enable_fallback
        self._stats = GeminiStats()
    
    @property
    def stats(self) -> GeminiStats:
        """Get usage statistics."""
        return self._stats
    
    @property
    def total_tokens_used(self) -> int:
        """Total tokens consumed across all extractions."""
        return self._stats.total_tokens
    
    def is_available(self) -> bool:
        """Check if Gemini is configured and available."""
        try:
            _ensure_genai()
            return _genai_configured
        except Exception:
            return False
    
    def _get_model(self, model_name: str):
        """Get Gemini model instance."""
        _ensure_genai()
        return genai.GenerativeModel(
            model_name=model_name,
            system_instruction=SYSTEM_INSTRUCTION_PT
        )
    
    def extract(
        self,
        file_path: Union[str, Path],
        use_fallback: bool = False
    ) -> GeminiExtractionResult:
        """
        Extract financial data from a PDF or image.
        
        Args:
            file_path: Path to PDF or image file
            use_fallback: Use pro model instead of flash
            
        Returns:
            GeminiExtractionResult with extracted data
        """
        file_path = Path(file_path)
        model_name = self.fallback_model if use_fallback else self.default_model
        
        if not file_path.exists():
            return GeminiExtractionResult(
                success=False,
                data={},
                confidence=0.0,
                model_used=model_name,
                error=f"File not found: {file_path}"
            )
        
        logger.info(f"Gemini extraction: {file_path.name} using {model_name}")
        start_time = time.time()
        
        try:
            _ensure_genai()
            
            # Upload file to Gemini
            logger.debug(f"Uploading file to Gemini: {file_path}")
            uploaded_file = genai.upload_file(str(file_path))
            
            # Wait for processing with timeout
            max_wait = 60  # seconds
            waited = 0
            while uploaded_file.state.name == "PROCESSING" and waited < max_wait:
                time.sleep(2)
                waited += 2
                uploaded_file = genai.get_file(uploaded_file.name)
            
            if uploaded_file.state.name != "ACTIVE":
                raise RuntimeError(
                    f"File processing failed: {uploaded_file.state.name} "
                    f"(waited {waited}s)"
                )
            
            # Run extraction with retry
            model = self._get_model(model_name)
            response_text, tokens = self._extract_with_retry(model, uploaded_file)
            
            # Parse response
            result = self._parse_response(response_text, model_name, tokens)
            result.processing_time_ms = int((time.time() - start_time) * 1000)
            result.raw_response = response_text
            
            # Cleanup uploaded file
            try:
                genai.delete_file(uploaded_file.name)
                logger.debug(f"Deleted uploaded file: {uploaded_file.name}")
            except Exception as e:
                logger.warning(f"Failed to delete uploaded file: {e}")
            
            self._stats.add_result(result)
            return result
            
        except Exception as e:
            logger.error(f"Gemini extraction failed: {e}")
            
            # Try fallback model if using flash and fallback is enabled
            if self.enable_fallback and not use_fallback and model_name == self.default_model:
                logger.info("Retrying with fallback model (pro)...")
                return self.extract(file_path, use_fallback=True)
            
            result = GeminiExtractionResult(
                success=False,
                data={},
                confidence=0.0,
                model_used=model_name,
                tokens_used=0,
                processing_time_ms=int((time.time() - start_time) * 1000),
                error=str(e)
            )
            self._stats.add_result(result)
            return result
    
    def extract_from_text(
        self,
        text_content: str,
        use_fallback: bool = False
    ) -> GeminiExtractionResult:
        """
        Extract financial data from raw text (email body, etc).
        
        Args:
            text_content: Text content to analyze
            use_fallback: Use pro model instead of flash
            
        Returns:
            GeminiExtractionResult with extracted data
        """
        model_name = self.fallback_model if use_fallback else self.default_model
        
        logger.info(f"Gemini text extraction using {model_name}")
        start_time = time.time()
        
        try:
            _ensure_genai()
            
            model = self._get_model(model_name)
            
            prompt = f"""Analise o seguinte texto de e-mail/documento e extraia os dados financeiros:

---
{text_content[:10000]}
---

Retorne APENAS o JSON com os dados extraídos."""
            
            response = model.generate_content(prompt)
            
            tokens = 0
            if hasattr(response, 'usage_metadata'):
                tokens = getattr(response.usage_metadata, 'total_token_count', 0)
            
            result = self._parse_response(response.text, model_name, tokens)
            result.processing_time_ms = int((time.time() - start_time) * 1000)
            
            self._stats.add_result(result)
            return result
            
        except Exception as e:
            logger.error(f"Gemini text extraction failed: {e}")
            result = GeminiExtractionResult(
                success=False,
                data={},
                confidence=0.0,
                model_used=model_name,
                error=str(e)
            )
            self._stats.add_result(result)
            return result
    
    def _extract_with_retry(self, model, uploaded_file) -> tuple:
        """Extract with exponential backoff retry. Returns (text, tokens)."""
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                response = model.generate_content([
                    "Extraia os dados financeiros deste documento. Retorne apenas JSON.",
                    uploaded_file
                ])
                
                # Track token usage
                tokens = 0
                if hasattr(response, 'usage_metadata'):
                    tokens = getattr(response.usage_metadata, 'total_token_count', 0)
                
                return response.text, tokens
                
            except Exception as e:
                last_error = e
                wait_time = (2 ** attempt) + 1
                logger.warning(
                    f"Attempt {attempt + 1}/{self.max_retries} failed, "
                    f"waiting {wait_time}s: {e}"
                )
                time.sleep(wait_time)
        
        raise last_error
    
    def _parse_response(
        self, 
        response_text: str, 
        model_name: str,
        tokens: int
    ) -> GeminiExtractionResult:
        """Parse Gemini response into structured result."""
        
        if not response_text:
            return GeminiExtractionResult(
                success=False,
                data={},
                confidence=0.0,
                model_used=model_name,
                tokens_used=tokens,
                error="Empty response from Gemini"
            )
        
        # Clean response (remove markdown code blocks if present)
        cleaned = response_text.strip()
        
        # Remove markdown code blocks
        if cleaned.startswith('```'):
            # Remove opening ```json or ``` 
            cleaned = re.sub(r'^```\w*\n?', '', cleaned)
            # Remove closing ```
            cleaned = re.sub(r'\n?```$', '', cleaned)
        
        # Try to extract JSON from response if mixed with text
        json_match = re.search(r'\{[^{}]*\}', cleaned, re.DOTALL)
        if json_match:
            cleaned = json_match.group()
        
        try:
            data = json.loads(cleaned)
            
            # Normalize keys to internal format
            normalized = self._normalize_data(data)
            
            # Calculate confidence based on fields extracted
            confidence = self._calculate_confidence(normalized)
            
            logger.info(
                f"Gemini extraction successful: confidence={confidence:.2f}, "
                f"tokens={tokens}"
            )
            
            return GeminiExtractionResult(
                success=True,
                data=normalized,
                confidence=confidence,
                model_used=model_name,
                tokens_used=tokens
            )
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse Gemini response as JSON: {e}")
            logger.debug(f"Raw response: {cleaned[:500]}")
            return GeminiExtractionResult(
                success=False,
                data={},
                confidence=0.0,
                model_used=model_name,
                tokens_used=tokens,
                error=f"JSON parse error: {e}"
            )
    
    def _normalize_data(self, data: Dict) -> Dict:
        """
        Normalize Gemini output to internal field names.
        
        Maps Portuguese keys to English internal keys.
        """
        # Value normalization
        valor = data.get('valor_total')
        if isinstance(valor, str):
            # Remove currency symbols and convert comma to dot
            valor = valor.replace('R$', '').replace('.', '').replace(',', '.').strip()
            try:
                valor = float(valor)
            except ValueError:
                valor = None
        
        # CNPJ normalization (remove formatting)
        cnpj = data.get('cnpj_fornecedor')
        if cnpj:
            cnpj = re.sub(r'[^\d]', '', str(cnpj))
        
        # Barcode normalization (remove spaces)
        barcode = data.get('codigo_barras')
        if barcode:
            barcode = re.sub(r'[^\d]', '', str(barcode))
            # Validate length
            if len(barcode) not in (47, 48, 44):
                barcode = None
        
        return {
            'due_date': data.get('data_vencimento'),
            'emission_date': data.get('data_emissao'),
            'amount': valor,
            'supplier_doc': cnpj,
            'supplier_name': data.get('nome_fornecedor'),
            'barcode': barcode,
            'document_number': data.get('numero_documento'),
            'document_type': data.get('tipo_documento'),
            'source': 'gemini'
        }
    
    def _calculate_confidence(self, data: Dict) -> float:
        """Calculate confidence based on extracted fields."""
        # Required fields for a "complete" extraction
        required_fields = ['due_date', 'amount', 'supplier_doc', 'supplier_name']
        optional_fields = ['emission_date', 'barcode', 'document_number']
        
        # Calculate score
        required_found = sum(1 for f in required_fields if data.get(f))
        optional_found = sum(1 for f in optional_fields if data.get(f))
        
        # Base confidence from required fields (70% weight)
        base_confidence = (required_found / len(required_fields)) * 0.7
        
        # Bonus from optional fields (30% weight)
        optional_confidence = (optional_found / len(optional_fields)) * 0.3
        
        confidence = base_confidence + optional_confidence
        
        # Boost if we have barcode (very reliable indicator)
        if data.get('barcode'):
            confidence = min(confidence + 0.1, 1.0)
        
        return confidence


# Singleton
_voter: Optional[GeminiVoter] = None


def get_gemini_voter() -> GeminiVoter:
    """Get or create Gemini voter singleton."""
    global _voter
    if _voter is None:
        _voter = GeminiVoter()
    return _voter


# =============================================================================
# CLI for testing
# =============================================================================
if __name__ == "__main__":
    import sys
    
    logging.basicConfig(level=logging.INFO)
    
    print("Gemini Voter Test")
    print("=" * 50)
    
    voter = get_gemini_voter()
    
    if not voter.is_available():
        print("ERROR: Gemini not configured!")
        print("Set GEMINI_API_KEY environment variable or update config.py")
        sys.exit(1)
    
    print(f"Default model: {voter.default_model}")
    print(f"Fallback model: {voter.fallback_model}")
    
    # Test with file if provided
    if len(sys.argv) > 1:
        test_file = sys.argv[1]
        print(f"\nExtracting from: {test_file}")
        
        result = voter.extract(test_file)
        
        print(f"\nSuccess: {result.success}")
        print(f"Confidence: {result.confidence:.2f}")
        print(f"Model: {result.model_used}")
        print(f"Tokens: {result.tokens_used}")
        print(f"Time: {result.processing_time_ms}ms")
        
        if result.success:
            print("\nExtracted data:")
            for key, value in result.data.items():
                if value:
                    print(f"  {key}: {value}")
        else:
            print(f"\nError: {result.error}")
    else:
        print("\nUsage: python gemini_voter.py <pdf_file>")
        
        # Test text extraction
        print("\nTesting text extraction...")
        test_text = """
        BOLETO BANCÁRIO
        Cedente: AGRO BAGGIO LTDA
        CNPJ: 12.345.678/0001-90
        Valor: R$ 1.547,89
        Vencimento: 27/01/2025
        Nosso Número: 123456789
        """
        
        result = voter.extract_from_text(test_text)
        print(f"Success: {result.success}")
        if result.success:
            print(f"Extracted: {result.data}")
