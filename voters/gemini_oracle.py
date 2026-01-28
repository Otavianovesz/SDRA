"""
Gemini Oracle - Central Intelligence Hub for SDRA
==================================================

This module implements the "Oracle" tier of the extraction hierarchy.
Gemini acts as the FINAL ARBITER for all document processing, providing:

1. EXTRACTION - Semantic understanding of financial documents
2. ARBITRATION - Resolving conflicts between OCR/regex results  
3. SCHEDULING - Detecting scheduled vs confirmed payments
4. VALIDATION - Cross-checking extracted data for consistency

Integration Points:
- Always called after local extraction (XML/OCR/regex)
- Compares and validates local results
- Provides confidence-weighted final decision

Cost Optimization:
- Uses gemini-1.5-flash by default (cheaper, faster)
- Falls back to gemini-1.5-pro for complex cases
- Caches results by file hash to avoid re-processing

Part of Project Cyborg - SDRA Autonomous Treasury Agent
"""

import json
import logging
import time
import re
import hashlib
from pathlib import Path
from typing import Optional, Dict, Any, Union, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime

# Lazy imports - using new google.genai library
genai_client = None
_genai_configured = False

logger = logging.getLogger('srda.gemini.oracle')


def _ensure_genai():
    """Lazy import and configure google.genai (new library)."""
    global genai_client, _genai_configured
    
    if genai_client is not None and _genai_configured:
        return
    
    try:
        from google import genai
        import config
        
        if config.GEMINI_API_KEY:
            genai_client = genai.Client(api_key=config.GEMINI_API_KEY)
            _genai_configured = True
            logger.info("Gemini Oracle configured successfully (using google.genai)")
        else:
            logger.warning("GEMINI_API_KEY not set - Gemini Oracle disabled. Add key to .env file.")
            _genai_configured = False
            return  # Don't raise, just disable Gemini
    except ImportError as e:
        logger.warning(f"google.genai not available: {e}. Trying fallback...")
        # Fallback to old library if new one not available
        try:
            import google.generativeai as old_genai
            import config
            
            if config.GEMINI_API_KEY:
                old_genai.configure(api_key=config.GEMINI_API_KEY)
                genai_client = old_genai
                _genai_configured = True
                logger.info("Gemini Oracle configured (using google.generativeai fallback)")
            else:
                logger.warning("GEMINI_API_KEY not set - Gemini Oracle disabled.")
                _genai_configured = False
        except ImportError as e2:
            logger.warning(f"Neither google.genai nor google.generativeai available: {e2}")
            _genai_configured = False


# Import config
try:
    import config
except ImportError:
    class config:
        GEMINI_API_KEY = ""
        GEMINI_MODEL_FLASH = "gemini-1.5-flash"
        GEMINI_MODEL_PRO = "gemini-1.5-pro"


# =============================================================================
# ADVANCED PROMPT ENGINEERING
# =============================================================================

# Master extraction prompt with JSON schema
EXTRACTION_PROMPT = """Você é o GEMINI ORACLE, motor de extração financeira de precisão máxima.
Análise este documento financeiro brasileiro e extraia TODOS os dados estruturados.

## SCHEMA JSON OBRIGATÓRIO
Retorne EXATAMENTE este formato, sem markdown:

{
  "document_type": "NFE|NFSE|BOLETO|COMPROVANTE|DUPLICATA|RECIBO|CONTRATO|UNKNOWN",
  "payment_status": "AGENDADO|CONFIRMADO|PENDENTE|CANCELADO|UNKNOWN",
  
  "supplier": {
    "name": "string (razão social ou nome fantasia)",
    "cnpj": "string (14 dígitos, só números)",
    "city": "string (opcional)",
    "uf": "string (2 letras, opcional)"
  },
  
  "values": {
    "total": number (valor total em decimal, ex: 1234.56),
    "discount": number (desconto, opcional),
    "fees": number (juros/multa, opcional),
    "final": number (valor final pago, se diferente do total)
  },
  
  "dates": {
    "emission": "YYYY-MM-DD (data de emissão)",
    "due": "YYYY-MM-DD (vencimento)",
    "payment": "YYYY-MM-DD (data do pagamento, se houver)",
    "scheduled": "YYYY-MM-DD (data agendada, se for agendamento)"
  },
  
  "identifiers": {
    "document_number": "string (número do documento)",
    "barcode": "string (linha digitável, 47-48 dígitos, opcional)",
    "nfe_key": "string (chave NFe 44 dígitos, opcional)",
    "sisbb_code": "string (código SISBB do BB, opcional)"
  },
  
  "bank_info": {
    "bank_name": "string (nome do banco)",
    "agency": "string (agência)",
    "account": "string (conta)",
    "payer": "string (nome do pagador)"
  },
  
  "confidence": {
    "overall": number (0.0 a 1.0),
    "reasoning": "string (breve explicação da confiança)"
  },
  
  "warnings": ["array de alertas importantes (agendamento, divergência, etc)"]
}

## REGRAS CRÍTICAS

### Normalização de Valores
- Valores SEMPRE em decimal com PONTO: 1234.56 (nunca 1.234,56)
- Converta R$ 1.234,56 → 1234.56
- Use null para campos ausentes

### Normalização de Datas
- SEMPRE formato ISO: YYYY-MM-DD
- Converta 27/01/2025 → 2025-01-27
- Se só tiver mês/ano, use primeiro dia

### Detecção de AGENDAMENTO vs PAGAMENTO
CRÍTICO: Diferencie pagamentos CONFIRMADOS de AGENDAMENTOS!

Sinais de AGENDAMENTO (payment_status = "AGENDADO"):
- Palavras: "AGENDAMENTO", "AGENDADO", "PREVISÃO DE DÉBITO", "DEBITAR EM"
- Data futura no corpo
- Ausência de código de autenticação/comprovante
- Termos: "SOLICITAÇÃO", "AUTORIZAÇÃO PENDENTE"

Sinais de PAGAMENTO CONFIRMADO (payment_status = "CONFIRMADO"):
- Palavras: "COMPROVANTE", "AUTENTICAÇÃO", "PAGO", "QUITADO", "LIQUIDADO"
- Código SISBB/autenticação presente
- Hora de processamento
- "DÉBITO EFETUADO", "PAGAMENTO REALIZADO"

### Extração de SISBB (Banco do Brasil)
Procure padrões como:
- SISBB: X.XXX.XXX.XXX
- AUTENTICAÇÃO: XXXXXXXX
- PAGAMENTO CONFIRMADO: SISBB=X.XXX.XXX.XXX

### Fornecedor
- Priorize CNPJ do CEDENTE/BENEFICIÁRIO (quem recebe)
- Nome: prefira razão social completa

### Código de Barras
- Extraia a linha digitável (47-48 dígitos)
- Remova espaços e pontos, apenas números
- Valide: boleto deve começar com código do banco (001=BB, 237=Bradesco, etc)

## IMPORTANTE
- Retorne APENAS o JSON puro, sem ```json ou markdown
- Se não encontrar um campo, use null
- Adicione warnings para situações suspeitas
"""


# Arbitration prompt for comparing OCR vs Gemini
ARBITRATION_PROMPT = """Você é o ÁRBITRO FINAL entre extração local (OCR/regex) e análise visual.

## DADOS DO OCR LOCAL
{ocr_data}

## SUA ANÁLISE (VISUAL)
{gemini_data}

## TAREFA
Compare os dois conjuntos de dados e determine o valor CORRETO para cada campo.
Prioridades:
1. Para valores numéricos: prefira o que tem formato válido
2. Para datas: valide formato ISO (YYYY-MM-DD)
3. Para CNPJ: deve ter 14 dígitos
4. Para fornecedor: prefira nome mais completo
5. Para código de barras: valide comprimento (47-48 dígitos)

## OUTPUT JSON
{{
  "final_values": {{
    "amount": number,
    "due_date": "YYYY-MM-DD",
    "emission_date": "YYYY-MM-DD",
    "supplier_name": "string",
    "supplier_doc": "string (14 dígitos)",
    "barcode": "string",
    "document_number": "string",
    "document_type": "string",
    "payment_status": "AGENDADO|CONFIRMADO|PENDENTE"
  }},
  "source_preference": {{
    "amount": "ocr|gemini|both_agree",
    "due_date": "ocr|gemini|both_agree",
    "supplier_name": "ocr|gemini|both_agree"
  }},
  "confidence": number (0.0 a 1.0),
  "arbitration_notes": "explicação das escolhas"
}}
"""


# Scheduling detection prompt
SCHEDULING_PROMPT = """Analise este documento para determinar se é um AGENDAMENTO ou PAGAMENTO CONFIRMADO.

## TEXTO DO DOCUMENTO
{document_text}

## INDICADORES

### AGENDAMENTO (scheduled)
- "AGENDAMENTO DE PAGAMENTO"
- "PREVISÃO DE DÉBITO"
- "DÉBITO AGENDADO PARA"
- "PAGAMENTO AGENDADO"
- Data futura sem autenticação
- "SOLICITAÇÃO DE TRANSFERÊNCIA"

### PAGAMENTO CONFIRMADO (confirmed)
- "COMPROVANTE DE PAGAMENTO"
- "PAGAMENTO EFETUADO"
- "DÉBITO REALIZADO"
- Código de autenticação/SISBB presente
- "QUITAÇÃO", "LIQUIDADO"
- Hora de processamento (ex: 14:35:22)

## OUTPUT JSON
{{
  "status": "AGENDADO|CONFIRMADO|PENDENTE",
  "confidence": number (0.0 a 1.0),
  "scheduled_date": "YYYY-MM-DD (se agendamento)",
  "payment_date": "YYYY-MM-DD (se confirmado)",
  "authentication_code": "string (se confirmado)",
  "warning_message": "string (alerta para o usuário)",
  "reasoning": "explicação da análise"
}}
"""


@dataclass
class OracleDecision:
    """Result from Oracle arbitration."""
    success: bool
    final_data: Dict[str, Any]
    confidence: float
    source_breakdown: Dict[str, str]  # field -> source (ocr/gemini/both)
    payment_status: str  # AGENDADO, CONFIRMADO, PENDENTE
    warnings: List[str]
    tokens_used: int
    processing_time_ms: int
    model_used: str
    raw_response: Optional[str] = None
    error: Optional[str] = None


@dataclass
class OracleStats:
    """Oracle usage statistics."""
    total_calls: int = 0
    extractions: int = 0
    arbitrations: int = 0
    scheduling_checks: int = 0
    total_tokens: int = 0
    cache_hits: int = 0
    ocr_preferred: int = 0
    gemini_preferred: int = 0
    both_agreed: int = 0


class GeminiOracle:
    """
    Central Intelligence Hub for document extraction and arbitration.
    
    This is the "brain" of SDRA - it uses Gemini to:
    1. Extract data from documents (visual understanding)
    2. Arbitrate between OCR results and its own analysis
    3. Detect scheduled vs confirmed payments
    4. Provide confidence-weighted final decisions
    """
    
    def __init__(
        self,
        default_model: str = None,
        fallback_model: str = None,
        always_call: bool = True,  # Always invoke Gemini
        cache_enabled: bool = True
    ):
        self.default_model = default_model or config.GEMINI_MODEL_FLASH
        self.fallback_model = fallback_model or config.GEMINI_MODEL_PRO
        self.always_call = always_call
        self.cache_enabled = cache_enabled
        self._stats = OracleStats()
        self._cache: Dict[str, Dict] = {}  # file_hash -> result
    
    @property
    def stats(self) -> OracleStats:
        return self._stats
    
    def is_available(self) -> bool:
        """Check if Gemini is configured."""
        try:
            _ensure_genai()
            return True
        except Exception:
            return False
    
    def _get_file_hash(self, file_path: Path) -> str:
        """Calculate MD5 hash of file for caching."""
        hasher = hashlib.md5()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                hasher.update(chunk)
        return hasher.hexdigest()
    
    def _get_model(self, model_name: str, system_instruction: str = None):
        """Get configured Gemini model (compatible with both old and new API)."""
        _ensure_genai()
        
        # New google.genai API uses client.models instead of GenerativeModel
        # We'll return a wrapper that's compatible with both
        return {
            'model_name': model_name,
            'system_instruction': system_instruction or EXTRACTION_PROMPT
        }
    
    # =========================================================================
    # MAIN API: Extract + Arbitrate
    # =========================================================================
    
    def process_document(
        self,
        file_path: Union[str, Path],
        ocr_data: Optional[Dict] = None,
        force_reprocess: bool = False
    ) -> OracleDecision:
        """
        Main entry point: Extract data and optionally arbitrate with OCR.
        
        Args:
            file_path: Path to PDF or image
            ocr_data: Results from local OCR/regex extraction (optional)
            force_reprocess: Skip cache
            
        Returns:
            OracleDecision with final arbitrated data
        """
        file_path = Path(file_path)
        start_time = time.time()
        
        # Check cache
        if self.cache_enabled and not force_reprocess:
            file_hash = self._get_file_hash(file_path)
            if file_hash in self._cache:
                self._stats.cache_hits += 1
                cached = self._cache[file_hash]
                logger.info(f"Cache hit for {file_path.name}")
                return OracleDecision(
                    success=True,
                    final_data=cached['data'],
                    confidence=cached['confidence'],
                    source_breakdown=cached.get('sources', {}),
                    payment_status=cached.get('payment_status', 'PENDENTE'),
                    warnings=cached.get('warnings', []),
                    tokens_used=0,
                    processing_time_ms=0,
                    model_used='cache'
                )
        
        self._stats.total_calls += 1
        
        # Step 1: Gemini extraction
        gemini_result = self._extract_visual(file_path)
        
        if not gemini_result['success']:
            return OracleDecision(
                success=False,
                final_data=ocr_data or {},
                confidence=0.3 if ocr_data else 0.0,
                source_breakdown={'all': 'ocr_fallback'},
                payment_status='PENDENTE',
                warnings=['Gemini extraction failed, using OCR fallback'],
                tokens_used=gemini_result.get('tokens', 0),
                processing_time_ms=int((time.time() - start_time) * 1000),
                model_used=self.default_model,
                error=gemini_result.get('error')
            )
        
        # Step 2: Arbitrate if OCR data provided
        if ocr_data:
            decision = self._arbitrate(ocr_data, gemini_result['data'])
        else:
            decision = {
                'final': gemini_result['data'],
                'sources': {k: 'gemini' for k in gemini_result['data'].keys()},
                'confidence': gemini_result.get('confidence', 0.8)
            }
        
        # Step 3: Determine payment status
        payment_status = decision['final'].get('payment_status', 'PENDENTE')
        warnings = gemini_result.get('warnings', [])
        
        # Add scheduling warning if applicable
        if payment_status == 'AGENDADO':
            scheduled_date = decision['final'].get('dates', {}).get('scheduled')
            warnings.append(f"⚠️ AGENDAMENTO DETECTADO - Data: {scheduled_date or 'N/A'}")
        
        # Cache result
        if self.cache_enabled:
            file_hash = self._get_file_hash(file_path)
            self._cache[file_hash] = {
                'data': decision['final'],
                'confidence': decision['confidence'],
                'sources': decision['sources'],
                'payment_status': payment_status,
                'warnings': warnings
            }
        
        return OracleDecision(
            success=True,
            final_data=decision['final'],
            confidence=decision['confidence'],
            source_breakdown=decision['sources'],
            payment_status=payment_status,
            warnings=warnings,
            tokens_used=gemini_result.get('tokens', 0),
            processing_time_ms=int((time.time() - start_time) * 1000),
            model_used=gemini_result.get('model', self.default_model),
            raw_response=gemini_result.get('raw_response')
        )
    
    def _extract_visual(self, file_path: Path) -> Dict:
        """Extract data using visual analysis (supports both old and new API)."""
        self._stats.extractions += 1
        
        try:
            _ensure_genai()
            
            model_config = self._get_model(self.default_model, EXTRACTION_PROMPT)
            
            logger.info(f"Processing {file_path.name} with Gemini...")
            
            # Check if using new google.genai or old google.generativeai
            if hasattr(genai_client, 'files'):
                # NEW API: google.genai
                return self._extract_with_new_api(file_path, model_config)
            else:
                # OLD API: google.generativeai (fallback)
                return self._extract_with_old_api(file_path, model_config)
                
        except Exception as e:
            logger.error(f"Gemini extraction error: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _extract_with_new_api(self, file_path: Path, model_config: Dict) -> Dict:
        """Extract using new google.genai API."""
        try:
            # Upload file using new API
            uploaded_file = genai_client.files.upload(file=str(file_path))
            
            # Wait for file to be ready
            import time
            max_wait = 60
            waited = 0
            while uploaded_file.state.name == "PROCESSING" and waited < max_wait:
                time.sleep(2)
                waited += 2
                uploaded_file = genai_client.files.get(name=uploaded_file.name)
            
            if uploaded_file.state.name != "ACTIVE":
                return {
                    'success': False,
                    'error': f"File processing failed: {uploaded_file.state.name}"
                }
            
            # Generate content using new API
            prompt = f"""{model_config['system_instruction']}

Extraia TODOS os dados financeiros deste documento. Retorne APENAS JSON válido."""
            
            response = genai_client.models.generate_content(
                model=model_config['model_name'],
                contents=[
                    prompt,
                    uploaded_file
                ]
            )
            
            # Track tokens
            tokens = 0
            if hasattr(response, 'usage_metadata'):
                tokens = getattr(response.usage_metadata, 'total_token_count', 0)
            self._stats.total_tokens += tokens
            
            # Parse response
            response_text = response.text if hasattr(response, 'text') else str(response.candidates[0].content.parts[0].text)
            result = self._parse_json_response(response_text)
            
            # Cleanup
            try:
                genai_client.files.delete(name=uploaded_file.name)
            except:
                pass
            
            if result['success']:
                normalized = self._normalize_extraction(result['data'])
                return {
                    'success': True,
                    'data': normalized,
                    'confidence': result['data'].get('confidence', {}).get('overall', 0.8),
                    'warnings': result['data'].get('warnings', []),
                    'tokens': tokens,
                    'model': self.default_model,
                    'raw_response': response_text
                }
            else:
                return result
                
        except Exception as e:
            logger.error(f"New API extraction error: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _extract_with_old_api(self, file_path: Path, model_config: Dict) -> Dict:
        """Extract using old google.generativeai API (fallback)."""
        try:
            # Create model using old API
            model = genai_client.GenerativeModel(
                model_name=model_config['model_name'],
                system_instruction=model_config['system_instruction']
            )
            
            # Upload file
            uploaded = genai_client.upload_file(str(file_path))
            
            # Wait for processing
            import time
            max_wait = 60
            waited = 0
            while uploaded.state.name == "PROCESSING" and waited < max_wait:
                time.sleep(2)
                waited += 2
                uploaded = genai_client.get_file(uploaded.name)
            
            if uploaded.state.name != "ACTIVE":
                return {
                    'success': False,
                    'error': f"File processing failed: {uploaded.state.name}"
                }
            
            # Generate content
            response = model.generate_content([
                "Extraia TODOS os dados financeiros deste documento. Retorne APENAS JSON válido.",
                uploaded
            ])
            
            # Track tokens
            tokens = 0
            if hasattr(response, 'usage_metadata'):
                tokens = getattr(response.usage_metadata, 'total_token_count', 0)
            self._stats.total_tokens += tokens
            
            # Parse response
            result = self._parse_json_response(response.text)
            
            # Cleanup
            try:
                genai_client.delete_file(uploaded.name)
            except:
                pass
            
            if result['success']:
                normalized = self._normalize_extraction(result['data'])
                return {
                    'success': True,
                    'data': normalized,
                    'confidence': result['data'].get('confidence', {}).get('overall', 0.8),
                    'warnings': result['data'].get('warnings', []),
                    'tokens': tokens,
                    'model': self.default_model,
                    'raw_response': response.text
                }
            else:
                return result
                
        except Exception as e:
            logger.error(f"Old API extraction error: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _arbitrate(self, ocr_data: Dict, gemini_data: Dict) -> Dict:
        """Arbitrate between OCR and Gemini results."""
        self._stats.arbitrations += 1
        
        final = {}
        sources = {}
        
        # Map Gemini's nested structure to flat
        gemini_flat = self._flatten_gemini_data(gemini_data)
        
        # Fields to compare
        fields = [
            ('amount', 'values.total', self._compare_numeric),
            ('due_date', 'dates.due', self._compare_dates),
            ('emission_date', 'dates.emission', self._compare_dates),
            ('supplier_name', 'supplier.name', self._compare_strings),
            ('supplier_doc', 'supplier.cnpj', self._compare_cnpj),
            ('barcode', 'identifiers.barcode', self._compare_barcode),
            ('document_number', 'identifiers.document_number', self._compare_strings),
            ('document_type', 'document_type', self._compare_strings),
            ('payment_status', 'payment_status', self._compare_strings)
        ]
        
        confidence_sum = 0
        field_count = 0
        
        for field_name, gemini_path, comparator in fields:
            ocr_val = ocr_data.get(field_name)
            gemini_val = gemini_flat.get(field_name)
            
            if ocr_val is None and gemini_val is None:
                continue
            
            winner, source, conf = comparator(ocr_val, gemini_val)
            
            if winner is not None:
                final[field_name] = winner
                sources[field_name] = source
                confidence_sum += conf
                field_count += 1
                
                # Track stats
                if source == 'ocr':
                    self._stats.ocr_preferred += 1
                elif source == 'gemini':
                    self._stats.gemini_preferred += 1
                else:
                    self._stats.both_agreed += 1
        
        # Copy nested structures from Gemini for additional context
        if 'values' in gemini_data:
            final['values'] = gemini_data['values']
        if 'dates' in gemini_data:
            final['dates'] = gemini_data['dates']
        if 'bank_info' in gemini_data:
            final['bank_info'] = gemini_data['bank_info']
        if 'identifiers' in gemini_data:
            final['identifiers'] = gemini_data['identifiers']
        
        avg_confidence = confidence_sum / field_count if field_count > 0 else 0.5
        
        return {
            'final': final,
            'sources': sources,
            'confidence': avg_confidence
        }
    
    def _flatten_gemini_data(self, data: Dict) -> Dict:
        """Flatten nested Gemini response to match OCR structure."""
        flat = {}
        
        # Direct fields
        flat['document_type'] = data.get('document_type')
        flat['payment_status'] = data.get('payment_status')
        
        # Supplier
        if 'supplier' in data:
            flat['supplier_name'] = data['supplier'].get('name')
            flat['supplier_doc'] = data['supplier'].get('cnpj')
        
        # Values
        if 'values' in data:
            flat['amount'] = data['values'].get('total') or data['values'].get('final')
        
        # Dates
        if 'dates' in data:
            flat['due_date'] = data['dates'].get('due')
            flat['emission_date'] = data['dates'].get('emission')
            flat['payment_date'] = data['dates'].get('payment')
        
        # Identifiers
        if 'identifiers' in data:
            flat['barcode'] = data['identifiers'].get('barcode')
            flat['document_number'] = data['identifiers'].get('document_number')
            flat['nfe_key'] = data['identifiers'].get('nfe_key')
            flat['sisbb_code'] = data['identifiers'].get('sisbb_code')
        
        return flat
    
    # =========================================================================
    # COMPARATORS
    # =========================================================================
    
    def _compare_numeric(self, ocr_val, gemini_val) -> Tuple[Any, str, float]:
        """Compare numeric values."""
        ocr_num = self._parse_number(ocr_val)
        gemini_num = self._parse_number(gemini_val)
        
        if ocr_num is None and gemini_num is None:
            return None, 'none', 0.0
        
        if ocr_num is None:
            return gemini_num, 'gemini', 0.9
        if gemini_num is None:
            return ocr_num, 'ocr', 0.7
        
        # Both have values - compare
        if abs(ocr_num - gemini_num) < 0.01:
            return ocr_num, 'both_agree', 1.0
        
        # Prefer Gemini for visual documents (more context)
        return gemini_num, 'gemini', 0.8
    
    def _compare_dates(self, ocr_val, gemini_val) -> Tuple[Any, str, float]:
        """Compare date values."""
        ocr_date = self._parse_date(ocr_val)
        gemini_date = self._parse_date(gemini_val)
        
        if ocr_date is None and gemini_date is None:
            return None, 'none', 0.0
        
        if ocr_date is None:
            return gemini_date, 'gemini', 0.9
        if gemini_date is None:
            return ocr_date, 'ocr', 0.7
        
        if ocr_date == gemini_date:
            return ocr_date, 'both_agree', 1.0
        
        # Prefer Gemini's date understanding
        return gemini_date, 'gemini', 0.85
    
    def _compare_strings(self, ocr_val, gemini_val) -> Tuple[Any, str, float]:
        """Compare string values, prefer more complete."""
        if not ocr_val and not gemini_val:
            return None, 'none', 0.0
        
        if not ocr_val:
            return gemini_val, 'gemini', 0.9
        if not gemini_val:
            return ocr_val, 'ocr', 0.7
        
        # Normalize for comparison
        ocr_clean = str(ocr_val).strip().upper()
        gemini_clean = str(gemini_val).strip().upper()
        
        if ocr_clean == gemini_clean:
            return ocr_val, 'both_agree', 1.0
        
        # Prefer longer/more complete
        if len(gemini_clean) > len(ocr_clean) * 1.2:
            return gemini_val, 'gemini', 0.85
        if len(ocr_clean) > len(gemini_clean) * 1.2:
            return ocr_val, 'ocr', 0.75
        
        # Default to Gemini for semantic understanding
        return gemini_val, 'gemini', 0.8
    
    def _compare_cnpj(self, ocr_val, gemini_val) -> Tuple[Any, str, float]:
        """Compare CNPJ values, validate format."""
        ocr_cnpj = self._parse_cnpj(ocr_val)
        gemini_cnpj = self._parse_cnpj(gemini_val)
        
        if ocr_cnpj is None and gemini_cnpj is None:
            return None, 'none', 0.0
        
        # Validate: must be 14 digits
        ocr_valid = ocr_cnpj and len(ocr_cnpj) == 14
        gemini_valid = gemini_cnpj and len(gemini_cnpj) == 14
        
        if ocr_valid and not gemini_valid:
            return ocr_cnpj, 'ocr', 0.9
        if gemini_valid and not ocr_valid:
            return gemini_cnpj, 'gemini', 0.9
        
        if ocr_cnpj == gemini_cnpj:
            return ocr_cnpj, 'both_agree', 1.0
        
        # Both valid but different - prefer Gemini (visual context)
        return gemini_cnpj, 'gemini', 0.7
    
    def _compare_barcode(self, ocr_val, gemini_val) -> Tuple[Any, str, float]:
        """Compare barcodes, validate format."""
        ocr_bc = self._parse_barcode(ocr_val)
        gemini_bc = self._parse_barcode(gemini_val)
        
        if ocr_bc is None and gemini_bc is None:
            return None, 'none', 0.0
        
        # Validate: 44, 47, or 48 digits
        valid_lengths = {44, 47, 48}
        ocr_valid = ocr_bc and len(ocr_bc) in valid_lengths
        gemini_valid = gemini_bc and len(gemini_bc) in valid_lengths
        
        if ocr_valid and not gemini_valid:
            return ocr_bc, 'ocr', 0.95
        if gemini_valid and not ocr_valid:
            return gemini_bc, 'gemini', 0.9
        
        if ocr_bc == gemini_bc:
            return ocr_bc, 'both_agree', 1.0
        
        # Prefer OCR for barcodes (more reliable for structured data)
        return ocr_bc, 'ocr', 0.85
    
    # =========================================================================
    # PARSING HELPERS
    # =========================================================================
    
    def _parse_number(self, val) -> Optional[float]:
        """Parse numeric value from various formats."""
        if val is None:
            return None
        if isinstance(val, (int, float)):
            return float(val)
        
        try:
            # Remove currency symbols, spaces
            cleaned = str(val).replace('R$', '').replace(' ', '').strip()
            # Handle Brazilian format: 1.234,56 → 1234.56
            if ',' in cleaned and '.' in cleaned:
                cleaned = cleaned.replace('.', '').replace(',', '.')
            elif ',' in cleaned:
                cleaned = cleaned.replace(',', '.')
            return float(cleaned)
        except:
            return None
    
    def _parse_date(self, val) -> Optional[str]:
        """Parse date to ISO format."""
        if not val:
            return None
        
        val = str(val).strip()
        
        # Already ISO format
        if re.match(r'^\d{4}-\d{2}-\d{2}$', val):
            return val
        
        # Brazilian format: DD/MM/YYYY
        match = re.match(r'^(\d{2})/(\d{2})/(\d{4})$', val)
        if match:
            return f"{match.group(3)}-{match.group(2)}-{match.group(1)}"
        
        return None
    
    def _parse_cnpj(self, val) -> Optional[str]:
        """Parse CNPJ, keeping only digits."""
        if not val:
            return None
        return re.sub(r'[^\d]', '', str(val))
    
    def _parse_barcode(self, val) -> Optional[str]:
        """Parse barcode, keeping only digits."""
        if not val:
            return None
        return re.sub(r'[^\d]', '', str(val))
    
    def _parse_json_response(self, text: str) -> Dict:
        """Parse JSON from Gemini response."""
        if not text:
            return {'success': False, 'error': 'Empty response'}
        
        # Clean markdown
        cleaned = text.strip()
        if cleaned.startswith('```'):
            cleaned = re.sub(r'^```\w*\n?', '', cleaned)
            cleaned = re.sub(r'\n?```$', '', cleaned)
        
        # Extract JSON object
        match = re.search(r'\{[\s\S]*\}', cleaned)
        if match:
            cleaned = match.group()
        
        try:
            data = json.loads(cleaned)
            return {'success': True, 'data': data}
        except json.JSONDecodeError as e:
            logger.error(f"JSON parse error: {e}")
            return {'success': False, 'error': str(e)}
    
    def _normalize_extraction(self, data: Dict) -> Dict:
        """Normalize extracted data structure."""
        # Ensure all expected keys exist
        normalized = {
            'document_type': data.get('document_type', 'UNKNOWN'),
            'payment_status': data.get('payment_status', 'PENDENTE'),
            'supplier': data.get('supplier', {}),
            'values': data.get('values', {}),
            'dates': data.get('dates', {}),
            'identifiers': data.get('identifiers', {}),
            'bank_info': data.get('bank_info', {}),
            'confidence': data.get('confidence', {}),
            'warnings': data.get('warnings', [])
        }
        
        # Normalize CNPJ
        if 'supplier' in normalized and 'cnpj' in normalized['supplier']:
            cnpj = normalized['supplier']['cnpj']
            if cnpj:
                normalized['supplier']['cnpj'] = re.sub(r'[^\d]', '', str(cnpj))
        
        # Normalize barcode
        if 'identifiers' in normalized and 'barcode' in normalized['identifiers']:
            barcode = normalized['identifiers']['barcode']
            if barcode:
                normalized['identifiers']['barcode'] = re.sub(r'[^\d]', '', str(barcode))
        
        return normalized
    
    # =========================================================================
    # SCHEDULING ANALYSIS
    # =========================================================================
    
    def analyze_scheduling(self, text: str) -> Dict:
        """
        Analyze text to determine if it's a scheduled or confirmed payment.
        
        Uses Gemini to understand context and detect scheduling indicators.
        """
        self._stats.scheduling_checks += 1
        
        try:
            _ensure_genai()
            
            model = self._get_model(self.default_model, SCHEDULING_PROMPT)
            
            prompt = SCHEDULING_PROMPT.format(document_text=text[:5000])
            response = model.generate_content(prompt)
            
            tokens = 0
            if hasattr(response, 'usage_metadata'):
                tokens = getattr(response.usage_metadata, 'total_token_count', 0)
            self._stats.total_tokens += tokens
            
            result = self._parse_json_response(response.text)
            
            if result['success']:
                return result['data']
            else:
                # Fallback to regex detection
                return self._regex_scheduling_detection(text)
                
        except Exception as e:
            logger.error(f"Scheduling analysis error: {e}")
            return self._regex_scheduling_detection(text)
    
    def _regex_scheduling_detection(self, text: str) -> Dict:
        """Fallback regex-based scheduling detection."""
        text_upper = text.upper()
        
        scheduled_keywords = [
            'AGENDAMENTO', 'AGENDADO', 'PREVISÃO DE DÉBITO',
            'DEBITAR EM', 'SOLICITAÇÃO DE', 'AUTORIZAÇÃO PENDENTE'
        ]
        
        confirmed_keywords = [
            'COMPROVANTE', 'AUTENTICAÇÃO', 'PAGO', 'QUITADO',
            'LIQUIDADO', 'DÉBITO EFETUADO', 'PAGAMENTO REALIZADO'
        ]
        
        scheduled_score = sum(1 for kw in scheduled_keywords if kw in text_upper)
        confirmed_score = sum(1 for kw in confirmed_keywords if kw in text_upper)
        
        # Check for SISBB code (strong confirmation indicator)
        sisbb_match = re.search(r'SISBB[=:\s]*[\dX\.]+', text_upper)
        if sisbb_match:
            confirmed_score += 3
        
        if scheduled_score > confirmed_score:
            return {
                'status': 'AGENDADO',
                'confidence': min(0.6 + scheduled_score * 0.1, 0.9),
                'warning_message': '⚠️ Possível AGENDAMENTO detectado (regex)',
                'reasoning': 'Keywords de agendamento detectados'
            }
        elif confirmed_score > 0:
            return {
                'status': 'CONFIRMADO',
                'confidence': min(0.7 + confirmed_score * 0.1, 0.95),
                'warning_message': None,
                'reasoning': 'Keywords de confirmação detectados'
            }
        else:
            return {
                'status': 'PENDENTE',
                'confidence': 0.5,
                'warning_message': None,
                'reasoning': 'Sem indicadores claros'
            }


# Singleton instance
_oracle: Optional[GeminiOracle] = None


def get_oracle() -> GeminiOracle:
    """Get or create Oracle singleton."""
    global _oracle
    if _oracle is None:
        _oracle = GeminiOracle()
    return _oracle


# =============================================================================
# INTEGRATION HELPER
# =============================================================================

def process_with_oracle(
    file_path: Union[str, Path],
    ocr_data: Optional[Dict] = None
) -> OracleDecision:
    """
    Convenience function to process a document with the Oracle.
    
    This is the main integration point for scanner.py.
    
    Args:
        file_path: Path to document
        ocr_data: OCR/regex extraction results (optional)
        
    Returns:
        OracleDecision with final data
    """
    oracle = get_oracle()
    return oracle.process_document(file_path, ocr_data)


# =============================================================================
# CLI TEST
# =============================================================================

if __name__ == "__main__":
    import sys
    
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 60)
    print("GEMINI ORACLE - Central Intelligence Hub")
    print("=" * 60)
    
    oracle = get_oracle()
    
    if not oracle.is_available():
        print("ERROR: Gemini not configured!")
        print("Set GEMINI_API_KEY in .env or config.py")
        sys.exit(1)
    
    print(f"Default model: {oracle.default_model}")
    print(f"Fallback model: {oracle.fallback_model}")
    print(f"Always call: {oracle.always_call}")
    
    if len(sys.argv) > 1:
        test_file = Path(sys.argv[1])
        print(f"\nProcessing: {test_file}")
        
        # Simulate OCR data
        mock_ocr = {
            'amount': 1500.00,
            'due_date': '2025-01-27',
            'supplier_name': 'EMPRESA TESTE'
        }
        
        decision = oracle.process_document(test_file, mock_ocr)
        
        print(f"\n{'=' * 40}")
        print(f"SUCCESS: {decision.success}")
        print(f"CONFIDENCE: {decision.confidence:.2f}")
        print(f"PAYMENT STATUS: {decision.payment_status}")
        print(f"MODEL: {decision.model_used}")
        print(f"TOKENS: {decision.tokens_used}")
        print(f"TIME: {decision.processing_time_ms}ms")
        
        print(f"\nFINAL DATA:")
        for key, value in decision.final_data.items():
            if value:
                print(f"  {key}: {value}")
        
        print(f"\nSOURCE BREAKDOWN:")
        for field, source in decision.source_breakdown.items():
            print(f"  {field}: {source}")
        
        if decision.warnings:
            print(f"\nWARNINGS:")
            for w in decision.warnings:
                print(f"  ⚠️ {w}")
    else:
        print("\nUsage: python gemini_oracle.py <pdf_file>")
        print("\nTesting scheduling detection...")
        
        test_text = """
        BANCO DO BRASIL S.A.
        AGENDAMENTO DE PAGAMENTO
        
        PREVISÃO DE DÉBITO EM: 28/01/2025
        VALOR: R$ 1.500,00
        FAVORECIDO: AGRO BAGGIO LTDA
        """
        
        result = oracle.analyze_scheduling(test_text)
        print(f"\nScheduling Analysis:")
        print(f"  Status: {result.get('status')}")
        print(f"  Confidence: {result.get('confidence')}")
        print(f"  Warning: {result.get('warning_message')}")
