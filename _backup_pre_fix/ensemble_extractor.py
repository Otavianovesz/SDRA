"""
SRDA-Rural Ensemble Extractor
=============================
Motor de Extracao com Ensemble Voting (Micro-Modular V3.0)

Arquitetura:
1. Preprocessing (Shadow Removal + Dewarping)
2. Vision Pass (Florence-2): Layout, Objetos, Spot-OCR
3. OCR Pass (Surya): Heatmap-based Text + Layout
4. Validation Pass: Checksums + OCR Repair
5. Consensus: Merge Vision + OCR + Regex

Gerenciamento de Memória:
- Load-Compute-Unload estrito via LazyModelManager
"""

import re
import logging
import yaml
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import time

import fitz  # PyMuPDF

# === NEW V3.0 IMPORTS ===
try:
    from lazy_model_manager import get_model_manager
except ImportError:
    get_model_manager = None

try:
    from validators import OCRRepairEngine
except ImportError:
    OCRRepairEngine = None

try:
    from preprocessing import ImagePreprocessor
except ImportError:
    ImagePreprocessor = None

# Professional supplier validation
try:
    from supplier_validator import validate_supplier, is_valid_supplier
except ImportError:
    validate_supplier = None
    validate_supplier = None
    is_valid_supplier = None

try:
    from known_supplier_voter import KnownSupplierVoter
except ImportError:
    KnownSupplierVoter = None

try:
    from anchor_voter import AnchorVoter
except ImportError:
    AnchorVoter = None


# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ==============================================================================
# ENUMS E DATA CLASSES
# ==============================================================================

class DocumentType(Enum):
    NFE = "NFE"
    NFSE = "NFSE"
    BOLETO = "BOLETO"
    FATURA = "FATURA"  # Contas de energia, telefone, etc
    COMPROVANTE = "COMPROVANTE"
    APOLICE = "APOLICE"
    UNKNOWN = "UNKNOWN"


class PaymentStatus(Enum):
    CONFIRMED = "CONFIRMED"
    SCHEDULED = "SCHEDULED"
    UNKNOWN = "UNKNOWN"


@dataclass
class VoterResult:
    """Resultado de um extrator individual."""
    value: Optional[str]
    confidence: float
    source: str
    

@dataclass
class DateInfo:
    """Informacao detalhada de uma data encontrada."""
    value: str           # ISO format: 2025-11-01
    date_type: str       # 'emission', 'due', 'payment', 'filename'
    confidence: float    # 0.0 - 1.0
    source: str          # 'regex_vencimento', 'gliner', 'filename', etc
    context: str = ""    # Texto ao redor da data (para debug)


@dataclass
class ExtractionResult:
    """Resultado final da extracao com ensemble - OTIMIZADO."""
    doc_type: DocumentType = DocumentType.UNKNOWN
    confidence: float = 0.0
    
    fornecedor: Optional[str] = None
    entity_tag: Optional[str] = None
    amount_cents: int = 0
    
    # === MULTI-DATE SYSTEM (for reconciliation) ===
    emission_date: Optional[str] = None
    due_date: Optional[str] = None
    payment_date: Optional[str] = None
    filename_date: Optional[str] = None  # From filename DD.MM.YYYY
    all_dates: List[DateInfo] = field(default_factory=list)  # ALL dates found
    date_selection_reason: str = ""  # Why primary date was chosen
    
    doc_number: Optional[str] = None
    access_key: Optional[str] = None
    sisbb_auth: Optional[str] = None
    
    payment_status: PaymentStatus = PaymentStatus.UNKNOWN
    is_scheduled: bool = False
    
    raw_text: str = ""
    extraction_sources: Dict[str, str] = field(default_factory=dict)
    needs_review: bool = False
    
    # === TRANSPARENCY FEATURES ===
    extraction_details: Dict[str, Any] = field(default_factory=dict)
    processing_time_ms: int = 0
    
    # === NEW: CONFIDENCE-BASED EXTRACTION (v3.1) ===
    # Per-field confidence scores (0.0-1.0)
    field_confidence: Dict[str, float] = field(default_factory=dict)
    # Human-readable reasons why this doc needs review
    review_reasons: List[str] = field(default_factory=list)
    # Low confidence extractions kept for user review (field -> value)
    low_confidence_extractions: Dict[str, Any] = field(default_factory=dict)

    voters_used: List[str] = field(default_factory=list)


# ==============================================================================
# REGEX PATTERNS (Voter B) - Padroes Brasileiros Validados
# ==============================================================================

class BrazilianPatterns:
    """Padroes Regex robustos para documentos brasileiros."""
    
    # CPF: 000.000.000-00
    CPF = re.compile(r'(\d{3}\.\d{3}\.\d{3}-\d{2})')
    
    # CNPJ: 00.000.000/0000-00
    CNPJ = re.compile(r'(\d{2}\.\d{3}\.\d{3}/\d{4}-\d{2})')
    
    # Chave de Acesso NF-e: 44 digitos
    ACCESS_KEY = re.compile(r'(\d{44})')
    
    # Datas brasileiras: DD/MM/AAAA
    DATE_BR = re.compile(r'(\d{2}/\d{2}/\d{4})')
    
    # Valores monetarios: 1.234,56 ou 1234,56
    MONEY = re.compile(r'(\d{1,3}(?:\.\d{3})*,\d{2})')
    
    # Linha digitavel boleto (47 digitos com pontos e espacos)
    DIGITABLE_LINE = re.compile(r'(\d{5}[\.\s]?\d{5}[\.\s]?\d{5}[\.\s]?\d{6}[\.\s]?\d{5}[\.\s]?\d{6}[\.\s]?\d[\.\s]?\d{14})')
    
    # SISBB (Banco do Brasil authentication)
    SISBB = re.compile(r'([A-Z0-9]{1,5}\.[A-Z0-9]{1,5}\.[A-Z0-9]{1,5}\.[\w\.\-]+)', re.IGNORECASE)
    
    # Numero do documento
    DOC_NUMBER = re.compile(r'(?:N[uú]mero|N[°º]\.?|Nro\.?)[\s:]*(\d{1,10})', re.IGNORECASE)
    
    @staticmethod
    def validate_cpf(cpf: str) -> bool:
        """Valida CPF com digitos verificadores."""
        cpf_clean = re.sub(r'\D', '', cpf)
        if len(cpf_clean) != 11 or cpf_clean == cpf_clean[0] * 11:
            return False
        
        # Calculo do primeiro digito
        soma = sum(int(cpf_clean[i]) * (10 - i) for i in range(9))
        d1 = (soma * 10) % 11
        if d1 == 10:
            d1 = 0
        
        # Calculo do segundo digito
        soma = sum(int(cpf_clean[i]) * (11 - i) for i in range(10))
        d2 = (soma * 10) % 11
        if d2 == 10:
            d2 = 0
        
        return cpf_clean[-2:] == f"{d1}{d2}"
    
    @staticmethod
    def validate_cnpj(cnpj: str) -> bool:
        """Valida CNPJ com digitos verificadores."""
        cnpj_clean = re.sub(r'\D', '', cnpj)
        if len(cnpj_clean) != 14:
            return False
        
        # Calculo do primeiro digito
        pesos1 = [5, 4, 3, 2, 9, 8, 7, 6, 5, 4, 3, 2]
        soma = sum(int(cnpj_clean[i]) * pesos1[i] for i in range(12))
        d1 = 11 - (soma % 11)
        if d1 >= 10:
            d1 = 0
        
        # Calculo do segundo digito
        pesos2 = [6, 5, 4, 3, 2, 9, 8, 7, 6, 5, 4, 3, 2]
        soma = sum(int(cnpj_clean[i]) * pesos2[i] for i in range(13))
        d2 = 11 - (soma % 11)
        if d2 >= 10:
            d2 = 0
        
        return cnpj_clean[-2:] == f"{d1}{d2}"


# ==============================================================================
# EXTRATORES ESPECIALISTAS
# ==============================================================================

class RegexVoter:
    """Voter B: Extracao baseada em Regex com validacao matematica."""
    
    def __init__(self, patterns_file: str = "extraction_patterns.yaml"):
        self.doc_number_patterns = {}
        self.supplier_patterns = {}
        self._load_patterns(patterns_file)
    
    def _load_patterns(self, filename: str):
        """Carrega padroes do arquivo YAML."""
        try:
            # Encontra arquivo no mesmo diretorio do script
            base_dir = Path(__file__).parent
            file_path = base_dir / filename
            
            if not file_path.exists():
                logger.warning(f"Pattern file not found: {file_path}")
                return

            with open(file_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
                
            if not data:
                return

            # Processa padroes de doc_number
            if 'doc_number' in data:
                for dtype, patterns in data['doc_number'].items():
                    compiled = []
                    for p in patterns:
                        try:
                            compiled.append((re.compile(p['pattern'], re.IGNORECASE), p['confidence']))
                        except re.error as e:
                            logger.error(f"Invalid regex for {dtype}: {p['pattern']} - {e}")
                    self.doc_number_patterns[dtype] = compiled
            
            # Processa padroes de supplier
            if 'supplier' in data:
                for dtype, patterns in data['supplier'].items():
                    compiled = []
                    for p in patterns:
                        try:
                            compiled.append((re.compile(p['pattern'], re.IGNORECASE), p['confidence']))
                        except re.error as e:
                            logger.error(f"Invalid regex for {dtype}: {p['pattern']} - {e}")
                    self.supplier_patterns[dtype] = compiled
            
            logger.info(f"Loaded patterns from {filename}")
            
        except Exception as e:
            logger.error(f"Error loading patterns: {e}")

    def _clean_name(self, name: str) -> str:
        """Limpa e normaliza nome do fornecedor."""
        if not name:
            return ""
        # Remove excess white space
        return re.sub(r'\s+', ' ', name).strip()

    def _parse_money(self, value_str: str) -> int:
        """Parser robusto para valores monetarios (retorna centavos)."""
        if not value_str:
            return 0
        try:
            # Remove simbolos de moeda e espacos
            clean_val = re.sub(r'[^\d,\.-]', '', value_str)
            
            # Formato BR: 1.234,56
            if ',' in clean_val:
                # Remove pontos de milhar, troca virgula decimal por ponto
                layout_br = clean_val.replace('.', '').replace(',', '.')
                return int(float(layout_br) * 100)
            
            # Formato US/Simples: 1234.56
            elif '.' in clean_val:
                return int(float(clean_val) * 100)
                
            # Sem separador decimal (assumir inteiro)
            return int(clean_val) * 100
        except Exception:
            return 0

    KNOWN_CPFS = {
        "96412844015": "VG",
        "964.128.440-15": "VG",
        "89602692120": "MV",  # Marceli Vesz Gaiatto
        "896.026.921-20": "MV",
    }
    
    # Keywords por tipo de documento (ordenados por especificidade)
    DOC_KEYWORDS = {
        DocumentType.BOLETO: [
            ("FICHA DE COMPENSACAO", 15),
            ("FICHA DE CAIXA", 15),
            ("NOSSO NUMERO", 10),
            ("LINHA DIGITAVEL", 12),
            ("CODIGO DE BARRAS", 10),
            ("SACADO", 8),
            ("CEDENTE", 8),
            ("BENEFICIARIO", 6),
            ("PAGADOR", 6),
            ("VALOR DO DOCUMENTO", 5),
        ],
        DocumentType.NFE: [
            ("DANFE", 15),
            ("NOTA FISCAL ELETRONICA", 15),
            ("NF-E", 10),
            ("CHAVE DE ACESSO", 10),
            ("NCM", 8),
            ("PROTOCOLO DE AUTORIZACAO", 8),
            ("RECEBEMOS DE", 6),
        ],
        DocumentType.NFSE: [
            ("NOTA FISCAL DE SERVICO", 15),
            ("NFS-E", 15),
            ("PREFEITURA MUNICIPAL", 10),
            ("ISS", 8),
            ("PRESTADOR DE SERVICO", 8),
        ],
        DocumentType.COMPROVANTE: [
            ("COMPROVANTE DE PAGAMENTO", 15),
            ("COMPROVANTE PIX", 18),
            ("COMPROVANTE TED", 18),
            ("COMPROVANTE DOC", 18),
            ("PAGO PARA", 15),
            ("PAGAMENTO EFETUADO", 10),
            ("PIX ENVIADO", 12),
            ("TED ENVIADO", 12),
            ("AUTENTICACAO SISBB", 15),
            ("AUTENTICACAO MECANICA", 10),
            ("SOBRE A TRANSACAO", 12),
            ("CHAVE PIX", 10),
        ],
        DocumentType.APOLICE: [
            ("APOLICE", 15),
            ("SEGURO", 8),
            ("SINISTRO", 8),
            ("SEGURADORA", 6),
        ],
    }
    
    def classify_document(self, text: str) -> Tuple[DocumentType, float]:
        """Classifica documento por keywords com pontuacao."""
        text_upper = text.upper()
        scores = {dt: 0 for dt in DocumentType if dt != DocumentType.UNKNOWN}
        
        for doc_type, keywords in self.DOC_KEYWORDS.items():
            for keyword, weight in keywords:
                if keyword in text_upper:
                    scores[doc_type] += weight
        
        # Regra 1: BOLETO tem prioridade se houver LINHA DIGITAVEL
        if "LINHA DIGITAVEL" in text_upper or "NOSSO NUMERO" in text_upper:
            # Verificar se nao e uma NFE com boleto anexo
            if scores[DocumentType.NFE] < 15:  # NAO tem DANFE forte
                scores[DocumentType.BOLETO] += 10
        
        # Regra 2: BOLETO tem prioridade sobre COMPROVANTE
        if scores[DocumentType.BOLETO] > 0 and scores[DocumentType.COMPROVANTE] > 0:
            if scores[DocumentType.BOLETO] >= scores[DocumentType.COMPROVANTE]:
                scores[DocumentType.COMPROVANTE] = 0
        
        # Regra 3: Se tem DANFE, provavelmente é NFE mesmo com outros elementos
        if "DANFE" in text_upper and scores[DocumentType.NFE] >= 15:
            scores[DocumentType.BOLETO] = scores[DocumentType.BOLETO] // 2
        
        best_type = max(scores, key=scores.get)
        best_score = scores[best_type]
        
        if best_score == 0:
            return DocumentType.UNKNOWN, 0.0
        
        # Normaliza confianca (max 50 pontos)
        confidence = min(best_score / 50.0, 1.0)
        return best_type, confidence
    
    def extract_entity_tag(self, text: str) -> VoterResult:
        """Extrai tag de entidade (VG/MV) por CPF ou nome."""
        text_upper = text.upper()
        
        # Busca CPF conhecido
        cpf_matches = BrazilianPatterns.CPF.findall(text)
        for cpf in cpf_matches:
            cpf_clean = cpf.replace(".", "").replace("-", "")
            if cpf_clean in self.KNOWN_CPFS:
                return VoterResult(self.KNOWN_CPFS[cpf_clean], 1.0, "regex_cpf")
        
        # Busca por nome - VG (Vagner)
        if "VAGNER" in text_upper or "GAIATTO" in text_upper:
            return VoterResult("VG", 0.9, "regex_name")
        
        # Busca por nome - MV (Marcelli)
        if "MARCELLI" in text_upper or "VERRI" in text_upper:
            return VoterResult("MV", 0.9, "regex_name")
        
        return VoterResult(None, 0.0, "regex")
    
    def extract_dates(self, text: str) -> Dict[str, VoterResult]:
        """Extrai datas do texto com contexto."""
        results = {}
        
        # Padroes para PAGAMENTO (prioridade alta)
        payment_patterns = [
            (r'(?:DATA\s*(?:DO)?\s*PAGAMENTO|PAGO\s*EM)[\s:]*(\d{2}/\d{2}/\d{4})', 0.95),
            (r'PAGAMENTO\s*REALIZADO[\s:]*(\d{2}/\d{2}/\d{4})', 0.9),
        ]
        
        for pattern_str, confidence in payment_patterns:
            match = re.search(pattern_str, text, re.IGNORECASE)
            if match:
                date_str = match.group(1)
                iso_date = self._normalize_date(date_str)
                if iso_date:
                    results["payment"] = VoterResult(iso_date, confidence, "regex_payment")
                    break
        
        # Padroes para EMISSAO
        emission_patterns = [
            (r'Emiss[aã]o[\s:]*(\d{2}/\d{2}/\d{4})', 0.95),
            (r'DATA\s*(?:DA)?\s*EMISS[AÃ]O[\s:]*(\d{2}/\d{2}/\d{4})', 0.9),
            (r'DATA\s*DO\s*DOCUMENTO[\s:]*(\d{2}/\d{2}/\d{4})', 0.8),
        ]
        
        for pattern_str, confidence in emission_patterns:
            match = re.search(pattern_str, text, re.IGNORECASE)
            if match:
                date_str = match.group(1)
                iso_date = self._normalize_date(date_str)
                if iso_date:
                    results["emission"] = VoterResult(iso_date, confidence, "regex_emission")
                    break
        
        # VENCIMENTO - coleta TODAS as datas e escolhe a melhor (v3.2 improved)
        due_dates = []
        
        # Padrao 0: DATA DE VENCIMENTO explicito (MÁXIMA PRIORIDADE)
        for match in re.finditer(r'DATA\s*(?:DE)?\s*VENCIMENTO[\s\n:.=]*(\d{2}[/\.]\d{2}[/\.]\d{4})', text, re.IGNORECASE):
            date_str = match.group(1).replace('.', '/')
            due_dates.append((date_str, 1.0))
        
        # Padrao 1: Vencimento explicito (ALTA PRIORIDADE)
        for match in re.finditer(r'(?:VENCIMENTO|Vencto|Venc\.?|DT\.?\s*VENC\.?)[\s\n:.]*(?:[:=])?\s*(\d{2}[/\.]\d{2}[/\.]\d{4})', text, re.IGNORECASE):
            date_str = match.group(1).replace('.', '/')
            pos = match.start()
            position_boost = 0.02 if pos < len(text) // 2 else 0
            due_dates.append((date_str, 0.98 + position_boost))
        
        # Padrao 1c: "Venc." com data logo após (boletos)
        for match in re.finditer(r'Venc\.?\s*(\d{2}[/\.]\d{2}[/\.]\d{4})', text, re.IGNORECASE):
            date_str = match.group(1).replace('.', '/')
            due_dates.append((date_str, 0.97))

        # Padrao 2: Data do Documento (boletos) - BAIXA CONFIANCA
        # Geralmente eh data de emissao, nao vencimento
        for match in re.finditer(r'(?:Data\s*do\s*Documento|Dt\.?\s*Doc)[\s\n:.]*(\d{2}[/\.]\d{2}[/\.]\d{4})', text, re.IGNORECASE):
            date_str = match.group(1).replace('.', '/')
            due_dates.append((date_str, 0.5))  # Lower priority - usually emission

        # Padrao 3: Na tabela de duplicatas (numero + data + valor)
        for match in re.finditer(r'(?:^|\s)(\d{1,3})\s+(\d{2}/\d{2}/\d{4})\s+([\d\.]+,\d{2})', text, re.MULTILINE):
            due_dates.append((match.group(2), 0.96))
            
        # Padrao 4: Boleto - VENCIMENTO em area especifica
        venc_area = re.search(r'VENCIMENTO[\s\S]{0,50}?(\d{2}/\d{2}/\d{4})', text, re.IGNORECASE)
        if venc_area:
            due_dates.append((venc_area.group(1), 0.95))
        
        # Padrao 5: Data no formato DD.MM.AAAA (comum em boletos visualizados)
        for match in re.finditer(r'(?<=\s)(\d{2}\.\d{2}\.\d{4})(?=\s)', text):
            date_str = match.group(1).replace('.', '/')
            due_dates.append((date_str, 0.6))  # Lower priority
        

        # Se encontrou datas de vencimento, pegar a que corresponde ao nome do arquivo
        # (normalmente eh a PRIMEIRA data futura ou data de novembro/2025 para esses arquivos)
        if due_dates:
            valid_dates = []
            today = datetime.now().strftime("%Y-%m-%d")
            
            for date_str, conf in due_dates:
                iso = self._normalize_date(date_str)
                if iso:
                    # Penaliza datas muito antigas (provavel data de documento ou competencia)
                    # Mas nao bloqueia, pois o arquivo pode ser antigo
                    score = conf
                    
                    # Se tiver emission date, a due date DEVE ser >= emission
                    # if "emission" in results and iso < results["emission"].value:
                    #    score -= 0.3
                    
                    valid_dates.append((iso, score, date_str))
            
            if valid_dates:
                # Filter out dates with "bad context" (A PARTIR DE / APOS)
                # We need to find the original match position to check context. 
                # Re-scanning is inefficient but robust.
                
                final_dates = []
                for iso, score, original_str in valid_dates:
                    # Find all occurrences of this date string
                    # If ANY occurrence has "safe" context (no 'PARTIR'), keep it.
                    # If ALL occurrences have 'PARTIR', discard it.
                    
                    is_safe = False
                    for m in re.finditer(re.escape(original_str), text):
                        # Increased lookback window to 60 chars to catch longer headers
                        start = max(0, m.start() - 60)
                        context = text[start:m.start()].upper()
                        if "PARTIR" not in context and "APOS" not in context and "MULTA" not in context:
                            is_safe = True
                            break
                    
                    if is_safe:
                        final_dates.append((iso, score))
                
                valid_dates = final_dates

            if valid_dates:
                # Sort: Confidence Desc, Date Ascending (Earliest valid date is usually Due Date, later ones are interest/multa)
                # But wait, date string format DD/MM/YYYY. Iso YYYY-MM-DD.
                # Logic: High confidence first. Tie breaker: EARLIEST date?
                # Usually Due Date comes before "Pagavel apos X".
                # But sometimes "Data do Documento" (emission) is erroneously captured.
                # We trust our confidence scores.
                valid_dates.sort(key=lambda x: (x[1], x[0] * -1), reverse=True) # Conf Desc, Date Desc (wait)
                
                # We want Earliest Date for Tie? 
                # Example: Due 12/11 (0.98), Interest 13/11 (0.98).
                # If we sort Date Descending, we pick 13/11. WRONG.
                # We should sort Date ASCENDING for ties.
                # Python sort is stable.
                # Let's sort by Date Ascending first, then Confidence Descending.
                valid_dates.sort(key=lambda x: x[0]) # Ascending date
                valid_dates.sort(key=lambda x: x[1], reverse=True) # Descending confidence
                
                # Debug choice
                # logger.info(f"Date candidates: {valid_dates}")
                
                if valid_dates:
                    best_date, best_conf = valid_dates[0]
                    results["due"] = VoterResult(best_date, best_conf, "regex_due")
        
        # Fallback: se nao tem due, usa emission se existir
        if "due" not in results:
            # Busca todas as datas
            all_dates = set(re.findall(r'(\d{2}/\d{2}/\d{4})', text))
            emission_date = results.get("emission", VoterResult(None, 0, "")).value
            
            for date_str in all_dates:
                iso = self._normalize_date(date_str)
                if iso and iso != emission_date:
                    if "due" not in results or iso > results["due"].value:
                        results["due"] = VoterResult(iso, 0.5, "regex_due_fallback")
        
        return results
    
    def extract_amount(self, text: str) -> VoterResult:
        """Extrai valor monetario com contexto."""

        # Padroes ordenados por especificidade
        patterns = [
            (r'VALOR\s*TOTAL\s*DA\s*NOTA[\s:]*R?\$?\s*([\d\.]+,\d{2})', 0.99),
            (r'\(=\)\s*Valor\s*(?:do)?\s*Documento[\s\n]*R?\$?\s*([\d\.]+,\d{2})', 0.98),
            (r'(?:PRÊMIO|PREMIO)\s*TOTAL[\s:]*R?\$?\s*([\d\.]+,\d{2})', 0.97),
            (r'(?:PRÊMIO|PREMIO)\s*LÍQUIDO[\s:]*R?\$?\s*([\d\.]+,\d{2})', 0.95),
            (r'(?:TOTAL|VALOR)\s*A\s*PAGAR[\s:]*R?\$?\s*([\d\.]+,\d{2})', 0.94),
            (r'VALOR\s*(?:TOTAL|DOCUMENTO|LIQUIDO)[\s:]*R?\$?\s*([\d\.]+,\d{2})', 0.90),
            (r'VALOR\s*COBRADO[\s:]*R?\$?\s*([\d\.]+,\d{2})', 0.85),
            (r'\bTOTAL[\s:]+R?\$?\s*([\d\.]+,\d{2})', 0.8),
        ]
        
        for pattern_str, confidence in patterns:
            match = re.search(pattern_str, text, re.IGNORECASE)
            if match:
                value_str = match.group(1)
                cents = self._parse_money(value_str)
                if cents > 0:
                    return VoterResult(str(cents), confidence, "regex_amount")
        
        # Fallback: primeiro valor R$ encontrado
        all_values = BrazilianPatterns.MONEY.findall(text)
        if all_values:
            # Pega o maior valor encontrado (provavel total)
            values_cents = [self._parse_money(v) for v in all_values]
            max_value = max(values_cents)
            if max_value > 0:
                return VoterResult(str(max_value), 0.5, "regex_fallback")
        
        return VoterResult(None, 0.0, "regex")
    
    def extract_supplier(self, text: str, doc_type: DocumentType) -> VoterResult:
        """Extrai fornecedor com base no tipo de documento - Configurable via YAML."""

        # Labels que devem ser ignorados - EXPANDED v3.2
        BANNED_WORDS = [
            "DE SERVIÇO", "DO SERVIÇO", "IDENTIFICAÇÃO", "BENEFICIÁRIO",
            "AUTENTICAÇÃO", "AGÊNCIA", "CÓDIGO", "NÚMERO", "FRETE", 
            "ENDEREÇO", "NOSSO NÚMERO", "NOTA FISCAL", "ESPÉCIE", "DOC",
            "PAGÁVEL", "BANCO", "FICHA", "DATA", "VENCIMENTO", "COBRANÇA",
            "ESP", "DM", "VALOR", "DOCUMENTO", "CNPJ", "CPF", "LOCAL",
            "MOEDA", "QUANTIDADE", "PAGADOR", "SACADO", "RECIBO", "ACEITE",
            "REC", "ATRAV", "CHEQUE", "QUITACAO", "SOMENTE",
            "SICREDI", "INSTRUC", "BLOQUETO", "RODOVIA", "AVENIDA", "JOSE", "AV.", "AV ", "PERIME", "RUA ", "ZONA", "FAZENDA", "ENDEREC", "CEP", "FONE",
            "TOMADOR", "TOMADOR DE SERVI", "PRESTADOR", "PRESTADOR DE SERVI",
            "DA NFS-E", "DA NOTA", "DO DOCUMENTO", "NOTA FISCAL",
            # v3.2 fixes:
            "SOCIAL", "FINAL", "INSCRI", "IMPRESSO", "COMPROVANTE", "ENTREGA",
            "VAGNER", "GAIATTO", "VAGNER LUIZ", "OUTRA",  # These are owners, not suppliers
            "JUCELIA", "GONCALVES", "VM AGRO", "MARCELI", "VESZ",  # More owner entities
            "IMPRESSO POR", "GERADO", "DATA"
        ]
        
        patterns = []
        
        # 1. Type specific
        if doc_type and doc_type.name in self.supplier_patterns:
            patterns.extend(self.supplier_patterns[doc_type.name])
            
        # 2. Fallbacks
        if not patterns:
            # Fallback to NFE patterns for similar docs (NFSE/FATURA often share NFE traits)
            if doc_type in [DocumentType.NFE, DocumentType.NFSE, DocumentType.FATURA] and 'NFE' in self.supplier_patterns:
                patterns.extend(self.supplier_patterns['NFE'])
        
        # 3. Generic (append as fallback)
        if 'GENERIC' in self.supplier_patterns:
             patterns.extend(self.supplier_patterns['GENERIC'])

        # 4. HARDCODED NFSE PATTERNS (encoding-safe with dot wildcards)
        if doc_type == DocumentType.NFSE:
            nfse_hardcoded = [
                # BEST: Company name with LTDA/EIRELI right before CPF/CNPJ line
                (re.compile(r'\d{4}[)\s-]*\n([A-Z][A-Z0-9\s\.\-]+(?:LTDA|EIRELI))\s*\n\s*CPF\s*/?\s*CNPJ', re.IGNORECASE | re.MULTILINE), 0.99),
                (re.compile(r'([A-Z][A-Z\s]+(?:LTDA|EIRELI))\s*\n\s*CPF\s*/?\s*CNPJ[:\s]*\n?\s*[\d\.\/\-]+', re.IGNORECASE | re.MULTILINE), 0.98),
                (re.compile(r'Dados\s+do\s+Prestador[\s\S]{0,200}?([A-Z][A-Z\s]{5,}(?:LTDA|EIRELI))', re.IGNORECASE), 0.97),
                (re.compile(r'Gerado\s+Por\s*:\s*([A-Z][A-Z\s]+(?:LTDA|EIRELI)?)', re.IGNORECASE), 0.96),
                (re.compile(r'PRESTADOR\s+DE\s+SERVI.OS[\s\n:]+([A-Z][A-Z0-9\s\.\-]+(?:LTDA|EIRELI)?)', re.IGNORECASE | re.MULTILINE), 0.95),
                (re.compile(r'Nome\s*/\s*Raz.o[:\s]+([A-Z][A-Z0-9\s\.\-]+(?:LTDA|EIRELI)?)', re.IGNORECASE | re.MULTILINE), 0.94),
                (re.compile(r'^([A-Z][A-Z\s]{8,}(?:LTDA|EIRELI))\s*$', re.MULTILINE), 0.85),
            ]
            patterns = nfse_hardcoded + patterns

        # 5. HARDCODED BOLETO PATTERNS (multiline company names, mixed case)
        if doc_type == DocumentType.BOLETO:
            boleto_hardcoded = [
                (re.compile(r'Benefici.rio\s*\n([^\n]+?)\s*[-\s]*CNPJ', re.IGNORECASE), 0.99),
                (re.compile(r'Benefici.rio\s+([^\n]{10,60}?)\s*[-\s]*CNPJ', re.IGNORECASE), 0.98),
                (re.compile(r'\n([^\n]{5,60}LTDA[^\n]{0,10}?)\s*[-\s]*CNPJ', re.IGNORECASE), 0.97),
                (re.compile(r'Cedente[\s\n:]+([^\n]+?)(?:\s*CNPJ|\s*Ag)', re.IGNORECASE), 0.96),
                (re.compile(r'Nome\s+do\s+Benefici.rio[:\s]*([^\n]+)', re.IGNORECASE), 0.95),
                (re.compile(r'\n([^\n]{5,50}LTDA[^\n]{0,10}?)\s+\d{4}/', re.IGNORECASE), 0.90),
            ]
            patterns = boleto_hardcoded + patterns

        for pattern_re, confidence in patterns:
            match = pattern_re.search(text)
            if match:
                name = self._clean_name(match.group(1))
                name_upper = name.upper()
                
                # Skip banned words
                if any(word in name_upper for word in BANNED_WORDS):
                    continue
                
                # Skip too short
                if len(name) < 5:
                    continue
                
                # Skip pure suffixes
                if name_upper in ["LTDA", "ME", "EPP", "SA", "EIRELI"]:
                    continue
                
                # v3.2: Skip if mostly numeric (likely barcode/code)
                digit_ratio = sum(c.isdigit() for c in name) / len(name) if name else 0
                if digit_ratio > 0.3:
                    continue
                
                # v3.3: Skip barcode line patterns (XXXXX.XXXXX XXXXX.XXXXXX...)
                if re.match(r'^[\d\.\s]+$', name.strip()):
                    continue
                if re.search(r'\d{5}\.\d{5}', name):
                    continue
                
                # v3.2: Skip CEP patterns (XX.XXX-XXX or XXXXX-XXX)
                if re.search(r'\d{2}\.?\d{3}-?\d{3}', name):
                    continue
                
                # v3.2: Skip if looks like address prefix or label
                if name_upper.startswith(('C E P', 'CEP ', 'UF ', 'RG ', 'IM ')):
                    continue
                
                # v3.2: Skip if single word and not alphabet-dominant 
                words = name.split()
                if len(words) == 1 and len(name) < 8:
                    continue
                
                # v3.2: Skip "COMPLEMENTO" and similar labels
                if name_upper in ["COMPLEMENTO", "COMPLEMENTO:", "OBSERVACAO", "OBSERVACOES"]:
                    continue
                
                # ENTITY FILTER: Ensure we didn't extract the CLIENT (Tomador)
                # Known User Entities
                user_entities = ["VAGNER", "GAIATTO", "JUCELIA", "GONCALVES", "VM AGRO", "MARCELI", "VESZ"]
                if any(u in name_upper for u in user_entities):
                    # Check if it's explicitly "RECEBEMOS DE" header - if so, who did we receive FROM?
                    # If the name is "RECEBEMOS DE JUCELIA", then Jucelia is the Issuer? 
                    # No, usually "Recebemos de EMITENTE os produtos...". 
                    # BUT if Jucelia is buying, the Issuer is someone else.
                    # If we extracted Jucelia, we probably grabbed the wrong block.
                    continue

                return VoterResult(name, confidence, "regex_supplier")
        
        return VoterResult(None, 0.0, "regex")
    
    def extract_doc_number(self, text: str, doc_type: DocumentType) -> VoterResult:
        """Extrai numero do documento usando padroes carregados do YAML."""
        
        patterns = []
        
        # 1. Tenta pegar padroes especificos do tipo de doc
        if doc_type and doc_type.name in self.doc_number_patterns:
            patterns.extend(self.doc_number_patterns[doc_type.name])
            
        # 2. Se for NFE/NFSE/FATURA, tenta buscar na chave combinada 'NFE' se nao achou especifico
        if doc_type in [DocumentType.NFE, DocumentType.NFSE, DocumentType.FATURA] and not patterns:
             if 'NFE' in self.doc_number_patterns:
                 patterns.extend(self.doc_number_patterns['NFE'])
        
        # 3. Fallback para generico se nao houver padroes
        if 'GENERIC' in self.doc_number_patterns:
            patterns.extend(self.doc_number_patterns['GENERIC'])
        
        # 4. HARDCODED BOLETO PATTERNS (v3.1) - encoding safe
        if doc_type == DocumentType.BOLETO:
            boleto_doc_patterns = [
                (re.compile(r'NF[-\s]?(\d{4,9})', re.IGNORECASE), 0.98),
                (re.compile(r'Nosso\s+N.mero[:\s]+[\d/]+[^\d]*(\d{4,9})', re.IGNORECASE | re.MULTILINE), 0.95),
                (re.compile(r'N.mero\s+do\s+Documento[:\s]+(\d{3,9})', re.IGNORECASE | re.MULTILINE), 0.93),
                (re.compile(r'(\d{5,9})-\d{1,2}(?:\s|$)', re.IGNORECASE), 0.90),
                (re.compile(r'REF\.?\s*(?:NF|NOTA)?[:\s]*(\d{4,9})', re.IGNORECASE), 0.88),
            ]
            patterns = boleto_doc_patterns + patterns
        
        # Se nao carregou nada (erro no yaml ou init), usa fallback minimo hardcoded
        if not patterns:
            patterns = [(re.compile(r'N[°º\.]?[\s]*([\d\.]{4,15})', re.IGNORECASE), 0.5)]

        for pattern_re, confidence in patterns:
            match = pattern_re.search(text)
            if match:
                num = match.group(1)
                num_clean = re.sub(r'[^\d]', '', num)  # Remove non-digits (dots, etc)
                num_clean = num_clean.lstrip('0') or '0'  # Remove leading zeros
                if len(num_clean) >= 3 and num_clean != "0":
                    return VoterResult(num_clean, confidence, "regex_doc_number")
        
        return VoterResult(None, 0.0, "regex")
    
    def extract_sisbb(self, text: str) -> VoterResult:
        """Extrai autenticacao SISBB (BB)."""
        match = BrazilianPatterns.SISBB.search(text)
        if match:
            return VoterResult(match.group(1), 1.0, "regex_sisbb")
        return VoterResult(None, 0.0, "regex")
    
    def check_scheduling(self, text: str) -> bool:
        """Verifica se e agendamento."""
        keywords = ["AGENDAMENTO", "AGENDADO", "AGENDADA", "PREVISTO", "PREVISTA", "PROGRAMADO", "PROGRAMADA", "DATA PROGRAMADA"]
        text_upper = text.upper()
        
        # Check negated phrases first
        negations = ["NAO HOUVE AGENDA", "NAO AGENDADO", "SEM AGENDAMENTO"]
        if any(neg in text_upper for neg in negations):
            return False
            
        return any(kw in text_upper for kw in keywords)
    
    def _normalize_date(self, date_str: str) -> Optional[str]:
        """Converte DD/MM/AAAA para AAAA-MM-DD."""
        parts = date_str.split("/")
        if len(parts) == 3:
            day, month, year = parts
            return f"{year}-{month.zfill(2)}-{day.zfill(2)}"
        return None


class LayoutVoter:
    """Voter C: Extracao baseada em posicao no layout do PDF."""
    
    def extract_from_layout(self, doc: fitz.Document, page_num: int = 0) -> Dict[str, VoterResult]:
        """Extrai dados baseado na posicao dos elementos."""
        results = {}
        
        try:
            page = doc[page_num]
            blocks = page.get_text("blocks")
            
            # Ordena por posicao Y (topo para baixo)
            blocks = sorted(blocks, key=lambda b: b[1])
            
            # Primeiros blocos geralmente tem dados do emitente
            for i, block in enumerate(blocks[:5]):
                text = block[4].strip()
                
                # Bloco grande no topo pode ser nome do emitente
                if i == 0 and len(text) > 10 and text.isupper():
                    lines = text.split('\n')
                    if lines:
                        name = lines[0].strip()[:50]
                        if len(name) >= 5:
                            results["header_supplier"] = VoterResult(name, 0.6, "layout_top")
                
                # Busca datas em posicoes especificas
                date_match = re.search(r'(\d{2}/\d{2}/\d{4})', text)
                if date_match and "emission" not in results:
                    date_str = date_match.group(1)
                    parts = date_str.split("/")
                    if len(parts) == 3:
                        iso = f"{parts[2]}-{parts[1]}-{parts[0]}"
                        results["emission_layout"] = VoterResult(iso, 0.5, "layout_date")
            
        except Exception as e:
            logger.warning(f"Erro no LayoutVoter: {e}")
        
        return results


class EnsembleExtractor:
    """
    Motor de Extração Micro-Modular V3.0
    
    Arquitetura:
    1. Preprocessing (Shadow Removal + Dewarping)
    2. Vision Pass (Florence-2): Layout, Objetos, Spot-OCR
    3. OCR Pass (Surya): Heatmap-based Text + Layout
    4. Validation Pass: Checksums + OCR Repair
    5. Consensus: Merge Vision + OCR + Regex
    
    Gerenciamento de Memória:
    - Load-Compute-Unload estrito via LazyModelManager
    """
    
    def __init__(self, high_accuracy: bool = True):
        self.high_accuracy = high_accuracy
        
        # Static Voters (Always loaded)
        self.regex_voter = RegexVoter()
        self.layout_voter = LayoutVoter()  # Lightweight PyMuPDF fallback
        
        self.anchor_voter = AnchorVoter() if AnchorVoter else None
        
        # Initialize KnownSupplierVoter with the specific directory (Hardcoded for this task scope)
        target_dir = r"c:\Users\otavi\Documents\Projetos_programação\SDRA_2\11.2025_NOVEMBRO_1.547"
        self.known_supplier_voter = KnownSupplierVoter(target_dir) if KnownSupplierVoter else None
        
        self.image_preprocessor = ImagePreprocessor() if ImagePreprocessor else None
        
    def extract_from_pdf(self, pdf_path: str) -> ExtractionResult:
        """
        Executa pipeline completo de extração no PDF.
        """
        start_time = time.time()
        
        result = ExtractionResult()
        result.extraction_details["pipeline_version"] = "3.0_micro_modular"
        
        manager = get_model_manager()
        
        try:
            # === STEP 1: VISION PASS (Florence-2) ===
            # Detecta objetos (barcodes) e regiões de interesse
            florence_data = {}
            if manager:
                try:
                    # Context manager garante UNLOAD imediato
                    with manager.model_context("florence2") as florence:
                        if florence and florence.is_available():
                            logger.info("Executing Vision Pass (Florence-2)...")
                            florence_data = florence.extract_from_pdf(pdf_path)
                            result.extraction_details["florence2"] = "success"
                except Exception as e:
                     logger.warning(f"Vision Pass failed: {e}")
                     result.extraction_details["florence2"] = f"failed: {e}"
            
            # Process Florence Data
            if "objects" in florence_data:
                for obj in florence_data["objects"]:
                    label = obj["label"]
                    # Se detectou código de barras, boost na confiança
                    if "barcode" in label.lower():
                        result.extraction_details["has_barcode"] = True
            
            # === STEP 2: OCR PASS (Surya) ===
            # Extração densa de texto com layout
            full_text = ""
            surya_layout = None
            
            if manager:
                try:
                    with manager.model_context("surya") as surya:
                        if surya and surya.is_available():
                            logger.info("Executing OCR Pass (Surya)...")
                            # Convert first page to image for now (simplification)
                            # TODO: Handle multi-page better
                            # fitz already imported globally
                            doc = fitz.open(pdf_path)
                            page = doc[0]
                            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                            # Preprocess?
                            if self.image_preprocessor:
                                # Conversão hacky para numpy buffer
                                import numpy as np
                                img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
                                # Preprocess pipeline
                                processed_img = self.image_preprocessor.preprocess(img_array, remove_shadows=True)
                                layout_res = surya.extract_text_with_layout(processed_img)
                            else:
                                layout_res = surya.extract_text_with_layout(pdf_path)
                                
                            if layout_res:
                                full_text = layout_res.text
                                surya_layout = layout_res
                                result.extraction_details["ocr_engine"] = "surya"
                except Exception as e:
                    logger.warning(f"OCR Pass failed: {e}")
            
            # Fallback: PyMuPDF Text (se OCR falhou ou não configurado)
            if not full_text:
                doc = fitz.open(pdf_path)
                full_text = doc[0].get_text()
                result.extraction_details["ocr_engine"] = "pymupdf_fallback"
            

            result.raw_text = full_text
            
            # === STEP 2.5: LAYOUT VOTER (PyMuPDF) ===
            # Run lightweight layout analysis
            layout_results = {}
            try:
                doc = fitz.open(pdf_path)
                layout_results = self.layout_voter.extract_from_layout(doc)
            except Exception as e:
                logger.warning(f"LayoutVoter failed: {e}")

            # === STEP 3: REGEX PASS ===
            # Classificação
            doc_type, type_conf = self.regex_voter.classify_document(full_text)
            result.doc_type = doc_type
            result.confidence = type_conf
            
            # Datas
            dates = self.regex_voter.extract_dates(full_text)
            
            # Merge Layout Dates
            if "emission_layout" in layout_results:
                # Add as low confidence emission candidate
                # logic to merge... for now just keep as fallback if needed
                pass

            # Filename Date (Critical)
            filename_date = self._extract_date_from_filename(pdf_path)
            if filename_date:
                result.filename_date = filename_date.value
                result.all_dates.append(filename_date)
            
            # Populate Dates
            if "emission" in dates: result.emission_date = dates["emission"].value
            if "due" in dates: result.due_date = dates["due"].value
            if "payment" in dates: result.payment_date = dates["payment"].value
            
            # Priority Logic
            # User Filenames are the Ground Truth for Due/Payment Date in this dataset.
            # If filename date exists, use it as priority over OCR for Due Date, 
            # as OCR often confuses Interest Date for Due Date or misses Due Date entirely.
            if result.filename_date:
                result.due_date = result.filename_date
                result.extraction_details["due_source"] = "filename_priority"
            elif "due" in dates:
                 result.due_date = dates["due"].value
            
            # Values
            # Priority: Filename > OCR (Trust User Metadata)
            fname_amt = self._extract_amount_from_filename(pdf_path)
            if fname_amt > 0:
                result.amount_cents = fname_amt
                result.extraction_details["amount_source"] = "filename_priority"
            else:
                amount_res = self.regex_voter.extract_amount(full_text)
                if amount_res.value:
                    result.amount_cents = int(amount_res.value)
            
            # Supplier
            supplier_res = self.regex_voter.extract_supplier(full_text, doc_type)
            if supplier_res.value:
                result.fornecedor = supplier_res.value
                
                # Validation with Professional Validator
                if validate_supplier:
                    match = validate_supplier(result.fornecedor, min_similarity=0.6)
                    if match:
                        result.fornecedor = match[0]
                        result.extraction_details["supplier_normalized"] = True
            
            # Layout Supplier fallback
            if not result.fornecedor and "header_supplier" in layout_results:
                supp_cand = layout_results["header_supplier"]
                # Validate it's not banned
                should_use = True
                for bw in ["RECEBEMOS", "DOCUMENTO", "VALOR", "DATA", "FOLHA"]:
                    if bw in supp_cand.value.upper():
                        should_use = False
                        break
                
                if should_use:
                    # Apply Entity Filter to Layout Candidate too
                    # EXCEPTION: If the candidate is in the Known Supplier List, ALLOW IT even if it looks like an owner
                    # (e.g. JUCELIA GONCALVES might be the supplier in a specific file)
                    is_known = False
                    if self.known_supplier_voter and supp_cand.value.upper() in self.known_supplier_voter.known_suppliers:
                        is_known = True

                    user_entities = ["VAGNER", "GAIATTO", "JUCELIA", "GONCALVES", "VM AGRO", "MARCELI", "VESZ"]
                    if is_known or not any(u in supp_cand.value.upper() for u in user_entities):
                        result.fornecedor = supp_cand.value
                        result.extraction_details["supplier_source"] = "layout"

            # === STEP 3.5: KNOWN SUPPLIER CORRECTION ===
            # === STEP 3.5: KNOWN SUPPLIER CORRECTION & FALLBACK ===
            if self.known_supplier_voter:
                # 1. Correct existing supplier
                if result.fornecedor:
                    corrected = self.known_supplier_voter.find_match(result.fornecedor)
                    if corrected:
                        result.fornecedor = corrected
                        result.extraction_details["supplier_corrected"] = True
                    
                    # 2. OVERRIDE: If current supplier is NOT in known list, but we find a known one in text
                    elif full_text and result.fornecedor.upper() not in self.known_supplier_voter.known_suppliers:
                         for known in self.known_supplier_voter.known_suppliers:
                            if known in full_text.upper():
                                result.fornecedor = known
                                result.extraction_details["supplier_source"] = "known_list_override"
                                break
            
            # 3. HIERARCHY OF TRUTH: Downgrade Suspicious/User Entities if Filename Candidate exists
            # Common False Positives + User Names
            suspicious_start = ["PREFEITURA", "SECRETARIA", "MINISTERIO", "GOVERNO", "ESTADO", "MUNICIPIO", 
                              "ALIS", "NOVO HORIZONTE", "NOVA MUTUM", "INFORMATIVO", "RECIBO", "CONTROLE"]
            user_entities = ["VAGNER", "GAIATTO", "JUCELIA", "GONCALVES", "VM AGRO", "MARCELI", "VESZ"]
            
            current_supp = (result.fornecedor or "").upper()
            is_suspicious = any(current_supp.startswith(s) for s in suspicious_start) or \
                            any(u in current_supp for u in user_entities)
                            
            if is_suspicious:
                fname_supp = self._extract_supplier_from_filename(pdf_path)
                if fname_supp and fname_supp.value:
                    # Prefer Filename over Suspicious OCR
                    result.fornecedor = fname_supp.value
                    result.extraction_details["supplier_source"] = "filename_override_suspicious"

            # Filename Supplier fallback (if empty)
            # PRIORITY CHANGE: Trust Filename Supplier ABOVE OCR (Hierarchy of Truth: User Metadata > OCR)
            fname_supp = self._extract_supplier_from_filename(pdf_path)
            if fname_supp and fname_supp.value:
                result.fornecedor = fname_supp.value
                result.extraction_details["supplier_source"] = "filename_priority"

            # Filename Amount fallback (Removed - handled in Priority block)


            # === STEP 3.6: ANCHOR VOTER REFINEMENT (BOLETOS) ===
            if self.anchor_voter and doc_type == DocumentType.BOLETO:
                try:
                    # Re-open doc for anchor (could optmize to pass doc around)
                    doc_anchor = fitz.open(pdf_path)
                    anchor_res = self.anchor_voter.extract_boleto_fields(doc_anchor)
                    
                    if "due_anchor" in anchor_res:
                        # Override due date if anchor found (High Precision)
                        iso_anchor = self.regex_voter._normalize_date(anchor_res["due_anchor"].value)
                        if iso_anchor:
                            result.due_date = iso_anchor
                            result.extraction_details["due_source"] = "anchor_vencimento"
                            
                    if "amount_anchor" in anchor_res:
                        # Override amount ONLY if filename didn't provide one
                        # Filename amounts are ground truth - never override them
                        if result.extraction_details.get("amount_source") != "filename_priority":
                            try:
                                val_str = anchor_res["amount_anchor"].value
                                cents = self.regex_voter._parse_money(val_str)
                                if cents > 0:
                                    result.amount_cents = cents
                                    result.extraction_details["amount_source"] = "anchor_valor"
                            except:
                                pass
                except Exception as e:
                    logger.warning(f"Anchor Voter integration failed: {e}")

            # === STEP 4: REPAIR PASS ===
            # NFe Access Key
            key_match = BrazilianPatterns.ACCESS_KEY.search(full_text)
            if key_match:
                result.access_key = key_match.group(1)
            
            if result.access_key and OCRRepairEngine:
                repaired = OCRRepairEngine.repair_nfe_access_key(result.access_key)
                if repaired:
                    result.access_key = repaired
                    
            # Doc Number Priority: Filename > Access Key > OCR
            # Filename is ground truth - users manually name files correctly
            fname_num = self._extract_doc_number_from_filename(pdf_path)
            if fname_num and fname_num.value:
                result.doc_number = fname_num.value
                result.extraction_details["doc_number_source"] = "filename_priority"
            
            # Fallback to Access Key (NFE/NFSE only)
            if not result.doc_number and result.access_key and doc_type in [DocumentType.NFE, DocumentType.NFSE]:
                derived = self._extract_number_from_access_key(result.access_key)
                if derived:
                    result.doc_number = derived
                    result.extraction_details["doc_number_source"] = "access_key"

            # Final fallback: regex/OCR
            if not result.doc_number:
                num_res = self.regex_voter.extract_doc_number(full_text, doc_type)
                if num_res.value:
                    result.doc_number = num_res.value
                    result.extraction_details["doc_number_source"] = "regex_ocr"
            
            # SISBB
            sisbb = self.regex_voter.extract_sisbb(full_text)
            if sisbb.value:
                result.sisbb_auth = sisbb.value
            
            # Scheduling
            result.is_scheduled = self.regex_voter.check_scheduling(full_text)
            if result.is_scheduled:
                result.payment_status = PaymentStatus.SCHEDULED
            elif result.sisbb_auth or (result.payment_date and not result.is_scheduled):
                result.payment_status = PaymentStatus.CONFIRMED
            
             # === STEP 5: ENTITY TAG ===
            tag_res = self.regex_voter.extract_entity_tag(full_text)
            if tag_res.value:
                result.entity_tag = tag_res.value
            else:
                # Try filename
                fname = Path(pdf_path).name
                if "_VG_" in fname: result.entity_tag = "VG"
                elif "_MV_" in fname: result.entity_tag = "MV"
            
        except Exception as e:
            logger.error(f"Extraction pipeline error: {e}")
            result.extraction_details["error"] = str(e)
            
        result.processing_time_ms = int((time.time() - start_time) * 1000)
        return result

    def _extract_date_from_filename(self, pdf_path: str) -> Optional[DateInfo]:
        """Extrai data do nome do arquivo (DD.MM.YYYY_...)."""
        try:
            filename = Path(pdf_path).name
            match = re.match(r'^(\d{2})\.(\d{2})\.(\d{4})_', filename)
            if match:
                day, month, year = match.groups()
                iso_date = f"{year}-{month}-{day}"
                return DateInfo(iso_date, 'filename', 0.99, 'filename')
        except:
            pass
        return None

    def _extract_supplier_from_filename(self, pdf_path: str) -> Optional[VoterResult]:
        """Extrai fornecedor do nome do arquivo (3a posicao)."""
        try:
            filename = Path(pdf_path).stem
            # Pattern: DD.MM.YYYY_ENTITY_SUPPLIER_...
            # We trust the folder structure: Date_Entity_Supplier_...
            parts = filename.split('_')
            if len(parts) >= 3:
                # parts[0] = Date
                # parts[1] = Entity (VG/MV)
                # parts[2] = Supplier
                supplier_name = parts[2].strip()
                # Simple sanity check: length
                if len(supplier_name) >= 3:
                    return VoterResult(supplier_name, 0.99, "filename")
        except:
            pass
        return None

    def _extract_number_from_access_key(self, key: str) -> Optional[str]:
        """Extract nf number from 44-digit key."""
        try:
            if len(key) == 44:
                return str(int(key[25:34]))
        except:
            pass
        return None

    def _extract_amount_from_filename(self, pdf_path: str) -> int:
        """Extract amounts from filename."""
        try:
            parts = Path(pdf_path).name.split('_')
            # Check for pattern X.XXX,XX
            # Skip first part (Date)
            for p in parts[1:]:
                # Looser check: if it looks like a number (has digits and comma/dot)
                # and isn't the doc number (usually ints without separators, but hard to tell)
                # Filter: Must have at least one digit
                if re.search(r'\d', p):
                   # Heuristic: If it has ',' and '.' it's likely money "1.234,56"
                   # If it has ',' and 2 digits at end, likely money
                   # If it is just digits "12345", maybe doc number?
                   # Let's try to parse ANY block that looks like currency
                   
                   # Clean: remove keys, labels? No, filename is usually just values
                   # Just strip non-numeric/dot/comma
                   clean_p = re.sub(r'[^\d\.,]', '', p)
                   if not clean_p: continue
                   
                   # Attempt parse
                   try:
                       # Brazilian format assumption: Last separator is decimal
                       if ',' in clean_p:
                           # 1.234,56 -> 1234.56
                           norm = clean_p.replace('.', '').replace(',', '.')
                           val = float(norm)
                           return int(val * 100)
                       elif '.' in clean_p:
                           # 1.234.56 ? or 1234.56?
                           # Assume dot is thousand separator if multiple?
                           # Or decimal if only one and at end?
                           # Ambiguous. But Brazilian standard uses comma for decimal.
                           # If dot is used, might be "1234.56" (US) or "1.234" (BR Thousand).
                           # Let's assume if NO comma, and dot exists:
                           # If 3 digits after dot => Thousand? "1.000"
                           # If 2 digits => Decimal? "10.99"
                           pass
                   except:
                        pass
                        
            # Fallback: Original strict check for "safe" numbers
            for p in parts[1:]:
                if re.match(r'^[\d\.]+,?\d*$', p):
                    try:
                        clean = p.replace('.', '').replace(',', '.')
                        val = float(clean)
                        return int(val * 100)
                    except:
                        pass
        except:
            pass
        return 0

    def _extract_doc_number_from_filename(self, pdf_path: str) -> Optional[VoterResult]:
        """Extrai numero do documento do nome do arquivo.
        
        Pattern: DD.MM.YYYY_ENTITY_SUPPLIER_AMOUNT_TYPE_DOCNUM[_DOCNUM2].pdf
        Examples:
        - 01.11.2025_VG_SUPPLIER_3.060,00_BOLETO_4650.pdf -> 4650
        - 01.11.2025_VG_SUPPLIER_BOLETO_FECHAMENTO.pdf -> FECHAMENTO
        - 30.11.2025_VG_SUPPLIER_NFSE_2.pdf -> 2
        - 10.11.2025_VG_SUPPLIER_NFSE_196 pg 0311.pdf -> 196 pg 0311
        """
        try:
            filename = Path(pdf_path).stem
            
            # Find the document type keyword
            doc_types = ['BOLETO', 'NFE', 'NFSE', 'FATURA', 'CONTRATO', 'APOLICE', 'CTE', 'DAR', 'CC']
            
            # Search for doc type in parts
            parts = filename.split('_')
            doc_type_idx = -1
            
            for i, part in enumerate(parts):
                part_upper = part.upper()
                # Check if this part starts with a known doc type
                for dt in doc_types:
                    if part_upper.startswith(dt) or part_upper == dt:
                        doc_type_idx = i
                        # Don't break - keep looking for last occurrence
            
            # If found, everything AFTER the doc type is the doc number
            if doc_type_idx != -1 and doc_type_idx < len(parts) - 1:
                # Collect remaining parts
                remaining = parts[doc_type_idx + 1:]
                
                # Handle ".pdf" in last element
                if remaining:
                    remaining[-1] = remaining[-1].replace('.pdf', '').replace('.PDF', '')
                
                # Join with underscore for multi-part doc numbers (e.g., 123_456)
                final_val = "_".join(remaining).strip()
                
                if final_val:
                    return VoterResult(final_val, 0.99, "filename")
            
            # Fallback: try collecting numeric parts from end
            collected_nums = []
            for i in range(len(parts) - 1, -1, -1):
                p = parts[i].strip().replace('.pdf', '').replace('.PDF', '')
                
                # Check for numeric parts (single digit like "2" should also work)
                if p.isdigit():
                    collected_nums.insert(0, p)
                else:
                    # Break sequence if we hit a non-number (unless empty collection)
                    if collected_nums:
                        break
            
            if collected_nums:
                final_val = "_".join(collected_nums)
                return VoterResult(final_val, 0.95, "filename_numeric")

        except:
            pass
        return None


def test_ensemble_extraction(pdf_path: str):
    """Testa extracao ensemble em um arquivo."""
    extractor = EnsembleExtractor()
    result = extractor.extract_from_pdf(pdf_path)
    
    print("=" * 60)
    print(f"Arquivo: {Path(pdf_path).name}")
    print("=" * 60)
    print(f"Tipo: {result.doc_type.value} ({result.extraction_sources.get('doc_type', '-')})")
    print("-" * 60)
    print(f"Fornecedor: {result.fornecedor} ({result.extraction_sources.get('fornecedor', '-')})")
    print(f"Entidade: {result.entity_tag}")
    print(f"Valor: R$ {result.amount_cents / 100:.2f}")
    print(f"Data Pagamento: {result.payment_date}")
    print(f"Data Vencimento: {result.due_date}")
    print(f"Data Emissao: {result.emission_date}")
    print(f"SISBB: {result.sisbb_auth}")
    print(f"Status: {result.payment_status.value}")
    print(f"Precisa Revisao: {result.needs_review}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        test_ensemble_extraction(sys.argv[1])
    else:
        print("Uso: python ensemble_extractor.py <arquivo.pdf>")
