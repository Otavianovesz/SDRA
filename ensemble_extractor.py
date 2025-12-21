"""
SRDA-Rural Ensemble Extractor
=============================
Motor de Extracao com Ensemble Voting (Multiplos Extratores com Fallback)

Arquitetura baseada no Relatorio Tecnico:
- Voter A: GLiNER (NER semantico)
- Voter B: Regex (padroes fixos - CPF, CNPJ, valores, datas)
- Voter C: Layout (posicao no PDF)
- Voter D: OCR bruto (fallback)

Logica de Consenso:
1. Regex valido = peso maximo (validacao matematica)
2. Consenso GLiNER + Layout = alta confianca
3. OCR bruto = fallback para campos faltantes
"""

import re
import logging
import yaml  # Added for YAML pattern loading
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

import fitz  # PyMuPDF

# Professional supplier validation (replaces legacy supplier_matcher)
try:
    from supplier_validator import validate_supplier, is_valid_supplier
except ImportError:
    validate_supplier = None
    is_valid_supplier = None

# SmolDocling VLM (optional, slow - disabled by default)
try:
    from smol_docling_voter import get_smol_docling_voter
    SMOL_DOCLING_AVAILABLE = True
except ImportError:
    get_smol_docling_voter = None
    SMOL_DOCLING_AVAILABLE = False

# Configuracao de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# NOTE: BANNED_ENTITIES moved to supplier_validator.py for professional centralization
# Use validate_supplier() or is_valid_supplier() from supplier_validator module instead

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
            ("BANCO DO BRASIL", 5),
            ("CAIXA ECONOMICA", 5),
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
            ("PAGAMENTO EFETUADO", 10),
            ("PIX ENVIADO", 12),
            ("TED ENVIADO", 12),
            ("AUTENTICACAO MECANICA", 10),
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
        
        # VENCIMENTO - coleta TODAS as datas e escolhe a melhor
        due_dates = []
        
        # Padrao 1: Vencimento explicito (ALTA PRIORIDADE)
        for match in re.finditer(r'(?:VENCIMENTO|Vencto|Venc\.?|DT\.?\s*VENC\.?)[\s\n:.]*(?:[:=])?\s*(\d{2}[/\.]\d{2}[/\.]\d{4})', text, re.IGNORECASE):
            date_str = match.group(1).replace('.', '/')
            # Check if this is in the first half of document (header area)
            pos = match.start()
            position_boost = 0.05 if pos < len(text) // 2 else 0
            due_dates.append((date_str, 0.98 + position_boost))
            
        # Padrao 1b: Data Vencimento (boletos)
        for match in re.finditer(r'DATA\s*(?:DE)?\s*VENCIMENTO[\s\n:.=]*(\d{2}[/\.]\d{2}[/\.]\d{4})', text, re.IGNORECASE):
            date_str = match.group(1).replace('.', '/')
            due_dates.append((date_str, 0.99))
        
        # Padrao 2: Data do Documento (boletos)
        for match in re.finditer(r'(?:Data\s*do\s*Documento|Dt\.?\s*Doc)[\s\n:.]*(\d{2}[/\.]\d{2}[/\.]\d{4})', text, re.IGNORECASE):
            date_str = match.group(1).replace('.', '/')
            due_dates.append((date_str, 0.85))
        
        # Padrao 3: Na tabela de duplicatas (numero + data + valor)
        for match in re.finditer(r'(\d{3})\s*[\n\t]+(\d{2}/\d{2}/\d{4})\s*[\n\t]+', text):
            due_dates.append((match.group(2), 0.9))
        
        # Padrao 4: Boleto - VENCIMENTO em area especifica
        venc_area = re.search(r'VENCIMENTO[\s\S]{0,50}?(\d{2}/\d{2}/\d{4})', text, re.IGNORECASE)
        if venc_area:
            due_dates.append((venc_area.group(1), 0.95))
        
        # Padrao 5: Data no formato DD.MM.AAAA (comum em boletos visualizados)
        for match in re.finditer(r'(?<=\s)(\d{2}\.\d{2}\.\d{4})(?=\s)', text):
            date_str = match.group(1).replace('.', '/')
            due_dates.append((date_str, 0.7))
        
        # Se encontrou datas de vencimento, pegar a que corresponde ao nome do arquivo
        # (normalmente eh a PRIMEIRA data futura ou data de novembro/2025 para esses arquivos)
        if due_dates:
            valid_dates = []
            for date_str, conf in due_dates:
                iso = self._normalize_date(date_str)
                if iso:
                    valid_dates.append((iso, conf, date_str))
            
            if valid_dates:
                # Pega a data de MAIOR CONFIANCA (pattern mais especifico)
                valid_dates.sort(key=lambda x: (x[1], x[0]), reverse=True)
                best_date, best_conf, _ = valid_dates[0]
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
            (r'\(=\)\s*Valor\s*(?:do)?\s*Documento[\s\n]*R?\$?\s*([\d\.]+,\d{2})', 0.95),
            (r'VALOR\s*TOTAL\s*DA\s*NOTA[\s:]*R?\$?\s*([\d\.]+,\d{2})', 0.95),
            (r'VALOR\s*(?:TOTAL|DOCUMENTO|LIQUIDO)[\s:]*R?\$?\s*([\d\.]+,\d{2})', 0.9),
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
        # Labels que devem ser ignorados
        BANNED_WORDS = [
            "DE SERVIÇO", "DO SERVIÇO", "IDENTIFICAÇÃO", "BENEFICIÁRIO",
            "AUTENTICAÇÃO", "AGÊNCIA", "CÓDIGO", "NÚMERO", "FRETE", 
            "ENDEREÇO", "NOSSO NÚMERO", "NOTA FISCAL", "ESPÉCIE", "DOC",
            "PAGÁVEL", "BANCO", "FICHA", "DATA", "VENCIMENTO", "COBRANÇA",
            "ESP", "DM", "VALOR", "DOCUMENTO", "CNPJ", "CPF", "LOCAL",
            "MOEDA", "QUANTIDADE", "PAGADOR", "SACADO", "RECIBO", "ACEITE",
            "REC", "ATRAV", "CHEQUE", "QUITACAO", "SOMENTE",
            "SICREDI", "INSTRUC", "BLOQUETO", "RODOVIA", "AVENIDA", "JOSE", "AV.", "AV ", "PERIME", "RUA ", "ZONA", "FAZENDA", "ENDEREC", "CEP", "FONE"
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

        for pattern_re, confidence in patterns:
            match = pattern_re.search(text)
            if match:
                name = self._clean_name(match.group(1))
                # Verifica se capturou um label em vez do nome
                name_upper = name.upper()
                # Skip se contém palavra banida
                if any(word in name_upper for word in BANNED_WORDS):
                    continue
                # Skip se muito curto (provavelmente label)
                if len(name) < 5:
                    continue
                # Skip se for apenas sufixo corporativo
                if name_upper in ["LTDA", "ME", "EPP", "SA", "EIRELI"]:
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
        
        # Se nao carregou nada (erro no yaml ou init), usa fallback minimo hardcoded
        if not patterns:
            patterns = [(re.compile(r'N[°º\.]?[\s]*([\d\.]{4,15})', re.IGNORECASE), 0.5)]

        for pattern_re, confidence in patterns:
            match = pattern_re.search(text)
            if match:
                # Clean the number - remove dots and leading zeros
                num = match.group(1)
                num_clean = re.sub(r'[^\d]', '', num)  # Remove non-digits (dots, etc)
                num_clean = num_clean.lstrip('0') or '0'  # Remove leading zeros
                # Accept numbers with at least 3 digits (after removing zeros)
                if len(num_clean) >= 3:
                     # Check if it looks valid
                     if num_clean == "0": 
                         continue
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
    
    def _parse_money(self, value_str: str) -> int:
        """Converte 1.234,56 para centavos."""
        try:
            clean = value_str.replace(".", "").replace(",", ".")
            return int(float(clean) * 100)
        except:
            return 0
    
    def _clean_name(self, name: str) -> str:
        """Limpa nome de fornecedor."""
        # Remove quebras de linha
        name = re.sub(r'[\n\r]+', ' ', name)
        name = re.sub(r'\s+', ' ', name).strip().upper()
        
        # Corta em palavras de parada
        stop_words = ['CNPJ', 'CPF', 'PAGADOR', 'CEP', 'ENDERECO', 'VENCIMENTO', 
                      'AGENCIA', 'SITE', 'TELEFONE', 'FONE', 'LOCAL', 'DATA']
        for stop in stop_words:
            if stop in name:
                name = name.split(stop)[0].strip()
        
        # Remove sufixos
        suffixes = [" LTDA", " ME", " EPP", " EIRELI", " S.A.", " S/A", " SA"]
        for suffix in suffixes:
            if name.endswith(suffix):
                name = name[:-len(suffix)].strip()
        
        # Remove pontuacao final
        name = re.sub(r'[\.\,\-\:]+$', '', name).strip()
        
        return name[:50]


class GLiNERVoter:
    """Voter A: Extracao semantica via GLiNER."""
    
    def __init__(self):
        self._model = None
        self._loaded = False
    
    @property
    def model(self):
        if self._model is None:
            try:
                from gliner import GLiNER
                logger.info("Carregando GLiNER...")
                self._model = GLiNER.from_pretrained("urchade/gliner_small-v2.1")
                self._loaded = True
                logger.info("GLiNER carregado")
            except Exception as e:
                logger.error(f"Erro ao carregar GLiNER: {e}")
                self._model = None
        return self._model
    
    def extract_entities(self, text: str) -> Dict[str, VoterResult]:
        """Extrai entidades via NER semantico com sliding windows baseado em tokens.
        
        Implementa janelas deslizantes para processar textos longos sem truncamento:
        - Divide texto em chunks de ~300 tokens (seguro para limite de 384)
        - Overlap de 50 tokens para nao perder entidades nas bordas
        - Processa TODO o documento, nao apenas primeiros 5000 chars
        """
        results = {}
        
        if self.model is None:
            return results
        
        try:
            # Configuracao de sliding windows (baseado em tokens, nao caracteres)
            # ~1 token = ~4 caracteres em portugues (mais conservador que ingles)
            chars_per_token = 3.0  # Conservador para portugues com acentos
            max_tokens = 150  # Maximo seguro para eliminar truncacao em OCR (limite = 384)
            overlap_tokens = 50  # Overlap para entidades nas bordas
            
            chunk_size_chars = int(max_tokens * chars_per_token)  # ~1050 chars
            overlap_chars = int(overlap_tokens * chars_per_token)  # ~175 chars
            
            labels = ["fornecedor", "banco", "pessoa", "municipio", "empresa"]
            all_predictions = []
            
            # Processa TODO o texto em sliding windows
            full_text = text  # Nao limita o texto!
            step_size = chunk_size_chars - overlap_chars
            
            for start in range(0, len(full_text), step_size):
                end = start + chunk_size_chars
                chunk = full_text[start:end]
                
                if len(chunk) < 100:  # Skip chunks muito pequenos
                    continue
                
                try:
                    predictions = self.model.predict_entities(chunk, labels, threshold=0.35)
                    # Adiciona offset para rastrear posicao no documento
                    for pred in predictions:
                        pred["_offset"] = start
                    all_predictions.extend(predictions)
                except Exception as chunk_err:
                    logger.debug(f"Erro no chunk {start}: {chunk_err}")
                    continue
            
            # Agregacao inteligente: remove duplicatas e entidades sobrepostas
            entities_by_type = {}
            seen_texts = set()
            
            for pred in all_predictions:
                label = pred["label"]
                text_val = pred["text"].strip()
                score = pred["score"]
                
                # Skip textos muito curtos ou duplicados
                if len(text_val) < 3:
                    continue
                if text_val.lower() in seen_texts:
                    continue
                    
                # Skip entidades que sao claramente labels/campos
                BANNED = ["CNPJ", "CPF", "INSS", "ISSQN", "ISS", "PIS", "COFINS", 
                         "CSLL", "VALOR", "DATA", "VENCIMENTO", "DOCUMENTO"]
                if text_val.upper() in BANNED:
                    continue
                    
                seen_texts.add(text_val.lower())
                
                if label not in entities_by_type:
                    entities_by_type[label] = []
                entities_by_type[label].append((text_val, score))
            
            # Pega a melhor de cada tipo
            for label, entities in entities_by_type.items():
                if not entities:
                    continue
                best = max(entities, key=lambda x: x[1])
                text_val, score = best
                
                # Mapeia para campos
                if label in ["fornecedor", "empresa"]:
                    results["fornecedor"] = VoterResult(text_val.strip(), score, "gliner")
                elif label == "banco":
                    results["banco"] = VoterResult(text_val.strip(), score, "gliner")
                elif label == "municipio":
                    results["municipio"] = VoterResult(text_val.strip(), score, "gliner")
            
        except Exception as e:
            logger.warning(f"Erro no GLiNER: {e}")
        
        return results


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
                    # Limpa o nome
                    lines = text.split('\n')
                    if lines:
                        name = lines[0].strip()[:50]
                        if len(name) >= 5:
                            results["fornecedor_layout"] = VoterResult(name, 0.6, "layout_top")
                
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


class OCRVoter:
    """Voter D: OCR fallback for scanned PDFs using Tesseract."""
    
    # Default path for Tesseract on Windows (winget install)
    TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    
    def __init__(self):
        self._tesseract_available = None
    
    def is_available(self) -> bool:
        """Check if Tesseract is available."""
        if self._tesseract_available is None:
            try:
                import pytesseract
                import os
                # Configure path if exists
                if os.path.exists(self.TESSERACT_PATH):
                    pytesseract.pytesseract.tesseract_cmd = self.TESSERACT_PATH
                # Try to get tesseract version
                pytesseract.get_tesseract_version()
                self._tesseract_available = True
            except:
                self._tesseract_available = False
        return self._tesseract_available
    
    def extract_text(self, doc: fitz.Document, page_num: int = 0) -> Optional[str]:
        """Extract text from a page using OCR."""
        if not self.is_available():
            return None
        
        try:
            import pytesseract
            from PIL import Image
            import io
            
            page = doc[page_num]
            # High-resolution rendering for OCR
            mat = fitz.Matrix(2.0, 2.0)
            pix = page.get_pixmap(matrix=mat)
            
            # Convert to PIL Image
            img_data = pix.tobytes("png")
            img = Image.open(io.BytesIO(img_data))
            
            # OCR with English language (fallback if Portuguese not available)
            text = pytesseract.image_to_string(img, lang='eng', config='--psm 6')
            return text
            
        except Exception as e:
            logger.warning(f"OCR failed: {e}")
            return None


# ==============================================================================
# ENSEMBLE EXTRACTOR
# ==============================================================================


class EnsembleExtractor:
    """
    Motor de Extracao com Ensemble Voting e Tiered Strategy.
    
    ARQUITETURA OTIMIZADA:
    - TIER 1 (Fast): Regex only (<100ms)
    - TIER 2 (Medium): Regex + GLiNER (~2s) - se regex <0.9 confiança
    - TIER 3 (Slow): + SmolDocling VLM (~8s) - DESABILITADO por padrão
    
    Combina multiplos extratores e usa votacao ponderada
    para obter o resultado mais confiavel.
    """
    
    def __init__(
        self, 
        use_gliner: bool = True, 
        use_ocr: bool = True,
        use_vlm: bool = False,  # VLM OFF by default - too slow for production
        lazy_load: bool = True  # Delay loading heavy models until needed
    ):
        """
        Inicializa o extrator com configuração otimizada.
        
        Args:
            use_gliner: Habilita GLiNER NER (TIER 2)
            use_ocr: Habilita OCR fallback para PDFs escaneados
            use_vlm: Habilita SmolDocling VLM (TIER 3) - LENTO, usar só para debug
            lazy_load: Atrasa carregamento do GLiNER até primeiro uso
        """
        self.use_gliner = use_gliner
        self.use_vlm = use_vlm
        self.lazy_load = lazy_load
        
        # TIER 1: Always loaded (fast)
        self.regex_voter = RegexVoter()
        self.layout_voter = LayoutVoter()
        
        # TIER 2: Lazy loaded on first use
        self._gliner_voter = None
        if use_gliner and not lazy_load:
            self._load_gliner()
        
        # OCR: Only for scanned PDFs
        self.ocr_voter = OCRVoter() if use_ocr else None
    
    def _load_gliner(self):
        """Carrega GLiNER sob demanda."""
        if self._gliner_voter is None and self.use_gliner:
            logger.info("Carregando GLiNER...")
            self._gliner_voter = GLiNERVoter()
            logger.info("GLiNER carregado")
    
    @property
    def gliner_voter(self):
        """Acesso lazy ao GLiNER."""
        if self._gliner_voter is None and self.use_gliner:
            self._load_gliner()
        return self._gliner_voter
    
    def _is_readable_text(self, text: str) -> bool:
        """Check if extracted text is readable (not binary garbage)."""
        if len(text) < 50:
            return False
        # Check for minimum ratio of readable characters
        readable = sum(1 for c in text[:200] if c.isalnum() or c.isspace() or c in '.,;:/-')
        return readable / min(200, len(text)) > 0.5
    
    def _extract_date_from_filename(self, pdf_path: str) -> Optional[DateInfo]:
        """
        Extrai data do nome do arquivo (DD.MM.YYYY_...).
        CRITICAL para NFSE que não tem vencimento.
        """
        try:
            filename = Path(pdf_path).name
            # Pattern: DD.MM.YYYY at start of filename
            match = re.match(r'^(\d{2})\.(\d{2})\.(\d{4})_', filename)
            if match:
                day, month, year = match.groups()
                iso_date = f"{year}-{month}-{day}"
                return DateInfo(
                    value=iso_date,
                    date_type='filename',
                    confidence=0.95,
                    source='filename',
                    context=f"From filename: {filename[:20]}..."
                )
        except Exception as e:
            logger.debug(f"Could not extract date from filename: {e}")
        return None
    
    def _is_valid_supplier(self, supplier: str) -> bool:
        """
        Validate a supplier name using centralized SupplierValidator.
        Delegates to professional is_valid_supplier from supplier_validator module.
        """
        if is_valid_supplier:
            return is_valid_supplier(supplier)
        # Fallback if import failed
        if not supplier or len(supplier.strip()) < 3:
            return False
        return True
    
    def _resolve_supplier(
        self, 
        pdf_path: str, 
        full_text: str, 
        doc_type: DocumentType, 
        layout_results: Dict[str, VoterResult], 
        gliner_result: Dict[str, VoterResult]
    ) -> Tuple[Optional[str], Dict[str, Any]]:
        """
        Resolves supplier using Ensemble logic: GLiNER -> Regex -> Layout -> SmolDocling.
        Returns: (supplier_name, sources_update_dict)
        """
        sources = {}
        final_supplier = None
        
        # Coleta votos
        supplier_votes = []
        
        # 1. GLiNER
        if "fornecedor" in gliner_result:
            supplier_votes.append(gliner_result["fornecedor"])
            
        # 2. Regex (agora usa YAML patterns)
        regex_supp = self.regex_voter.extract_supplier(full_text, doc_type)
        if regex_supp.value:
            supplier_votes.append(regex_supp)
            
        # 3. Layout sources
        if "header_supplier" in layout_results:
            supplier_votes.append(layout_results["header_supplier"])
            
        # Consenso
        if supplier_votes:
            # Ordena por confianca
            sorted_votes = sorted(supplier_votes, key=lambda v: v.confidence, reverse=True)
            
            best_supplier_vote = None
            cleaned_supplier = None
            
            for vote in sorted_votes:
                # Usa cleaner do EnsembleExtractor (que limpa LTDA, stop words, etc)
                candidate = self._clean_supplier(vote.value)
                if not candidate:
                    continue
                
                # Validação centralizada
                if not self._is_valid_supplier(candidate):
                    logger.debug(f"Skipping invalid/banned supplier: {candidate}")
                    continue
                
                best_supplier_vote = vote
                cleaned_supplier = candidate
                break
            
            # Fuzzy match com known suppliers
            if validate_supplier and cleaned_supplier:
                match_result = validate_supplier(cleaned_supplier, min_similarity=0.5)
                if match_result:
                    canonical, similarity = match_result
                    cleaned_supplier = canonical
                    sources["fornecedor_match_score"] = f"{similarity:.0%}"

            if cleaned_supplier:
                final_supplier = cleaned_supplier
                sources["fornecedor"] = best_supplier_vote.source if best_supplier_vote else "unknown"
                
                # SmolDocling Fallback for Low Confidence (< 0.6)
                if best_supplier_vote and best_supplier_vote.confidence < 0.6:
                    sources["fornecedor_needs_review"] = True
                    
                    if self.use_vlm and SMOL_DOCLING_AVAILABLE and get_smol_docling_voter:
                        try:
                            smol_voter = get_smol_docling_voter()
                            smol_result = smol_voter.extract_from_pdf(pdf_path)
                            if smol_result and smol_result.get("supplier_name"):
                                smol_supplier = smol_result["supplier_name"]
                                # Validate
                                if self._is_valid_supplier(smol_supplier):
                                    # Match
                                    if validate_supplier:
                                        smol_match = validate_supplier(smol_supplier, min_similarity=0.5)
                                        if smol_match and self._is_valid_supplier(smol_match[0]):
                                            final_supplier = smol_match[0]
                                            sources["fornecedor"] = "smol_docling"
                                            sources["fornecedor_match_score"] = f"{smol_match[1]:.0%}"
                                            sources["fornecedor_needs_review"] = False
                                            logger.info(f"SmolDocling found supplier: {final_supplier}")
                        except Exception as e:
                            logger.debug(f"SmolDocling fallback error: {e}")

        # SmolDocling Rescue (No supplier found)
        if not final_supplier and self.use_vlm and SMOL_DOCLING_AVAILABLE and get_smol_docling_voter:
            try:
                smol_voter = get_smol_docling_voter()
                smol_result = smol_voter.extract_from_pdf(pdf_path)
                if smol_result and smol_result.get("supplier_name"):
                    smol_supplier = smol_result["supplier_name"]
                    if self._is_valid_supplier(smol_supplier):
                        if validate_supplier:
                            smol_match = validate_supplier(smol_supplier, min_similarity=0.5)
                            if smol_match and self._is_valid_supplier(smol_match[0]):
                                final_supplier = smol_match[0]
                                sources["fornecedor"] = "smol_docling_rescue"
                                logger.info(f"SmolDocling rescued supplier: {final_supplier}")
            except Exception as e:
                logger.debug(f"SmolDocling rescue error: {e}")

        return final_supplier, sources

    def _resolve_dates(
        self, 
        pdf_path: str, 
        full_text: str, 
        doc_type: DocumentType, 
        layout_results: Dict[str, VoterResult],
        result: ExtractionResult
    ):
        """
        Resolves dates (Emission, Due, Payment) and Filename Date.
        Updates result object in-place with dates, sources, and priorities.
        """
        # 1. Regex dates
        dates = self.regex_voter.extract_dates(full_text)
        
        # 2. Filename date (Critical fallback)
        filename_date_info = self._extract_date_from_filename(pdf_path)
        if filename_date_info:
            result.filename_date = filename_date_info.value
            result.all_dates.append(filename_date_info)
        
        # 3. Populate basic dates
        if "payment" in dates:
            result.payment_date = dates["payment"].value
            result.extraction_sources["payment_date"] = dates["payment"].source
            result.all_dates.append(DateInfo(
                value=dates["payment"].value,
                date_type='payment',
                confidence=dates["payment"].confidence,
                source=dates["payment"].source
            ))
        
        if "due" in dates:
            result.due_date = dates["due"].value
            result.extraction_sources["due_date"] = dates["due"].source
            result.all_dates.append(DateInfo(
                value=dates["due"].value,
                date_type='due',
                confidence=dates["due"].confidence,
                source=dates["due"].source
            ))
        
        if "emission" in dates:
            result.emission_date = dates["emission"].value
            result.extraction_sources["emission_date"] = dates["emission"].source
            result.all_dates.append(DateInfo(
                value=dates["emission"].value,
                date_type='emission',
                confidence=dates["emission"].confidence,
                source=dates["emission"].source
            ))
            
        # 4. Prioritization Logic
        primary_date = None
        
        if doc_type == DocumentType.COMPROVANTE:
            primary_date = result.payment_date or result.due_date or result.emission_date
            result.date_selection_reason = "comprovante_prioritizes_payment"
        elif doc_type == DocumentType.BOLETO:
            primary_date = result.due_date or result.payment_date
            result.date_selection_reason = "boleto_prioritizes_due"
        elif doc_type == DocumentType.NFSE:
            if result.due_date:
                primary_date = result.due_date
                result.date_selection_reason = "nfse_has_due_date"
            elif result.filename_date:
                primary_date = result.filename_date
                result.date_selection_reason = "nfse_uses_filename_date(no_due)"
            else:
                primary_date = result.emission_date
                result.date_selection_reason = "nfse_fallback_emission"
        elif doc_type == DocumentType.NFE:
            if result.due_date:
                primary_date = result.due_date
                result.date_selection_reason = "nfe_has_due_date"
            elif result.filename_date:
                primary_date = result.filename_date
                result.date_selection_reason = "nfe_uses_filename_date(no_due)"
            else:
                primary_date = result.emission_date
                result.date_selection_reason = "nfe_fallback_emission"
        else:
            primary_date = result.due_date or result.payment_date or result.emission_date
            result.date_selection_reason = "default_priority"
        
        # Update due_date if it was empty but primary was found
        if primary_date and not result.due_date:
            result.due_date = primary_date
            
        # 5. Layout Fallback for emission
        if not result.emission_date and "emission_layout" in layout_results:
            result.emission_date = layout_results["emission_layout"].value
            result.extraction_sources["emission_date"] = "layout"

            result.extraction_sources["emission_date"] = "layout"

    def _resolve_financials(
        self,
        full_text: str,
        doc_type: DocumentType,
        result: ExtractionResult
    ):
        """
        Resolves Amount, Doc Number, SISBB Auth, Payment Status, and Final Confidence.
        Updates result object in-place.
        """
        # 1. Extract Amount
        amount_result = self.regex_voter.extract_amount(full_text)
        if amount_result.value:
            result.amount_cents = int(amount_result.value)
            result.extraction_sources["amount"] = amount_result.source
        
        # 2. Extract Document Number
        doc_num_result = self.regex_voter.extract_doc_number(full_text, doc_type)
        if doc_num_result.value:
            result.doc_number = doc_num_result.value
            result.extraction_sources["doc_number"] = doc_num_result.source
        
        # 3. SISBB Validation (Banco do Brasil)
        sisbb_result = self.regex_voter.extract_sisbb(full_text)
        if sisbb_result.value:
            result.sisbb_auth = sisbb_result.value
        
        # 4. Determine Payment Status
        result.is_scheduled = self.regex_voter.check_scheduling(full_text)
        
        if result.is_scheduled:
            result.payment_status = PaymentStatus.SCHEDULED
        elif result.sisbb_auth:
            result.payment_status = PaymentStatus.CONFIRMED
        else:
            result.payment_status = PaymentStatus.UNKNOWN
        
        # 5. Calculate Final Confidence
        result.confidence = self._calculate_confidence(result)
        result.needs_review = result.confidence < 0.6

    def extract_from_pdf(
        self, 
        pdf_path: str, 
        page_range: Optional[Tuple[int, int]] = None
    ) -> ExtractionResult:
        """
        Extrai dados de um PDF usando ensemble voting (Decomposed Logic).
        """
        start_time = datetime.now()
        result = ExtractionResult()
        doc = None
        
        try:
            # ========================================
            # STEP 0: Leitura e Texto
            # ========================================
            try:
                doc = fitz.open(pdf_path)
            except Exception as e:
                logger.error(f"Erro ao abrir PDF: {e}")
                result.needs_review = True
                return result

            # Extrai texto
            if page_range:
                start, end = page_range
                pages = range(start - 1, min(end, len(doc)))
            else:
                pages = range(len(doc))
            
            full_text = ""
            for i in pages:
                page = doc[i]
                text = page.get_text()
                full_text += text + "\n\n"
            
            # OCR Fallback
            if not self._is_readable_text(full_text):
                logger.info(f"Text unreadable, attempting OCR for {pdf_path}")
                if self.ocr_voter and self.ocr_voter.is_available():
                    ocr_text = ""
                    for i in pages:
                        ocr_page = self.ocr_voter.extract_text(doc, i)
                        if ocr_page:
                            ocr_text += ocr_page + "\n\n"
                    if self._is_readable_text(ocr_text):
                        full_text = ocr_text
                        result.extraction_sources["method"] = "ocr"
            
            result.raw_text = full_text

            # ========================================
            # STEP 1: Classificacao e Entity Tag
            # ========================================
            doc_type, type_conf = self.regex_voter.classify_document(full_text)
            result.doc_type = doc_type
            result.extraction_sources["doc_type"] = "regex"
            
            entity_result = self.regex_voter.extract_entity_tag(full_text)
            if entity_result.value:
                result.entity_tag = entity_result.value
                result.extraction_sources["entity"] = entity_result.source

            # ========================================
            # STEP 2: Executa Voters (GLiNER + Layout)
            # ========================================
            # GLiNER Vote
            gliner_entities = {}
            if self.gliner_voter:
                raw_entities = self.gliner_voter.extract_entities(full_text)
                # Ensure we have VoterResults with penalty if needed (handled in resolve?)
                # We'll pass raw entities, _resolve_supplier logic handles specific keys
                gliner_entities = raw_entities

            # Layout Vote
            layout_results = self.layout_voter.extract_from_layout(doc, 0)
            
            # ========================================
            # STEP 3: Resolve Componentes (Decomposed)
            # ========================================
            
            # 3.1 Fornecedor
            final_supplier, supp_sources = self._resolve_supplier(
                pdf_path, full_text, doc_type, layout_results, gliner_entities
            )
            result.fornecedor = final_supplier
            result.extraction_sources.update(supp_sources)

            # 3.2 Datas (inclui filename date e priorização)
            self._resolve_dates(pdf_path, full_text, doc_type, layout_results, result)
            
            # 3.3 Valor, Numero Doc, Status
            self._resolve_financials(full_text, doc_type, result)
            
            # ========================================
            # STEP 4: Finaliza
            # ========================================
            end_time = datetime.now()
            result.processing_time_ms = int((end_time - start_time).total_seconds() * 1000)
            result.voters_used = list(result.extraction_sources.keys())
            
        except Exception as e:
            logger.error(f"Erro na extracao: {e}")
            result.needs_review = True
        finally:
            if doc:
                doc.close()
        
        return result
        
        return result
    
    def _clean_supplier(self, name: str) -> str:
        """Limpa nome do fornecedor."""
        if not name:
            return ""
        
        # Remove quebras de linha
        name = re.sub(r'[\n\r]+', ' ', name)
        name = re.sub(r'\s+', ' ', name).strip().upper()
        
        # Corta em palavras de parada
        stop_words = ['CNPJ', 'CPF', 'PAGADOR', 'CEP', 'ENDERECO', 
                      'AGENCIA', 'SITE', 'TELEFONE', 'AG', 'DATA',
                      'VENCIMENTO', 'LOCAL']
        for stop in stop_words:
            if f" {stop}" in name:
                name = name.split(f" {stop}")[0].strip()
        
        # Remove sufixos
        suffixes = [" LTDA", " ME", " EPP", " EIRELI", " SA", " S.A."]
        for suffix in suffixes:
            if name.endswith(suffix):
                name = name[:-len(suffix)].strip()
        
        return name[:50]
    
    def _calculate_confidence(self, result: ExtractionResult) -> float:
        """Calcula confianca geral."""
        score = 0.0
        
        if result.doc_type != DocumentType.UNKNOWN:
            score += 0.2
        
        if result.amount_cents > 0:
            score += 0.25
        
        if result.due_date or result.emission_date or result.payment_date:
            score += 0.2
        
        if result.fornecedor and len(result.fornecedor) >= 3:
            score += 0.2
        
        if result.entity_tag:
            score += 0.15
        
        return min(score, 1.0)


# ==============================================================================
# FUNCOES AUXILIARES
# ==============================================================================

def test_ensemble_extraction(pdf_path: str):
    """Testa extracao ensemble em um arquivo."""
    extractor = EnsembleExtractor(use_gliner=True)
    result = extractor.extract_from_pdf(pdf_path)
    
    print("=" * 60)
    print(f"Arquivo: {Path(pdf_path).name}")
    print("=" * 60)
    print(f"Tipo: {result.doc_type.value} ({result.extraction_sources.get('doc_type', '-')})")
    print(f"Confianca: {result.confidence:.1%}")
    print("-" * 60)
    print(f"Fornecedor: {result.fornecedor} ({result.extraction_sources.get('fornecedor', '-')})")
    print(f"Entidade: {result.entity_tag}")
    print(f"Valor: R$ {result.amount_cents / 100:.2f}")
    print(f"Data Pagamento: {result.payment_date}")
    print(f"Data Vencimento: {result.due_date}")
    print(f"Data Emissao: {result.emission_date}")
    print(f"SISBB: {result.sisbb_auth}")
    print(f"Agendamento: {result.is_scheduled}")
    print(f"Precisa Revisao: {result.needs_review}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        test_ensemble_extraction(sys.argv[1])
    else:
        print("Uso: python ensemble_extractor.py <arquivo.pdf>")
