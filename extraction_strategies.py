"""
SDRA Extraction Strategies - Contextual Extraction Engine
==========================================================

This module implements the Strategy pattern for document extraction.
Each document type (NFe, Boleto, Comprovante) has specialized extraction
logic that knows WHERE and HOW to find data.

Architecture:
    ClassifyDocument → GetStrategy → ExtractWithContext
           │                │                │
           ▼                ▼                ▼
       "BOLETO"      BoletoStrategy    Regex após "Beneficiário"
       "NFE"         NfeStrategy       XML tags ou quadrante 1
       "COMPROVANTE" ComprovanteStrategy  Regex após "Favorecido"

Key Features:
- Blacklist: Prevents extracting dates/keywords as supplier names
- Fuzzy matching: Corrects supplier names using known_suppliers.txt
- Zone-based extraction: Looks in document-specific regions
"""

import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path

logger = logging.getLogger('srda.extraction_strategies')

# Try to import fuzzy matching
try:
    from rapidfuzz import fuzz, process
    FUZZY_AVAILABLE = True
except ImportError:
    FUZZY_AVAILABLE = False
    logger.warning("rapidfuzz not available - fuzzy matching disabled")


# =============================================================================
# CONSTANTS AND BLACKLISTS
# =============================================================================

# Words that should NEVER be extracted as supplier names
SUPPLIER_BLACKLIST = {
    # Generic payment terms
    "PAGAMENTO", "PAGTO", "DATA", "VENCIMENTO", "VENC", "VALOR",
    "AUTENTICAÇÃO", "AUTENTICACAO", "BANCO", "AGÊNCIA", "AGENCIA",
    "CONTA", "CORRENTE", "POUPANÇA", "POUPANCA", "BENEFICIÁRIO", "BENEFICIARIO",
    
    # Dates and numbers often mistaken for names
    "JANEIRO", "FEVEREIRO", "MARÇO", "MARCO", "ABRIL", "MAIO", "JUNHO",
    "JULHO", "AGOSTO", "SETEMBRO", "OUTUBRO", "NOVEMBRO", "DEZEMBRO",
    "JAN", "FEV", "MAR", "ABR", "MAI", "JUN", "JUL", "AGO", "SET", "OUT", "NOV", "DEZ",
    
    # Document type keywords
    "BOLETO", "NOTA", "FISCAL", "NFE", "NFSE", "COMPROVANTE", "RECIBO",
    "DANFE", "FATURA", "DUPLICATA", "PIX", "TED", "DOC", "TRANSFERÊNCIA",
    "TRANSFERENCIA", "DEPÓSITO", "DEPOSITO",
    
    # Bank names (should use full name, not fragments)
    "BRADESCO", "ITAÚ", "ITAU", "SANTANDER", "CAIXA", "BRASIL", "INTER",
    "NUBANK", "SICOOB", "SICREDI", "SAFRA", "BTG",
    
    # Common false positives
    "TOTAL", "SUBTOTAL", "DESCONTO", "ACRÉSCIMO", "ACRESCIMO", "MULTA",
    "JUROS", "MORA", "ENCARGOS", "TAXA", "IMPOSTO", "ISS", "ICMS", "IPI",
    "COFINS", "PIS", "CSLL", "IRRF", "INSS",
    
    # Titles
    "LTDA", "EIRELI", "ME", "EPP", "SA", "S/A", "S.A.", "CNPJ", "CPF",
    
    # Generic words
    "OUTROS", "DIVERSOS", "SERVIÇO", "SERVICO", "PRODUTO", "ITEM",
    "QUANTIDADE", "QTD", "UNIDADE", "UN", "CX", "PCT", "KG", "LT",
}

# Minimum length for valid supplier name
MIN_SUPPLIER_LENGTH = 3

# Maximum length for supplier name (avoid extracting entire paragraphs)
MAX_SUPPLIER_LENGTH = 100

# Fuzzy match threshold (0-100)
FUZZY_THRESHOLD = 85


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class ExtractionResult:
    """Result from extraction strategy."""
    supplier: Optional[str] = None
    supplier_confidence: float = 0.0
    value: Optional[float] = None
    value_confidence: float = 0.0
    date: Optional[str] = None
    date_type: Optional[str] = None  # "vencimento", "emissao", "pagamento"
    document_number: Optional[str] = None
    parcela: Optional[str] = None  # "1/3", "2/3", etc.
    extra_data: Dict[str, Any] = field(default_factory=dict)
    strategy_used: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "supplier": self.supplier,
            "supplier_confidence": self.supplier_confidence,
            "value": self.value,
            "value_confidence": self.value_confidence,
            "date": self.date,
            "date_type": self.date_type,
            "document_number": self.document_number,
            "parcela": self.parcela,
            "extra_data": self.extra_data,
            "strategy_used": self.strategy_used,
        }


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def is_blacklisted(text: str) -> bool:
    """Check if text contains only blacklisted words."""
    if not text:
        return True
    
    # Clean and uppercase
    clean = text.strip().upper()
    
    # Check direct match
    if clean in SUPPLIER_BLACKLIST:
        return True
    
    # Check if ALL words are blacklisted
    words = clean.split()
    if all(w in SUPPLIER_BLACKLIST for w in words):
        return True
    
    return False


def is_date_like(text: str) -> bool:
    """Check if text looks like a date."""
    if not text:
        return False
    
    # Common date patterns
    date_patterns = [
        r'^\d{1,2}/\d{1,2}/\d{2,4}$',  # DD/MM/YYYY
        r'^\d{1,2}-\d{1,2}-\d{2,4}$',  # DD-MM-YYYY
        r'^\d{1,2}\.\d{1,2}\.\d{2,4}$',  # DD.MM.YYYY
        r'^\d{4}-\d{2}-\d{2}$',  # YYYY-MM-DD (ISO)
    ]
    
    clean = text.strip()
    for pattern in date_patterns:
        if re.match(pattern, clean):
            return True
    
    return False


def clean_supplier_name(name: str) -> Optional[str]:
    """
    Clean and validate extracted supplier name.
    
    Returns None if name is invalid (blacklisted, too short, looks like date).
    """
    if not name:
        return None
    
    # Basic cleanup
    cleaned = name.strip()
    cleaned = re.sub(r'\s+', ' ', cleaned)  # Normalize whitespace
    
    # Length checks
    if len(cleaned) < MIN_SUPPLIER_LENGTH:
        return None
    if len(cleaned) > MAX_SUPPLIER_LENGTH:
        cleaned = cleaned[:MAX_SUPPLIER_LENGTH]
    
    # Check blacklist
    if is_blacklisted(cleaned):
        logger.debug(f"Supplier '{cleaned}' is blacklisted")
        return None
    
    # Check date-like
    if is_date_like(cleaned):
        logger.debug(f"Supplier '{cleaned}' looks like a date")
        return None
    
    # Check if it's mostly numbers (likely a document number)
    digits = sum(c.isdigit() for c in cleaned)
    if digits / len(cleaned) > 0.7:
        logger.debug(f"Supplier '{cleaned}' is mostly digits")
        return None
    
    return cleaned


def fuzzy_match_supplier(
    extracted: str,
    known_suppliers: List[str],
    threshold: int = FUZZY_THRESHOLD
) -> Tuple[Optional[str], float]:
    """
    Match extracted supplier against known suppliers using fuzzy matching.
    
    Returns:
        (matched_name, confidence) or (None, 0) if no match
    """
    if not FUZZY_AVAILABLE or not known_suppliers or not extracted:
        return None, 0.0
    
    try:
        # Use process.extractOne for best match
        result = process.extractOne(
            extracted.upper(),
            [s.upper() for s in known_suppliers],
            scorer=fuzz.token_sort_ratio
        )
        
        if result and result[1] >= threshold:
            # Find original case supplier
            idx = [s.upper() for s in known_suppliers].index(result[0])
            return known_suppliers[idx], result[1] / 100.0
        
        return None, 0.0
        
    except Exception as e:
        logger.warning(f"Fuzzy match error: {e}")
        return None, 0.0


def load_known_suppliers(path: Path = None) -> List[str]:
    """Load known suppliers from file."""
    if path is None:
        # Default path
        path = Path(__file__).parent / "known_suppliers.txt"
    
    if not path.exists():
        logger.warning(f"Known suppliers file not found: {path}")
        return []
    
    try:
        with open(path, 'r', encoding='utf-8') as f:
            suppliers = [line.strip() for line in f if line.strip()]
        logger.info(f"Loaded {len(suppliers)} known suppliers")
        return suppliers
    except Exception as e:
        logger.error(f"Error loading known suppliers: {e}")
        return []


# =============================================================================
# EXTRACTION STRATEGIES (Strategy Pattern)
# =============================================================================

class BaseExtractionStrategy(ABC):
    """
    Abstract base class for extraction strategies.
    
    Each strategy knows how to extract data from a specific document type.
    """
    
    def __init__(self, known_suppliers: List[str] = None):
        self.known_suppliers = known_suppliers or []
    
    @property
    @abstractmethod
    def document_type(self) -> str:
        """Return the document type this strategy handles."""
        pass
    
    @abstractmethod
    def extract(self, text: str, metadata: Dict[str, Any] = None) -> ExtractionResult:
        """
        Extract data from document text.
        
        Args:
            text: Full document text (OCR or parsed)
            metadata: Optional metadata (filename, page count, etc.)
            
        Returns:
            ExtractionResult with extracted data
        """
        pass
    
    def _extract_value_brl(self, text: str) -> Tuple[Optional[float], float]:
        """
        Extract Brazilian currency value.
        
        Returns:
            (value, confidence) tuple
        """
        # Patterns for BRL values
        patterns = [
            # R$ 1.234,56 or R$1.234,56
            r'R\$\s*(\d{1,3}(?:\.\d{3})*,\d{2})',
            # 1.234,56 (without R$, but in value context)
            r'(?:VALOR|TOTAL|PAGAR)[\s:]+(\d{1,3}(?:\.\d{3})*,\d{2})',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                value_str = match.group(1)
                try:
                    # Convert BR format to float
                    value = float(value_str.replace('.', '').replace(',', '.'))
                    return value, 0.95
                except ValueError:
                    continue
        
        return None, 0.0
    
    def _extract_date(self, text: str, date_type: str = None) -> Tuple[Optional[str], Optional[str]]:
        """
        Extract date from text.
        
        Args:
            text: Text to search
            date_type: Optional hint ("vencimento", "emissao", "pagamento")
            
        Returns:
            (date_string, date_type) tuple
        """
        # Context patterns for different date types
        contexts = {
            "vencimento": r'(?:VENCIMENTO|VENC\.?|DATA\s+VENC)[\s:]+(\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4})',
            "emissao": r'(?:EMISSÃO|EMISSAO|DATA\s+EMISS)[\s:]+(\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4})',
            "pagamento": r'(?:PAGAMENTO|PAGTO|DATA\s+PAG)[\s:]+(\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4})',
        }
        
        # If date_type specified, try that first
        if date_type and date_type in contexts:
            match = re.search(contexts[date_type], text, re.IGNORECASE)
            if match:
                return match.group(1), date_type
        
        # Try all contexts
        for dt, pattern in contexts.items():
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1), dt
        
        # Generic date extraction (less reliable)
        generic = re.search(r'\b(\d{1,2}/\d{1,2}/\d{2,4})\b', text)
        if generic:
            return generic.group(1), "unknown"
        
        return None, None
    
    def _apply_fuzzy_correction(self, supplier: str) -> Tuple[str, float]:
        """Apply fuzzy matching to correct supplier name."""
        if not supplier or not self.known_suppliers:
            return supplier, 0.5
        
        matched, confidence = fuzzy_match_supplier(supplier, self.known_suppliers)
        if matched:
            logger.info(f"Fuzzy corrected '{supplier}' → '{matched}' ({confidence:.0%})")
            return matched, confidence
        
        return supplier, 0.5  # Keep original with lower confidence


class BoletoStrategy(BaseExtractionStrategy):
    """
    Extraction strategy for Boletos (payment slips).
    
    Key features:
    - Looks for supplier after "Beneficiário" keyword
    - Extracts due date (vencimento)
    - Extracts value from "Valor do Documento"
    - Ignores lines with dates when looking for supplier
    """
    
    @property
    def document_type(self) -> str:
        return "BOLETO"
    
    def extract(self, text: str, metadata: Dict[str, Any] = None) -> ExtractionResult:
        result = ExtractionResult(strategy_used=self.__class__.__name__)
        
        # Extract value
        result.value, result.value_confidence = self._extract_value_brl(text)
        
        # Extract due date
        result.date, result.date_type = self._extract_date(text, "vencimento")
        
        # Extract supplier - look AFTER "Beneficiário"
        result.supplier, result.supplier_confidence = self._extract_beneficiario(text)
        
        # Extract parcela if present
        result.parcela = self._extract_parcela(text)
        
        # Extract document number (linha digitável)
        linha = self._extract_linha_digitavel(text)
        if linha:
            result.document_number = linha
        
        return result
    
    def _extract_beneficiario(self, text: str) -> Tuple[Optional[str], float]:
        """Extract beneficiário (supplier) from boleto."""
        
        # Pattern: Look for text after "Beneficiário" keyword
        patterns = [
            r'BENEFICI[ÁA]RIO[\s:]+([^\n]+)',
            r'CEDENTE[\s:]+([^\n]+)',
            r'FAVOR(?:ECIDO)?[\s:]+([^\n]+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                raw = match.group(1).strip()
                
                # Split by common delimiters and take first meaningful part
                parts = re.split(r'[|\-/]', raw)
                for part in parts:
                    cleaned = clean_supplier_name(part)
                    if cleaned:
                        # Apply fuzzy correction
                        return self._apply_fuzzy_correction(cleaned)
        
        return None, 0.0
    
    def _extract_parcela(self, text: str) -> Optional[str]:
        """Extract parcela info (1/3, 2/3, etc.)."""
        match = re.search(r'PARCELA[\s:]+(\d+\s*/\s*\d+)', text, re.IGNORECASE)
        if match:
            return match.group(1).replace(' ', '')
        
        # Alternative pattern
        match = re.search(r'(\d+)\s*(?:DE|/)\s*(\d+)\s*(?:PARCELA|PARC)', text, re.IGNORECASE)
        if match:
            return f"{match.group(1)}/{match.group(2)}"
        
        return None
    
    def _extract_linha_digitavel(self, text: str) -> Optional[str]:
        """Extract linha digitável."""
        # 47 digits pattern
        match = re.search(r'\b(\d{5}\.?\d{5}\s*\d{5}\.?\d{6}\s*\d{5}\.?\d{6}\s*\d\s*\d{14})\b', text)
        if match:
            return re.sub(r'[\s.]', '', match.group(1))
        return None


class NfeStrategy(BaseExtractionStrategy):
    """
    Extraction strategy for NFe (Notas Fiscais Eletrônicas).
    
    Key features:
    - Looks for supplier in "Emitente" section (usually top-left)
    - Extracts total value
    - Extracts emission date
    - Extracts chave de acesso (44 digits)
    """
    
    @property
    def document_type(self) -> str:
        return "NFE"
    
    def extract(self, text: str, metadata: Dict[str, Any] = None) -> ExtractionResult:
        result = ExtractionResult(strategy_used=self.__class__.__name__)
        
        # Extract value
        result.value, result.value_confidence = self._extract_nfe_value(text)
        
        # Extract emission date
        result.date, result.date_type = self._extract_date(text, "emissao")
        
        # Extract supplier (emitente)
        result.supplier, result.supplier_confidence = self._extract_emitente(text)
        
        # Extract chave de acesso
        chave = self._extract_chave_acesso(text)
        if chave:
            result.document_number = chave
        
        return result
    
    def _extract_nfe_value(self, text: str) -> Tuple[Optional[float], float]:
        """Extract NFe total value."""
        # NFe specific patterns
        patterns = [
            r'VALOR\s+TOTAL\s+DA\s+(?:NOTA|NF)[\s:]+R?\$?\s*(\d{1,3}(?:\.\d{3})*,\d{2})',
            r'TOTAL\s+GERAL[\s:]+R?\$?\s*(\d{1,3}(?:\.\d{3})*,\d{2})',
            r'V(?:ALOR)?\.?\s*TOTAL[\s:]+R?\$?\s*(\d{1,3}(?:\.\d{3})*,\d{2})',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                value_str = match.group(1)
                try:
                    value = float(value_str.replace('.', '').replace(',', '.'))
                    return value, 0.95
                except ValueError:
                    continue
        
        # Fallback to generic
        return self._extract_value_brl(text)
    
    def _extract_emitente(self, text: str) -> Tuple[Optional[str], float]:
        """Extract emitente (supplier) from NFe."""
        
        # Pattern: Look for text after "Emitente" or "Razão Social"
        patterns = [
            r'EMITENTE[\s\n:]+(?:RAZ[ÃA]O\s+SOCIAL)?[\s:]*([^\n]+)',
            r'RAZ[ÃA]O\s+SOCIAL(?:\s+DO\s+EMITENTE)?[\s:]+([^\n]+)',
            r'NOME\s*/\s*RAZ[ÃA]O\s+SOCIAL[\s:]+([^\n]+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                raw = match.group(1).strip()
                cleaned = clean_supplier_name(raw)
                if cleaned:
                    return self._apply_fuzzy_correction(cleaned)
        
        # Try zone-based extraction (top portion of document)
        lines = text.split('\n')
        top_lines = lines[:20]  # First 20 lines
        
        for line in top_lines:
            # Look for company names (contain LTDA, EIRELI, etc)
            if any(suffix in line.upper() for suffix in ['LTDA', 'EIRELI', 'S/A', 'S.A.', 'ME', 'EPP']):
                cleaned = clean_supplier_name(line)
                if cleaned:
                    return self._apply_fuzzy_correction(cleaned)
        
        return None, 0.0
    
    def _extract_chave_acesso(self, text: str) -> Optional[str]:
        """Extract chave de acesso (44 digits)."""
        # Remove common OCR artifacts
        clean_text = re.sub(r'[\s.-]', '', text)
        
        # Look for 44 consecutive digits
        match = re.search(r'\b(\d{44})\b', clean_text)
        if match:
            return match.group(1)
        
        # Try pattern with spaces
        match = re.search(r'(\d{4}\s*\d{4}\s*\d{4}\s*\d{4}\s*\d{4}\s*\d{4}\s*\d{4}\s*\d{4}\s*\d{4}\s*\d{4}\s*\d{4})', text)
        if match:
            return re.sub(r'\s', '', match.group(1))
        
        return None


class ComprovanteStrategy(BaseExtractionStrategy):
    """
    Extraction strategy for Comprovantes (payment receipts).
    
    Key features:
    - Looks for supplier in "Favorecido" section
    - Extracts payment date (not due date)
    - Extracts paid value
    - Looks for authentication codes (SISBB, etc.)
    """
    
    @property
    def document_type(self) -> str:
        return "COMPROVANTE"
    
    def extract(self, text: str, metadata: Dict[str, Any] = None) -> ExtractionResult:
        result = ExtractionResult(strategy_used=self.__class__.__name__)
        
        # Extract value
        result.value, result.value_confidence = self._extract_comprovante_value(text)
        
        # Extract payment date
        result.date, result.date_type = self._extract_date(text, "pagamento")
        
        # Extract favorecido (supplier)
        result.supplier, result.supplier_confidence = self._extract_favorecido(text)
        
        # Extract authentication code
        auth_code = self._extract_auth_code(text)
        if auth_code:
            result.document_number = auth_code
            result.extra_data["auth_code"] = auth_code
        
        return result
    
    def _extract_comprovante_value(self, text: str) -> Tuple[Optional[float], float]:
        """Extract comprovante value."""
        patterns = [
            r'VALOR\s+(?:PAGO|DEBITADO|TRANSFERIDO)[\s:]+R?\$?\s*(\d{1,3}(?:\.\d{3})*,\d{2})',
            r'VALOR\s+DA\s+TRANSFER[ÊE]NCIA[\s:]+R?\$?\s*(\d{1,3}(?:\.\d{3})*,\d{2})',
            r'VALOR[\s:]+R?\$?\s*(\d{1,3}(?:\.\d{3})*,\d{2})',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                value_str = match.group(1)
                try:
                    value = float(value_str.replace('.', '').replace(',', '.'))
                    return value, 0.95
                except ValueError:
                    continue
        
        return self._extract_value_brl(text)
    
    def _extract_favorecido(self, text: str) -> Tuple[Optional[str], float]:
        """Extract favorecido (supplier) from comprovante."""
        
        patterns = [
            r'FAVORECIDO[\s:]+([^\n]+)',
            r'DESTINAT[ÁA]RIO[\s:]+([^\n]+)',
            r'PARA[\s:]+([^\n]+)',
            r'NOME[\s:]+([^\n]+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                raw = match.group(1).strip()
                cleaned = clean_supplier_name(raw)
                if cleaned:
                    return self._apply_fuzzy_correction(cleaned)
        
        return None, 0.0
    
    def _extract_auth_code(self, text: str) -> Optional[str]:
        """Extract authentication code."""
        patterns = [
            r'SISBB[\s:]*(\d+)',
            r'AUTENT(?:ICA[ÇC][ÃA]O)?[\s:]+([A-Z0-9]+)',
            r'C[ÓO]D(?:IGO)?\.?\s*AUTH?[\s:]+([A-Z0-9]+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1)
        
        return None


class NfseStrategy(BaseExtractionStrategy):
    """
    Extraction strategy for NFSe (Notas Fiscais de Serviço).
    
    Similar to NFe but looks for service-specific fields.
    """
    
    @property
    def document_type(self) -> str:
        return "NFSE"
    
    def extract(self, text: str, metadata: Dict[str, Any] = None) -> ExtractionResult:
        result = ExtractionResult(strategy_used=self.__class__.__name__)
        
        # Extract value (liquid value for services)
        result.value, result.value_confidence = self._extract_nfse_value(text)
        
        # Extract emission date
        result.date, result.date_type = self._extract_date(text, "emissao")
        
        # Extract prestador (service provider)
        result.supplier, result.supplier_confidence = self._extract_prestador(text)
        
        # Extract NFSe number
        nfse_num = self._extract_nfse_number(text)
        if nfse_num:
            result.document_number = nfse_num
        
        return result
    
    def _extract_nfse_value(self, text: str) -> Tuple[Optional[float], float]:
        """Extract NFSe value."""
        patterns = [
            r'VALOR\s+L[ÍI]QUIDO[\s:]+R?\$?\s*(\d{1,3}(?:\.\d{3})*,\d{2})',
            r'VALOR\s+(?:DOS?\s+)?SERVI[ÇC]OS?[\s:]+R?\$?\s*(\d{1,3}(?:\.\d{3})*,\d{2})',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                value_str = match.group(1)
                try:
                    value = float(value_str.replace('.', '').replace(',', '.'))
                    return value, 0.95
                except ValueError:
                    continue
        
        return self._extract_value_brl(text)
    
    def _extract_prestador(self, text: str) -> Tuple[Optional[str], float]:
        """Extract prestador (service provider)."""
        patterns = [
            r'PRESTADOR(?:\s+DE\s+SERVI[ÇC]OS?)?[\s\n:]+(?:RAZ[ÃA]O\s+SOCIAL)?[\s:]*([^\n]+)',
            r'RAZ[ÃA]O\s+SOCIAL(?:\s+DO\s+PRESTADOR)?[\s:]+([^\n]+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                raw = match.group(1).strip()
                cleaned = clean_supplier_name(raw)
                if cleaned:
                    return self._apply_fuzzy_correction(cleaned)
        
        return None, 0.0
    
    def _extract_nfse_number(self, text: str) -> Optional[str]:
        """Extract NFSe number."""
        match = re.search(r'N[ÚU]MERO\s+(?:DA\s+)?NFS?-?E?[\s:]+(\d+)', text, re.IGNORECASE)
        if match:
            return match.group(1)
        return None


# =============================================================================
# STRATEGY FACTORY
# =============================================================================

class ExtractionStrategyFactory:
    """
    Factory for creating extraction strategies based on document type.
    """
    
    _strategies = {
        "NFE": NfeStrategy,
        "NFSE": NfseStrategy,
        "BOLETO": BoletoStrategy,
        "COMPROVANTE": ComprovanteStrategy,
    }
    
    def __init__(self, known_suppliers: List[str] = None):
        self.known_suppliers = known_suppliers or load_known_suppliers()
        self._cache = {}
    
    def get_strategy(self, document_type: str) -> BaseExtractionStrategy:
        """
        Get extraction strategy for document type.
        
        Args:
            document_type: "NFE", "NFSE", "BOLETO", "COMPROVANTE"
            
        Returns:
            Appropriate extraction strategy
        """
        doc_type = document_type.upper()
        
        # Check cache
        if doc_type in self._cache:
            return self._cache[doc_type]
        
        # Create strategy
        strategy_class = self._strategies.get(doc_type, BoletoStrategy)  # Default to Boleto
        strategy = strategy_class(known_suppliers=self.known_suppliers)
        
        self._cache[doc_type] = strategy
        return strategy
    
    @classmethod
    def register_strategy(cls, document_type: str, strategy_class: type):
        """Register a new strategy class."""
        cls._strategies[document_type.upper()] = strategy_class


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================

def extract_with_strategy(
    text: str,
    document_type: str,
    known_suppliers: List[str] = None,
    metadata: Dict[str, Any] = None
) -> ExtractionResult:
    """
    Convenience function for one-shot extraction.
    
    Args:
        text: Document text (OCR or parsed)
        document_type: Type of document (NFE, BOLETO, etc.)
        known_suppliers: Optional list of known suppliers for fuzzy matching
        metadata: Optional document metadata
        
    Returns:
        ExtractionResult with extracted data
    """
    factory = ExtractionStrategyFactory(known_suppliers=known_suppliers)
    strategy = factory.get_strategy(document_type)
    return strategy.extract(text, metadata)
