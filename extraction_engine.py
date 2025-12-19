"""
SRDA-Rural Extraction Engine
=============================
Motor de Extracao Cognitiva com OCR e Validacao

Este modulo implementa extracao precisa de dados de documentos financeiros:
- Camada 1: Extracao nativa via PyMuPDF (PDFs digitais)
- Camada 2: OCR com Tesseract + pre-processamento OpenCV (PDFs escaneados)
- Validacao de dados extraidos (CPF/CNPJ, linha digitavel)
- Identificacao correta de destinatario (nao confundir com banco/cidade)

Referencia: Otimizacao de Codigo para Documentos Financeiros.txt
"""

import re
import cv2
import numpy as np
import fitz  # PyMuPDF
import pytesseract
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from rapidfuzz import fuzz

from database import (
    SRDADatabase,
    DocumentType,
    DocumentStatus,
    EntityTag
)


# ==============================================================================
# CONFIGURACOES
# ==============================================================================

# Threshold minimo de texto para considerar PDF digital (vs escaneado)
MIN_TEXT_LENGTH = 50

# Confianca minima para aceitar extracao automatica
MIN_CONFIDENCE = 0.7

# Nomes de bancos para ignorar na extracao de destinatario
BANK_NAMES = [
    "BANCO DO BRASIL", "BB", "CAIXA ECONOMICA", "CEF", "CAIXA",
    "ITAU", "ITAU UNIBANCO", "BRADESCO", "SANTANDER", "SICOOB",
    "SICREDI", "CRESOL", "BANRISUL", "SAFRA", "BTG", "INTER",
    "NUBANK", "C6 BANK", "ORIGINAL", "PAN", "VOTORANTIM"
]

# Padroes de cidades para ignorar (Cidade - UF)
CITY_PATTERN = re.compile(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s*[-/]\s*[A-Z]{2}\b')

# Lista de ancoras para encontrar destinatario
RECIPIENT_ANCHORS = [
    "DESTINATARIO", "TOMADOR", "SACADO", "CLIENTE", 
    "PAGADOR", "COMPRADOR", "ADQUIRENTE"
]

# Lista de ancoras para encontrar fornecedor/emissor
SUPPLIER_ANCHORS = [
    "EMITENTE", "PRESTADOR", "CEDENTE", "FORNECEDOR",
    "REMETENTE", "VENDEDOR", "BENEFICIARIO"
]

# Regex para valores monetarios brasileiros (mais preciso)
AMOUNT_PATTERNS = [
    # Valor apos "VALOR TOTAL", "VALOR DO DOCUMENTO", etc
    re.compile(r'VALOR\s*(?:TOTAL|DOCUMENTO|LIQUIDO|COBRADO|PAGO|A\s*PAGAR)[:\s]*R?\$?\s*([\d\.]+,\d{2})', re.IGNORECASE),
    # Valor apos "TOTAL" isolado
    re.compile(r'\bTOTAL[:\s]+R?\$?\s*([\d\.]+,\d{2})', re.IGNORECASE),
    # R$ seguido de valor
    re.compile(r'R\$\s*([\d\.]+,\d{2})'),
    # Valor grande isolado (fallback)
    re.compile(r'\b(\d{1,3}(?:\.\d{3})*,\d{2})\b'),
]

# Regex para datas
DATE_PATTERNS = {
    "payment": re.compile(r'(?:DATA\s*(?:DO)?\s*PAGAMENTO|PAGO\s*EM|PAGAMENTO\s*REALIZADO)[:\s]*(\d{2}[/\.\-]\d{2}[/\.\-]\d{4})', re.IGNORECASE),
    "due": re.compile(r'(?:VENCIMENTO|VENC\.|DATA\s*DE\s*VENCIMENTO)[:\s]*(\d{2}[/\.\-]\d{2}[/\.\-]\d{4})', re.IGNORECASE),
    "emission": re.compile(r'(?:EMISSAO|DATA\s*(?:DA)?\s*EMISSAO|EMITIDO\s*EM)[:\s]*(\d{2}[/\.\-]\d{2}[/\.\-]\d{4})', re.IGNORECASE),
    "generic": re.compile(r'\b(\d{2}[/\.\-]\d{2}[/\.\-]\d{4})\b'),
}


# ==============================================================================
# DATA CLASSES
# ==============================================================================

@dataclass
class ExtractedData:
    """Dados extraidos de um documento."""
    doc_type: DocumentType = DocumentType.UNKNOWN
    confidence: float = 0.0
    
    # Identificacao
    recipient_name: Optional[str] = None      # Destinatario/Tomador
    supplier_name: Optional[str] = None       # Fornecedor/Emitente
    entity_tag: Optional[EntityTag] = None    # VG ou MV
    
    # Valores
    amount_cents: int = 0
    
    # Datas
    emission_date: Optional[str] = None
    due_date: Optional[str] = None
    payment_date: Optional[str] = None        # Data principal para comprovantes
    
    # Identificadores
    doc_number: Optional[str] = None
    access_key: Optional[str] = None          # Chave 44 digitos NF-e
    digitable_line: Optional[str] = None      # Linha digitavel boleto
    cpf: Optional[str] = None
    cnpj: Optional[str] = None
    
    # Metadados
    raw_text: str = ""
    is_scheduled: bool = False                # Agendamento vs pagamento
    needs_review: bool = False
    extraction_method: str = "native"         # native, ocr, manual
    
    # Parcelas (NF-e)
    installments: List[Dict] = field(default_factory=list)


# ==============================================================================
# CLASSE PRINCIPAL: ExtractionEngine
# ==============================================================================

class ExtractionEngine:
    """
    Motor de extracao cognitiva para documentos financeiros.
    
    Usa abordagem em camadas:
    1. Extracao nativa (PDF digital)
    2. OCR com pre-processamento (PDF escaneado)
    3. Validacao e normalizacao
    """
    
    def __init__(self, tesseract_cmd: Optional[str] = None):
        """
        Inicializa o motor de extracao.
        
        Args:
            tesseract_cmd: Caminho para o executavel do Tesseract (opcional)
        """
        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
    
    # ==========================================================================
    # EXTRACAO PRINCIPAL
    # ==========================================================================
    
    def extract_from_pdf(self, pdf_path: str, page_range: Tuple[int, int] = None) -> ExtractedData:
        """
        Extrai dados de um arquivo PDF.
        
        Args:
            pdf_path: Caminho do arquivo PDF
            page_range: Tupla (inicio, fim) para paginas especificas (1-indexed)
            
        Returns:
            ExtractedData com todos os campos extraidos
        """
        result = ExtractedData()
        
        try:
            doc = fitz.open(pdf_path)
        except Exception as e:
            result.needs_review = True
            result.confidence = 0.0
            return result
        
        try:
            # Determina paginas a processar
            if page_range:
                start, end = page_range
                pages = range(start - 1, min(end, len(doc)))
            else:
                pages = range(len(doc))
            
            # Extrai texto de todas as paginas
            full_text = ""
            for i in pages:
                page = doc[i]
                text = page.get_text()
                
                # Se texto muito curto, tenta OCR
                if len(text.strip()) < MIN_TEXT_LENGTH:
                    text = self._extract_with_ocr(page)
                    result.extraction_method = "ocr"
                
                full_text += text + "\n\n"
            
            result.raw_text = full_text
            
            # Classifica o documento
            result.doc_type, type_conf = self._classify_document(full_text)
            
            # Extrai dados especificos
            result.recipient_name = self._extract_recipient(full_text)
            result.supplier_name = self._extract_supplier(full_text)
            result.entity_tag = self._extract_entity(full_text)
            result.amount_cents = self._extract_amount(full_text)
            
            # Extrai datas (prioriza data de pagamento para comprovantes)
            result.payment_date = self._extract_date(full_text, "payment")
            result.due_date = self._extract_date(full_text, "due")
            result.emission_date = self._extract_date(full_text, "emission")
            
            # Extrai identificadores
            result.doc_number = self._extract_doc_number(full_text, result.doc_type)
            result.access_key = self._extract_access_key(full_text)
            result.digitable_line = self._extract_digitable_line(full_text)
            result.cpf, result.cnpj = self._extract_documents(full_text)
            
            # Detecta agendamento vs pagamento
            result.is_scheduled = self._detect_scheduling(full_text)
            
            # Extrai parcelas (duplicatas) de NF-e
            if result.doc_type in [DocumentType.NFE, DocumentType.NFSE]:
                result.installments = self._extract_installments(full_text)
            
            # Calcula confianca geral
            result.confidence = self._calculate_confidence(result)
            result.needs_review = result.confidence < MIN_CONFIDENCE
            
        finally:
            doc.close()
        
        return result
    
    # ==========================================================================
    # OCR COM PRE-PROCESSAMENTO
    # ==========================================================================
    
    def _extract_with_ocr(self, page: fitz.Page) -> str:
        """
        Extrai texto via OCR com pre-processamento de imagem.
        
        Args:
            page: Pagina do PDF (fitz.Page)
            
        Returns:
            Texto extraido via OCR
        """
        # Renderiza pagina como imagem
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x para melhor qualidade
        img_data = np.frombuffer(pix.samples, dtype=np.uint8)
        img = img_data.reshape(pix.height, pix.width, pix.n)
        
        # Converte para BGR (OpenCV)
        if pix.n == 4:  # RGBA
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        elif pix.n == 3:  # RGB
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        # Pre-processamento
        img = self._preprocess_image(img)
        
        # OCR com Tesseract
        try:
            text = pytesseract.image_to_string(img, lang='por', config='--psm 6')
            return text
        except Exception as e:
            print(f"[OCR] Erro: {e}")
            return ""
    
    def _preprocess_image(self, img: np.ndarray) -> np.ndarray:
        """
        Aplica pre-processamento para melhorar OCR.
        
        - Converte para escala de cinza
        - Binarizacao adaptativa
        - Remocao de ruido
        """
        # Escala de cinza
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Binarizacao adaptativa (lida com fundos de seguranca)
        binary = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11, 2
        )
        
        # Remove ruido
        denoised = cv2.fastNlMeansDenoising(binary, h=10)
        
        return denoised
    
    # ==========================================================================
    # CLASSIFICACAO DE DOCUMENTO
    # ==========================================================================
    
    def _classify_document(self, text: str) -> Tuple[DocumentType, float]:
        """Classifica o tipo de documento baseado em palavras-chave."""
        text_upper = text.upper()
        
        # Scores por tipo
        scores = {
            DocumentType.NFE: 0,
            DocumentType.NFSE: 0,
            DocumentType.BOLETO: 0,
            DocumentType.COMPROVANTE: 0,
        }
        
        # NFE keywords
        nfe_kw = ["DANFE", "NOTA FISCAL ELETRONICA", "CHAVE DE ACESSO", 
                  "NF-E", "PROTOCOLO DE AUTORIZACAO"]
        for kw in nfe_kw:
            if kw in text_upper:
                scores[DocumentType.NFE] += 2
        
        # NFSE keywords
        nfse_kw = ["NOTA FISCAL DE SERVICO", "NFS-E", "PREFEITURA MUNICIPAL",
                   "IMPOSTO SOBRE SERVICOS", "ISS"]
        for kw in nfse_kw:
            if kw in text_upper:
                scores[DocumentType.NFSE] += 2
        
        # Boleto keywords
        boleto_kw = ["LINHA DIGITAVEL", "FICHA DE COMPENSACAO", "CEDENTE",
                     "SACADO", "NOSSO NUMERO", "CODIGO DE BARRAS"]
        for kw in boleto_kw:
            if kw in text_upper:
                scores[DocumentType.BOLETO] += 2
        
        # Comprovante keywords
        comp_kw = ["COMPROVANTE", "PAGAMENTO EFETUADO", "TRANSACAO REALIZADA",
                   "AUTENTICACAO", "PIX ENVIADO", "TED", "DOC"]
        for kw in comp_kw:
            if kw in text_upper:
                scores[DocumentType.COMPROVANTE] += 2
        
        # Encontra o melhor
        best_type = max(scores, key=scores.get)
        best_score = scores[best_type]
        
        if best_score == 0:
            return DocumentType.UNKNOWN, 0.0
        
        confidence = min(best_score / 6.0, 1.0)
        return best_type, confidence
    
    # ==========================================================================
    # EXTRACAO DE DESTINATARIO (CORRIGIDA)
    # ==========================================================================
    
    def _extract_recipient(self, text: str) -> Optional[str]:
        """
        Extrai o destinatario/tomador do documento.
        
        NAO confunde com:
        - Nomes de bancos
        - Nomes de cidades
        - Nomes de fornecedores
        """
        text_upper = text.upper()
        
        # Busca por ancoras
        for anchor in RECIPIENT_ANCHORS:
            pattern = re.compile(
                rf'{anchor}[:\s]*\n?\s*([A-Z][A-Z\s\.\-]+)',
                re.IGNORECASE | re.MULTILINE
            )
            match = pattern.search(text_upper)
            if match:
                name = match.group(1).strip()
                
                # Valida que nao e banco ou cidade
                if not self._is_bank_name(name) and not self._is_city_name(name):
                    # Limpa e normaliza
                    name = self._normalize_name(name)
                    if len(name) > 3:
                        return name
        
        # Fallback: busca por CPF/CNPJ e pega nome proximo
        cpf_pattern = re.compile(r'(\d{3}\.\d{3}\.\d{3}-\d{2})')
        match = cpf_pattern.search(text)
        if match:
            # Busca nome antes ou depois do CPF
            pos = match.start()
            context = text[max(0, pos-100):pos+100]
            lines = context.split('\n')
            for line in lines:
                line = line.strip().upper()
                if len(line) > 5 and not self._is_bank_name(line) and not self._is_city_name(line):
                    if not re.search(r'\d{3}\.\d{3}', line):  # Nao e CPF/CNPJ
                        return self._normalize_name(line)
        
        return None
    
    def _extract_supplier(self, text: str) -> Optional[str]:
        """Extrai o fornecedor/emitente do documento."""
        text_upper = text.upper()
        
        for anchor in SUPPLIER_ANCHORS:
            pattern = re.compile(
                rf'{anchor}[:\s]*\n?\s*([A-Z][A-Z\s\.\-]+)',
                re.IGNORECASE | re.MULTILINE
            )
            match = pattern.search(text_upper)
            if match:
                name = match.group(1).strip()
                if not self._is_bank_name(name) and not self._is_city_name(name):
                    return self._normalize_name(name)
        
        return None
    
    def _is_bank_name(self, name: str) -> bool:
        """Verifica se o nome e de um banco."""
        name_upper = name.upper()
        for bank in BANK_NAMES:
            if bank in name_upper or fuzz.ratio(name_upper, bank) > 80:
                return True
        return False
    
    def _is_city_name(self, name: str) -> bool:
        """Verifica se o nome e uma cidade (Cidade - UF)."""
        return bool(CITY_PATTERN.search(name))
    
    def _normalize_name(self, name: str) -> str:
        """Normaliza nome removendo sufixos juridicos."""
        suffixes = [" LTDA", " ME", " EPP", " EIRELI", " S.A.", " S/A", " SA"]
        name = name.strip().upper()
        for suffix in suffixes:
            if name.endswith(suffix):
                name = name[:-len(suffix)].strip()
        # Remove pontuacao extra
        name = re.sub(r'[\.\,\-]+$', '', name)
        return name
    
    # ==========================================================================
    # EXTRACAO DE VALORES
    # ==========================================================================
    
    def _extract_amount(self, text: str) -> int:
        """
        Extrai valor monetario do documento.
        
        Prioriza:
        1. "VALOR TOTAL" ou "VALOR DO DOCUMENTO"
        2. Maior valor encontrado (provavelmente o total)
        """
        for pattern in AMOUNT_PATTERNS:
            match = pattern.search(text)
            if match:
                value_str = match.group(1)
                cents = self._parse_brazilian_currency(value_str)
                if cents > 0:
                    return cents
        
        # Fallback: encontra todos os valores e pega o maior
        all_values = []
        for pattern in AMOUNT_PATTERNS:
            matches = pattern.findall(text)
            for m in matches:
                cents = self._parse_brazilian_currency(m)
                if cents > 0:
                    all_values.append(cents)
        
        return max(all_values) if all_values else 0
    
    def _parse_brazilian_currency(self, value_str: str) -> int:
        """Converte string de moeda brasileira para centavos."""
        try:
            # Remove pontos de milhar e troca virgula por ponto
            clean = value_str.replace(".", "").replace(",", ".")
            return int(float(clean) * 100)
        except:
            return 0
    
    # ==========================================================================
    # EXTRACAO DE DATAS
    # ==========================================================================
    
    def _extract_date(self, text: str, date_type: str = "generic") -> Optional[str]:
        """
        Extrai data do texto e normaliza para ISO8601.
        
        Args:
            text: Texto para busca
            date_type: Tipo de data (payment, due, emission, generic)
            
        Returns:
            Data no formato YYYY-MM-DD
        """
        pattern = DATE_PATTERNS.get(date_type, DATE_PATTERNS["generic"])
        match = pattern.search(text)
        
        if match:
            date_str = match.group(1)
            return self._normalize_date(date_str)
        
        return None
    
    def _normalize_date(self, date_str: str) -> Optional[str]:
        """Normaliza data para formato ISO8601."""
        parts = re.split(r'[/\.\-]', date_str)
        if len(parts) == 3:
            day, month, year = parts
            try:
                return f"{year}-{month.zfill(2)}-{day.zfill(2)}"
            except:
                pass
        return None
    
    # ==========================================================================
    # EXTRACAO DE IDENTIFICADORES
    # ==========================================================================
    
    def _extract_doc_number(self, text: str, doc_type: DocumentType) -> Optional[str]:
        """Extrai numero do documento."""
        patterns = {
            DocumentType.NFE: re.compile(r'(?:N[uU]MERO|Nro?\.?|N[°º])[:\s]*(\d{3,9})', re.IGNORECASE),
            DocumentType.NFSE: re.compile(r'(?:N[uU]MERO|Nro?\.?)[:\s]*(\d{3,9})', re.IGNORECASE),
            DocumentType.BOLETO: re.compile(r'NOSSO\s*N[uU]MERO[:\s]*([0-9\-\.\/]+)', re.IGNORECASE),
        }
        
        pattern = patterns.get(doc_type)
        if pattern:
            match = pattern.search(text)
            if match:
                return match.group(1).strip()
        
        return None
    
    def _extract_access_key(self, text: str) -> Optional[str]:
        """Extrai chave de acesso NF-e (44 digitos)."""
        # Chave com ou sem espacos
        pattern = re.compile(r'\b(\d{4}\s?\d{4}\s?\d{4}\s?\d{4}\s?\d{4}\s?\d{4}\s?\d{4}\s?\d{4}\s?\d{4}\s?\d{4}\s?\d{4})\b')
        match = pattern.search(text)
        if match:
            return match.group(1).replace(" ", "")
        return None
    
    def _extract_digitable_line(self, text: str) -> Optional[str]:
        """Extrai linha digitavel do boleto."""
        # Padrao: 5+5+5+6+5+6+1+14 digitos (com espacos/pontos opcionais)
        pattern = re.compile(r'(\d{5}[\.\s]?\d{5}[\.\s]?\d{5}[\.\s]?\d{6}[\.\s]?\d{5}[\.\s]?\d{6}[\.\s]?\d[\.\s]?\d{14})')
        match = pattern.search(text.replace(" ", "").replace(".", ""))
        if match:
            return match.group(1)
        return None
    
    def _extract_documents(self, text: str) -> Tuple[Optional[str], Optional[str]]:
        """Extrai CPF e CNPJ do documento."""
        cpf_pattern = re.compile(r'\b(\d{3}\.\d{3}\.\d{3}-\d{2})\b')
        cnpj_pattern = re.compile(r'\b(\d{2}\.\d{3}\.\d{3}/\d{4}-\d{2})\b')
        
        cpf_match = cpf_pattern.search(text)
        cnpj_match = cnpj_pattern.search(text)
        
        cpf = cpf_match.group(1) if cpf_match else None
        cnpj = cnpj_match.group(1) if cnpj_match else None
        
        return cpf, cnpj
    
    # ==========================================================================
    # EXTRACAO DE ENTIDADE (VG/MV)
    # ==========================================================================
    
    def _extract_entity(self, text: str) -> Optional[EntityTag]:
        """Identifica a entidade financeira (Vagner ou Marcelli)."""
        # CPF do Vagner
        if "964.128.440-15" in text or "96412844015" in text:
            return EntityTag.VG
        
        # Busca por nomes
        text_upper = text.upper()
        if "VAGNER" in text_upper:
            return EntityTag.VG
        if "MARCELLI" in text_upper:
            return EntityTag.MV
        
        return None
    
    # ==========================================================================
    # DETECCAO DE AGENDAMENTO
    # ==========================================================================
    
    def _detect_scheduling(self, text: str) -> bool:
        """Detecta se e agendamento (nao pagamento efetivo)."""
        scheduling_words = [
            "AGENDAMENTO", "AGENDADO", "PREVISTO", "PROGRAMADO",
            "DATA PROGRAMADA", "ORDEM FUTURA", "AGUARDANDO"
        ]
        text_upper = text.upper()
        for word in scheduling_words:
            if word in text_upper:
                return True
        return False
    
    # ==========================================================================
    # EXTRACAO DE PARCELAS
    # ==========================================================================
    
    def _extract_installments(self, text: str) -> List[Dict]:
        """Extrai tabela de duplicatas/parcelas de NF-e."""
        installments = []
        
        # Padrao: numero + data + valor
        pattern = re.compile(r'(\d{1,3})\s+(\d{2}/\d{2}/\d{4})\s+([\d\.]+,\d{2})')
        matches = pattern.findall(text)
        
        for seq, date_str, amount_str in matches:
            installments.append({
                "seq_num": int(seq),
                "due_date": self._normalize_date(date_str),
                "amount_cents": self._parse_brazilian_currency(amount_str)
            })
        
        return installments
    
    # ==========================================================================
    # CALCULO DE CONFIANCA
    # ==========================================================================
    
    def _calculate_confidence(self, data: ExtractedData) -> float:
        """Calcula confianca geral da extracao."""
        score = 0.0
        max_score = 0.0
        
        # Tipo identificado
        max_score += 0.2
        if data.doc_type != DocumentType.UNKNOWN:
            score += 0.2
        
        # Valor extraido
        max_score += 0.25
        if data.amount_cents > 0:
            score += 0.25
        
        # Data extraida
        max_score += 0.2
        if data.payment_date or data.due_date or data.emission_date:
            score += 0.2
        
        # Nome extraido (destinatario ou fornecedor)
        max_score += 0.2
        if data.recipient_name or data.supplier_name:
            score += 0.2
        
        # Entidade identificada
        max_score += 0.15
        if data.entity_tag:
            score += 0.15
        
        return score / max_score if max_score > 0 else 0.0


# ==============================================================================
# FUNCOES UTILITARIAS
# ==============================================================================

def test_extraction(pdf_path: str):
    """Testa a extracao em um arquivo PDF."""
    engine = ExtractionEngine()
    result = engine.extract_from_pdf(pdf_path)
    
    print("=" * 60)
    print(f"Arquivo: {Path(pdf_path).name}")
    print("=" * 60)
    print(f"Tipo: {result.doc_type.value}")
    print(f"Confianca: {result.confidence:.1%}")
    print(f"Metodo: {result.extraction_method}")
    print(f"Precisa revisao: {result.needs_review}")
    print("-" * 60)
    print(f"Destinatario: {result.recipient_name}")
    print(f"Fornecedor: {result.supplier_name}")
    print(f"Entidade: {result.entity_tag.value if result.entity_tag else '-'}")
    print(f"Valor: R$ {result.amount_cents / 100:.2f}")
    print(f"Data Pagamento: {result.payment_date}")
    print(f"Data Vencimento: {result.due_date}")
    print(f"Data Emissao: {result.emission_date}")
    print(f"Numero Doc: {result.doc_number}")
    print(f"Agendamento: {result.is_scheduled}")
    
    if result.installments:
        print(f"Parcelas: {len(result.installments)}")
        for inst in result.installments:
            print(f"  {inst['seq_num']}: {inst['due_date']} - R$ {inst['amount_cents']/100:.2f}")
    
    return result


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        test_extraction(sys.argv[1])
    else:
        print("Uso: python extraction_engine.py <arquivo.pdf>")
