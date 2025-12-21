"""
SRDA-Rural Cognitive Extractor
==============================
Motor de Extracao Cognitiva usando GLiNER (Zero-Shot NER)

Arquitetura de 3 camadas (Vol I, Secao 3.2):
1. GLiNER: Extracao semantica de entidades (FORNECEDOR vs BANCO vs MUNICIPIO)
2. Regex: Dados estruturados (valores, datas, chaves de acesso)
3. OCR: Fallback para PDFs escaneados (Tesseract com pre-processamento)

Este modulo substitui a abordagem "burra" baseada em listas estaticas.
"""

import re
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum

import fitz  # PyMuPDF
import cv2
import numpy as np
from gliner import GLiNER
from rapidfuzz import fuzz

# Configuracao de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ==============================================================================
# ENUMS E CONSTANTES
# ==============================================================================

class DocumentType(Enum):
    """Tipos de documentos financeiros."""
    NFE = "NFE"           # Nota Fiscal Eletronica (Modelo 55)
    NFSE = "NFSE"         # Nota Fiscal de Servico Eletronica
    BOLETO = "BOLETO"     # Boleto Bancario
    COMPROVANTE = "COMPROVANTE"  # Comprovante de Pagamento
    UNKNOWN = "UNKNOWN"


class PaymentStatus(Enum):
    """Status de pagamento (blindagem cognitiva)."""
    CONFIRMED = "CONFIRMED"    # Pagamento efetivo (tem SISBB ou sem keywords)
    SCHEDULED = "SCHEDULED"    # Agendamento (nao gera baixa)
    UNKNOWN = "UNKNOWN"


class EntityType(Enum):
    """Tipos de entidades extraidas via NER."""
    FORNECEDOR = "FORNECEDOR"  # Quem vende/presta servico
    BANCO = "BANCO"            # Instituicao financeira
    PESSOA = "PESSOA"          # Pessoa fisica (Vagner, Marcelli)
    MUNICIPIO = "MUNICIPIO"    # Cidade/Municipio


# CPFs conhecidos para classificacao automatica
KNOWN_CPFS = {
    "96412844015": "VG",
    "964.128.440-15": "VG",
}

# Bancos brasileiros para desambiguacao
BRAZILIAN_BANKS = [
    "BANCO DO BRASIL", "BB", "CAIXA ECONOMICA", "CEF", "CAIXA",
    "ITAU", "ITAU UNIBANCO", "BRADESCO", "SANTANDER", "SICOOB",
    "SICREDI", "CRESOL", "BANRISUL", "SAFRA", "BTG", "INTER",
    "NUBANK", "C6 BANK", "ORIGINAL", "PAN", "VOTORANTIM"
]


# ==============================================================================
# DATA CLASSES
# ==============================================================================

@dataclass
class ExtractedEntity:
    """Entidade extraida via NER."""
    text: str
    type: EntityType
    confidence: float
    start: int = 0
    end: int = 0


@dataclass
class ExtractedData:
    """Dados extraidos de um documento."""
    doc_type: DocumentType = DocumentType.UNKNOWN
    confidence: float = 0.0
    
    # Entidades identificadas semanticamente
    fornecedor: Optional[str] = None      # Quem vende (sem confundir com banco)
    destinatario: Optional[str] = None    # Quem compra (Vagner/Marcelli)
    banco: Optional[str] = None           # Instituicao financeira
    municipio: Optional[str] = None       # Cidade do documento
    entity_tag: Optional[str] = None      # VG ou MV
    
    # Valores
    amount_cents: int = 0
    
    # Datas
    emission_date: Optional[str] = None
    due_date: Optional[str] = None
    payment_date: Optional[str] = None
    
    # Identificadores
    doc_number: Optional[str] = None
    access_key: Optional[str] = None          # Chave 44 digitos NF-e
    digitable_line: Optional[str] = None      # Linha digitavel boleto
    sisbb_auth: Optional[str] = None          # Autenticacao SISBB
    
    # Status de pagamento (blindagem)
    payment_status: PaymentStatus = PaymentStatus.UNKNOWN
    is_scheduled: bool = False
    
    # Duplicatas (parcelas)
    installments: List[Dict] = field(default_factory=list)
    
    # Metadados
    raw_text: str = ""
    extraction_method: str = "gliner"
    needs_review: bool = False


# ==============================================================================
# CLASSE PRINCIPAL: CognitiveExtractor
# ==============================================================================

class CognitiveExtractor:
    """
    Motor de Extracao Cognitiva com GLiNER.
    
    Supera a abordagem de regex simples ao usar um modelo NER zero-shot
    que entende semanticamente a diferenca entre:
    - FORNECEDOR (quem vende)
    - BANCO (instituicao financeira)
    - PESSOA (Vagner/Marcelli)
    - MUNICIPIO (cidade)
    """
    
    # Labels para o GLiNER
    NER_LABELS = ["fornecedor", "banco", "pessoa", "municipio", "empresa"]
    
    # Padroes Regex para dados estruturados
    PATTERNS = {
        "amount": [
            re.compile(r'(=)\s*Valor\s*(?:do)?\s*Documento[\s\n]*R?\$?\s*([\d\.]+,\d{2})', re.IGNORECASE),
            re.compile(r'VALOR\s*(?:TOTAL|DOCUMENTO|LIQUIDO|COBRADO|PAGO|A\s*PAGAR)[\s:]*R?\$?\s*([\d\.]+,\d{2})', re.IGNORECASE),
            re.compile(r'\bTOTAL[\s:]+R?\$?\s*([\d\.]+,\d{2})', re.IGNORECASE),
            re.compile(r'VALOR\s*TOTAL\s*DA\s*NOTA[\s:]*R?\$?\s*([\d\.]+,\d{2})', re.IGNORECASE),
            re.compile(r'R\$\s*([\d\.]+,\d{2})'),  # Fallback genérico
        ],
        "date": {
            "payment": re.compile(r'(?:DATA\s*(?:DO)?\s*PAGAMENTO|PAGO\s*EM|PAGAMENTO\s*REALIZADO)[\s:]*(\\d{2}[/\\.\\-]\\d{2}[/\\.\\-]\\d{4})', re.IGNORECASE),
            "due": [
                re.compile(r'Vencimento[\s\n]*(\\d{2}/\\d{2}/\\d{4})', re.IGNORECASE),
                re.compile(r'(?:VENCIMENTO|VENC\\.)[\s:]*(\\d{2}[/\\.\\-]\\d{2}[/\\.\\-]\\d{4})', re.IGNORECASE),
            ],
            "emission": [
                re.compile(r'Emiss[aã]o[\s:]*(\\d{2}/\\d{2}/\\d{4})', re.IGNORECASE),
                re.compile(r'DATA\s*DA\s*EMISS[AÃ]O[\s\n]*(\\d{2}/\\d{2}/\\d{4})', re.IGNORECASE),
            ],
        },
        "beneficiario": re.compile(r'Benefici[aá]rio[\s\n]+(.+?)(?:\n|\r|CNPJ|CPF|$)', re.IGNORECASE),
        "prestador": [
            re.compile(r'Recebemos\s+de\s+(.+?)\s+os\s+produtos', re.IGNORECASE),
            re.compile(r'Raz[aã]o\s*Social[\s\n:]+(.+?)(?:\n|\r|CNPJ|$)', re.IGNORECASE),
        ],
        "access_key": re.compile(r'(\d{4}\s?\d{4}\s?\d{4}\s?\d{4}\s?\d{4}\s?\d{4}\s?\d{4}\s?\d{4}\s?\d{4}\s?\d{4}\s?\d{4})'),
        "cpf": re.compile(r'(\d{3}\.\d{3}\.\d{3}-\d{2})'),
        "cnpj": re.compile(r'(\d{2}\.\d{3}\.\d{3}/\d{4}-\d{2})'),
        "sisbb": re.compile(r'(\d+\.[A-F0-9]{2,4}\.[A-F0-9]{2,4}\.[\w\-]+)', re.IGNORECASE),
        "doc_number": re.compile(r'(?:NUMERO|Nro?\.?|N[°º])[\s:]*(\\d{3,9})', re.IGNORECASE),
        "installment_row": re.compile(r'(\d{1,3})\s+(\d{2}/\d{2}/\d{4})\s+([\d\.]+,\d{2})'),
        "digitable_line": re.compile(r'(\d{5}\.\d{5}\s*\d{5}\.\d{6}\s*\d{5}\.\d{6}\s*\d\s*\d{14})'),
    }
    
    # Keywords para classificacao de documentos (ordem importa - mais especifico primeiro)
    DOC_KEYWORDS = {
        DocumentType.BOLETO: ["FICHA DE COMPENSACAO", "FICHA DE CAIXA", "NOSSO NUMERO", "CEDENTE", "SACADO", "LINHA DIGITAVEL", "CODIGO DE BARRAS", "BANCO", "ESPECIE MOEDA"],
        DocumentType.NFE: ["DANFE", "NOTA FISCAL ELETRONICA", "NF-E", "CHAVE DE ACESSO", "PROTOCOLO DE AUTORIZACAO", "NCM"],
        DocumentType.NFSE: ["NOTA FISCAL DE SERVICO", "NFS-E", "PREFEITURA MUNICIPAL", "ISS", "IMPOSTO SOBRE SERVICOS", "PRESTADOR"],
        DocumentType.COMPROVANTE: ["COMPROVANTE DE PAGAMENTO", "PAGAMENTO EFETUADO", "TRANSACAO REALIZADA", "AUTENTICACAO MECANICA", "PIX ENVIADO", "TED ENVIADO", "DOC ENVIADO"],
    }
    
    # Keywords de agendamento (blindagem cognitiva)
    SCHEDULING_KEYWORDS = ["AGENDAMENTO", "AGENDADO", "PREVISTO", "PROGRAMADO", "DATA PROGRAMADA", "ORDEM FUTURA"]
    
    def __init__(self, model_name: str = "urchade/gliner_small-v2.1"):
        """
        Inicializa o extrator cognitivo.
        
        Args:
            model_name: Nome do modelo GLiNER no HuggingFace
        """
        self.model_name = model_name
        self._model: Optional[GLiNER] = None
        self._model_loaded = False
    
    @property
    def model(self) -> GLiNER:
        """Carrega o modelo GLiNER sob demanda (lazy loading)."""
        if self._model is None:
            try:
                logger.info(f"Carregando modelo GLiNER: {self.model_name}")
                self._model = GLiNER.from_pretrained(self.model_name)
                self._model_loaded = True
                logger.info("Modelo GLiNER carregado com sucesso")
            except Exception as e:
                logger.error(f"Erro ao carregar GLiNER: {e}")
                raise
        return self._model
    
    # ==========================================================================
    # EXTRACAO PRINCIPAL
    # ==========================================================================
    
    def extract_from_pdf(
        self, 
        pdf_path: str, 
        page_range: Optional[Tuple[int, int]] = None
    ) -> ExtractedData:
        """
        Extrai dados de um arquivo PDF usando abordagem cognitiva.
        
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
            logger.error(f"Erro ao abrir PDF: {e}")
            result.needs_review = True
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
                if len(text.strip()) < 50:
                    text = self._extract_with_ocr(page)
                    result.extraction_method = "ocr"
                
                full_text += text + "\n\n"
            
            result.raw_text = full_text
            
            # 1. Classifica o tipo de documento
            result.doc_type = self._classify_document(full_text)
            
            # 2. Extrai fornecedor via Regex especifico (mais confiavel que GLiNER para BR)
            result.fornecedor = self._extract_supplier(full_text, result.doc_type)
            
            # 3. Se nao encontrou, tenta via GLiNER
            if not result.fornecedor:
                entities = self._extract_entities_gliner(full_text)
                result.fornecedor = self._get_entity_by_type(entities, EntityType.FORNECEDOR)
                result.banco = self._get_entity_by_type(entities, EntityType.BANCO)
                result.destinatario = self._get_entity_by_type(entities, EntityType.PESSOA)
                result.municipio = self._get_entity_by_type(entities, EntityType.MUNICIPIO)
            
            # 4. Identifica entidade financeira (VG/MV)
            result.entity_tag = self._identify_entity_tag(full_text)
            
            # 5. Extrai dados estruturados via Regex
            result.amount_cents = self._extract_amount(full_text, result.doc_type)
            result.payment_date = self._extract_date(full_text, "payment")
            result.due_date = self._extract_date(full_text, "due")
            result.emission_date = self._extract_date(full_text, "emission")
            result.access_key = self._extract_access_key(full_text)
            result.doc_number = self._extract_doc_number(full_text)
            
            # 5. Validacao de pagamento (blindagem cognitiva)
            result.sisbb_auth = self._extract_sisbb(full_text)
            result.payment_status = self._validate_payment(full_text, result.banco)
            result.is_scheduled = (result.payment_status == PaymentStatus.SCHEDULED)
            
            # 6. Extrai duplicatas (parcelas)
            if result.doc_type in [DocumentType.NFE, DocumentType.NFSE]:
                result.installments = self._extract_installments(full_text)
            
            # 7. Calcula confianca geral
            result.confidence = self._calculate_confidence(result)
            result.needs_review = result.confidence < 0.7
            
        except Exception as e:
            logger.error(f"Erro na extracao: {e}")
            result.needs_review = True
            
        finally:
            doc.close()
        
        return result
    
    # ==========================================================================
    # EXTRACAO VIA GLiNER (NER SEMANTICO)
    # ==========================================================================
    
    def _extract_entities_gliner(self, text: str) -> List[ExtractedEntity]:
        """
        Extrai entidades usando GLiNER (zero-shot NER).
        
        Diferente de regex, o GLiNER entende o CONTEXTO semântico:
        - "Banco do Brasil" → BANCO (nao FORNECEDOR)
        - "Nova Mutum - MT" → MUNICIPIO (nao PESSOA)
        - "VAGNER LUIZ GAIATTO" → PESSOA
        """
        entities = []
        
        try:
            # Limita texto para performance (primeiros 3000 chars)
            text_chunk = text[:3000]
            
            # Executa NER
            predictions = self.model.predict_entities(
                text_chunk,
                self.NER_LABELS,
                threshold=0.4
            )
            
            for pred in predictions:
                entity_type = self._map_label_to_type(pred["label"])
                
                # Desambiguacao: verifica se o texto e um banco conhecido
                entity_text = pred["text"].strip().upper()
                if self._is_known_bank(entity_text):
                    entity_type = EntityType.BANCO
                
                entities.append(ExtractedEntity(
                    text=pred["text"],
                    type=entity_type,
                    confidence=pred["score"],
                    start=pred.get("start", 0),
                    end=pred.get("end", 0)
                ))
            
            # Ordena por confianca
            entities.sort(key=lambda x: x.confidence, reverse=True)
            
        except Exception as e:
            logger.warning(f"Erro no GLiNER, usando fallback: {e}")
            # Fallback: tenta extrair via regex
            entities = self._fallback_entity_extraction(text)
        
        return entities
    
    def _map_label_to_type(self, label: str) -> EntityType:
        """Mapeia label do GLiNER para EntityType."""
        mapping = {
            "fornecedor": EntityType.FORNECEDOR,
            "empresa": EntityType.FORNECEDOR,
            "banco": EntityType.BANCO,
            "pessoa": EntityType.PESSOA,
            "municipio": EntityType.MUNICIPIO,
        }
        return mapping.get(label.lower(), EntityType.FORNECEDOR)
    
    def _is_known_bank(self, text: str) -> bool:
        """Verifica se o texto e um banco conhecido."""
        for bank in BRAZILIAN_BANKS:
            if bank in text or fuzz.ratio(text, bank) > 80:
                return True
        return False
    
    def _get_entity_by_type(
        self, 
        entities: List[ExtractedEntity], 
        entity_type: EntityType
    ) -> Optional[str]:
        """Retorna a entidade de maior confianca de um tipo."""
        for entity in entities:
            if entity.type == entity_type:
                return self._normalize_name(entity.text)
        return None
    
    def _fallback_entity_extraction(self, text: str) -> List[ExtractedEntity]:
        """Fallback para extracao de entidades quando GLiNER falha."""
        entities = []
        
        # Busca por padroes de nome (uppercase com mais de 3 palavras)
        pattern = re.compile(r'\b([A-Z][A-Z\s\.]+(?:\s[A-Z][A-Z\s\.]+){2,})\b')
        matches = pattern.findall(text)
        
        for match in matches[:5]:  # Limita a 5 matches
            text_clean = match.strip()
            
            # Determina tipo
            if self._is_known_bank(text_clean):
                entity_type = EntityType.BANCO
            elif re.search(r'\b[A-Z][a-z]+\s*-\s*[A-Z]{2}\b', text_clean):
                entity_type = EntityType.MUNICIPIO
            else:
                entity_type = EntityType.FORNECEDOR
            
            entities.append(ExtractedEntity(
                text=text_clean,
                type=entity_type,
                confidence=0.5
            ))
        
        return entities
    
    # ==========================================================================
    # VALIDACAO DE PAGAMENTO (BLINDAGEM COGNITIVA)
    # ==========================================================================
    
    def _validate_payment(self, text: str, banco: Optional[str]) -> PaymentStatus:
        """
        Valida se o documento e pagamento efetivo ou agendamento.
        
        REGRA ABSOLUTA (Vol II, Secao 2.1.3):
        Para Banco do Brasil, a presenca do hash SISBB e OBRIGATORIA.
        Sem SISBB = Agendamento, independente de outras palavras.
        """
        text_upper = text.upper()
        
        # Regra especifica para Banco do Brasil
        if banco and "BRASIL" in banco.upper():
            sisbb = self._extract_sisbb(text)
            if sisbb:
                logger.info(f"SISBB encontrado: {sisbb} - Pagamento CONFIRMADO")
                return PaymentStatus.CONFIRMED
            else:
                logger.warning("Banco do Brasil sem SISBB - Classificado como AGENDAMENTO")
                return PaymentStatus.SCHEDULED
        
        # Para outros bancos: verifica keywords de agendamento
        for keyword in self.SCHEDULING_KEYWORDS:
            if keyword in text_upper:
                logger.info(f"Keyword de agendamento encontrado: {keyword}")
                return PaymentStatus.SCHEDULED
        
        return PaymentStatus.CONFIRMED
    
    def _extract_sisbb(self, text: str) -> Optional[str]:
        """Extrai autenticacao SISBB (Banco do Brasil)."""
        match = self.PATTERNS["sisbb"].search(text)
        return match.group(1) if match else None
    
    # ==========================================================================
    # EXTRACAO DE DADOS ESTRUTURADOS (REGEX)
    # ==========================================================================
    
    def _classify_document(self, text: str) -> DocumentType:
        """Classifica o tipo de documento baseado em keywords."""
        text_upper = text.upper()
        
        scores = {doc_type: 0 for doc_type in DocumentType if doc_type != DocumentType.UNKNOWN}
        
        # Ordem de verificacao importa - BOLETO tem prioridade sobre COMPROVANTE
        for doc_type, keywords in self.DOC_KEYWORDS.items():
            for kw in keywords:
                if kw in text_upper:
                    # Peso maior para keywords especificos
                    weight = 3 if kw in ["FICHA DE COMPENSACAO", "NOSSO NUMERO", "DANFE", "NFS-E"] else 2
                    scores[doc_type] += weight
        
        # Penaliza COMPROVANTE se tem keywords de BOLETO
        if scores[DocumentType.BOLETO] > 0 and scores[DocumentType.COMPROVANTE] > 0:
            scores[DocumentType.COMPROVANTE] = 0
        
        best_type = max(scores, key=scores.get)
        return best_type if scores[best_type] > 0 else DocumentType.UNKNOWN
    
    def _extract_supplier(self, text: str, doc_type: DocumentType) -> Optional[str]:
        """Extrai fornecedor com regex especifico por tipo de documento."""
        # Para BOLETO: Beneficiario e o fornecedor
        if doc_type == DocumentType.BOLETO:
            match = self.PATTERNS["beneficiario"].search(text)
            if match:
                supplier = match.group(1).strip()
                return self._normalize_name(supplier)
        
        # Para NFE/NFSE: Tenta multiplos padroes
        if doc_type in [DocumentType.NFE, DocumentType.NFSE]:
            patterns = self.PATTERNS["prestador"]
            for pattern in patterns:
                match = pattern.search(text)
                if match:
                    supplier = match.group(1).strip()
                    return self._normalize_name(supplier)
        
        return None
    
    def _extract_amount(self, text: str, doc_type: DocumentType = None) -> int:
        """Extrai valor monetario e converte para centavos."""
        for pattern in self.PATTERNS["amount"]:
            match = pattern.search(text)
            if match:
                value = self._parse_brazilian_currency(match.group(1))
                if value > 0:
                    return value
        return 0
    
    def _extract_date(self, text: str, date_type: str) -> Optional[str]:
        """Extrai data e normaliza para ISO8601."""
        pattern_or_list = self.PATTERNS["date"].get(date_type)
        
        if pattern_or_list is None:
            return None
        
        # Suporta lista de padroes ou padrao unico
        patterns = pattern_or_list if isinstance(pattern_or_list, list) else [pattern_or_list]
        
        for pattern in patterns:
            match = pattern.search(text)
            if match:
                return self._normalize_date(match.group(1))
        return None
    
    def _extract_access_key(self, text: str) -> Optional[str]:
        """Extrai chave de acesso NF-e (44 digitos)."""
        match = self.PATTERNS["access_key"].search(text)
        if match:
            return match.group(1).replace(" ", "")
        return None
    
    def _extract_doc_number(self, text: str) -> Optional[str]:
        """Extrai numero do documento."""
        match = self.PATTERNS["doc_number"].search(text)
        if match:
            # Remove zeros a esquerda
            return str(int(match.group(1)))
        return None
    
    def _extract_installments(self, text: str) -> List[Dict]:
        """Extrai tabela de duplicatas/parcelas."""
        installments = []
        matches = self.PATTERNS["installment_row"].findall(text)
        
        for seq, date_str, amount_str in matches:
            installments.append({
                "seq_num": int(seq),
                "due_date": self._normalize_date(date_str),
                "amount_cents": self._parse_brazilian_currency(amount_str)
            })
        
        return installments
    
    def _identify_entity_tag(self, text: str) -> Optional[str]:
        """Identifica se o documento pertence a VG ou MV."""
        # Primeiro: busca por CPF
        cpf_match = self.PATTERNS["cpf"].search(text)
        if cpf_match:
            cpf = cpf_match.group(1)
            cpf_clean = cpf.replace(".", "").replace("-", "")
            if cpf_clean in KNOWN_CPFS or cpf in KNOWN_CPFS:
                return "VG"
        
        # Segundo: busca por nome
        text_upper = text.upper()
        if "VAGNER" in text_upper:
            return "VG"
        if "MARCELLI" in text_upper:
            return "MV"
        
        return None
    
    # ==========================================================================
    # OCR COM PRE-PROCESSAMENTO
    # ==========================================================================
    
    def _extract_with_ocr(self, page: fitz.Page) -> str:
        """Extrai texto via OCR com pre-processamento."""
        try:
            import pytesseract
            
            # Renderiza pagina como imagem
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
            img_data = np.frombuffer(pix.samples, dtype=np.uint8)
            img = img_data.reshape(pix.height, pix.width, pix.n)
            
            # Converte para BGR
            if pix.n == 4:
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
            elif pix.n == 3:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            
            # Pre-processamento
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
            denoised = cv2.fastNlMeansDenoising(binary, h=10)
            
            # OCR
            text = pytesseract.image_to_string(denoised, lang='por', config='--psm 6')
            return text
            
        except Exception as e:
            logger.error(f"Erro no OCR: {e}")
            return ""
    
    # ==========================================================================
    # UTILITARIOS
    # ==========================================================================
    
    def _normalize_name(self, name: str) -> str:
        """Normaliza nome removendo sufixos juridicos e limpando."""
        # Remove quebras de linha e espacos extras
        name = re.sub(r'[\n\r]+', ' ', name)
        name = re.sub(r'\s+', ' ', name).strip().upper()
        
        # Corta em palavras-chave comuns que indicam fim do nome
        stop_words = ['CNPJ', 'CPF', 'PAGADOR', 'CEP', 'ENDERECO', 'VENCIMENTO', 'BENEFICI', 
                      'AG', 'AGENCIA', 'FL.', 'PAG', 'SITE', 'SEFAZ', 'TELEFONE', 'FONE',
                      'LOCAL', 'DATA', 'OU NO']
        for stop in stop_words:
            if stop in name:
                name = name.split(stop)[0].strip()
        
        # Remove sufixos juridicos
        suffixes = [" LTDA", " ME", " EPP", " EIRELI", " S.A.", " S/A", " SA", " LTDA."]
        for suffix in suffixes:
            if name.endswith(suffix):
                name = name[:-len(suffix)].strip()
        
        # Remove caracteres finais invalidos
        name = re.sub(r'[\.\,\-\:]+$', '', name).strip()
        
        return name[:50]  # Limita tamanho
    
    def _normalize_date(self, date_str: str) -> Optional[str]:
        """Normaliza data para formato ISO8601."""
        parts = re.split(r'[/\.\-]', date_str)
        if len(parts) == 3:
            day, month, year = parts
            if len(year) == 2:
                year = "20" + year
            return f"{year}-{month.zfill(2)}-{day.zfill(2)}"
        return None
    
    def _parse_brazilian_currency(self, value_str: str) -> int:
        """Converte string de moeda brasileira para centavos."""
        try:
            clean = value_str.replace(".", "").replace(",", ".")
            return int(float(clean) * 100)
        except:
            return 0
    
    def _calculate_confidence(self, data: ExtractedData) -> float:
        """Calcula confianca geral da extracao."""
        score = 0.0
        
        # Tipo identificado
        if data.doc_type != DocumentType.UNKNOWN:
            score += 0.2
        
        # Valor extraido
        if data.amount_cents > 0:
            score += 0.25
        
        # Data extraida
        if data.payment_date or data.due_date or data.emission_date:
            score += 0.2
        
        # Entidade identificada (fornecedor ou destinatario)
        if data.fornecedor or data.destinatario:
            score += 0.2
        
        # Tag de entidade (VG/MV)
        if data.entity_tag:
            score += 0.15
        
        return score


# ==============================================================================
# TESTE
# ==============================================================================

def test_cognitive_extraction(pdf_path: str):
    """Testa a extracao cognitiva em um arquivo PDF."""
    extractor = CognitiveExtractor()
    result = extractor.extract_from_pdf(pdf_path)
    
    print("=" * 60)
    print(f"Arquivo: {Path(pdf_path).name}")
    print("=" * 60)
    print(f"Tipo: {result.doc_type.value}")
    print(f"Confianca: {result.confidence:.1%}")
    print(f"Metodo: {result.extraction_method}")
    print("-" * 60)
    print(f"Fornecedor (GLiNER): {result.fornecedor}")
    print(f"Banco (GLiNER): {result.banco}")
    print(f"Destinatario (GLiNER): {result.destinatario}")
    print(f"Municipio (GLiNER): {result.municipio}")
    print(f"Entidade: {result.entity_tag}")
    print(f"Valor: R$ {result.amount_cents / 100:.2f}")
    print(f"Data Pagamento: {result.payment_date}")
    print(f"Data Vencimento: {result.due_date}")
    print(f"SISBB: {result.sisbb_auth}")
    print(f"Status Pagamento: {result.payment_status.value}")
    print(f"Agendamento: {result.is_scheduled}")
    
    if result.installments:
        print(f"Parcelas: {len(result.installments)}")
        for inst in result.installments:
            print(f"  {inst['seq_num']}: {inst['due_date']} - R$ {inst['amount_cents']/100:.2f}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        test_cognitive_extraction(sys.argv[1])
    else:
        print("Uso: python cognitive_extractor.py <arquivo.pdf>")
