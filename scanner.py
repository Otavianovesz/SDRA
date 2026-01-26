"""
SRDA-Rural Scanner Module
=========================
Camada de Ingestao e Percepcao (The Scanner)

Este modulo atua como os "olhos" do sistema, responsavel por:
- Varredura de diretorios
- Calculo de integridade (hashing MD5)
- Segmentacao visual de arquivos PDF combinados
- Extracao de texto bruto via PyMuPDF
- Classificacao de tipo de documento (NFE, NFSE, BOLETO, COMPROVANTE)

Referencia: Automacao e Reconciliacao Financeira com IA.txt (Secao 3)
"""

import fitz  # PyMuPDF
import re
import os
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple, Generator
from dataclasses import dataclass
from enum import Enum

import config
from resource_manager import get_resource_manager

# Importa o modulo de banco de dados
from database import (
    SRDADatabase,
    DocumentType,
    DocumentStatus,
    EntityTag,
    KNOWN_CPFS
)

# Importa o EnsembleExtractor para extracao avancada (OCR, GLiNER, etc)
try:
    from ensemble_extractor import EnsembleExtractor, ExtractionResult
    ENSEMBLE_AVAILABLE = True
except ImportError:
    EnsembleExtractor = None
    ENSEMBLE_AVAILABLE = False
    print("[AVISO] EnsembleExtractor nao disponivel, usando extracao basica")

# Importa barcode extractor para boletos (optional)
try:
    from barcode_extractor import get_barcode_extractor
    BARCODE_AVAILABLE = True
except ImportError:
    get_barcode_extractor = None
    BARCODE_AVAILABLE = False


# ==============================================================================
# CONFIGURACOES E CONSTANTES
# ==============================================================================

# Pasta padrao de entrada
DEFAULT_INPUT_FOLDER = "Input"

# Palavras-chave para classificacao de documentos
KEYWORDS = {
    DocumentType.NFE: [
        "DANFE",
        "DOCUMENTO AUXILIAR DA NOTA FISCAL",
        "NF-e",
        "NFe",
        "NOTA FISCAL ELETRONICA",
        "CHAVE DE ACESSO",
        "Protocolo de Autorizacao",
    ],
    DocumentType.NFSE: [
        "NOTA FISCAL DE SERVICO",
        "NFS-e",
        "NFSe",
        "NOTA FISCAL ELETRONICA DE SERVICOS",
        "PREFEITURA MUNICIPAL",
        "IMPOSTO SOBRE SERVICOS",
        "ISS",
    ],
    DocumentType.BOLETO: [
        "LINHA DIGITAVEL",
        "FICHA DE COMPENSACAO",
        "AUTENTICACAO MECANICA",
        "CODIGO DE BARRAS",
        "SACADO",
        "CEDENTE",
        "VENCIMENTO",
        "ATE O VENCIMENTO",
        "APOS O VENCIMENTO",
        "INSTRUCOES",
    ],
    DocumentType.COMPROVANTE: [
        "COMPROVANTE DE PAGAMENTO",
        "COMPROVANTE DE TRANSFERENCIA",
        "AUTENTICACAO SISBB",
        "TRANSACAO EFETUADA",
        "PAGAMENTO EFETUADO",
        "PAGAMENTO REALIZADO",
        "RECIBO DE PAGAMENTO",
        "PIX ENVIADO",
        "TED ENVIADA",
        "DOC ENVIADO",
    ],
}

# Palavras-chave NEGATIVAS que indicam agendamento (nao pagamento efetivo)
SCHEDULING_KEYWORDS = [
    "AGENDAMENTO",
    "AGENDADO",
    "PREVISTO",
    "DATA PROGRAMADA",
    "PAGAMENTO PROGRAMADO",
    "ORDEM FUTURA",
]

# Regex para extracao de dados
REGEX_PATTERNS = {
    # Valor monetario brasileiro (ex: R$ 3.300,00 ou 1.740,00)
    "amount": re.compile(
        r'(?:R\$|Total|Valor|VALOR)\s*[:\.]?\s*([\d\.]+,\d{2})',
        re.IGNORECASE
    ),
    
    # Valor alternativo sem prefixo (busca valores grandes)
    "amount_plain": re.compile(
        r'\b(\d{1,3}(?:\.\d{3})*,\d{2})\b'
    ),
    
    # Data brasileira (DD/MM/AAAA ou DD.MM.AAAA)
    "date_br": re.compile(
        r'\b(\d{2})[/\.\-](\d{2})[/\.\-](\d{4})\b'
    ),
    
    # Data vencimento especifica
    "due_date": re.compile(
        r'(?:Vencimento|Venc\.|DATA DE VENCIMENTO)\s*[:\s]*(\d{2}[/\.\-]\d{2}[/\.\-]\d{4})',
        re.IGNORECASE
    ),
    
    # Data emissao
    "emission_date": re.compile(
        r'(?:Emiss.o|Data da Emiss.o|DATA DE EMISSAO)\s*[:\s]*(\d{2}[/\.\-]\d{2}[/\.\-]\d{4})',
        re.IGNORECASE
    ),
    
    # Chave de acesso NF-e (44 digitos)
    "access_key": re.compile(
        r'\b(\d{4}\s?\d{4}\s?\d{4}\s?\d{4}\s?\d{4}\s?\d{4}\s?\d{4}\s?\d{4}\s?\d{4}\s?\d{4}\s?\d{4})\b'
    ),
    
    # CPF (com ou sem formatacao)
    "cpf": re.compile(
        r'\b(\d{3}\.?\d{3}\.?\d{3}\-?\d{2})\b'
    ),
    
    # CNPJ
    "cnpj": re.compile(
        r'\b(\d{2}\.?\d{3}\.?\d{3}\/?\d{4}\-?\d{2})\b'
    ),
    
    # Linha digitavel do boleto (47 digitos com espacos)
    "digitable_line": re.compile(
        r'\b(\d{5}[\.\s]?\d{5}[\.\s]?\d{5}[\.\s]?\d{6}[\.\s]?\d{5}[\.\s]?\d{6}[\.\s]?\d{1}[\.\s]?\d{14})\b'
    ),
    
    # Autenticacao SISBB (Banco do Brasil)
    "sisbb_auth": re.compile(
        r'(?:AUTENTICACAO|Autentica..o)\s*[:\s]*([A-Z0-9\.]+)',
        re.IGNORECASE
    ),
    
    # Numero da nota fiscal
    "nf_number": re.compile(
        r'(?:N.?\s*(?:mero)?|NF|Nota\s*Fiscal)[\s:]*(\d{3,9})',
        re.IGNORECASE
    ),
    
    # Tabela de duplicatas (parcelas)
    "duplicata": re.compile(
        r'(\d{3})\s+(\d{2}/\d{2}/\d{4})\s+([\d\.]+,\d{2})',
    ),
}


# ==============================================================================
# DATA CLASSES
# ==============================================================================

@dataclass
class PageInfo:
    """Informacoes extraidas de uma pagina."""
    page_num: int
    doc_type: DocumentType
    text: str
    confidence: float
    
    # Dados extraidos
    amount_cents: int = 0
    due_date: Optional[str] = None
    emission_date: Optional[str] = None
    access_key: Optional[str] = None
    entity_tag: Optional[EntityTag] = None
    is_scheduled: bool = False


@dataclass
class DocumentSegment:
    """Representa um segmento logico de um PDF (pode ser parte de arquivo combinado)."""
    file_path: str
    page_start: int
    page_end: int
    doc_type: DocumentType
    pages_info: List[PageInfo]
    
    @property
    def text(self) -> str:
        """Retorna o texto concatenado de todas as paginas."""
        return "\n\n".join(p.text for p in self.pages_info)
    
    @property
    def primary_amount(self) -> int:
        """Retorna o valor principal do segmento."""
        for page in self.pages_info:
            if page.amount_cents > 0:
                return page.amount_cents
        return 0
    
    @property
    def primary_date(self) -> Optional[str]:
        """Retorna a data principal (vencimento ou emissao)."""
        for page in self.pages_info:
            if page.due_date:
                return page.due_date
            if page.emission_date:
                return page.emission_date
        return None


# ==============================================================================
# CLASSE PRINCIPAL: CognitiveScanner
# ==============================================================================

class CognitiveScanner:
    """
    Scanner cognitivo para documentos financeiros.
    
    Implementa a Camada de Ingestao e Percepcao do SRDA-Rural:
    - Varredura de diretorios com geradores (baixo uso de memoria)
    - Classificacao de documentos por palavras-chave
    - Segmentacao inteligente de PDFs combinados
    - Extracao de valores e datas via Regex
    - Identificacao de entidades (Vagner/Marcelli)
    """
    
    def __init__(
        self,
        input_folder: str = DEFAULT_INPUT_FOLDER,
        db: Optional[SRDADatabase] = None
    ):
        """
        Inicializa o scanner.
        
        Args:
            input_folder: Pasta de entrada para varredura
            db: Instancia do banco de dados (cria nova se None)
        """
        self.input_folder = Path(input_folder)
        self.db = db or SRDADatabase()
        
        # Inicializa EnsembleExtractor para extracao avancada (com OCR)
        self.ensemble = None
        if ENSEMBLE_AVAILABLE:
            try:
                self.ensemble = EnsembleExtractor(use_gliner=True, use_ocr=True)
                print("[OK] EnsembleExtractor ativado (GLiNER + OCR)")
            except Exception as e:
                print(f"[AVISO] Falha ao iniciar EnsembleExtractor: {e}")
        
        # Inicializa barcode extractor para boletos
        self.barcode_extractor = None
        if BARCODE_AVAILABLE:
            try:
                self.barcode_extractor = get_barcode_extractor()
                if self.barcode_extractor.is_available():
                    print("[OK] BarcodeExtractor ativado")
                else:
                    self.barcode_extractor = None
            except:
                pass
        
        # Cria pasta de entrada se nao existir
        self.input_folder.mkdir(parents=True, exist_ok=True)
    
    # ==========================================================================
    # VARREDURA DE DIRETORIOS
    # ==========================================================================
    
    def scan_directory(self) -> Generator[Path, None, None]:
        """
        Varre a pasta de entrada usando gerador (memoria constante).
        
        Yields:
            Path para cada arquivo PDF encontrado
        """
        if not self.input_folder.exists():
            print(f"[AVISO] Pasta nao encontrada: {self.input_folder}")
            return
        
        for file_path in self.input_folder.rglob("*.pdf"):
            yield file_path
        
        # Tambem busca PDFs com extensao maiuscula
        for file_path in self.input_folder.rglob("*.PDF"):
            yield file_path
    
    def count_files(self) -> int:
        """Conta o numero de PDFs na pasta de entrada."""
        return sum(1 for _ in self.scan_directory())
    
    # ==========================================================================
    # CLASSIFICACAO DE DOCUMENTOS
    # ==========================================================================
    
    def classify_page(self, text: str) -> Tuple[DocumentType, float]:
        """
        Classifica uma pagina baseado em palavras-chave.
        
        Args:
            text: Texto extraido da pagina
            
        Returns:
            Tuple (DocumentType, confidence)
        """
        text_upper = text.upper()
        scores = {doc_type: 0 for doc_type in DocumentType}
        
        # Conta matches de palavras-chave
        for doc_type, keywords in KEYWORDS.items():
            for keyword in keywords:
                if keyword.upper() in text_upper:
                    scores[doc_type] += 1
        
        # Encontra o tipo com maior score
        max_type = DocumentType.UNKNOWN
        max_score = 0
        
        for doc_type, score in scores.items():
            if score > max_score:
                max_score = score
                max_type = doc_type
        
        # Calcula confianca (0.0 a 1.0)
        total_keywords = len(KEYWORDS.get(max_type, []))
        confidence = min(max_score / max(total_keywords, 1), 1.0) if max_score > 0 else 0.0
        
        return max_type, confidence
    
    def detect_scheduling(self, text: str) -> bool:
        """
        Detecta se um comprovante e de AGENDAMENTO (nao pagamento efetivo).
        
        Args:
            text: Texto do documento
            
        Returns:
            True se for agendamento, False se for pagamento efetivo
        """
        text_upper = text.upper()
        for keyword in SCHEDULING_KEYWORDS:
            if keyword in text_upper:
                return True
        return False
    
    # ==========================================================================
    # EXTRACAO DE DADOS
    # ==========================================================================
    
    def extract_amount(self, text: str) -> int:
        """
        Extrai valor monetario do texto.
        
        Args:
            text: Texto para busca
            
        Returns:
            Valor em centavos (0 se nao encontrado)
        """
        # Tenta primeiro com prefixo (R$, Total, Valor)
        match = REGEX_PATTERNS["amount"].search(text)
        if match:
            return SRDADatabase.amount_to_cents(match.group(1))
        
        # Busca todos os valores no texto
        matches = REGEX_PATTERNS["amount_plain"].findall(text)
        if matches:
            # Pega o maior valor encontrado (geralmente o total)
            amounts = [SRDADatabase.amount_to_cents(m) for m in matches]
            return max(amounts) if amounts else 0
        
        return 0
    
    def extract_date(self, text: str, pattern_name: str = "date_br") -> Optional[str]:
        """
        Extrai data do texto e normaliza para ISO8601.
        
        Args:
            text: Texto para busca
            pattern_name: Nome do padrao de regex
            
        Returns:
            Data no formato YYYY-MM-DD ou None
        """
        pattern = REGEX_PATTERNS.get(pattern_name)
        if not pattern:
            return None
        
        match = pattern.search(text)
        if match:
            if pattern_name == "date_br":
                day, month, year = match.groups()
            else:
                # Para padroes que capturam a data completa
                date_str = match.group(1)
                parts = re.split(r'[/\.\-]', date_str)
                if len(parts) == 3:
                    day, month, year = parts
                else:
                    return None
            
            try:
                return f"{year}-{month.zfill(2)}-{day.zfill(2)}"
            except:
                return None
        
        return None
    
    def _parse_date(self, date_str: str) -> Optional[str]:
        """
        Converte string de data para formato ISO (YYYY-MM-DD).
        
        Aceita formatos: DD/MM/YYYY, DD.MM.YYYY, DD-MM-YYYY
        """
        if not date_str:
            return None
        
        # Padrão brasileiro
        import re
        match = re.match(r'(\d{1,2})[/.\-](\d{1,2})[/.\-](\d{2,4})', date_str)
        if match:
            day, month, year = match.groups()
            # Normaliza ano de 2 dígitos
            if len(year) == 2:
                year = '20' + year if int(year) < 50 else '19' + year
            try:
                return f"{year}-{month.zfill(2)}-{day.zfill(2)}"
            except:
                pass
        
        return None
    
    def extract_due_date(self, text: str) -> Optional[str]:
        """Extrai data de vencimento especificamente."""
        match = REGEX_PATTERNS["due_date"].search(text)
        if match:
            date_str = match.group(1)
            parts = re.split(r'[/\.\-]', date_str)
            if len(parts) == 3:
                day, month, year = parts
                return f"{year}-{month.zfill(2)}-{day.zfill(2)}"
        
        # Fallback: busca qualquer data
        return self.extract_date(text)
    
    def extract_emission_date(self, text: str) -> Optional[str]:
        """Extrai data de emissao especificamente."""
        match = REGEX_PATTERNS["emission_date"].search(text)
        if match:
            date_str = match.group(1)
            parts = re.split(r'[/\.\-]', date_str)
            if len(parts) == 3:
                day, month, year = parts
                return f"{year}-{month.zfill(2)}-{day.zfill(2)}"
        return None
    
    def extract_access_key(self, text: str) -> Optional[str]:
        """Extrai chave de acesso NF-e (44 digitos)."""
        match = REGEX_PATTERNS["access_key"].search(text)
        if match:
            # Remove espacos
            return match.group(1).replace(" ", "")
        return None
    
    def extract_entity(self, text: str) -> Optional[EntityTag]:
        """
        Identifica a entidade financeira (Vagner ou Marcelli) pelo CPF.
        
        Args:
            text: Texto do documento
            
        Returns:
            EntityTag.VG, EntityTag.MV ou None
        """
        text_upper = text.upper()
        
        # 1. Busca CPFs conhecidos (mais confiável)
        cpf_matches = REGEX_PATTERNS["cpf"].findall(text)
        for cpf in cpf_matches:
            entity = SRDADatabase.parse_cpf_to_entity(cpf)
            if entity:
                return entity
        
        # 2. Busca nomes conhecidos de sócios/empresas
        # VG - Vagner / Vesz Participações
        vg_keywords = ["VAGNER", "VESZ PARTICIPACOES", "VESZ PARTIC"]
        for kw in vg_keywords:
            if kw in text_upper:
                return EntityTag.VG
        
        # MV - Marcelli / Mauro
        mv_keywords = ["MARCELLI", "MAURO VESZ", "M V AGRO"]
        for kw in mv_keywords:
            if kw in text_upper:
                return EntityTag.MV
        
        # 3. Busca CNPJ do SACADO/DESTINATÁRIO (quem paga)
        # Apenas aceita CNPJ se estiver próximo de "SACADO" ou "PAGADOR"
        sacado_section = ""
        for marker in ["SACADO", "PAGADOR", "DESTINAT"]:
            idx = text_upper.find(marker)
            if idx != -1:
                sacado_section = text[idx:idx+200]  # Próximos 200 chars
                break
        
        if sacado_section:
            # CNPJs conhecidos para VG e MV (empresas pagadoras)
            VG_CNPJS = {
                "01.696.819",  # Vesz Participacoes
                "04.176.760",  # Outros VG  
            }
            MV_CNPJS = {
                # Adicionar CNPJ da empresa MV quando souber
            }
            
            cnpj_matches = REGEX_PATTERNS["cnpj"].findall(sacado_section)
            for cnpj in cnpj_matches:
                cnpj_clean = cnpj.replace('.', '').replace('/', '').replace('-', '').replace(' ', '')
                if len(cnpj_clean) >= 8:
                    prefix = f"{cnpj_clean[:2]}.{cnpj_clean[2:5]}.{cnpj_clean[5:8]}"
                    if prefix in MV_CNPJS:
                        return EntityTag.MV
                    if prefix in VG_CNPJS:
                        return EntityTag.VG
        
        return None
    
    def extract_duplicatas(self, text: str) -> List[Dict[str, Any]]:
        """
        Extrai tabela de duplicatas/parcelas de uma NF-e.
        
        Args:
            text: Texto da NF-e
            
        Returns:
            Lista de duplicatas com seq_num, due_date, amount_cents
        """
        duplicatas = []
        matches = REGEX_PATTERNS["duplicata"].findall(text)
        
        for seq_str, date_str, amount_str in matches:
            parts = date_str.split("/")
            if len(parts) == 3:
                day, month, year = parts
                iso_date = f"{year}-{month}-{day}"
            else:
                iso_date = date_str
            
            duplicatas.append({
                "seq_num": int(seq_str),
                "due_date": iso_date,
                "amount_cents": SRDADatabase.amount_to_cents(amount_str)
            })
        
        return duplicatas
    
    # ==========================================================================
    # BATCH PIPELINE (WATERFALL PROCESSING - ELIMINATES MODEL THRASHING)
    # ==========================================================================
    
    def scan_and_classify_all(self, files: List[Path] = None, 
                               progress_callback=None) -> Dict[str, List[Path]]:
        """
        Estágio 1 do Pipeline Waterfall: Classifica todos os arquivos.
        
        Passa por todos os arquivos usando PyMuPDF (fast, sem IA) e categoriza:
        - DIGITAL_OK: Texto nativo suficiente, não precisa de OCR
        - NEEDS_OCR: Texto insuficiente, precisa de PaddleOCR
        - NEEDS_VLM: OCR falhou ou documento muito complexo, precisa de Florence/Surya
        
        Args:
            files: Lista de arquivos a processar (se None, usa scan_directory)
            progress_callback: Callback(current, total, filename, category)
            
        Returns:
            Dict com listas separadas: {'DIGITAL_OK': [...], 'NEEDS_OCR': [...], 'NEEDS_VLM': [...]}
        """
        if files is None:
            files = list(self.scan_directory())
        
        result = {
            'DIGITAL_OK': [],
            'NEEDS_OCR': [],
            'NEEDS_VLM': [],
            'ERRORS': []
        }
        
        total = len(files)
        print(f"[STAGE 1] Classificando {total} arquivos...")
        
        for i, file_path in enumerate(files, 1):
            if progress_callback:
                progress_callback(i, total, file_path.name, "classifying")
            
            try:
                doc = fitz.open(str(file_path))
                if len(doc) == 0:
                    doc.close()
                    result['ERRORS'].append(file_path)
                    continue
                
                page = doc[0]
                text = page.get_text()
                text_len = len(text.strip())
                doc.close()
                
                # Classifica baseado na quantidade de texto
                if text_len >= 200:
                    # Texto rico - PDF digital, fast path
                    result['DIGITAL_OK'].append(file_path)
                elif text_len >= 50:
                    # Texto parcial - pode precisar de OCR para completar
                    result['NEEDS_OCR'].append(file_path)
                else:
                    # Pouco ou nenhum texto - necessita OCR completo
                    result['NEEDS_OCR'].append(file_path)
                    
            except Exception as e:
                print(f"  [ERROR] {file_path.name}: {e}")
                result['ERRORS'].append(file_path)
        
        print(f"[STAGE 1] Resultado: {len(result['DIGITAL_OK'])} digitais, "
              f"{len(result['NEEDS_OCR'])} para OCR, {len(result['ERRORS'])} erros")
        
        return result
    
    def process_batch_digital(self, files: List[Path], 
                               progress_callback=None) -> Dict[Path, Dict[str, Any]]:
        """
        Processa arquivos digitais (texto nativo) em lote.
        
        Usa apenas PyMuPDF e SpatialExtractor - sem modelos pesados.
        
        Args:
            files: Lista de arquivos a processar
            progress_callback: Callback(current, total, filename, stage)
            
        Returns:
            Dict[file_path -> extraction_result]
        """
        results = {}
        total = len(files)
        
        print(f"[STAGE 2a] Processando {total} arquivos digitais...")
        
        for i, file_path in enumerate(files, 1):
            if progress_callback:
                progress_callback(i, total, file_path.name, "digital")
            
            try:
                # Usa o hierarchical_extract mas força o fast path
                result = self.hierarchical_extract(file_path, 0, DocumentType.UNKNOWN)
                results[file_path] = result
            except Exception as e:
                print(f"  [ERROR] {file_path.name}: {e}")
                results[file_path] = {'error': str(e), 'confidence': 0.0}
        
        return results
    
    def process_batch_ocr(self, files: List[Path], 
                          progress_callback=None) -> Dict[Path, Dict[str, Any]]:
        """
        Estágio 2b: Processa arquivos que precisam de OCR em LOTE.
        
        OTIMIZAÇÃO CRÍTICA: Carrega PaddleOCR UMA VEZ, processa todos os arquivos,
        depois descarrega. Isso elimina o thrashing que consumia 2-5s por arquivo.
        
        Args:
            files: Lista de arquivos que precisam de OCR
            progress_callback: Callback(current, total, filename, stage)
            
        Returns:
            Dict[file_path -> extraction_result]
        """
        if not files:
            return {}
        
        results = {}
        total = len(files)
        paddle_voter = None
        
        print(f"[STAGE 2b] Processando {total} arquivos com OCR...")
        
        try:
            # CARREGA PADDLE UMA VEZ
            from paddle_voter import get_paddle_voter
            paddle_voter = get_paddle_voter()
            paddle_voter.load_model()  # Force load
            print("  [PADDLE] Modelo carregado para batch processing")
            
            for i, file_path in enumerate(files, 1):
                if progress_callback:
                    progress_callback(i, total, file_path.name, "ocr")
                
                try:
                    doc = fitz.open(str(file_path))
                    if len(doc) == 0:
                        doc.close()
                        continue
                    
                    page = doc[0]
                    
                    # Render at 150 DPI
                    matrix = fitz.Matrix(1.5, 1.5)
                    pix = page.get_pixmap(matrix=matrix)
                    
                    from PIL import Image
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    del pix
                    
                    # OCR com modelo já carregado
                    ocr_text = paddle_voter.extract_text(img)
                    
                    # Extrai dados via regex
                    result = {
                        'amount_cents': self.extract_amount(ocr_text) if ocr_text else 0,
                        'due_date': self.extract_due_date(ocr_text) if ocr_text else None,
                        'emission_date': self.extract_emission_date(ocr_text) if ocr_text else None,
                        'cnpj': None,
                        'entity_tag': self.extract_entity(ocr_text) if ocr_text else None,
                        'confidence': 0.7,
                        'extraction_path': 'batch_ocr',
                        'needs_review': False
                    }
                    
                    # CNPJ
                    cnpj_match = REGEX_PATTERNS["cnpj"].search(ocr_text) if ocr_text else None
                    if cnpj_match:
                        result['cnpj'] = cnpj_match.group(1)
                    
                    results[file_path] = result
                    
                    del img
                    doc.close()
                    
                except Exception as e:
                    print(f"  [ERROR] {file_path.name}: {e}")
                    results[file_path] = {'error': str(e), 'confidence': 0.0}
                
                # Cleanup periódico
                if i % 10 == 0:
                    gc.collect()
                    
        except ImportError:
            print("  [WARN] PaddleOCR não disponível, usando fallback")
            # Fallback: processa sem OCR
            for file_path in files:
                results[file_path] = self.hierarchical_extract(file_path, 0, DocumentType.UNKNOWN)
        
        finally:
            # DESCARREGA PADDLE
            if paddle_voter:
                try:
                    paddle_voter.unload_model()
                    print("  [PADDLE] Modelo descarregado após batch")
                except:
                    pass
            gc.collect()
            gc.collect()
        
        return results
    
    def process_batch_vlm(self, files: List[Path], 
                          progress_callback=None) -> Dict[Path, Dict[str, Any]]:
        """
        Estágio 3: Processa arquivos difíceis com VLM em LOTE.
        
        OTIMIZAÇÃO CRÍTICA: Carrega Florence-2/Surya UMA VEZ, processa todos,
        depois descarrega.
        
        Args:
            files: Lista de arquivos que precisam de VLM
            progress_callback: Callback(current, total, filename, stage)
            
        Returns:
            Dict[file_path -> extraction_result]
        """
        if not files:
            return {}
        
        results = {}
        total = len(files)
        vlm_voter = None
        
        print(f"[STAGE 3] Processando {total} arquivos com VLM...")
        
        try:
            # Tenta Florence-2 primeiro
            from florence2_voter import get_florence2_voter
            vlm_voter = get_florence2_voter()
            vlm_voter.load_model()
            vlm_name = "Florence-2"
            print(f"  [{vlm_name}] Modelo carregado para batch processing")
        except ImportError:
            try:
                from surya_voter import get_surya_voter
                vlm_voter = get_surya_voter()
                vlm_voter.load_model()
                vlm_name = "Surya"
                print(f"  [{vlm_name}] Modelo carregado para batch processing")
            except ImportError:
                print("  [WARN] Nenhum VLM disponível, retornando vazio")
                return results
        
        try:
            for i, file_path in enumerate(files, 1):
                if progress_callback:
                    progress_callback(i, total, file_path.name, "vlm")
                
                try:
                    doc = fitz.open(str(file_path))
                    if len(doc) == 0:
                        doc.close()
                        continue
                    
                    page = doc[0]
                    
                    # Render at 300 DPI para VLM
                    matrix = fitz.Matrix(2.0, 2.0)
                    pix = page.get_pixmap(matrix=matrix)
                    
                    from PIL import Image
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    del pix
                    
                    # VLM extraction
                    vlm_text = vlm_voter.extract_text(img)
                    
                    # Extrai dados via regex
                    result = {
                        'amount_cents': self.extract_amount(vlm_text) if vlm_text else 0,
                        'due_date': self.extract_due_date(vlm_text) if vlm_text else None,
                        'emission_date': self.extract_emission_date(vlm_text) if vlm_text else None,
                        'cnpj': None,
                        'entity_tag': self.extract_entity(vlm_text) if vlm_text else None,
                        'confidence': 0.65,
                        'extraction_path': f'batch_{vlm_name.lower()}',
                        'needs_review': True
                    }
                    
                    # CNPJ
                    cnpj_match = REGEX_PATTERNS["cnpj"].search(vlm_text) if vlm_text else None
                    if cnpj_match:
                        result['cnpj'] = cnpj_match.group(1)
                    
                    results[file_path] = result
                    
                    del img
                    doc.close()
                    
                except Exception as e:
                    print(f"  [ERROR] {file_path.name}: {e}")
                    results[file_path] = {'error': str(e), 'confidence': 0.0}
                
                # Cleanup periódico
                if i % 5 == 0:
                    gc.collect()
                    
        finally:
            # DESCARREGA VLM
            if vlm_voter:
                try:
                    vlm_voter.unload_model()
                    print(f"  [{vlm_name}] Modelo descarregado após batch")
                except:
                    pass
            gc.collect()
            gc.collect()
            gc.collect()
        
        return results

    # ==========================================================================
    # HIERARCHICAL EXTRACTION (8GB RAM OPTIMIZED)
    # ==========================================================================
    
    def hierarchical_extract(
        self, 
        file_path: Path,
        page_num: int = 0,
        doc_type: DocumentType = DocumentType.UNKNOWN
    ) -> Dict[str, Any]:
        """
        Step-by-Step Extraction Algorithm (8GB RAM Survival Mode).
        
        Priority: Native Text > Regex > Paddle > VLM (last resort)
        
        Algorithm:
        1. Extract native PDF text (free, fast)
        2. Run regex for CNPJ/Date/Amount
        3. If confidence > 0.9 -> RETURN (skip OCR)
        4. If confidence low -> run PaddleOCR at 150 DPI
        5. If still low -> run VLM at 300 DPI (last resort)
        6. Cleanup memory after each stage
        
        Args:
            file_path: Path to PDF
            page_num: Page to process (0-indexed)
            doc_type: Expected document type
            
        Returns:
            Dict with extracted data and confidence scores
        """
        result = {
            "amount_cents": 0,
            "due_date": None,
            "emission_date": None,
            "supplier_name": None,
            "cnpj": None,
            "entity_tag": None,
            "confidence": 0.0,
            "extraction_path": "none",  # fast/slow/heavy
            "needs_review": False
        }
        
        try:
            doc = fitz.open(str(file_path))
            if page_num >= len(doc):
                doc.close()
                return result
            
            page = doc[page_num]
            
            # ================================================================
            # STEP 1: FAST PATH - Native Text + Spatial Extraction (FREE)
            # ================================================================
            native_text = page.get_text()
            text_length = len(native_text.strip())
            
            print(f"    [FAST] Native text: {text_length} chars")
            
            if text_length >= 100:
                # Good text layer - use Spatial Extraction first, then regex fallback
                result["extraction_path"] = "fast"
                
                # Try SpatialExtractor for anchor-based extraction
                spatial_results = {}
                try:
                    from spatial_extractor import get_spatial_extractor
                    spatial = get_spatial_extractor()
                    
                    if spatial.load_pdf(file_path, page_num):
                        spatial_results = spatial.extract_all()
                        
                        # Use spatial results if available
                        if 'amount' in spatial_results:
                            result["amount_cents"] = SRDADatabase.amount_to_cents(
                                spatial_results['amount'].value
                            )
                        if 'due_date' in spatial_results:
                            result["due_date"] = self._parse_date(
                                spatial_results['due_date'].value
                            )
                        if 'emission_date' in spatial_results:
                            result["emission_date"] = self._parse_date(
                                spatial_results['emission_date'].value
                            )
                        if 'cnpj_emissor' in spatial_results:
                            result["cnpj"] = spatial_results['cnpj_emissor'].value
                        
                        print(f"    [FAST] Spatial extracted {len(spatial_results)} fields")
                except Exception as e:
                    print(f"    [FAST] Spatial extraction failed: {e}")
                
                # Fallback/supplement with regex for missing fields
                if result["amount_cents"] == 0:
                    amount = self.extract_amount(native_text)
                    result["amount_cents"] = amount
                
                if not result["due_date"]:
                    due_date = self.extract_due_date(native_text)
                    result["due_date"] = due_date
                
                if not result["emission_date"]:
                    emission_date = self.extract_emission_date(native_text)
                    result["emission_date"] = emission_date
                
                if not result["cnpj"]:
                    cnpj_match = REGEX_PATTERNS["cnpj"].search(native_text)
                    if cnpj_match:
                        result["cnpj"] = cnpj_match.group(1)
                
                # Always extract entity from text
                entity = self.extract_entity(native_text)
                result["entity_tag"] = entity
                
                # Calculate confidence based on what we found
                confidence = 0.0
                if result["amount_cents"] > 0:
                    # Higher confidence if from spatial extraction
                    conf_boost = 0.35 if 'amount' in spatial_results else 0.3
                    confidence += conf_boost
                if result["due_date"] or result["emission_date"]:
                    conf_boost = 0.35 if 'due_date' in spatial_results else 0.3
                    confidence += conf_boost
                if result["cnpj"]:
                    conf_boost = 0.25 if 'cnpj_emissor' in spatial_results else 0.2
                    confidence += conf_boost
                if entity:
                    confidence += 0.1
                
                result["confidence"] = min(confidence, 1.0)
                
                # If high confidence, we're done
                if result["confidence"] >= 0.9:
                    print(f"    [FAST] High confidence ({result['confidence']:.2f}), skipping OCR")
                    doc.close()
                    return result
            
            # ================================================================
            # STEP 2: SLOW PATH - PaddleOCR at 150 DPI
            # ================================================================
            if text_length < 100 or result["confidence"] < 0.7:
                print(f"    [SLOW] Running PaddleOCR...")
                result["extraction_path"] = "slow"
                
                try:
                    # Render at 150 DPI (1.5x)
                    matrix = fitz.Matrix(1.5, 1.5)
                    pix = page.get_pixmap(matrix=matrix)
                    
                    from PIL import Image
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    del pix  # Free memory
                    
                    # Try PaddleOCR
                    try:
                        from paddle_voter import get_paddle_voter
                        paddle = get_paddle_voter()
                        ocr_text = paddle.extract_text(img)
                        
                        if ocr_text and len(ocr_text) > len(native_text):
                            # Run regex on OCR text
                            amount = self.extract_amount(ocr_text)
                            due_date = self.extract_due_date(ocr_text)
                            emission_date = self.extract_emission_date(ocr_text)
                            cnpj_match = REGEX_PATTERNS["cnpj"].search(ocr_text)
                            
                            # Update if better
                            if amount > result["amount_cents"]:
                                result["amount_cents"] = amount
                            if due_date and not result["due_date"]:
                                result["due_date"] = due_date
                            if emission_date and not result["emission_date"]:
                                result["emission_date"] = emission_date
                            if cnpj_match and not result["cnpj"]:
                                result["cnpj"] = cnpj_match.group(1)
                            
                            # Recalculate confidence
                            confidence = 0.0
                            if result["amount_cents"] > 0:
                                confidence += 0.3
                            if result["due_date"] or result["emission_date"]:
                                confidence += 0.3
                            if result["cnpj"]:
                                confidence += 0.3
                            
                            result["confidence"] = confidence
                            print(f"    [SLOW] Paddle confidence: {confidence:.2f}")
                        
                        # Cleanup paddle
                        paddle.unload_model()
                        
                    except ImportError:
                        print("    [SLOW] PaddleOCR not available")
                    
                    del img
                    gc.collect()
                    
                    # Check if we need heavy path
                    if result["confidence"] >= 0.7:
                        doc.close()
                        return result
                        
                except Exception as e:
                    print(f"    [SLOW] Error: {e}")
            
            # ================================================================
            # STEP 3: HEAVY PATH - VLM at 300 DPI (Last Resort)
            # ================================================================
            if result["confidence"] < 0.7:
                print(f"    [HEAVY] Running VLM (last resort)...")
                result["extraction_path"] = "heavy"
                result["needs_review"] = True  # Flag for human review
                
                try:
                    # Render at 300 DPI (2x)
                    matrix = fitz.Matrix(2.0, 2.0)
                    pix = page.get_pixmap(matrix=matrix)
                    
                    from PIL import Image
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    del pix
                    
                    # Try Florence-2 first (lighter)
                    vlm_text = ""
                    try:
                        from florence2_voter import get_florence2_voter
                        florence = get_florence2_voter()
                        vlm_text = florence.extract_text(img)
                        florence.unload_model()
                    except Exception as e:
                        print(f"    [HEAVY] Florence-2 failed: {e}")
                    
                    # If Florence failed, try Surya
                    if not vlm_text or len(vlm_text) < 50:
                        try:
                            from surya_voter import get_surya_voter
                            surya = get_surya_voter()
                            surya_result = surya.extract_text_with_layout(img)
                            if surya_result:
                                vlm_text = surya_result.text
                            surya.unload_model()
                        except Exception as e:
                            print(f"    [HEAVY] Surya failed: {e}")
                    
                    if vlm_text:
                        # Final regex extraction
                        amount = self.extract_amount(vlm_text)
                        due_date = self.extract_due_date(vlm_text)
                        cnpj_match = REGEX_PATTERNS["cnpj"].search(vlm_text)
                        
                        if amount > result["amount_cents"]:
                            result["amount_cents"] = amount
                        if due_date and not result["due_date"]:
                            result["due_date"] = due_date
                        if cnpj_match and not result["cnpj"]:
                            result["cnpj"] = cnpj_match.group(1)
                        
                        # Final confidence
                        confidence = 0.0
                        if result["amount_cents"] > 0:
                            confidence += 0.35
                        if result["due_date"]:
                            confidence += 0.35
                        if result["cnpj"]:
                            confidence += 0.3
                        
                        result["confidence"] = confidence
                        print(f"    [HEAVY] VLM confidence: {confidence:.2f}")
                    
                    del img
                    gc.collect()
                    gc.collect()
                    
                except Exception as e:
                    print(f"    [HEAVY] Error: {e}")
            
            doc.close()
            
        except Exception as e:
            print(f"    [ERROR] Hierarchical extract failed: {e}")
        
        finally:
            # Final cleanup
            get_resource_manager().force_cleanup()
        
        return result
    
    # ==========================================================================
    # PROCESSAMENTO DE PDF
    # ==========================================================================
    
    def process_page(self, page: fitz.Page, page_num: int) -> PageInfo:
        """
        Processa uma unica pagina do PDF.
        
        Args:
            page: Objeto fitz.Page
            page_num: Numero da pagina (1-indexed)
            
        Returns:
            PageInfo com dados extraidos
        """
        # Extrai texto (metodo rapido)
        text = page.get_text()
        
        # Fallback para OCR se texto insuficiente (para PDFs escaneados)
        if len(text.strip()) < 10 and self.ensemble and self.ensemble.ocr_voter:
            try:
                # Usa OCR Voter do ensemble que ja tem preprocessamento v3.0
                ocr_text = self.ensemble.ocr_voter.extract_text(page.parent, page_num - 1)
                if ocr_text:
                    text = ocr_text
                    print(f"  [OCR] Pagina {page_num} processada via OCR")
            except Exception as e:
                print(f"  [AVISO] Falha no OCR fallback: {e}")
        
        # Classifica a pagina
        doc_type, confidence = self.classify_page(text)
        
        # Extrai dados
        amount = self.extract_amount(text)
        due_date = self.extract_due_date(text)
        emission_date = self.extract_emission_date(text)
        access_key = self.extract_access_key(text)
        entity = self.extract_entity(text)
        is_scheduled = self.detect_scheduling(text)
        
        return PageInfo(
            page_num=page_num,
            doc_type=doc_type,
            text=text,
            confidence=confidence,
            amount_cents=amount,
            due_date=due_date,
            emission_date=emission_date,
            access_key=access_key,
            entity_tag=entity,
            is_scheduled=is_scheduled
        )
    
    def segment_pdf(self, file_path: Path) -> List[DocumentSegment]:
        """
        Segmenta um PDF em documentos logicos (detecta arquivos combinados).
        
        Um arquivo combinado e identificado quando o tipo de documento muda
        entre paginas (ex: NFE na pag 1, BOLETO na pag 2).
        
        Args:
            file_path: Caminho do arquivo PDF
            
        Returns:
            Lista de DocumentSegment (cada um e um documento logico)
        """
        segments = []
        
        try:
            doc = fitz.open(str(file_path))
        except Exception as e:
            print(f"[ERRO] Nao foi possivel abrir {file_path}: {e}")
            return segments
        
        try:
            if len(doc) == 0:
                doc.close()
                return segments
            
            # Processa todas as paginas
            pages_info: List[PageInfo] = []
            for i in range(len(doc)):
                page = doc[i]
                page_info = self.process_page(page, i + 1)
                pages_info.append(page_info)
            
            # Detecta transicoes de tipo (segmentacao inteligente)
            current_segment_start = 0
            current_type = pages_info[0].doc_type
            
            for i in range(1, len(pages_info)):
                # Detecta transicao se:
                # 1. O tipo mudou E
                # 2. O novo tipo nao e UNKNOWN E
                # 3. Pelo menos um dos tipos nao e UNKNOWN
                prev_type = pages_info[i - 1].doc_type
                curr_type = pages_info[i].doc_type
                
                is_transition = (
                    curr_type != prev_type and
                    curr_type != DocumentType.UNKNOWN and
                    prev_type != DocumentType.UNKNOWN
                )
                
                if is_transition:
                    # Fecha o segmento anterior
                    segment = DocumentSegment(
                        file_path=str(file_path),
                        page_start=current_segment_start + 1,  # 1-indexed
                        page_end=i,  # 1-indexed
                        doc_type=current_type,
                        pages_info=pages_info[current_segment_start:i]
                    )
                    segments.append(segment)
                    
                    # Inicia novo segmento
                    current_segment_start = i
                    current_type = curr_type
            
            # Adiciona o ultimo segmento
            segment = DocumentSegment(
                file_path=str(file_path),
                page_start=current_segment_start + 1,
                page_end=len(pages_info),
                doc_type=current_type,
                pages_info=pages_info[current_segment_start:]
            )
            segments.append(segment)
            
        finally:
            doc.close()
        
        return segments
    
    # ==========================================================================
    # PROCESSAMENTO PRINCIPAL
    # ==========================================================================
    
    def process_file(self, file_path: Path) -> List[int]:
        """
        Processa um arquivo PDF e salva no banco de dados.
        
        Args:
            file_path: Caminho do arquivo
            
        Returns:
            Lista de IDs dos documentos inseridos
        """
        doc_ids = []
        
        # Verifica se o arquivo ja foi processado (idempotencia)
        file_hash = SRDADatabase.calculate_file_hash(str(file_path))
        if self.db.document_exists(file_hash):
            print(f"  [SKIP] Arquivo ja processado: {file_path.name}")
            return doc_ids
        
        # Segmenta o PDF
        segments = self.segment_pdf(file_path)
        
        if not segments:
            print(f"  [AVISO] Nenhum segmento extraido: {file_path.name}")
            return doc_ids
        
        # Se ha multiplos segmentos, e um arquivo combinado
        is_combined = len(segments) > 1
        if is_combined:
            print(f"  [INFO] Arquivo combinado detectado: {len(segments)} documentos")
        
        for segment in segments:
            # Determina a entidade (VG ou MV)
            entity = None
            for page in segment.pages_info:
                if page.entity_tag:
                    entity = page.entity_tag
                    break
            
            # Insere documento no banco
            doc_id = self.db.insert_document(
                file_path=str(file_path),
                doc_type=segment.doc_type,
                entity_tag=entity,
                page_start=segment.page_start,
                page_end=segment.page_end
            )
            
            if doc_id:
                doc_ids.append(doc_id)
                
                # Atualiza status e salva texto (cache)
                self.db.update_document_status(
                    doc_id=doc_id,
                    status=DocumentStatus.PARSED,
                    raw_text=segment.text
                )
                
                # =====================================================
                # EXTRACAO AVANCADA (Ensemble com OCR + GLiNER)
                # =====================================================
                amount = segment.primary_amount
                due_date = segment.primary_date
                emission_date = None
                is_scheduled = False
                supplier_name = None
                doc_number = None
                
                # Fallback: extrai dados basicos do segmento
                for page in segment.pages_info:
                    if page.emission_date:
                        emission_date = page.emission_date
                    if page.is_scheduled:
                        is_scheduled = True
                
                # Se EnsembleExtractor disponivel, usa extracao avancada
                # Se EnsembleExtractor disponivel, usa extracao avancada
                # FIX: Usar hierarchical_extract (Pipeline 8GB Survival de 3 Estágios)
                try:
                    # Usa o pipeline otimizado em vez do ensemble direto
                    extraction_result = self.hierarchical_extract(
                        file_path=file_path,
                        page_num=segment.page_start - 1, # 0-indexed
                        doc_type=segment.doc_type
                    )
                    
                    # Mapeia resultados
                    if extraction_result["amount_cents"] > 0:
                        amount = extraction_result["amount_cents"]
                    
                    if extraction_result["due_date"]:
                        due_date = extraction_result["due_date"]
                    
                    if extraction_result["emission_date"]:
                        emission_date = extraction_result["emission_date"]
                        
                    if extraction_result["cnpj"]:
                        # Tenta encontrar entidade pelo CNPJ
                        pass # Ja tratado na extracao de entidade
                    
                    # Fornecedor/CNPJ (TODO: Melhorar mapeamento)
                    if extraction_result.get("supplier_name"):
                         supplier_name = extraction_result["supplier_name"]
                    
                    print(f"    [PIPELINE] Tipo: {extraction_result['extraction_path']} | Confiança: {extraction_result['confidence']:.2f}")

                except Exception as e:
                     print(f"    [AVISO] Pipeline falhou, usando basico: {e}")
                
                # Para BOLETOs, tenta barcode extractor para dados precisos
                if segment.doc_type == DocumentType.BOLETO and self.barcode_extractor:
                    try:
                        barcode_result = self.barcode_extractor.extract_from_pdf(str(file_path), segment.page_start - 1)
                        if barcode_result and barcode_result.get('success'):
                            # Barcode tem valor e vencimento EXATOS
                            if barcode_result.get('amount'):
                                barcode_amount = int(barcode_result['amount'] * 100)
                                if barcode_amount > 0:
                                    amount = barcode_amount
                                    print(f"    [BARCODE] Valor: {SRDADatabase.cents_to_display(amount)}")
                            
                            if barcode_result.get('due_date'):
                                due_date_barcode = barcode_result['due_date']
                                # Converte de DD/MM/YYYY para ISO
                                if '/' in due_date_barcode:
                                    parts = due_date_barcode.split('/')
                                    if len(parts) == 3:
                                        due_date = f"{parts[2]}-{parts[1]}-{parts[0]}"
                                print(f"    [BARCODE] Vencimento: {due_date}")
                    except Exception as e:
                        print(f"    [AVISO] Barcode falhou: {e}")
                
                # Atualiza doc_number no documento se encontrado
                if doc_number:
                    cursor = self.db.connection.cursor()
                    cursor.execute("UPDATE documentos SET doc_number = ? WHERE id = ?", (doc_number, doc_id))
                    self.db.connection.commit()
                
                # Insere transacao com dados completos
                if amount > 0:
                    self.db.insert_transaction(
                        doc_id=doc_id,
                        amount_cents=amount,
                        entidade_pagadora=entity,
                        emission_date=emission_date,
                        due_date=due_date,
                        is_scheduled=is_scheduled,
                        supplier_clean=supplier_name  # Novo campo
                    )
                
                # Se for NF-e, extrai duplicatas
                if segment.doc_type == DocumentType.NFE:
                    duplicatas = self.extract_duplicatas(segment.text)
                    for dup in duplicatas:
                        self.db.insert_installment(
                            nfe_id=doc_id,
                            seq_num=dup["seq_num"],
                            due_date=dup["due_date"],
                            amount_cents=dup["amount_cents"]
                        )
                        print(f"    [DUP] Parcela {dup['seq_num']}: {SRDADatabase.cents_to_display(dup['amount_cents'])}")
                
                print(f"  [{segment.doc_type.value}] {segment.page_start}-{segment.page_end} | {SRDADatabase.cents_to_display(amount)} | {entity.value if entity else '?'}")
        
        return doc_ids
    
    def process_all(self, batch_size: int = 10, progress_callback=None) -> Dict[str, Any]:
        """
        Processa todos os arquivos da pasta de entrada.
        
        Usa processamento em lotes para manter baixo uso de memoria
        conforme especificado no Volume II (Secao 8.1).
        
        Args:
            batch_size: Numero de arquivos por lote
            progress_callback: Funcao callback(current, total, filename, stage) para updates
            
        Returns:
            Dicionario com estatisticas do processamento
        """
        print("=" * 60)
        print("SRDA-Rural Scanner - Iniciando Varredura")
        print(f"Pasta: {self.input_folder.absolute()}")
        print("=" * 60)
        
        stats = {
            "total_files": 0,
            "processed": 0,
            "skipped": 0,
            "errors": 0,
            "documents_created": 0,
            "combined_files": 0,
            "by_type": {t.value: 0 for t in DocumentType},
            "by_entity": {"VG": 0, "MV": 0, "UNKNOWN": 0},
        }
        
        files = list(self.scan_directory())
        stats["total_files"] = len(files)
        
        if stats["total_files"] == 0:
            print(f"\n[AVISO] Nenhum arquivo PDF encontrado em {self.input_folder}")
            print("Crie a pasta 'Input' e adicione arquivos PDF para processar.")
            if progress_callback:
                progress_callback(0, 0, "", "nenhum_arquivo")
            return stats
        
        print(f"\nEncontrados {stats['total_files']} arquivos PDF")
        print("-" * 60)
        
        # Processa em lotes com progress callbacks
        for i, file_path in enumerate(files, 1):
            filename = file_path.name
            print(f"\n[{i}/{stats['total_files']}] {filename}")
            
            # Callback: iniciando arquivo
            if progress_callback:
                progress_callback(i, stats["total_files"], filename, "lendo")
            
            try:
                doc_ids = self.process_file(file_path)
                
                if doc_ids:
                    stats["processed"] += 1
                    stats["documents_created"] += len(doc_ids)
                    
                    if len(doc_ids) > 1:
                        stats["combined_files"] += 1
                    
                    # Callback: arquivo processado
                    if progress_callback:
                        progress_callback(i, stats["total_files"], filename, "ok")
                else:
                    stats["skipped"] += 1
                    if progress_callback:
                        progress_callback(i, stats["total_files"], filename, "skip")
                    
            except Exception as e:
                stats["errors"] += 1
                print(f"  [ERRO] {e}")
                if progress_callback:
                    progress_callback(i, stats["total_files"], filename, "erro")
        
        # Atualiza estatisticas do banco
        db_stats = self.db.get_statistics()
        stats["by_type"] = db_stats.get("by_status", {})
        stats["by_entity"] = db_stats.get("by_entity", {})
        
        # Imprime resumo
        print("\n" + "=" * 60)
        print("RESUMO DO PROCESSAMENTO")
        print("=" * 60)
        print(f"Arquivos encontrados:  {stats['total_files']}")
        print(f"Processados:           {stats['processed']}")
        print(f"Ignorados (duplicata): {stats['skipped']}")
        print(f"Erros:                 {stats['errors']}")
        print(f"Documentos criados:    {stats['documents_created']}")
        print(f"Arquivos combinados:   {stats['combined_files']}")
        print(f"\nDocumentos por entidade:")
        print(f"  Vagner (VG):   {stats['by_entity'].get('VG', 0)}")
        print(f"  Marcelli (MV): {stats['by_entity'].get('MV', 0)}")
        
        return stats
    
    # ==========================================================================
    # IMPORTACAO RAPIDA (SEM AI)
    # ==========================================================================
    
    def quick_import(self, progress_callback=None) -> Dict[str, Any]:
        """
        Importa PDFs rapidamente SEM processamento de AI.
        Apenas registra arquivos no banco com status PENDING.
        """
        stats = {"total_files": 0, "imported": 0, "skipped": 0, "errors": 0}
        
        files = list(self.scan_directory())
        stats["total_files"] = len(files)
        
        if stats["total_files"] == 0:
            if progress_callback:
                progress_callback(0, 0, "", "nenhum_arquivo")
            return stats
        
        for i, file_path in enumerate(files, 1):
            filename = file_path.name
            
            if progress_callback:
                progress_callback(i, stats["total_files"], filename, "importando")
            
            try:
                file_hash = SRDADatabase.calculate_file_hash(str(file_path))
                if self.db.document_exists(file_hash):
                    stats["skipped"] += 1
                    if progress_callback:
                        progress_callback(i, stats["total_files"], filename, "skip")
                    continue
                
                # Apenas registra - SEM processar AI
                cursor = self.db.connection.cursor()
                cursor.execute("""
                    INSERT INTO documentos 
                    (original_path, file_hash, doc_type, status, page_start, page_end)
                    VALUES (?, ?, 'UNKNOWN', 'PENDING', 1, 1)
                """, (str(file_path), file_hash))
                self.db.connection.commit()
                
                stats["imported"] += 1
                if progress_callback:
                    progress_callback(i, stats["total_files"], filename, "ok")
                    
            except Exception as e:
                stats["errors"] += 1
                if progress_callback:
                    progress_callback(i, stats["total_files"], filename, "erro")
        
        return stats


# ==============================================================================
# EXEMPLO DE USO
# ==============================================================================

if __name__ == "__main__":
    # Cria instancia do scanner
    scanner = CognitiveScanner(input_folder="Input")
    
    # Processa todos os arquivos
    stats = scanner.process_all()
    
    # Mostra estatisticas finais do banco
    print("\n" + "-" * 60)
    print("Estatisticas do Banco de Dados:")
    db_stats = scanner.db.get_statistics()
    print(f"  Total de documentos: {db_stats['total_documents']}")
    print(f"  Total de transacoes: {db_stats['total_transactions']}")
    print(f"  Total de matches:    {db_stats['total_matches']}")
    
    scanner.db.close()
    print("\n[OK] Processamento concluido!")
