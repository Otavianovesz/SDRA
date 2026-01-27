"""
SpatialExtractor - Motor "Sniper" de Extração Espacial
=======================================================
Implementa extração baseada em coordenadas geométricas,
substituindo Regex cego por âncoras espaciais.

Fase 2 do Master Protocol (Steps 21-50)
"""

import re
import gc
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any, Union
from enum import Enum

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

logger = logging.getLogger(__name__)


class ExtractionMethod(Enum):
    """Método usado para extração."""
    DIGITAL = "DIGITAL"  # Texto nativo do PDF
    OCR = "OCR"          # Via PaddleOCR/Surya
    HYBRID = "HYBRID"    # Combinação


@dataclass
class TextBlock:
    """Bloco de texto com coordenadas."""
    text: str
    x0: float
    y0: float
    x1: float
    y1: float
    confidence: float = 1.0
    
    @property
    def center_x(self) -> float:
        return (self.x0 + self.x1) / 2
    
    @property
    def center_y(self) -> float:
        return (self.y0 + self.y1) / 2
    
    @property
    def width(self) -> float:
        return self.x1 - self.x0
    
    @property
    def height(self) -> float:
        return self.y1 - self.y0
    
    def intersects(self, other: 'Rect') -> bool:
        """Verifica se centro do bloco está dentro do Rect."""
        return (other.x0 <= self.center_x <= other.x1 and 
                other.y0 <= self.center_y <= other.y1)


@dataclass
class Rect:
    """Retângulo de região de interesse (ROI)."""
    x0: float
    y0: float
    x1: float
    y1: float
    
    def expand(self, factor: float = 0.1) -> 'Rect':
        """Expande o retângulo em uma porcentagem."""
        w = self.x1 - self.x0
        h = self.y1 - self.y0
        dx = w * factor / 2
        dy = h * factor / 2
        return Rect(
            self.x0 - dx,
            self.y0 - dy,
            self.x1 + dx,
            self.y1 + dy
        )
    
    def contains_point(self, x: float, y: float) -> bool:
        return self.x0 <= x <= self.x1 and self.y0 <= y <= self.y1


@dataclass
class Anchor:
    """Âncora de referência espacial."""
    keyword: str
    rect: Rect
    quadrant: str  # 'TL', 'TR', 'BL', 'BR'
    priority: int = 0


@dataclass
class ExtractionResult:
    """Resultado de uma extração espacial."""
    field_name: str
    value: str
    confidence: float
    anchor_used: Optional[str] = None
    roi: Optional[Rect] = None
    method: ExtractionMethod = ExtractionMethod.DIGITAL


# ============================================================================
# ÂNCORAS CONHECIDAS POR TIPO DE CAMPO
# ============================================================================
ANCHOR_KEYWORDS = {
    'due_date': [
        'VENCIMENTO', 'VENC.', 'VENC', 'DT VENC', 'DATA VENCIMENTO',
        'DT. VENCIMENTO', 'VENCTO', 'DATA VENC.', 'DT VCTO'
    ],
    'emission_date': [
        'EMISSÃO', 'EMISSAO', 'DATA EMISSÃO', 'DT EMISSÃO', 'DATA DE EMISSÃO',
        'DOCUMENTO', 'DT DOCUMENTO', 'DATA', 'DT. EMISSÃO'
    ],
    'amount': [
        # NFE-specific (higher priority - estas devem vir primeiro)
        'VALOR TOTAL DA NOTA', 'VALOR TOTAL DA NF', 'VALOR TOTAL',
        'VALOR DA NOTA', 'TOTAL DA NOTA', 'VL. TOTAL DA NOTA',
        'V. TOTAL TRIB.', 'VALOR TOTAL TRIB',
        # Boleto-specific
        'VALOR DO DOCUMENTO', 'VALOR A PAGAR', 'TOTAL A PAGAR',
        '(=) VALOR DOCUMENTO', 'VLR. DOCUMENTO',
        # Generic (lower priority)
        'VALOR', 'TOTAL', 'QUANTIA', 'R$'
    ],
    'nfe_total': [
        # Específico para NFE - valor total da nota fiscal
        'VALOR TOTAL DA NOTA', 'VALOR TOTAL DA NF-E', 'VALOR TOTAL DA NFE',
        'TOTAL DA NOTA', 'VL TOTAL NF', 'VALOR DA NOTA'
    ],
    'cnpj_emissor': [
        'CNPJ', 'CNPJ/CPF', 'CNPJ EMISSOR', 'CNPJ/MF', 
        'CEDENTE', 'BENEFICIÁRIO', 'EMITENTE'
    ],
    'cnpj_destinatario': [
        'SACADO', 'PAGADOR', 'CLIENTE', 'DESTINATÁRIO',
        'CNPJ/CPF DO SACADO', 'CNPJ DO PAGADOR'
    ],
    'nfe_number': [
        'Nº', 'NUMERO', 'NF-e', 'NFE', 'NOTA FISCAL',
        'DANFE', 'SÉRIE', 'NUMERO DA NOTA'
    ],
    'boleto_number': [
        'NOSSO NÚMERO', 'NOSSO NUMERO', 'NOSSO Nº',
        'SEU NÚMERO', 'NÚMERO DO DOCUMENTO'
    ]
}

# Padrões Regex para validação
PATTERNS = {
    'date': re.compile(r'\b(\d{1,2})[/.\-](\d{1,2})[/.\-](\d{2,4})\b'),
    'amount': re.compile(r'\b(\d{1,3}(?:[.,]\d{3})*[.,]\d{2})\b'),
    'cnpj': re.compile(r'\b(\d{2}[.\s]?\d{3}[.\s]?\d{3}[/\s]?\d{4}[-.\s]?\d{2})\b'),
    'cpf': re.compile(r'\b(\d{3}[.\s]?\d{3}[.\s]?\d{3}[-.\s]?\d{2})\b'),
    'nfe_key': re.compile(r'\b(\d{44})\b'),
    'parcela': re.compile(r'\b(\d{1,2})\s*/\s*(\d{1,2})\b')
}


class SpatialExtractor:
    """
    Extrator espacial baseado em âncoras geométricas.
    
    Fluxo:
    1. Carrega PDF e extrai blocos de texto com coordenadas
    2. Identifica âncoras (palavras-chave conhecidas)
    3. Define ROI (região de interesse) relativa à âncora
    4. Extrai e valida conteúdo da ROI
    
    Normalização DPI:
    - Todas as coordenadas são normalizadas para uma "página canônica" de 612pt de largura
    - Isso equivale a 72 DPI x 8.5" (tamanho Letter)
    - ROIs são definidas em unidades relativas (% da página)
    """
    
    # Configurações
    MIN_WORDS_DIGITAL = 50  # Mínimo de palavras para considerar PDF digital
    ROI_EXPANSION = 0.15    # Expansão da ROI se primeira tentativa falhar
    DPI_NORMALIZE = 72      # DPI base para normalização
    
    # Página canônica para normalização de coordenadas
    CANONICAL_WIDTH = 612.0   # 72 DPI × 8.5" = largura Letter
    CANONICAL_HEIGHT = 792.0  # 72 DPI × 11" = altura Letter
    
    # ROI defaults em unidades relativas (% da página)
    DEFAULT_ROI_WIDTH_PCT = 0.25   # 25% da largura da página
    DEFAULT_ROI_HEIGHT_PCT = 0.06  # 6% da altura da página
    
    def __init__(self, db=None):
        self.page_width: float = self.CANONICAL_WIDTH
        self.page_height: float = self.CANONICAL_HEIGHT
        self.text_blocks: List[TextBlock] = []
        self.anchors: Dict[str, List[Anchor]] = {}
        self.method: ExtractionMethod = ExtractionMethod.DIGITAL
        self._force_ocr: bool = False
        self._scale_factor: float = 1.0  # Fator de escala para normalização
        self._original_width: float = 0   # Dimensões originais (antes da normalização)
        self._original_height: float = 0
        self.db = db  # Referência ao banco para Active Learning
    
    def load_pdf(self, path: Union[str, Path], page_num: int = 0) -> bool:
        """
        Carrega PDF e extrai blocos de texto.
        
        Args:
            path: Caminho do PDF
            page_num: Número da página (0-indexed)
        
        Returns:
            True se carregado com sucesso, False se precisa OCR
        """
        if fitz is None:
            logger.error("PyMuPDF não disponível")
            return False
        
        path = Path(path)
        if not path.exists():
            logger.error(f"Arquivo não encontrado: {path}")
            return False
        
        try:
            doc = fitz.open(str(path))
            
            # Verificar criptografia
            if doc.is_encrypted:
                if not doc.authenticate(""):  # Tenta senha vazia
                    logger.warning(f"PDF criptografado: {path}")
                    doc.close()
                    return False
            
            if page_num >= len(doc):
                page_num = 0
            
            page = doc[page_num]
            
            # Guardar dimensões originais
            self._original_width = page.rect.width
            self._original_height = page.rect.height
            
            # Calcular fator de escala para normalizar para página canônica
            self._scale_factor = self.CANONICAL_WIDTH / self._original_width
            
            # Usar dimensões normalizadas
            self.page_width = self.CANONICAL_WIDTH
            self.page_height = self._original_height * self._scale_factor
            
            logger.debug(f"DPI Normalization: {self._original_width:.0f}x{self._original_height:.0f} -> "
                        f"{self.page_width:.0f}x{self.page_height:.0f} (scale={self._scale_factor:.3f})")
            
            # Extrair palavras com coordenadas NORMALIZADAS
            words = page.get_text("words")
            
            self.text_blocks = []
            for w in words:
                # w = (x0, y0, x1, y1, "word", block_no, line_no, word_no)
                if len(w) >= 5:
                    # Aplicar normalização às coordenadas
                    self.text_blocks.append(TextBlock(
                        text=w[4],
                        x0=w[0] * self._scale_factor,
                        y0=w[1] * self._scale_factor,
                        x1=w[2] * self._scale_factor,
                        y1=w[3] * self._scale_factor
                    ))
            
            doc.close()
            
            # Verificar se precisa OCR
            if len(self.text_blocks) < self.MIN_WORDS_DIGITAL:
                logger.info(f"PDF com pouco texto ({len(self.text_blocks)} palavras), sinalizando OCR")
                self._force_ocr = True
                return False
            
            self._force_ocr = False
            self.method = ExtractionMethod.DIGITAL
            logger.debug(f"PDF carregado: {len(self.text_blocks)} blocos de texto")
            return True
            
        except Exception as e:
            logger.error(f"Erro ao carregar PDF: {e}")
            return False
    
    def load_from_ocr(self, ocr_results: List[Dict], 
                      page_width: float, page_height: float,
                      source_dpi: int = 300) -> None:
        """
        Carrega resultados do OCR convertendo para formato padronizado.
        
        Args:
            ocr_results: Lista de dicts com 'text', 'bbox', 'confidence'
            page_width: Largura da página em pixels no DPI fonte
            page_height: Altura da página em pixels no DPI fonte
            source_dpi: DPI da imagem usada no OCR
        """
        # Fator de escala para normalizar para 72 DPI (padrão PDF)
        scale = self.DPI_NORMALIZE / source_dpi
        
        self.page_width = page_width * scale
        self.page_height = page_height * scale
        
        self.text_blocks = []
        for item in ocr_results:
            bbox = item.get('bbox', [0, 0, 0, 0])
            self.text_blocks.append(TextBlock(
                text=item.get('text', ''),
                x0=bbox[0] * scale,
                y0=bbox[1] * scale,
                x1=bbox[2] * scale,
                y1=bbox[3] * scale,
                confidence=item.get('confidence', 0.5)
            ))
        
        self._force_ocr = False
        self.method = ExtractionMethod.OCR
        logger.debug(f"OCR carregado: {len(self.text_blocks)} blocos")
    
    @property
    def needs_ocr(self) -> bool:
        """Indica se o documento precisa de OCR."""
        return self._force_ocr
    
    def _normalize_text(self, text: str) -> str:
        """Normaliza texto para comparação (uppercase, sem acentos)."""
        import unicodedata
        text = text.upper().strip()
        # Remove acentos
        text = unicodedata.normalize('NFD', text)
        text = ''.join(c for c in text if unicodedata.category(c) != 'Mn')
        return text
    
    def find_anchors(self, field_name: str) -> List[Anchor]:
        """
        Encontra âncoras para um campo específico.
        
        Args:
            field_name: Nome do campo (ex: 'due_date', 'amount')
        
        Returns:
            Lista de âncoras encontradas, ordenadas por prioridade
        """
        if field_name not in ANCHOR_KEYWORDS:
            return []
        
        keywords = ANCHOR_KEYWORDS[field_name]
        anchors = []
        
        # Agrupa blocos consecutivos para formar frases
        full_text_blocks = self._get_text_lines()
        
        for keyword in keywords:
            keyword_norm = self._normalize_text(keyword)
            
            # Busca em blocos individuais
            for block in self.text_blocks:
                if keyword_norm in self._normalize_text(block.text):
                    quadrant = self._get_quadrant(block.center_x, block.center_y)
                    priority = self._anchor_priority(field_name, quadrant)
                    anchors.append(Anchor(
                        keyword=keyword,
                        rect=Rect(block.x0, block.y0, block.x1, block.y1),
                        quadrant=quadrant,
                        priority=priority
                    ))
            
            # Busca em linhas compostas
            for line in full_text_blocks:
                line_text = ' '.join(b.text for b in line)
                if keyword_norm in self._normalize_text(line_text):
                    # Usa o bounding box da linha
                    x0 = min(b.x0 for b in line)
                    y0 = min(b.y0 for b in line)
                    x1 = max(b.x1 for b in line)
                    y1 = max(b.y1 for b in line)
                    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
                    quadrant = self._get_quadrant(cx, cy)
                    priority = self._anchor_priority(field_name, quadrant)
                    anchors.append(Anchor(
                        keyword=keyword,
                        rect=Rect(x0, y0, x1, y1),
                        quadrant=quadrant,
                        priority=priority
                    ))
        
        # Ordena por prioridade (maior primeiro) e remove duplicatas
        seen = set()
        unique = []
        for a in sorted(anchors, key=lambda x: x.priority, reverse=True):
            key = (round(a.rect.x0), round(a.rect.y0))
            if key not in seen:
                seen.add(key)
                unique.append(a)
        
        self.anchors[field_name] = unique
        return unique
    
    def _get_text_lines(self) -> List[List[TextBlock]]:
        """Agrupa blocos de texto em linhas baseado em Y."""
        if not self.text_blocks:
            return []
        
        # Ordena por Y, depois por X
        sorted_blocks = sorted(self.text_blocks, key=lambda b: (b.y0, b.x0))
        
        lines = []
        current_line = [sorted_blocks[0]]
        
        for block in sorted_blocks[1:]:
            # Se Y está próximo (mesma linha)
            if abs(block.center_y - current_line[-1].center_y) < 10:
                current_line.append(block)
            else:
                lines.append(current_line)
                current_line = [block]
        
        if current_line:
            lines.append(current_line)
        
        # Ordena cada linha por X
        for line in lines:
            line.sort(key=lambda b: b.x0)
        
        return lines
    
    def _get_quadrant(self, x: float, y: float) -> str:
        """Retorna o quadrante da página (TL, TR, BL, BR)."""
        mid_x = self.page_width / 2
        mid_y = self.page_height / 2
        
        if y < mid_y:
            return 'TL' if x < mid_x else 'TR'
        else:
            return 'BL' if x < mid_x else 'BR'
    
    def _anchor_priority(self, field_name: str, quadrant: str) -> int:
        """
        Calcula prioridade de uma âncora baseado no campo e quadrante.
        
        Regras baseadas em layout típico de boletos brasileiros:
        - Vencimento: prioriza quadrante superior direito (TR)
        - Valor: prioriza quadrante inferior (BL, BR)
        - CNPJ emissor: prioriza topo esquerdo (TL)
        """
        priorities = {
            'due_date': {'TR': 10, 'TL': 8, 'BR': 5, 'BL': 3},
            'emission_date': {'TL': 10, 'TR': 8, 'BL': 5, 'BR': 3},
            'amount': {'BR': 10, 'BL': 10, 'TR': 5, 'TL': 3},
            'cnpj_emissor': {'TL': 10, 'TR': 8, 'BL': 5, 'BR': 3},
            'cnpj_destinatario': {'BL': 10, 'BR': 8, 'TL': 5, 'TR': 3},
        }
        
        return priorities.get(field_name, {}).get(quadrant, 5)
    
    def define_roi(self, anchor: Anchor, direction: str = 'RIGHT', 
                   width: float = None, height: float = None,
                   width_pct: float = None, height_pct: float = None) -> Rect:
        """
        Define ROI relativa à âncora usando unidades relativas.
        
        Args:
            anchor: Âncora de referência
            direction: 'RIGHT', 'BELOW', 'LEFT', 'ABOVE'
            width: Largura em pixels (retrocompat) - será convertida para %
            height: Altura em pixels (retrocompat) - será convertida para %
            width_pct: Largura como % da página (0.0 a 1.0)
            height_pct: Altura como % da página (0.0 a 1.0)
        
        Returns:
            Retângulo da ROI em coordenadas normalizadas
        """
        # Converter pixels antigos para percentuais (retrocompatibilidade)
        if width is not None and width > 1.0:
            width_pct = width / self.CANONICAL_WIDTH
        if height is not None and height > 1.0:
            height_pct = height / self.CANONICAL_HEIGHT
        
        # Usar defaults se não especificado
        if width_pct is None:
            width_pct = self.DEFAULT_ROI_WIDTH_PCT
        if height_pct is None:
            height_pct = self.DEFAULT_ROI_HEIGHT_PCT
        
        # Calcular dimensões absolutas na página canônica
        roi_width = self.page_width * width_pct
        roi_height = self.page_height * height_pct
        
        # Margem proporcional (1% da página)
        margin = self.page_width * 0.01
        
        ar = anchor.rect
        
        if direction == 'RIGHT':
            return Rect(ar.x1, ar.y0 - margin, ar.x1 + roi_width, ar.y1 + margin)
        elif direction == 'BELOW':
            return Rect(ar.x0 - margin, ar.y1, ar.x1 + roi_width * 0.3, ar.y1 + roi_height)
        elif direction == 'LEFT':
            return Rect(ar.x0 - roi_width, ar.y0 - margin, ar.x0, ar.y1 + margin)
        elif direction == 'ABOVE':
            return Rect(ar.x0 - margin, ar.y0 - roi_height, ar.x1 + roi_width * 0.3, ar.y0)
        else:
            # Default: direita e abaixo
            return Rect(ar.x1, ar.y0 - margin, ar.x1 + roi_width, ar.y1 + roi_height)
    
    def extract_from_roi(self, roi: Rect, pattern_name: str = None) -> Optional[str]:
        """
        Extrai texto de uma ROI.
        
        Args:
            roi: Região de interesse
            pattern_name: Nome do padrão para validar ('date', 'amount', etc.)
        
        Returns:
            Texto extraído ou None se inválido
        """
        # Filtra blocos dentro da ROI
        blocks_in_roi = [b for b in self.text_blocks if b.intersects(roi)]
        
        if not blocks_in_roi:
            return None
        
        # Ordena da esquerda para direita
        blocks_in_roi.sort(key=lambda b: (b.y0, b.x0))
        
        # Concatena texto
        text = ' '.join(b.text for b in blocks_in_roi)
        
        # Valida com padrão se especificado
        if pattern_name and pattern_name in PATTERNS:
            match = PATTERNS[pattern_name].search(text)
            if match:
                return match.group(0)
            return None
        
        return text.strip()
    
    def extract_field(self, field_name: str, pattern_name: str = None) -> Optional[ExtractionResult]:
        """
        Extrai um campo usando âncoras espaciais.
        
        Args:
            field_name: Nome do campo
            pattern_name: Padrão regex para validação
        
        Returns:
            Resultado da extração ou None
        """
        anchors = self.find_anchors(field_name)
        
        if not anchors:
            logger.debug(f"Nenhuma âncora encontrada para '{field_name}'")
            return None
        
        # Direções a tentar baseado no campo
        directions = {
            'due_date': ['RIGHT', 'BELOW'],
            'emission_date': ['RIGHT', 'BELOW'],
            'amount': ['RIGHT', 'BELOW'],
            'cnpj_emissor': ['RIGHT', 'BELOW'],
            'cnpj_destinatario': ['BELOW', 'RIGHT'],
            'nfe_number': ['RIGHT', 'BELOW'],
            'boleto_number': ['RIGHT', 'BELOW']
        }
        
        dirs = directions.get(field_name, ['RIGHT', 'BELOW'])
        
        for anchor in anchors[:3]:  # Tenta top 3 âncoras
            for direction in dirs:
                roi = self.define_roi(anchor, direction)
                value = self.extract_from_roi(roi, pattern_name)
                
                if value:
                    return ExtractionResult(
                        field_name=field_name,
                        value=value,
                        confidence=0.9 if self.method == ExtractionMethod.DIGITAL else 0.7,
                        anchor_used=anchor.keyword,
                        roi=roi,
                        method=self.method
                    )
                
                # Tenta expandir ROI
                expanded_roi = roi.expand(self.ROI_EXPANSION)
                value = self.extract_from_roi(expanded_roi, pattern_name)
                
                if value:
                    return ExtractionResult(
                        field_name=field_name,
                        value=value,
                        confidence=0.7 if self.method == ExtractionMethod.DIGITAL else 0.5,
                        anchor_used=anchor.keyword,
                        roi=expanded_roi,
                        method=self.method
                    )
        
        return None
    
    def extract_value_bottom_heavy(self) -> Optional[ExtractionResult]:
        """
        Extrai valor total buscando o maior número no terço inferior.
        Método de fallback quando âncoras não funcionam.
        """
        # Define terço inferior da página
        bottom_third = Rect(0, self.page_height * 0.6, self.page_width, self.page_height)
        
        candidates = []
        
        for block in self.text_blocks:
            if not block.intersects(bottom_third):
                continue
            
            # Busca padrão de valor monetário
            match = PATTERNS['amount'].search(block.text)
            if match:
                value_str = match.group(0)
                # Converte para float para comparação
                try:
                    value = float(
                        value_str.replace('.', '').replace(',', '.')
                    )
                    candidates.append((value, value_str, block))
                except ValueError:
                    pass
        
        if not candidates:
            return None
        
        # Retorna o maior valor
        candidates.sort(key=lambda x: x[0], reverse=True)
        best = candidates[0]
        
        return ExtractionResult(
            field_name='amount',
            value=best[1],
            confidence=0.6,  # Menor confiança por ser fallback
            anchor_used='BOTTOM_HEAVY',
            method=self.method
        )
    
    def extract_cnpj_intelligent(self) -> Optional[ExtractionResult]:
        """
        Extrai CNPJ penalizando os que estão na área de 'Pagador'.
        
        O CNPJ correto (Fornecedor) está perto de palavras como 
        "Beneficiário", "Cedente", "Emitente".
        O CNPJ errado está perto de "Pagador", "Sacado".
        
        Returns:
            ExtractionResult com o CNPJ do fornecedor (não do pagador)
        """
        full_text = self.get_full_text()
        text_upper = full_text.upper()
        
        candidates = []
        
        # Encontra todos os CNPJs no documento
        for match in PATTERNS['cnpj'].finditer(full_text):
            cnpj = match.group(1)
            start, end = match.span()
            
            # Contexto: Olhar 150 caracteres antes
            context_start = max(0, start - 150)
            context_window = text_upper[context_start:start]
            
            score = 1.0
            anchor_used = "CNPJ_GENERIC"
            
            # Penalidades (área de pagador - CNPJ do cliente, não do fornecedor)
            penalty_keywords = ["PAGADOR", "SACADO", "DESTINAT", "CLIENTE"]
            for kw in penalty_keywords:
                if kw in context_window:
                    score -= 0.8  # Quase certeza que é o cliente, não o fornecedor
                    anchor_used = f"CNPJ_PENALIZED_{kw}"
                    break
            
            # Bônus (área de fornecedor)
            bonus_keywords = ["BENEFICIARIO", "CEDENTE", "EMITENTE", "FAVORECIDO", "FORNECEDOR"]
            for kw in bonus_keywords:
                if kw in context_window:
                    score += 0.5
                    anchor_used = f"CNPJ_BONUS_{kw}"
                    break
            
            candidates.append({
                'cnpj': cnpj,
                'score': score,
                'anchor': anchor_used,
                'position': start
            })
        
        if not candidates:
            return None
        
        # Ordena por score (maior primeiro), depois por posição (primeiro no doc)
        candidates.sort(key=lambda x: (-x['score'], x['position']))
        
        # Retorna o CNPJ com maior score, desde que score > 0.3
        best = candidates[0]
        if best['score'] > 0.3:
            return ExtractionResult(
                field_name='cnpj_emissor',
                value=best['cnpj'],
                confidence=min(0.9, 0.5 + best['score'] * 0.4),
                anchor_used=best['anchor'],
                method=self.method
            )
        
        return None
    
    def extract_all(self) -> Dict[str, ExtractionResult]:
        """
        Extrai todos os campos disponíveis.
        
        Returns:
            Dicionário com resultados por campo
        """
        results = {}
        
        # Campos com padrões
        field_patterns = {
            'due_date': 'date',
            'emission_date': 'date',
            'amount': 'amount',
            'cnpj_emissor': 'cnpj',
            'cnpj_destinatario': 'cnpj',
        }
        
        for field, pattern in field_patterns.items():
            result = self.extract_field(field, pattern)
            if result:
                results[field] = result
        
        # Fallback para valor se não encontrado
        if 'amount' not in results:
            result = self.extract_value_bottom_heavy()
            if result:
                results['amount'] = result
        
        # MELHORIA: Fallback inteligente para CNPJ do fornecedor
        # Usa extract_cnpj_intelligent se a extração por âncora falhou ou tem baixa confiança
        if 'cnpj_emissor' not in results or results.get('cnpj_emissor', {}).confidence < 0.7:
            intelligent_cnpj = self.extract_cnpj_intelligent()
            if intelligent_cnpj:
                results['cnpj_emissor'] = intelligent_cnpj
        
        return results
    
    def get_full_text(self) -> str:
        """Retorna todo o texto da página concatenado."""
        lines = self._get_text_lines()
        return '\n'.join(' '.join(b.text for b in line) for line in lines)


# ============================================================================
# SINGLETON PARA USO GLOBAL
# ============================================================================
_spatial_extractor = None

def get_spatial_extractor() -> SpatialExtractor:
    """Retorna instância singleton do SpatialExtractor."""
    global _spatial_extractor
    if _spatial_extractor is None:
        _spatial_extractor = SpatialExtractor()
    return _spatial_extractor
