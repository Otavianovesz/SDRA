import fitz  # PyMuPDF
import re
import logging
from typing import Optional, Tuple, List

logger = logging.getLogger(__name__)

class SpatialExtractor:
    def __init__(self, pdf_path):
        self.doc = fitz.open(pdf_path)
        self.first_page = self.doc[0]
        # Carrega todas as palavras com coordenadas: (x0, y0, x1, y1, "texto", block, line, word)
        self.words = self.first_page.get_text("words")

    def find_anchor(self, text_pattern: str) -> Optional[fitz.Rect]:
        """
        Encontra a caixa (Rect) de uma palavra-chave ou frase.
        Usa search_for para suportar frases multi-palavras.
        """
        # search_for é mais robusto para frases do que iterar sobre words individuais
        rects = self.first_page.search_for(text_pattern)
        if rects:
            # Retorna o primeiro match como um fitz.Rect
            return rects[0]
        
        # Fallback para regex manual se necessário (se search_for for rígido demais)
        pattern = re.compile(text_pattern, re.IGNORECASE)
        for w in self.words:
            if pattern.search(w[4]):
                return fitz.Rect(w[0], w[1], w[2], w[3])
        return None

    def get_text_in_region(self, region: fitz.Rect) -> str:
        """
        Retorna todo o texto dentro de um retângulo, ordenado logicamente.
        """
        found_words = []
        for w in self.words:
            # Verifica se a palavra está contida ou cruza a região
            w_rect = fitz.Rect(w[0], w[1], w[2], w[3])
            # Intersect check: se pelo menos 50% da palavra está dentro
            if (w_rect & region).get_area() > (w_rect.get_area() * 0.4):
                found_words.append(w)
        
        # Ordenação Snaking (Top-down, then Left-to-right)
        # Usamos uma tolerância de 3 pixels para alinhar palavras na mesma linha
        found_words.sort(key=lambda w: (w[1] // 3, w[0]))
        
        return " ".join([w[4] for w in found_words]).strip()

    def extract_value_below(self, anchor_text: str, vertical_scan=40, width_scan=150) -> Optional[str]:
        """Estratégia: Busca a âncora e lê estritamente ABAIXO dela."""
        anchor = self.find_anchor(anchor_text)
        if not anchor:
            return None
        
        # Define ROI: Mesma largura da âncora (com folga), logo abaixo
        roi = fitz.Rect(anchor.x0 - 5, anchor.y1, anchor.x0 + width_scan, anchor.y1 + vertical_scan)
        return self.get_text_in_region(roi)

    def extract_value_right(self, anchor_text: str, horizontal_scan=200) -> Optional[str]:
        """Estratégia: Busca a âncora e lê estritamente à DIREITA dela."""
        anchor = self.find_anchor(anchor_text)
        if not anchor:
            return None
        
        # Define ROI: Começa no fim da âncora, mesma faixa de altura
        roi = fitz.Rect(anchor.x1, anchor.y0 - 2, anchor.x1 + horizontal_scan, anchor.y1 + 2)
        return self.get_text_in_region(roi)

    def extract_cnpj_near(self, anchor_text="CNPJ", scan_radius=150) -> Optional[str]:
        """Busca padrão de CNPJ próximo a uma âncora 'CNPJ'."""
        anchor = self.find_anchor(anchor_text)
        if not anchor:
            # Se não houver âncora, tenta varrer o topo do documento
            roi = fitz.Rect(0, 0, self.first_page.rect.width, 300)
        else:
            # Região expandida ao redor da âncora
            roi = fitz.Rect(anchor.x0 - 10, anchor.y0 - 10, anchor.x1 + scan_radius, anchor.y1 + 50)
        
        text = self.get_text_in_region(roi)
        match = re.search(r"\d{2}\.\d{3}\.\d{3}/\d{4}-\d{2}", text)
        return match.group(0) if match else None

    def extract_supplier_header(self) -> Optional[str]:
        """
        Tenta extrair o nome do fornecedor olhando para o topo do documento.
        Normalmente o maior texto no primeiro bloco do topo.
        """
        # Pega as primeiras 200 unidades verticais
        roi = fitz.Rect(0, 0, self.first_page.rect.width, 200)
        # Filtra palavras no topo
        top_words = [w for w in self.words if w[3] < 200]
        
        if not top_words: return None
        
        # Agrupa por blocos de texto (PyMuPDF já dá o block_id em w[5])
        blocks = {}
        for w in top_words:
            bid = w[5]
            if bid not in blocks: blocks[bid] = []
            blocks[bid].append(w)
            
        # O fornecedor costuma ser o bloco mais 'importante' no topo
        # Vamos assumir que é o primeiro bloco significativo que não seja 
        # tags manjadas (como 'DANFE', 'PREFEITURA')
        for bid in sorted(blocks.keys()):
            block_text = " ".join([w[4] for w in blocks[bid]])
            if len(block_text) > 5 and not any(tag in block_text.upper() for tag in ["DANFE", "NOTA FISCAL", "BOLETO", "COMPROVANTE"]):
                return block_text
        return None

    def close(self):
        """Fecha o documento PDF."""
        if hasattr(self, 'doc'):
            self.doc.close()

# Singleton helper
_instance = None
def get_spatial_extractor(pdf_path=None):
    global _instance
    if pdf_path:
        if _instance:
            _instance.close()
        try:
            _instance = SpatialExtractor(pdf_path)
        except Exception as e:
            logger.error(f"Erro ao carregar SpatialExtractor para {pdf_path}: {e}")
            return None
    return _instance
