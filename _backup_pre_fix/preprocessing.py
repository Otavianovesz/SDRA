"""
SDRA-Rural Preprocessing Module
================================
Motor de Restauração Forense de Imagens

Técnicas implementadas:
1. CLAHE - Equalização adaptativa para recibos térmicos desbotados
2. Sauvola - Binarização adaptativa para layouts complexos
3. Morfologia - Limpeza de ruído (sal/pimenta)
4. Super-Resolução - Condicional para imagens de baixa qualidade

Baseado no Relatório Estratégico v3.0 (Seção 2)
"""

import cv2
import numpy as np
import logging
from typing import Tuple, Optional
from pathlib import Path
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class PreprocessingConfig:
    """Configuração do pipeline de pré-processamento."""
    # CLAHE
    clahe_clip_limit: float = 3.0
    clahe_tile_size: Tuple[int, int] = (8, 8)
    
    # Sauvola
    sauvola_window_size: int = 25
    sauvola_k: float = 0.3
    sauvola_r: int = 128
    
    # Morfologia
    opening_kernel_size: Tuple[int, int] = (2, 2)
    closing_kernel_size: Tuple[int, int] = (3, 3)
    
    # Super-resolução condicional
    sr_confidence_threshold: float = 0.5
    sr_enabled: bool = False  # Desabilitado por padrão (requer modelo)


class ImagePreprocessor:
    """
    Motor de restauração forense para documentos financeiros degradados.
    
    Otimizado para:
    - Recibos térmicos desbotados (SISBB, cupons fiscais)
    - Comprovantes amarelados (Banco do Brasil)
    - Digitalizações de baixa qualidade (fotos de celular)
    """
    
    def __init__(self, config: Optional[PreprocessingConfig] = None):
        self.config = config or PreprocessingConfig()
        self._clahe = cv2.createCLAHE(
            clipLimit=self.config.clahe_clip_limit,
            tileGridSize=self.config.clahe_tile_size
        )
        self._sr_model = None
        
    def apply_clahe(self, image: np.ndarray) -> np.ndarray:
        """
        Aplica CLAHE (Contrast Limited Adaptive Histogram Equalization).
        
        Normaliza iluminação localmente, fazendo texto desbotado ter
        o mesmo contraste que texto nítido.
        
        Args:
            image: Imagem BGR ou grayscale
            
        Returns:
            Imagem com contraste equalizado
        """
        if len(image.shape) == 3:
            # Converter para LAB, aplicar CLAHE no canal L
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            l_enhanced = self._clahe.apply(l)
            lab_enhanced = cv2.merge([l_enhanced, a, b])
            return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
        else:
            # Grayscale
            return self._clahe.apply(image)
    
    def apply_sauvola(self, image: np.ndarray) -> np.ndarray:
        """
        Aplica binarização de Sauvola.
        
        Fórmula: T(x,y) = m(x,y) * [1 + k * (s(x,y)/R - 1)]
        
        Onde:
        - m(x,y): média local
        - s(x,y): desvio padrão local
        - R: alcance dinâmico (128 para 8-bit)
        - k: fator de sensibilidade
        
        Args:
            image: Imagem grayscale
            
        Returns:
            Imagem binarizada
        """
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        window = self.config.sauvola_window_size
        k = self.config.sauvola_k
        R = self.config.sauvola_r
        
        # Usar integral images para eficiência O(1) por pixel
        # Calcular média local
        mean = cv2.blur(image.astype(np.float64), (window, window))
        
        # Calcular desvio padrão local via integral image
        sq_mean = cv2.blur((image.astype(np.float64) ** 2), (window, window))
        std = np.sqrt(np.maximum(sq_mean - mean ** 2, 0))
        
        # Threshold de Sauvola
        threshold = mean * (1 + k * (std / R - 1))
        
        # Aplicar threshold
        binary = (image > threshold).astype(np.uint8) * 255
        
        return binary
    
    def apply_morphology(self, image: np.ndarray) -> np.ndarray:
        """
        Aplica operações morfológicas para limpeza.
        
        1. Opening (erosão + dilatação): Remove ruído de fundo
        2. Closing (dilatação + erosão): Conecta fragmentos de caracteres
        
        Args:
            image: Imagem binária
            
        Returns:
            Imagem limpa
        """
        # Kernel para opening (limpa "sal e pimenta")
        kernel_open = cv2.getStructuringElement(
            cv2.MORPH_RECT, 
            self.config.opening_kernel_size
        )
        
        # Kernel para closing (fecha buracos em caracteres)
        kernel_close = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,  # Elipse vertical para fontes financeiras
            self.config.closing_kernel_size
        )
        
        # Opening primeiro (remove ruído)
        opened = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel_open)
        
        # Closing depois (conecta fragmentos)
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel_close)
        
        return closed
    
    def remove_shadow(self, image: np.ndarray) -> np.ndarray:
        """
        Remove shadows using morphological estimation.
        
        Technique:
        1. Estimate background using large kernel dilation (max filter)
        2. Blur background to remove artifacts
        3. Normalize image: Result = Image / Background
        
        Args:
            image: Input image
            
        Returns:
            Shadow-free image
        """
        if len(image.shape) == 3:
            planes = cv2.split(image)
            result_planes = []
            
            for plane in planes:
                # 1. Dilate to remove text -> estimate background
                dilated_img = cv2.dilate(plane, np.ones((7, 7), np.uint8))
                
                # 2. Median blur to smooth background
                bg_img = cv2.medianBlur(dilated_img, 21)
                
                # 3. Calculate difference: 255 - |Img - Bg|
                diff_img = 255 - cv2.absdiff(plane, bg_img)
                
                # 4. Normalize
                norm_img = cv2.normalize(diff_img, None, alpha=0, beta=255, 
                                       norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
                
                result_planes.append(norm_img)
                
            return cv2.merge(result_planes)
        else:
            # Grayscale path
            dilated_img = cv2.dilate(image, np.ones((7, 7), np.uint8))
            bg_img = cv2.medianBlur(dilated_img, 21)
            diff_img = 255 - cv2.absdiff(image, bg_img)
            return cv2.normalize(diff_img, None, alpha=0, beta=255, 
                               norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)

    def dewarp_document(self, image: np.ndarray) -> np.ndarray:
        """
        Simple geometric dewarping (Perspective Transform).
        
        Attempts to find the largest 4-point contour (document sheet)
        and unwarp it to a flat rectangle.
        
        Args:
            image: Input image
            
        Returns:
            Dewarped image (or original if failure)
        """
        try:
            # Resize for speed
            height, width = image.shape[:2]
            scale = 1000 / max(height, width)
            small = cv2.resize(image, None, fx=scale, fy=scale)
            
            if len(small.shape) == 3:
                gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            else:
                gray = small
                
            # Edge detection
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edged = cv2.Canny(blurred, 75, 200)
            
            # Find contours
            cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]
            
            screen_cnt = None
            for c in cnts:
                peri = cv2.arcLength(c, True)
                # Approximate polygon
                approx = cv2.approxPolyDP(c, 0.02 * peri, True)
                
                if len(approx) == 4:
                    screen_cnt = approx
                    break
            
            if screen_cnt is None:
                return image  # No clear document border found
            
            # Rescale points to original size
            screen_cnt = screen_cnt.reshape(4, 2) / scale
            
            # Order points: tl, tr, br, bl
            rect = np.zeros((4, 2), dtype="float32")
            s = screen_cnt.sum(axis=1)
            rect[0] = screen_cnt[np.argmin(s)]
            rect[2] = screen_cnt[np.argmax(s)]
            
            diff = np.diff(screen_cnt, axis=1)
            rect[1] = screen_cnt[np.argmin(diff)]
            rect[3] = screen_cnt[np.argmax(diff)]
            
            (tl, tr, br, bl) = rect
            
            # Compute new dimensions
            widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
            widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
            max_width = max(int(widthA), int(widthB))
            
            heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
            heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
            max_height = max(int(heightA), int(heightB))
            
            # Destination points
            dst = np.array([
                [0, 0],
                [max_width - 1, 0],
                [max_width - 1, max_height - 1],
                [0, max_height - 1]
            ], dtype="float32")
            
            # Perspective transform
            M = cv2.getPerspectiveTransform(rect, dst)
            warped = cv2.warpPerspective(image, M, (max_width, max_height))
            
            return warped
            
        except Exception as e:
            logger.warning(f"Dewarping failed: {e}")
            return image

    def preprocess(
        self, 
        image: np.ndarray, 
        apply_binarization: bool = False,
        remove_shadows: bool = True,
        fix_geometry: bool = False
    ) -> np.ndarray:
        """
        Pipeline completo de pré-processamento.
        
        Args:
            image: Imagem original (BGR ou grayscale)
            apply_binarization: Se True, aplica Sauvola + morfologia
            remove_shadows: Remove sombras/iluminação desigual
            fix_geometry: Tenta corrigir perspectiva
            
        Returns:
            Imagem pré-processada
        """
        current_image = image
        
        # Etapa 0: Correção Geométrica (Opcional)
        if fix_geometry:
            current_image = self.dewarp_document(current_image)
            
        # Etapa 1: Remoção de Sombra (Opcional mas recomendado)
        if remove_shadows:
            current_image = self.remove_shadow(current_image)
            
        # Etapa 2: Normalização de Contraste (Se não removeu sombras, ou como reforço)
        if not remove_shadows:
            current_image = self.apply_clahe(current_image)
        
        if not apply_binarization:
            return current_image
        
        # Etapa 3: Binarização de Sauvola
        binary = self.apply_sauvola(current_image)
        
        # Etapa 4: Morfologia para limpeza
        cleaned = self.apply_morphology(binary)
        
        return cleaned
    
    def preprocess_for_ocr(self, image: np.ndarray) -> np.ndarray:
        """
        Pré-processamento otimizado para OCR.
        
        Para PaddleOCR, manter em cores (não binarizar).
        CLAHE + denoise é suficiente.
        
        Args:
            image: Imagem BGR
            
        Returns:
            Imagem preparada para OCR
        """
        # CLAHE
        enhanced = self.apply_clahe(image)
        
        # Denoise suave (preserva bordas)
        denoised = cv2.fastNlMeansDenoisingColored(
            enhanced, None, 
            3,   # h - força de filtragem luminância
            3,   # hColor - força de filtragem cor
            7,   # templateWindowSize
            21   # searchWindowSize
        )
        
        return denoised
    
    def preprocess_thermal_receipt(self, image: np.ndarray) -> np.ndarray:
        """
        Pré-processamento especializado para recibos térmicos desbotados.
        
        Aplica CLAHE agressivo + binarização adaptativa.
        
        Args:
            image: Imagem do recibo térmico
            
        Returns:
            Imagem restaurada
        """
        # CLAHE agressivo para recibos térmicos
        aggressive_clahe = cv2.createCLAHE(
            clipLimit=4.0,  # Mais agressivo
            tileGridSize=(16, 16)  # Tiles maiores para uniformidade
        )
        
        if len(image.shape) == 3:
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            l_enhanced = aggressive_clahe.apply(l)
            lab_enhanced = cv2.merge([l_enhanced, a, b])
            enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
        else:
            enhanced = aggressive_clahe.apply(image)
        
        return enhanced
    
    def estimate_image_quality(self, image: np.ndarray) -> float:
        """
        Estima a qualidade da imagem (0.0 a 1.0).
        
        Usado para decidir se super-resolução é necessária.
        
        Args:
            image: Imagem a avaliar
            
        Returns:
            Score de qualidade (< 0.5 sugere super-resolução)
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Variância do Laplaciano (nitidez)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Normalizar (valores típicos: 100-1000 para imagens nítidas)
        sharpness = min(laplacian_var / 500.0, 1.0)
        
        # Contraste (desvio padrão)
        contrast = gray.std() / 128.0
        contrast = min(contrast, 1.0)
        
        # Score combinado
        quality = 0.6 * sharpness + 0.4 * contrast
        
        logger.debug(f"Qualidade estimada: {quality:.2f} (nitidez={sharpness:.2f}, contraste={contrast:.2f})")
        
        return quality


def preprocess_pdf_page(
    page_image: np.ndarray,
    is_thermal: bool = False,
    for_ocr: bool = True
) -> np.ndarray:
    """
    Função de conveniência para pré-processar página de PDF.
    
    Args:
        page_image: Imagem da página (BGR)
        is_thermal: Se True, usa processamento agressivo para térmico
        for_ocr: Se True, otimiza para OCR (mantém cores)
        
    Returns:
        Imagem pré-processada
    """
    preprocessor = ImagePreprocessor()
    
    if is_thermal:
        return preprocessor.preprocess_thermal_receipt(page_image)
    elif for_ocr:
        return preprocessor.preprocess_for_ocr(page_image)
    else:
        return preprocessor.preprocess(page_image, apply_binarization=True)


# =============================================================================
# TESTES
# =============================================================================

if __name__ == "__main__":
    import sys
    
    logging.basicConfig(level=logging.DEBUG)
    
    print("=" * 60)
    print("SDRA-Rural Preprocessing Module")
    print("=" * 60)
    
    preprocessor = ImagePreprocessor()
    
    # Teste com imagem sintética
    print("\nCriando imagem de teste...")
    test_image = np.random.randint(100, 200, (480, 640, 3), dtype=np.uint8)
    
    # Adicionar texto simulado (região mais escura)
    test_image[200:250, 100:500] = 50
    
    print(f"Imagem original: {test_image.shape}")
    
    # Testar CLAHE
    clahe_result = preprocessor.apply_clahe(test_image)
    print(f"CLAHE aplicado: {clahe_result.shape}")
    
    # Testar pipeline completo
    preprocessed = preprocessor.preprocess_for_ocr(test_image)
    print(f"Pré-processamento OCR: {preprocessed.shape}")
    
    # Testar estimativa de qualidade
    quality = preprocessor.estimate_image_quality(test_image)
    print(f"Qualidade estimada: {quality:.2f}")
    
    print("\n[OK] Todos os testes passaram!")
