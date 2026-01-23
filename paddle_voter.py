"""
SDRA-Rural PaddleOCR Voter
===========================
Implementação do OCR de Alta Precisão (Fase 2)

Substitui o Tesseract pelo PaddleOCR (PP-OCRv4) para:
- Melhor suporte a layouts complexos
- Reconhecimento robusto de textos distorcidos
- Detecção de texto orientado (rotacionado)

Integração com LazyModelManager para gestão de memória (8GB RAM).
"""

import logging
import numpy as np
import cv2
from typing import Optional, List, Dict, Any, Union
from PIL import Image

from lazy_model_manager import get_model_manager
try:
    from preprocessing import ImagePreprocessor
except ImportError:
    ImagePreprocessor = None

logger = logging.getLogger(__name__)


class PaddleOCRVoter:
    """
    Voter baseado no PaddleOCR (PP-OCRv4).
    
    Características:
    - Multilíngue (foco em PT-BR)
    - Ângulo de classificação ativo
    - Otimizado para CPU (MKLDNN) via LazyModelManager
    """
    
    def __init__(self):
        self.manager = get_model_manager()
        self.preprocessor = ImagePreprocessor() if ImagePreprocessor else None
        
    def _get_model(self):
        """Obtém instância do PaddleOCR via gerenciador de memória."""
        return self.manager.load_paddle()
        
    def extract_text(self, image: Union[str, np.ndarray, Image.Image]) -> str:
        """
        Extrai texto puro de uma imagem.
        
        Args:
            image: Caminho, array numpy (BGR) ou PIL Image
            
        Returns:
            Texto concatenado
        """
        try:
            model = self._get_model()
            if not model:
                return ""
            
            # Converter entrada para numpy BGR
            img_array = self._prepare_image(image)
            
            # Pré-processamento (Restoration Phase 1)
            # PaddleOCR já tem preprocessamento interno, mas o nosso
            # é focado em restauração de degradação (CLAHE/Denoise)
            if self.preprocessor:
                # Usa preprocessamento leve para OCR (mantém cores/cinza)
                img_array = self.preprocessor.preprocess_for_ocr(img_array)
            
            # Inferência
            # cls=True habilita classificador de ângulo (importante para boletos tortos)
            result = model.ocr(img_array, cls=True)
            
            if not result or result[0] is None:
                return ""
            
            # Paddle retorna lista de linhas: [ [[points], (text, conf)], ... ]
            # extrair apenas o texto
            lines = []
            for line in result[0]:
                text_content = line[1][0]
                confidence = line[1][1]
                
                # Filtra lixo de baixa confiança
                if confidence > 0.5:
                    lines.append(text_content)
            
            return "\n".join(lines)
            
        except Exception as e:
            logger.error(f"Erro no PaddleOCR: {e}")
            return ""

    def extract_structured(self, image: Union[str, np.ndarray]) -> List[Dict[str, Any]]:
        """
        Extrai dados estruturados (texto + bbox + conf).
        
        Útil para arquitetura Localizar-Recortar-Reconhecer.
        """
        try:
            model = self._get_model()
            if not model:
                return []
            
            img_array = self._prepare_image(image)
            if self.preprocessor:
                img_array = self.preprocessor.preprocess_for_ocr(img_array)
                
            result = model.ocr(img_array, cls=True)
            
            output = []
            if result and result[0]:
                for line in result[0]:
                    points = line[0]
                    text = line[1][0]
                    conf = line[1][1]
                    output.append({
                        "text": text,
                        "confidence": conf,
                        "bbox": points  # [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                    })
            return output
            
        except Exception as e:
            logger.error(f"Erro estruturado PaddleOCR: {e}")
            return []

    def _prepare_image(self, image: Union[str, np.ndarray, Image.Image]) -> np.ndarray:
        """Converte entrada para numpy array BGR compatível com CV2/Paddle."""
        if isinstance(image, str):
            # Caminho arquivo
            return cv2.imread(image)
        
        if isinstance(image, Image.Image):
            # PIL -> Numpy -> BGR
            img_np = np.array(image)
            if len(img_np.shape) == 3 and img_np.shape[2] == 3:
                return cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            elif len(img_np.shape) == 2: # Grayscale
                return cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)
            elif len(img_np.shape) == 3 and img_np.shape[2] == 4: # RGBA
                return cv2.cvtColor(img_np, cv2.COLOR_RGBA2BGR)
            return img_np
            
        if isinstance(image, np.ndarray):
            return image
            
        raise ValueError(f"Tipo de imagem desconhecido: {type(image)}")


# =============================================================================
# TESTES
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Testando PaddleOCRVoter...")
    
    voter = PaddleOCRVoter()
    
    # Criar imagem sintética com texto
    img = np.zeros((100, 300, 3), dtype=np.uint8) + 255
    cv2.putText(img, "TESTE 123", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    
    print("Rodando OCR em imagem sintética...")
    # Nota: Pode falhar se paddleocr não estiver instalado
    try:
        text = voter.extract_text(img)
        print(f"Texto extraído: '{text}'")
    except MemoryError:
        print("Erro de memória (esperado se não houver RAM suficiente)")
    except Exception as e:
        print(f"Erro ao rodar teste (possivelmente falta libs): {e}")
