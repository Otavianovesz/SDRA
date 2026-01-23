import logging
import fitz
from typing import Optional, Dict, Any, List
from PIL import Image
from dataclasses import dataclass
import torch
import torchvision

logger = logging.getLogger(__name__)

@dataclass
class SuryaResult:
    text: str
    layout: Any # Surya LayoutResult

class SuryaVoter:
    """
    Adapter for Surya OCR & Layout Analysis.
    High accuracy, moderate resource usage.
    """
    
    def __init__(self):
        self.ocr_model = None
        self.layout_model = None
        self.processor = None
        self.loaded = False
        
    def _load_models(self):
        """Lazy load models."""
        if self.loaded:
            return

        try:
            # Fix circular imports by loading torch/vision first
            import torch
            import torchvision
            
            logger.info("Loading Surya models (API v0.17)...")
            from surya.detection import DetectionPredictor
            from surya.foundation import FoundationPredictor
            from surya.recognition import RecognitionPredictor
            
            # Instantiate models
            # Note: FoundationPredictor loads the heavy weights
            self.foundation = FoundationPredictor()
            self.detector = DetectionPredictor()
            self.recognizer = RecognitionPredictor(self.foundation)
            
            self.loaded = True
            logger.info("Surya models ready")
            
        except ImportError:
            logger.warning("Surya not installed or incompatible version. Surya Voter disabled.")
            self.loaded = False
            return
        except Exception as e:
            logger.warning(f"Failed to load Surya: {e}")
            self.loaded = False
            return

    def unload_model(self):
        """Free memory."""
        self.foundation = None
        self.detector = None
        self.recognizer = None
        self.loaded = False
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Surya unloaded")

    def is_available(self) -> bool:
        return self.loaded

    def extract_text_with_layout(self, content: Any) -> Optional[SuryaResult]:
        """
        Run OCR on content (PDF path or PIL Image).
        """
        self._load_models()
        if not self.loaded:
            return None
        
        try:
            from surya.common.surya.schema import TaskNames
            
            images = []
            if isinstance(content, str):
                # PDF Path - Convert first page (simplification)
                doc = fitz.open(content)
                page = doc[0]
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                images = [Image.frombytes("RGB", [pix.width, pix.height], pix.samples)]
            elif isinstance(content, Image.Image):
                images = [content]
            elif isinstance(content, list): # List of images
                images = content
            else:
                # Numpy array
                images = [Image.fromarray(content)]
            
            # Run Inference using new API
            task_names = [TaskNames.ocr_with_boxes] * len(images)
            
            predictions = self.recognizer(
                images, 
                task_names=task_names, 
                det_predictor=self.detector,
                math_mode=False # Disable math for speed/finance docs
            )
            
            # Aggregate text
            full_text = ""
            layout_data = None
            
            for pred in predictions:
                if hasattr(pred, 'text_lines'):
                    for line in pred.text_lines:
                        full_text += line.text + "\n"
                layout_data = pred # Keep last page layout
                
            return SuryaResult(full_text, layout_data)
            
        except Exception as e:
            logger.warning(f"Surya OCR inference failed: {e}")
            return None

# Singleton
_voter = None

def get_surya_voter():
    global _voter
    if _voter is None:
        _voter = SuryaVoter()
    return _voter
