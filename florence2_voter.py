import logging
import torch
import torchvision
from pathlib import Path
from typing import Dict, Any, List
from PIL import Image
import fitz  # PyMuPDF

logger = logging.getLogger(__name__)

class Florence2Voter:
    """
    Adapter for Microsoft Florence-2 Model.
    Provides layout analysis, object detection, and spot OCR.
    """
    
    def __init__(self, model_id: str = "microsoft/Florence-2-base"):
        self.model_id = model_id
        self.model = None
        self.processor = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    def _load_model(self):
        """Lazy load the model."""
        if self.model is not None:
            return

        try:
            import torch
            import torchvision
            
            try:
                from transformers import AutoProcessor
            except ImportError:
                # Fallback for some versions/environments
                from transformers.models.auto.processing_auto import AutoProcessor

            from transformers import AutoModelForCausalLM
            
            logger.info(f"Loading Florence-2 ({self.model_id}) on {self.device}...")
            self.processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id, 
                trust_remote_code=True,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            ).to(self.device)
            logger.info("Florence-2 loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load Florence-2: {e}")
            self.model = None
            self.processor = None
            # Don't raise, just stay disabled

    def unload_model(self):
        """Free memory."""
        if self.model:
            del self.model
            del self.processor
            self.model = None
            self.processor = None
            if self.device == "cuda":
                torch.cuda.empty_cache()
            logger.info("Florence-2 unloaded")

    def is_available(self) -> bool:
        return self.model is not None

    def extract_from_pdf(self, pdf_path: str, page_num: int = 0) -> Dict[str, Any]:
        """
        Run Florence-2 on a PDF page.
        Tasks: <OD> (Object Detection), <OCR> (if needed), <CAPTION>
        """
        self._load_model()
        results = {}
        
        try:
            # Render page to image
            doc = fitz.open(pdf_path)
            page = doc[page_num]
            pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5)) # 1.5x zoom for better details
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            
            # 1. Object Detection (Find barcodes, tables, signatures)
            task = "<OD>"
            inputs = self.processor(text=task, images=img, return_tensors="pt").to(self.device, self.torch_dtype)
            
            generated_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
                do_sample=False,
                num_beams=3
            )
            
            generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
            parsed_answer = self.processor.post_process_generation(generated_text, task=task, image_size=(img.width, img.height))
            
            results["objects"] = []
            if task in parsed_answer:
                # Convert to simple list
                for label, bbox in zip(parsed_answer[task]['labels'], parsed_answer[task]['bboxes']):
                    results["objects"].append({
                        "label": label,
                        "bbox": bbox  # [x1, y1, x2, y2]
                    })
            
            # 2. Dense Region Caption (Optional - skipped for speed)
            
        except Exception as e:
            logger.error(f"Florence-2 inference failed: {e}")
            results["error"] = str(e)
            
        return results
        
    @property
    def torch_dtype(self):
        return torch.float16 if self.device == "cuda" else torch.float32

# Singleton factory
_voter = None

def get_florence2_voter():
    global _voter
    if _voter is None:
        _voter = Florence2Voter()
    return _voter
