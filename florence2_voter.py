"""
Florence-2 Vision Voter - Fixed for 8GB RAM, CPU-only
======================================================

Fixes applied:
1. Changed task from <OD> to <OCR_WITH_REGION> (OD is for object detection, not OCR!)
2. Forced torch_dtype=float32 for CPU (float16 is inefficient on CPU)
3. Reduced num_beams from 3 to 1 (greedy search, 3x faster)
4. Added image resize for large images (max 1024px)
5. Added try/except for RuntimeError memory crashes
6. Immediate model cleanup after inference
"""

import os
import gc
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from PIL import Image
import fitz  # PyMuPDF

import config
from resource_manager import get_resource_manager

logger = logging.getLogger(__name__)

# Try to import torch
try:
    import torch
    torch.set_grad_enabled(False)
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class Florence2Voter:
    """
    Adapter for Microsoft Florence-2 Model.
    
    Provides:
    - OCR with region detection (<OCR_WITH_REGION>)
    - Dense captioning (<CAPTION> / <DETAILED_CAPTION>)
    
    Optimized for 8GB RAM, CPU-only:
    - float32 dtype (CPU doesn't benefit from float16)
    - Greedy decoding (num_beams=1)
    - Image resize to max 1024px
    - Immediate cleanup after use
    """
    
    # Maximum image dimension (larger images are resized)
    MAX_IMAGE_SIZE = 1024
    
    # Available tasks
    TASK_OCR = config.FLORENCE_CONFIG["task"]
    TASK_CAPTION = "<CAPTION>"
    TASK_DETAILED_CAPTION = "<DETAILED_CAPTION>"
    TASK_OD = "<OD>"  # Object Detection - NOT for OCR!
    
    def __init__(self, model_id: str = "microsoft/Florence-2-base"):
        self.model_id = model_id
        self.model = None
        self.processor = None
        self.device = "cpu"  # Force CPU
        
    def _load_model(self):
        """Lazy load the model with CPU optimizations."""
        if self.model is not None:
            return
        
        if not TORCH_AVAILABLE:
            logger.warning("PyTorch not available, Florence-2 disabled")
            return
        
        try:
            logger.info(f"Loading Florence-2 ({self.model_id}) on CPU...")
            
            # Try different processor imports based on transformers version
            processor_cls = None
            try:
                from transformers import AutoProcessor
                processor_cls = AutoProcessor
            except ImportError:
                try:
                    from transformers import AutoImageProcessor
                    processor_cls = AutoImageProcessor
                except ImportError:
                    try:
                        from transformers import CLIPProcessor as processor_cls
                    except ImportError:
                        logger.warning("No suitable processor class found in transformers")
                        return
            
            from transformers import AutoModelForCausalLM
            
            self.processor = processor_cls.from_pretrained(
                self.model_id, 
                trust_remote_code=True
            )
            
            # CRITICAL: Use float32 on CPU (float16 is slower and less accurate on CPU)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                trust_remote_code=True,
                torch_dtype=torch.float32,  # FIX: Was float16
                low_cpu_mem_usage=True
            ).to(self.device)
            
            # Put in eval mode
            self.model.eval()
            
            logger.info("Florence-2 loaded successfully (CPU, float32)")
            
        except Exception as e:
            logger.warning(f"Failed to load Florence-2: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            self.model = None
            self.processor = None

    def unload_model(self):
        """Free memory immediately."""
        if self.model:
            del self.model
        if self.processor:
            del self.processor
        
        self.model = None
        self.processor = None
        
        # Aggressive cleanup
        get_resource_manager().force_cleanup()
        
        logger.info("Florence-2 unloaded")

    def is_available(self) -> bool:
        """Check if model is loaded and ready."""
        return self.model is not None

    def _resize_image(self, img: Image.Image) -> Image.Image:
        """
        Resize image if too large (saves RAM and inference time).
        
        Args:
            img: PIL Image
            
        Returns:
            Resized image (or original if small enough)
        """
        max_dim = max(img.width, img.height)
        if max_dim <= self.MAX_IMAGE_SIZE:
            return img
        
        # Calculate scale factor
        scale = self.MAX_IMAGE_SIZE / max_dim
        new_width = int(img.width * scale)
        new_height = int(img.height * scale)
        
        logger.debug(f"Resizing image from {img.width}x{img.height} to {new_width}x{new_height}")
        return img.resize((new_width, new_height), Image.Resampling.LANCZOS)

    def _run_inference(
        self, 
        img: Image.Image, 
        task: str = "<OCR_WITH_REGION>"
    ) -> Dict[str, Any]:
        """
        Run Florence-2 inference on a single image.
        
        Args:
            img: PIL Image
            task: Task prompt (e.g., <OCR_WITH_REGION>, <CAPTION>)
            
        Returns:
            Parsed model output
        """
        try:
            # Resize if needed
            img = self._resize_image(img)
            original_size = (img.width, img.height)
            
            # Prepare inputs
            inputs = self.processor(
                text=task, 
                images=img, 
                return_tensors="pt"
            ).to(self.device)
            
            # Generate with GREEDY decoding (num_beams=1 is 3x faster)
            with torch.inference_mode():
                generated_ids = self.model.generate(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs["pixel_values"],
                    max_new_tokens=512,  # Reduced for speed
                    do_sample=False,
                    num_beams=config.FLORENCE_CONFIG.get("num_beams", 1),  # Greedy is faster
                )
            
            # Decode output
            generated_text = self.processor.batch_decode(
                generated_ids, 
                skip_special_tokens=False
            )[0]
            
            # Post-process with correct image size
            parsed = self.processor.post_process_generation(
                generated_text, 
                task=task, 
                image_size=original_size  # FIX: Use actual dimensions
            )
            
            return parsed
            
        except RuntimeError as e:
            # Memory errors
            if "out of memory" in str(e).lower():
                logger.error("Out of memory during Florence-2 inference")
                gc.collect()
            raise
        except Exception as e:
            logger.warning(f"Florence-2 inference error: {e}")
            return {}

    def extract_text(self, content: Any) -> str:
        """
        Extract text using OCR_WITH_REGION task.
        
        This is the PRIMARY method for text extraction.
        
        Args:
            content: PDF path, PIL Image, or numpy array
            
        Returns:
            Extracted text
        """
        self._load_model()
        if not self.is_available():
            return ""
        
        try:
            # Convert to PIL if needed
            if isinstance(content, str):
                # PDF path - extract first page
                img = self._pdf_page_to_image(content, page_num=0)
                if img is None:
                    return ""
            elif isinstance(content, Image.Image):
                img = content
            else:
                # Assume numpy array
                img = Image.fromarray(content)
            
            # Run OCR task (FIX: Was using <OD> which is for objects, not text!)
            result = self._run_inference(img, self.TASK_OCR)
            
            # Extract text from OCR result
            if self.TASK_OCR in result:
                ocr_data = result[self.TASK_OCR]
                if isinstance(ocr_data, dict) and 'labels' in ocr_data:
                    return "\n".join(ocr_data['labels'])
                elif isinstance(ocr_data, str):
                    return ocr_data
            
            return ""
            
        except Exception as e:
            logger.error(f"Florence-2 text extraction failed: {e}")
            return ""
        
        finally:
            # Immediate cleanup
            self.unload_model()

    def extract_from_pdf(
        self, 
        pdf_path: str, 
        page_num: int = 0,
        task: str = "<OCR_WITH_REGION>"
    ) -> Dict[str, Any]:
        """
        Run Florence-2 on a PDF page.
        
        Args:
            pdf_path: Path to PDF file
            page_num: Page number (0-indexed)
            task: Task to run (default: OCR_WITH_REGION)
            
        Returns:
            Dict with extracted data (text, regions, etc.)
        """
        self._load_model()
        results = {}
        
        if not self.is_available():
            results["error"] = "Model not available"
            return results
        
        try:
            # Render page to image
            img = self._pdf_page_to_image(pdf_path, page_num)
            if img is None:
                results["error"] = "Failed to render PDF page"
                return results
            
            # Run inference
            parsed = self._run_inference(img, task)
            
            # Format results based on task
            if task == self.TASK_OCR:
                results["text"] = ""
                results["regions"] = []
                if self.TASK_OCR in parsed:
                    ocr_data = parsed[self.TASK_OCR]
                    if isinstance(ocr_data, dict):
                        if 'labels' in ocr_data:
                            results["text"] = "\n".join(ocr_data['labels'])
                        if 'bboxes' in ocr_data and 'labels' in ocr_data:
                            for label, bbox in zip(ocr_data['labels'], ocr_data['bboxes']):
                                results["regions"].append({
                                    "text": label,
                                    "bbox": bbox
                                })
                                
            elif task in (self.TASK_CAPTION, self.TASK_DETAILED_CAPTION):
                results["caption"] = parsed.get(task, "")
                
            elif task == self.TASK_OD:
                # Object detection (NOT for OCR - use for detecting tables, barcodes, etc.)
                results["objects"] = []
                if self.TASK_OD in parsed:
                    od_data = parsed[self.TASK_OD]
                    if isinstance(od_data, dict) and 'labels' in od_data:
                        for label, bbox in zip(od_data['labels'], od_data['bboxes']):
                            results["objects"].append({
                                "label": label,
                                "bbox": bbox
                            })
            
        except RuntimeError as e:
            logger.error(f"Florence-2 RuntimeError: {e}")
            results["error"] = str(e)
            gc.collect()
            
        except Exception as e:
            logger.error(f"Florence-2 inference failed: {e}")
            results["error"] = str(e)
        
        finally:
            # Immediate cleanup
            self.unload_model()
            
        return results

    def _pdf_page_to_image(self, pdf_path: str, page_num: int = 0) -> Optional[Image.Image]:
        """
        Convert a PDF page to PIL Image.
        
        Uses 1.5x zoom (150 DPI) to balance quality and memory.
        """
        try:
            doc = fitz.open(pdf_path)
            if page_num >= len(doc):
                page_num = 0
            
            page = doc[page_num]
            # 1.5x zoom = ~150 DPI (good balance of quality and memory)
            matrix = fitz.Matrix(1.5, 1.5)
            pix = page.get_pixmap(matrix=matrix)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            
            doc.close()
            return img
            
        except Exception as e:
            logger.warning(f"Failed to convert PDF page to image: {e}")
            return None


# =============================================================================
# SINGLETON
# =============================================================================

_voter: Optional[Florence2Voter] = None


def get_florence2_voter() -> Florence2Voter:
    """Get or create the Florence-2 voter singleton."""
    global _voter
    if _voter is None:
        _voter = Florence2Voter()
    return _voter
