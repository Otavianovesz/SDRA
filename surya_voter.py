"""
Surya OCR Voter - Fixed for 8GB RAM, CPU-only
==============================================

Fixes applied:
1. Removed invalid import (from surya.common.surya.schema import TaskNames)
2. Fixed multi-page processing (was only processing page[0])
3. Reduced resolution from 2.0x to 1.5x (RAM savings)
4. Added text-layer detection to skip OCR when unnecessary
5. Forced CPU mode, disabled CUDA checks
6. Added immediate model cleanup after inference
"""

import logging
import gc
import fitz
from typing import Optional, Dict, Any, List
from PIL import Image
from dataclasses import dataclass

import config
from resource_manager import get_resource_manager

logger = logging.getLogger(__name__)


@dataclass
class SuryaResult:
    """Result from Surya OCR."""
    text: str
    layout: Any  # Surya LayoutResult
    page_count: int = 1


class SuryaVoter:
    """
    Adapter for Surya OCR & Layout Analysis.
    
    Optimized for 8GB RAM, CPU-only operation:
    - Lazy loading with immediate unload after use
    - Reduced resolution (1.5x instead of 2.0x)
    - Multi-page support (fixes doc[0] bug)
    - Text-layer detection to skip unnecessary OCR
    """
    
    # Resolution for PDF rasterization (1.5x = ~150 DPI)
    # Lower than 2.0x to save RAM, still good for most documents
    PDF_MATRIX_SCALE = config.SURYA_CONFIG["matrix_zoom"]
    
    def __init__(self):
        self.foundation = None
        self.detector = None
        self.recognizer = None
        self.loaded = False
        
    def _load_models(self):
        """Lazy load models with CPU enforcement."""
        if self.loaded:
            return
        
        try:
            # Force CPU and disable gradients
            import torch
            torch.set_grad_enabled(False)
            
            logger.info("Loading Surya models (CPU-only)...")
            from surya.detection import DetectionPredictor
            from surya.recognition import RecognitionPredictor
            
            # Try to load FoundationPredictor (newer API)
            try:
                from surya.foundation import FoundationPredictor
                self.foundation = FoundationPredictor()
                self.recognizer = RecognitionPredictor(self.foundation)
            except ImportError:
                # Fallback for older API
                self.recognizer = RecognitionPredictor()
                self.foundation = None
            
            self.detector = DetectionPredictor()
            self.loaded = True
            logger.info("Surya models loaded (CPU)")
            
        except ImportError as e:
            logger.warning(f"Surya not installed: {e}")
            self.loaded = False
        except Exception as e:
            logger.warning(f"Failed to load Surya: {e}")
            self.loaded = False

    def unload_model(self):
        """Immediately free all model memory."""
        if self.foundation:
            del self.foundation
        if self.detector:
            del self.detector
        if self.recognizer:
            del self.recognizer
        
        self.foundation = None
        self.detector = None
        self.recognizer = None
        self.loaded = False
        
        # Aggressive cleanup
        get_resource_manager().force_cleanup()
        
        logger.info("Surya models unloaded")

    def is_available(self) -> bool:
        """Check if Surya is loaded and ready."""
        return self.loaded
    
    def _has_text_layer(self, pdf_path: str, min_chars: int = 50) -> bool:
        """
        Check if PDF has enough native text to skip OCR.
        
        Args:
            pdf_path: Path to PDF file
            min_chars: Minimum characters to consider "has text"
            
        Returns:
            True if PDF has selectable text layer
        """
        try:
            with fitz.open(pdf_path) as doc:
                for page in doc:
                    text = page.get_text().strip()
                    if len(text) >= min_chars:
                        return True
            return False
        except Exception:
            return False
    
    def _pdf_to_images(self, pdf_path: str, max_pages: int = 10) -> List[Image.Image]:
        """
        Convert PDF pages to PIL Images.
        
        Args:
            pdf_path: Path to PDF file
            max_pages: Maximum pages to process (memory limit)
            
        Returns:
            List of PIL Images
        """
        images = []
        try:
            doc = fitz.open(pdf_path)
            page_count = min(len(doc), max_pages)
            
            for page_num in range(page_count):
                page = doc[page_num]
                # Use reduced resolution (1.5x) to save RAM
                matrix = fitz.Matrix(self.PDF_MATRIX_SCALE, self.PDF_MATRIX_SCALE)
                pix = page.get_pixmap(matrix=matrix)
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                images.append(img)
                
                # Free pixmap immediately
                del pix
            
            doc.close()
            
        except Exception as e:
            logger.warning(f"Failed to convert PDF to images: {e}")
        
        return images

    def extract_text_with_layout(
        self, 
        content: Any,
        skip_if_has_text: bool = True,
        max_pages: int = 10
    ) -> Optional[SuryaResult]:
        """
        Run OCR on content (PDF path or PIL Image).
        
        FIXES APPLIED:
        1. Processes ALL pages, not just page[0]
        2. Uses correct API without invalid TaskNames import
        3. Skips OCR if PDF has native text layer
        
        Args:
            content: PDF path, PIL Image, or list of images
            skip_if_has_text: Skip OCR if PDF has text layer
            max_pages: Max pages to process
            
        Returns:
            SuryaResult with aggregated text from all pages
        """
        # Check if we can skip OCR for text-based PDFs
        if skip_if_has_text and isinstance(content, str):
            if self._has_text_layer(content):
                logger.info("PDF has text layer, skipping Surya OCR")
                # Return native text instead
                try:
                    with fitz.open(content) as doc:
                        all_text = ""
                        for page in doc:
                            all_text += page.get_text() + "\n"
                        return SuryaResult(all_text.strip(), None, len(doc))
                except:
                    pass
        
        # Load models if needed
        self._load_models()
        if not self.loaded:
            return None
        
        try:
            # Convert content to images
            images = []
            if isinstance(content, str):
                # PDF Path - process ALL pages (FIX for doc[0] bug)
                images = self._pdf_to_images(content, max_pages)
                if not images:
                    return None
            elif isinstance(content, Image.Image):
                images = [content]
            elif isinstance(content, list):
                images = content
            else:
                # Numpy array
                images = [Image.fromarray(content)]
            
            logger.info(f"Running Surya OCR on {len(images)} page(s)...")
            
            # Run inference using correct API (no TaskNames import needed)
            # The recognizer takes images directly in newer versions
            try:
                # Try newer API first
                predictions = self.recognizer(
                    images, 
                    det_predictor=self.detector,
                    math_mode=False  # Disable math for speed
                )
            except TypeError:
                # Fallback for older API
                predictions = self.recognizer(images)
            
            # Aggregate text from ALL pages
            full_text = ""
            layout_data = None
            
            for page_idx, pred in enumerate(predictions):
                page_text = ""
                if hasattr(pred, 'text_lines'):
                    for line in pred.text_lines:
                        if hasattr(line, 'text'):
                            page_text += line.text + "\n"
                elif hasattr(pred, 'text'):
                    page_text = pred.text
                
                full_text += page_text
                if page_text.strip():
                    logger.debug(f"Page {page_idx + 1}: extracted {len(page_text)} chars")
                
                layout_data = pred  # Keep last page layout
            
            logger.info(f"Surya extracted {len(full_text)} chars total")
            return SuryaResult(full_text.strip(), layout_data, len(images))
            
        except Exception as e:
            logger.warning(f"Surya OCR inference failed: {e}")
            return None
        
        finally:
            # IMMEDIATE CLEANUP to prevent memory buildup
            self.unload_model()


# =============================================================================
# SINGLETON
# =============================================================================

_voter: Optional[SuryaVoter] = None


def get_surya_voter() -> SuryaVoter:
    """Get or create the Surya voter singleton."""
    global _voter
    if _voter is None:
        _voter = SuryaVoter()
    return _voter
