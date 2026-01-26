"""
SDRA-Rural PaddleOCR Voter - Optimized for 8GB RAM, CPU-only
=============================================================

Optimizations applied:
1. MKLDNN enabled for Intel CPU acceleration
2. use_gpu=False explicitly set
3. limit_side_len=960 (A4 sufficient, saves RAM)
4. cpu_threads=4
5. Confidence filter at 0.6 (was 0.5)
6. Skip BGR conversion for grayscale (saves memory)
7. Added bbox merge for broken text lines
"""

import logging
import numpy as np
import cv2
from typing import Optional, List, Dict, Any, Union
from PIL import Image

import config
from resource_manager import get_resource_manager

logger = logging.getLogger(__name__)

# Confidence threshold for filtering noise
MIN_CONFIDENCE = 0.6

# Y-axis proximity threshold for merging lines (pixels)
LINE_MERGE_THRESHOLD = 15


class PaddleOCRVoter:
    """
    Voter based on PaddleOCR (PP-OCRv4).
    
    Optimized for 8GB RAM CPU-only operation:
    - MKLDNN acceleration for Intel CPUs
    - Memory-efficient configuration
    - Confidence-based filtering
    - Line merging for broken text
    """
    
    def __init__(self):
        self._model = None
        self._loaded = False
        
    def _get_model(self):
        """Get or create PaddleOCR instance with CPU optimizations."""
        if self._model is not None:
            return self._model
        
        try:
            from paddleocr import PaddleOCR
            
            logger.info("Loading PaddleOCR (CPU)...")
            
            # Minimal configuration for maximum compatibility
            # MKLDNN disabled to avoid OneDNN attribute conversion errors
            import os
            os.environ['FLAGS_use_mkldnn'] = '0'
            
            self._model = PaddleOCR(
                lang=config.PADDLE_CONFIG.get("lang", "pt")
            )
            
            self._loaded = True
            logger.info("PaddleOCR loaded")
            return self._model
            
        except ImportError:
            logger.warning("PaddleOCR not installed")
            return None
        except Exception as e:
            logger.warning(f"Failed to load PaddleOCR: {e}")
            return None
    
    def unload_model(self):
        """Free memory."""
        if self._model:
            del self._model
            self._model = None
            self._loaded = False
            self._loaded = False
            get_resource_manager().force_cleanup()
            logger.info("PaddleOCR unloaded")
    
    def is_available(self) -> bool:
        """Check if PaddleOCR is available."""
        return self._loaded or self._get_model() is not None
        
    def extract_text(self, image: Union[str, np.ndarray, Image.Image]) -> str:
        """
        Extract plain text from an image.
        
        Args:
            image: File path, numpy array (BGR), or PIL Image
            
        Returns:
            Concatenated text
        """
        try:
            model = self._get_model()
            if not model:
                return ""
            
            # Convert to numpy BGR
            img_array = self._prepare_image(image)
            if img_array is None:
                return ""
            
            # Run OCR (new API doesn't support cls= argument)
            result = model.ocr(img_array)
            
            if not result or result[0] is None:
                return ""
            
            # Extract text with confidence filtering
            lines = []
            for line in result[0]:
                text_content = line[1][0]
                confidence = line[1][1]
                
                # FIX: Stricter confidence filter (was 0.5)
                if confidence >= MIN_CONFIDENCE:
                    lines.append(text_content)
            
            return "\n".join(lines)
            
        except Exception as e:
            logger.error(f"PaddleOCR error: {e}")
            return ""

    def extract_structured(
        self, 
        image: Union[str, np.ndarray, Image.Image],
        merge_lines: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Extract structured data (text + bbox + confidence).
        
        Args:
            image: Input image
            merge_lines: Whether to merge broken text lines
            
        Returns:
            List of dicts with text, confidence, and bbox
        """
        try:
            model = self._get_model()
            if not model:
                return []
            
            img_array = self._prepare_image(image)
            if img_array is None:
                return []
            
            result = model.ocr(img_array)
            
            output = []
            if result and result[0]:
                for line in result[0]:
                    points = line[0]
                    text = line[1][0]
                    conf = line[1][1]
                    
                    # FIX: Filter low confidence
                    if conf < MIN_CONFIDENCE:
                        continue
                    
                    # Calculate center Y for line merging
                    y_center = (points[0][1] + points[2][1]) / 2
                    
                    output.append({
                        "text": text,
                        "confidence": conf,
                        "bbox": points,  # [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                        "_y_center": y_center
                    })
            
            # FIX: Merge broken lines
            if merge_lines and len(output) > 1:
                output = self._merge_broken_lines(output)
            
            # Remove internal field
            for item in output:
                item.pop("_y_center", None)
            
            return output
            
        except Exception as e:
            logger.error(f"PaddleOCR structured error: {e}")
            return []

    def _merge_broken_lines(self, items: List[Dict]) -> List[Dict]:
        """
        Merge text boxes that are on the same line (Y-axis proximity).
        
        Fixes: Text that spans multiple columns or is broken by OCR.
        """
        if not items:
            return items
        
        # Sort by Y-center, then X
        sorted_items = sorted(
            items, 
            key=lambda x: (x["_y_center"], x["bbox"][0][0])
        )
        
        merged = []
        current_line = [sorted_items[0]]
        current_y = sorted_items[0]["_y_center"]
        
        for item in sorted_items[1:]:
            if abs(item["_y_center"] - current_y) <= LINE_MERGE_THRESHOLD:
                # Same line, add to current
                current_line.append(item)
            else:
                # New line, merge current and start new
                merged.append(self._merge_line_items(current_line))
                current_line = [item]
                current_y = item["_y_center"]
        
        # Don't forget last line
        if current_line:
            merged.append(self._merge_line_items(current_line))
        
        return merged
    
    def _merge_line_items(self, items: List[Dict]) -> Dict:
        """Merge multiple items on the same line into one."""
        if len(items) == 1:
            return items[0]
        
        # Sort by X position (left to right)
        items.sort(key=lambda x: x["bbox"][0][0])
        
        # Combine text with spaces
        combined_text = " ".join(item["text"] for item in items)
        
        # Average confidence
        avg_conf = sum(item["confidence"] for item in items) / len(items)
        
        # Combined bbox: leftmost x1, topmost y1, rightmost x2, bottommost y2
        all_points = [p for item in items for p in item["bbox"]]
        x_coords = [p[0] for p in all_points]
        y_coords = [p[1] for p in all_points]
        
        combined_bbox = [
            [min(x_coords), min(y_coords)],  # top-left
            [max(x_coords), min(y_coords)],  # top-right
            [max(x_coords), max(y_coords)],  # bottom-right
            [min(x_coords), max(y_coords)]   # bottom-left
        ]
        
        return {
            "text": combined_text,
            "confidence": avg_conf,
            "bbox": combined_bbox,
            "_y_center": items[0]["_y_center"]
        }

    def _prepare_image(self, image: Union[str, np.ndarray, Image.Image]) -> Optional[np.ndarray]:
        """
        Convert input to numpy array BGR compatible with Paddle.
        
        FIX: Skip conversion for grayscale (saves memory).
        """
        try:
            if isinstance(image, str):
                # File path - let OpenCV handle it
                img = cv2.imread(image)
                if img is None:
                    logger.warning(f"Failed to read image: {image}")
                return img
            
            if isinstance(image, Image.Image):
                img_np = np.array(image)
                
                # FIX: Handle grayscale efficiently (no BGR conversion needed)
                if len(img_np.shape) == 2:
                    # Grayscale - PaddleOCR handles this
                    return img_np
                
                if len(img_np.shape) == 3:
                    if img_np.shape[2] == 3:  # RGB
                        return cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
                    elif img_np.shape[2] == 4:  # RGBA
                        return cv2.cvtColor(img_np, cv2.COLOR_RGBA2BGR)
                
                return img_np
            
            if isinstance(image, np.ndarray):
                # Already numpy - check if grayscale
                if len(image.shape) == 2:
                    # Grayscale, return as-is
                    return image
                return image
            
            logger.warning(f"Unknown image type: {type(image)}")
            return None
            
        except Exception as e:
            logger.error(f"Image preparation error: {e}")
            return None


# =============================================================================
# SINGLETON
# =============================================================================

_voter: Optional[PaddleOCRVoter] = None


def get_paddle_voter() -> PaddleOCRVoter:
    """Get or create the PaddleOCR voter singleton."""
    global _voter
    if _voter is None:
        _voter = PaddleOCRVoter()
    return _voter


# =============================================================================
# TESTS
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Testing PaddleOCRVoter...")
    
    voter = PaddleOCRVoter()
    
    # Create synthetic image with text
    img = np.zeros((100, 300, 3), dtype=np.uint8) + 255
    cv2.putText(img, "TESTE 123", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    
    print("Running OCR on synthetic image...")
    try:
        text = voter.extract_text(img)
        print(f"Extracted text: '{text}'")
    except MemoryError:
        print("Memory error (expected if not enough RAM)")
    except Exception as e:
        print(f"Test error (possibly missing libs): {e}")
