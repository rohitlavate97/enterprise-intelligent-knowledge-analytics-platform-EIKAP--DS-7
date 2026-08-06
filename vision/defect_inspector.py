import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Union, Optional
from shared.logging import get_logger

logger = get_logger(__name__)

class ProductDefectInspector:
    """
    Standard maturity computer vision component for manufacturing product defect detection.
    Analyzes images for structural anomalies, surface scratches, discoloration, or edge irregularities.
    """

    def __init__(self, threshold: float = 0.3):
        self.threshold = threshold
        logger.info(f"Initialized ProductDefectInspector with threshold {self.threshold}")

    def _load_image(self, image: Union[str, Path, bytes, np.ndarray]) -> Optional[np.ndarray]:
        try:
            if isinstance(image, (str, Path)):
                img = cv2.imread(str(image))
                if img is None:
                    logger.error(f"Failed to load image from path: {image}")
                    return None
                return img
            elif isinstance(image, bytes):
                nparr = np.frombuffer(image, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                return img
            elif isinstance(image, np.ndarray):
                return image
            else:
                logger.error(f"Unsupported image type: {type(image)}")
                return None
        except Exception as e:
            logger.error(f"Error loading image: {e}")
            return None

    def inspect_image(self, image: Union[str, Path, bytes, np.ndarray]) -> Dict[str, Any]:
        """
        Analyzes image for structural anomalies, surface scratches, discoloration, or edge irregularities.
        """
        img = self._load_image(image)
        if img is None:
            return {
                "is_defective": False,
                "defect_score": 0.0,
                "defect_type": "none",
                "confidence": 0.0,
                "bounding_box": None
            }

        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blurred, 50, 150)
            
            # Feature variance to detect structural anomalies
            variance = np.var(gray)
            
            # Detect contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            defect_score = 0.0
            defect_type = "none"
            bounding_box = None
            
            if len(contours) > 0:
                largest_contour = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(largest_contour)
                x, y, w, h = cv2.boundingRect(largest_contour)
                
                # Simple logic for defect types
                if area < 500 and variance > 2000:
                    defect_type = "discoloration"
                    defect_score = min(1.0, 0.4 + (variance / 10000.0))
                    bounding_box = (x, y, w, h)
                elif area > 1000 and (w / max(1, h) > 3 or h / max(1, w) > 3):
                    defect_type = "scratch"
                    defect_score = min(1.0, 0.5 + (area / 10000.0))
                    bounding_box = (x, y, w, h)
                elif edges.sum() > 500000:
                    defect_type = "structural_crack"
                    defect_score = min(1.0, 0.6 + (edges.sum() / 5000000.0))
                    bounding_box = (x, y, w, h)

            is_defective = defect_score > self.threshold
            confidence = float(np.clip(defect_score + 0.2, 0.0, 1.0))

            result = {
                "is_defective": is_defective,
                "defect_score": float(defect_score),
                "defect_type": defect_type,
                "confidence": confidence,
                "bounding_box": bounding_box
            }
            logger.debug(f"Inspection complete: {result}")
            return result

        except Exception as e:
            logger.error(f"Error during inspection: {e}")
            return {
                "is_defective": False,
                "defect_score": 0.0,
                "defect_type": "none",
                "confidence": 0.0,
                "bounding_box": None
            }

    def batch_inspect(self, images: List[Any]) -> List[Dict[str, Any]]:
        """
        Batch inspect images.
        """
        logger.info(f"Batch inspecting {len(images)} images.")
        results = []
        for img in images:
            results.append(self.inspect_image(img))
        return results
