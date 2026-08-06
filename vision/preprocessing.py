import cv2
import numpy as np
from PIL import Image
import io
from pathlib import Path
from typing import Union, Optional, Tuple

from shared.logging import get_logger
from shared.exceptions import PipelineError

logger = get_logger(__name__)

class ImagePreprocessor:
    """Preprocesses images for computer vision tasks."""

    def __init__(self, target_size: Tuple[int, int] = (224, 224), normalize: bool = True):
        """
        Initializes the ImagePreprocessor.

        Args:
            target_size (tuple[int, int]): Target dimensions for resizing (width, height).
            normalize (bool): Whether to apply ImageNet normalization by default.
        """
        self.target_size = target_size
        self.normalize = normalize
        logger.info(f"Initialized ImagePreprocessor with target_size={self.target_size}, normalize={self.normalize}")

    def load_image(self, input_data: Union[str, Path, bytes, Image.Image, np.ndarray]) -> np.ndarray:
        """
        Loads an image from various input types and converts it to an RGB uint8 numpy array.
        
        Args:
            input_data: The image source (file path, bytes, PIL Image, or numpy array).
            
        Returns:
            np.ndarray: Image as an RGB uint8 numpy array (H, W, C).
        """
        try:
            if isinstance(input_data, (str, Path)):
                img = Image.open(str(input_data))
                img = img.convert("RGB")
                return np.array(img, dtype=np.uint8)
            elif isinstance(input_data, bytes):
                img = Image.open(io.BytesIO(input_data))
                img = img.convert("RGB")
                return np.array(img, dtype=np.uint8)
            elif isinstance(input_data, Image.Image):
                img = input_data.convert("RGB")
                return np.array(img, dtype=np.uint8)
            elif isinstance(input_data, np.ndarray):
                # Ensure the array is valid
                if input_data.size == 0:
                    raise ValueError("Empty numpy array provided.")
                    
                if input_data.ndim == 2:
                    # Grayscale to RGB
                    return cv2.cvtColor(input_data, cv2.COLOR_GRAY2RGB)
                elif input_data.ndim == 3:
                    if input_data.shape[2] == 4:
                        # RGBA to RGB
                        return cv2.cvtColor(input_data, cv2.COLOR_RGBA2RGB)
                    elif input_data.shape[2] == 3:
                        # Convert float [0, 1] to uint8 [0, 255] if necessary
                        if input_data.dtype in (np.float32, np.float64) and input_data.max() <= 1.0:
                            input_data = (input_data * 255).astype(np.uint8)
                        elif input_data.dtype != np.uint8:
                            input_data = input_data.astype(np.uint8)
                        return input_data
                raise ValueError(f"Unsupported numpy array shape: {input_data.shape}")
            else:
                raise TypeError(f"Unsupported input type: {type(input_data)}")
        except Exception as e:
            logger.error(f"Failed to load image: {e}")
            raise PipelineError(f"Image loading failed: {str(e)}") from e

    def resize(self, image: np.ndarray, target_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """
        Resizes the image to the target size.
        
        Args:
            image: Image to resize.
            target_size: Optional size (W, H). If None, uses self.target_size.
            
        Returns:
            np.ndarray: Resized image.
        """
        try:
            if image is None or image.size == 0:
                raise ValueError("Invalid image array.")
                
            size = target_size if target_size is not None else self.target_size
            if size is None:
                return image
            return cv2.resize(image, size, interpolation=cv2.INTER_AREA)
        except Exception as e:
            logger.error(f"Failed to resize image: {e}")
            raise PipelineError(f"Image resize failed: {str(e)}") from e

    def to_grayscale(self, image: np.ndarray) -> np.ndarray:
        """
        Converts an RGB image to grayscale.
        """
        try:
            if image is None or image.size == 0:
                raise ValueError("Invalid image array.")
                
            if image.ndim == 2:
                return image
            if image.shape[2] == 3:
                return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            elif image.shape[2] == 4:
                return cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
            return image
        except Exception as e:
            logger.error(f"Failed to convert to grayscale: {e}")
            raise PipelineError(f"Grayscale conversion failed: {str(e)}") from e

    def normalize_imagenet(self, image: np.ndarray) -> np.ndarray:
        """
        Normalizes a float32 array using ImageNet mean [0.485, 0.456, 0.406] and std [0.229, 0.224, 0.225].
        """
        try:
            if image is None or image.size == 0:
                raise ValueError("Invalid image array.")
                
            img_float = image.astype(np.float32) / 255.0
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            return (img_float - mean) / std
        except Exception as e:
            logger.error(f"Failed to normalize image: {e}")
            raise PipelineError(f"ImageNet normalization failed: {str(e)}") from e

    def apply_gaussian_blur(self, image: np.ndarray, kernel_size: Tuple[int, int] = (5, 5)) -> np.ndarray:
        """
        Applies Gaussian blur to the image.
        """
        try:
            if image is None or image.size == 0:
                raise ValueError("Invalid image array.")
                
            return cv2.GaussianBlur(image, kernel_size, 0)
        except Exception as e:
            logger.error(f"Failed to apply Gaussian blur: {e}")
            raise PipelineError(f"Gaussian blur failed: {str(e)}") from e

    def threshold_otsu(self, image: np.ndarray) -> np.ndarray:
        """
        Applies Otsu's binary thresholding.
        """
        try:
            if image is None or image.size == 0:
                raise ValueError("Invalid image array.")
                
            gray = self.to_grayscale(image)
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            return thresh
        except Exception as e:
            logger.error(f"Failed to apply Otsu thresholding: {e}")
            raise PipelineError(f"Otsu thresholding failed: {str(e)}") from e
