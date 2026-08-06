import cv2
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from typing import Optional

from shared.logging import get_logger
from shared.exceptions import PipelineError

logger = get_logger(__name__)

class VisionFeatureExtractor:
    """Extracts visual features from images using traditional CV and deep learning."""

    def __init__(self, model_name: str = "resnet18", use_pretrained: bool = True):
        """
        Initializes the VisionFeatureExtractor.
        
        Args:
            model_name (str): The name of the torchvision model to use.
            use_pretrained (bool): Whether to use pretrained ImageNet weights.
        """
        self.model_name = model_name
        self.use_pretrained = use_pretrained
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model()
        
        # Standard transform for ImageNet models
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        logger.info(f"Initialized VisionFeatureExtractor with {model_name} on {self.device}")

    def _load_model(self) -> torch.nn.Module:
        """Loads the specified torchvision model, stripping the classification head."""
        try:
            if self.model_name == "resnet18":
                weights = models.ResNet18_Weights.DEFAULT if self.use_pretrained else None
                model = models.resnet18(weights=weights)
                # Remove fully connected layer to get feature extractor
                model = torch.nn.Sequential(*(list(model.children())[:-1]))
            elif self.model_name.startswith("mobilenet"):
                weights = models.MobileNet_V3_Small_Weights.DEFAULT if self.use_pretrained else None
                model = models.mobilenet_v3_small(weights=weights)
                model = model.features
            else:
                logger.warning(f"Model {self.model_name} not explicitly handled, defaulting to resnet18")
                weights = models.ResNet18_Weights.DEFAULT if self.use_pretrained else None
                model = models.resnet18(weights=weights)
                model = torch.nn.Sequential(*(list(model.children())[:-1]))
            
            model = model.to(self.device)
            model.eval()
            return model
        except Exception as e:
            logger.error(f"Failed to load model {self.model_name}: {e}")
            logger.info("Falling back to randomly initialized ResNet18 due to load failure.")
            try:
                model = models.resnet18(weights=None)
                model = torch.nn.Sequential(*(list(model.children())[:-1]))
                model = model.to(self.device)
                model.eval()
                return model
            except Exception as fallback_err:
                raise PipelineError(f"Model loading and fallback failed: {fallback_err}") from fallback_err

    def extract_deep_features(self, image: np.ndarray) -> np.ndarray:
        """
        Passes image tensor through pre-trained CNN feature backbone, returns 1D feature vector.
        
        Args:
            image (np.ndarray): Image as a numpy array.
            
        Returns:
            np.ndarray: 1D feature vector.
        """
        if image is None or image.size == 0:
            raise PipelineError("Invalid image provided for deep feature extraction.")
            
        try:
            # Ensure 3 channels
            if image.ndim == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
            elif image.shape[2] != 3:
                raise ValueError(f"Expected 3 channels, got {image.shape[2]}")

            tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                features = self.model(tensor)
            
            # Flatten to 1D
            feature_vector = features.cpu().numpy().flatten()
            return feature_vector
        except Exception as e:
            logger.error(f"Deep feature extraction failed: {e}")
            raise PipelineError(f"Deep feature extraction failed: {str(e)}") from e

    def extract_color_histogram(self, image: np.ndarray, bins: int = 32) -> np.ndarray:
        """
        Computes concatenated RGB color histogram.
        
        Args:
            image (np.ndarray): Image as a numpy array.
            bins (int): Number of bins for the histogram.
            
        Returns:
            np.ndarray: Concatenated and normalized histogram features.
        """
        if image is None or image.size == 0:
            raise PipelineError("Invalid image provided for color histogram extraction.")

        try:
            if image.ndim == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

            hist_features = []
            for i in range(3): # R, G, B channels
                hist = cv2.calcHist([image], [i], None, [bins], [0, 256])
                hist = cv2.normalize(hist, hist).flatten()
                hist_features.extend(hist)
            
            return np.array(hist_features, dtype=np.float32)
        except Exception as e:
            logger.error(f"Color histogram extraction failed: {e}")
            raise PipelineError(f"Color histogram extraction failed: {str(e)}") from e

    def extract_edges(self, image: np.ndarray, low_threshold: int = 50, high_threshold: int = 150) -> np.ndarray:
        """
        Performs Canny edge detection.
        
        Args:
            image (np.ndarray): Image as a numpy array.
            low_threshold (int): Low threshold for Canny.
            high_threshold (int): High threshold for Canny.
            
        Returns:
            np.ndarray: Binary image representing edges.
        """
        if image is None or image.size == 0:
            raise PipelineError("Invalid image provided for edge extraction.")

        try:
            if image.ndim == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image
            
            edges = cv2.Canny(gray, low_threshold, high_threshold)
            return edges
        except Exception as e:
            logger.error(f"Edge extraction failed: {e}")
            raise PipelineError(f"Edge extraction failed: {str(e)}") from e
