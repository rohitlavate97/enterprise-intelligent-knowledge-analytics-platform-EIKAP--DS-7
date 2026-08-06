import numpy as np
import pytest
from vision.feature_extraction import VisionFeatureExtractor

def test_extract_deep_features(sample_synthetic_image):
    # Enable deep feature extraction for this test only
    extractor = VisionFeatureExtractor(use_pretrained=True)
    features = extractor.extract_deep_features(sample_synthetic_image)
    
    assert isinstance(features, np.ndarray)
    assert features.ndim == 1
    # ResNet18 output feature size is typically 1000 or 512 depending on layer
    assert len(features) in [1000, 512]

def test_extract_color_histogram(feature_extractor, sample_synthetic_image):
    hist = feature_extractor.extract_color_histogram(sample_synthetic_image, bins=32)
    assert isinstance(hist, np.ndarray)
    # 3 channels * 32 bins
    assert len(hist) == 32 * 3
