import pytest
import numpy as np
import cv2
from vision.preprocessing import ImagePreprocessor
from vision.feature_extraction import VisionFeatureExtractor
from vision.defect_inspector import ProductDefectInspector
from vision.ocr_parser import DocumentOCRParser

@pytest.fixture
def sample_synthetic_image():
    # 224x224x3 RGB numpy array, clean image
    img = np.ones((224, 224, 3), dtype=np.uint8) * 200  # Light gray background
    return img

@pytest.fixture
def sample_defective_image():
    # 224x224x3 numpy array with artificial scratch
    img = np.ones((224, 224, 3), dtype=np.uint8) * 200
    # Add a white "scratch" (edge)
    cv2.line(img, (50, 50), (150, 150), (255, 255, 255), 3)
    # Add some noise
    noise = np.random.randint(0, 50, (224, 224, 3), dtype=np.uint8)
    img = cv2.add(img, noise)
    return img

@pytest.fixture
def sample_receipt_image():
    # 224x224x3 numpy array representing receipt document
    img = np.ones((224, 224, 3), dtype=np.uint8) * 255 # White background
    # Add some "text" lines
    cv2.rectangle(img, (20, 20), (100, 30), (0, 0, 0), -1)
    cv2.rectangle(img, (20, 40), (150, 50), (0, 0, 0), -1)
    cv2.rectangle(img, (20, 60), (80, 70), (0, 0, 0), -1)
    return img

@pytest.fixture
def preprocessor():
    return ImagePreprocessor(target_size=(224, 224))

@pytest.fixture
def feature_extractor():
    return VisionFeatureExtractor(use_pretrained=False)

@pytest.fixture
def defect_inspector():
    return ProductDefectInspector(threshold=0.3)

@pytest.fixture
def ocr_parser():
    return DocumentOCRParser()
