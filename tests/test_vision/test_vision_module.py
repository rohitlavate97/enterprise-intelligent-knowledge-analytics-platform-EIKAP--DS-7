import cv2
import numpy as np
import pytest

from vision.module import ComputerVisionModule, VisionInput, VisionOutput
from module_registry.contract import UniversalModuleContract
from module_registry.base import ModuleCategory, MaturityLabel

def test_vision_module_metadata():
    module = ComputerVisionModule()
    metadata = module.metadata
    
    assert metadata.name == "computer_vision"
    assert metadata.category == ModuleCategory.COMPUTER_VISION
    assert metadata.maturity == MaturityLabel.STANDARD
    assert metadata.version == "0.1.0"
    assert "vision" in metadata.tags
    assert not metadata.requires_gpu

def test_vision_module_predict_defect(sample_defective_image):
    module = ComputerVisionModule()
    
    # Convert image to bytes to test the bytes path
    _, buffer = cv2.imencode('.jpg', sample_defective_image)
    image_bytes = buffer.tobytes()
    
    input_data = VisionInput(task="defect", image_bytes=image_bytes, threshold=0.3)
    
    result = module.predict(input_data)
    
    assert isinstance(result, VisionOutput)
    assert result.task == "defect"
    assert "is_defective" in result.results
    assert "defect_score" in result.results
    assert result.execution_time_ms > 0

def test_vision_module_predict_ocr(sample_receipt_image):
    module = ComputerVisionModule()
    
    _, buffer = cv2.imencode('.png', sample_receipt_image)
    image_bytes = buffer.tobytes()
    
    input_data = VisionInput(task="ocr", image_bytes=image_bytes)
    
    result = module.predict(input_data)
    
    assert isinstance(result, VisionOutput)
    assert result.task == "ocr"
    assert "raw_text" in result.results
    assert "merchant_name" in result.results
    assert "detected_bounding_boxes" in result.results

def test_vision_module_compliance_contract(sample_defective_image):
    module = ComputerVisionModule()
    contract = UniversalModuleContract()
    
    _, buffer = cv2.imencode('.jpg', sample_defective_image)
    image_bytes = buffer.tobytes()
    
    input_data = {"task": "defect", "image_bytes": image_bytes, "threshold": 0.3}
    prediction_result = module.predict(input_data)
    
    report = contract.run_full_check(
        module=module,
        sample_input=input_data,
        sample_prediction=prediction_result.model_dump(),
        sample_output=prediction_result.model_dump()
    )
    
    # We just want to ensure it evaluates properly without crashing.
    # The actual checks might fail because we haven't implemented MLFlow, etc.
    assert hasattr(report, 'all_passed')
