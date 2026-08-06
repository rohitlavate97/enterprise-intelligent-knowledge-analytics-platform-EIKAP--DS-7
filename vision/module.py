import time
import importlib
from typing import Any, Dict, Optional, Type
from pydantic import BaseModel, Field

from module_registry.base import EIKAPModule, ModuleMetadata, ModuleCategory, MaturityLabel
from vision.preprocessing import ImagePreprocessor
from vision.feature_extraction import VisionFeatureExtractor
from vision.defect_inspector import ProductDefectInspector
from vision.ocr_parser import DocumentOCRParser

class VisionInput(BaseModel):
    task: str = Field(default="defect", description="Task to perform: 'defect' or 'ocr'")
    image_bytes: Optional[bytes] = Field(default=None, description="Raw image bytes")
    image_path: Optional[str] = Field(default=None, description="Path to image file")
    threshold: float = Field(default=0.3, description="Threshold for defect detection")

class VisionOutput(BaseModel):
    task: str
    results: Dict[str, Any]
    execution_time_ms: float

class ComputerVisionModule(EIKAPModule):
    """Computer Vision & Image Analytics Engine."""

    def __init__(self):
        self._metadata = ModuleMetadata(
            name="computer_vision",
            version="0.1.0",
            description="Computer Vision & Image Analytics Engine",
            category=ModuleCategory.COMPUTER_VISION,
            maturity=MaturityLabel.STANDARD,
            author="EIKAP AI Engineer",
            dependencies=["opencv-python", "pillow", "torchvision", "numpy"],
            tags=["vision", "defect", "ocr", "image"],
            requires_gpu=False
        )
        self.preprocessor = ImagePreprocessor()
        self.feature_extractor = VisionFeatureExtractor(use_pretrained=False) 

    @property
    def metadata(self) -> ModuleMetadata:
        return self._metadata

    @property
    def input_schema(self) -> Type[BaseModel]:
        return VisionInput

    @property
    def output_schema(self) -> Type[BaseModel]:
        return VisionOutput

    def train(self, data: Any, **kwargs) -> Dict[str, Any]:
        """
        Fine-tunes defect inspection threshold or feature extractor on training image dataset.
        For this prototype, it adjusts the threshold based on average defect scores.
        """
        if not data or not isinstance(data, list):
            return {"status": "error", "message": "Training data must be a list of images."}
        
        inspector = ProductDefectInspector(threshold=0.5)
        scores = []
        for img_data in data:
            img = self.preprocessor.load_image(img_data)
            img = self.preprocessor.resize(img)
            result = inspector.inspect_image(img)
            scores.append(result.get("defect_score", 0.0))
            
        avg_score = sum(scores) / len(scores) if scores else 0.3
        new_threshold = avg_score * 1.2 # Set threshold slightly above average normal score
        
        return {
            "status": "success",
            "message": "Fine-tuned defect inspection threshold.",
            "old_threshold": 0.5,
            "new_threshold": new_threshold,
            "samples_processed": len(data)
        }

    def predict(self, input_data: Any, **kwargs) -> VisionOutput:
        """
        Routes to product defect inspection or document OCR receipt parsing based on task.
        """
        start_time = time.time()
        validated_input = self.validate_input(input_data)
        
        if validated_input.image_bytes is not None:
            image_source = validated_input.image_bytes
        elif validated_input.image_path is not None:
            image_source = validated_input.image_path
        else:
            raise ValueError("Either image_bytes or image_path must be provided.")
            
        img = self.preprocessor.load_image(image_source)
        img = self.preprocessor.resize(img)
        
        results = {}
        if validated_input.task == "defect":
            inspector = ProductDefectInspector(threshold=validated_input.threshold)
            results = inspector.inspect_image(img)
        elif validated_input.task == "ocr":
            ocr_parser = DocumentOCRParser()
            results = ocr_parser.parse_receipt(img)
        else:
            raise ValueError(f"Unknown task: {validated_input.task}. Supported tasks: 'defect', 'ocr'.")
            
        execution_time_ms = (time.time() - start_time) * 1000
        
        output_data = {
            "task": validated_input.task,
            "results": results,
            "execution_time_ms": execution_time_ms
        }
        
        return self.validate_output(output_data)

    def explain(self, input_data: Any, prediction: Any, **kwargs) -> Dict[str, Any]:
        """
        Returns defect bounding box or OCR text region contours.
        """
        if isinstance(prediction, VisionOutput):
            pred_dict = prediction.model_dump()
        else:
            pred_dict = prediction if isinstance(prediction, dict) else {}
            
        results = pred_dict.get("results", {})
        task = pred_dict.get("task", "unknown")
        
        explanation = {"task": task}
        if task == "defect":
            explanation["bounding_boxes"] = results.get("bounding_boxes", [])
            explanation["defect_score"] = results.get("defect_score", 0.0)
            explanation["reasoning"] = "Highlighted regions indicate areas with high edge density."
        elif task == "ocr":
            explanation["text_regions"] = results.get("text_regions", [])
            explanation["reasoning"] = "Highlighted contours indicate detected text blocks."
            
        return explanation

    def evaluate(self, test_data: Any, **kwargs) -> Dict[str, float]:
        """
        Evaluates defect detection precision & recall.
        test_data should be a list of dicts: {"image": data, "label": bool}
        """
        if not test_data or not isinstance(test_data, list):
            return {"precision": 0.0, "recall": 0.0, "f1_score": 0.0}
            
        inspector = ProductDefectInspector(threshold=0.3)
        tp = 0
        fp = 0
        fn = 0
        tn = 0
        
        for item in test_data:
            img = self.preprocessor.load_image(item["image"])
            img = self.preprocessor.resize(img)
            true_label = item["label"]
            result = inspector.inspect_image(img)
            pred_label = result.get("is_defective", False)
            
            if pred_label and true_label:
                tp += 1
            elif pred_label and not true_label:
                fp += 1
            elif not pred_label and true_label:
                fn += 1
            else:
                tn += 1
                
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1_score),
            "accuracy": float((tp + tn) / len(test_data)) if len(test_data) > 0 else 0.0
        }

    def health_check(self) -> Dict[str, Any]:
        """
        Verifies OpenCV, Pillow, Torchvision dependencies.
        """
        health_status = {
            "status": "healthy",
            "dependencies": {}
        }
        
        deps_to_check = {
            "cv2": "opencv-python",
            "PIL": "pillow",
            "torchvision": "torchvision"
        }
        
        all_ok = True
        for module_name, pip_name in deps_to_check.items():
            try:
                importlib.import_module(module_name)
                health_status["dependencies"][pip_name] = "installed"
            except ImportError:
                health_status["dependencies"][pip_name] = "missing"
                all_ok = False
                
        if not all_ok:
            health_status["status"] = "unhealthy"
            
        return health_status
