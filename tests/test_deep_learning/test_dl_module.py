import pytest
import numpy as np
from deep_learning.module import DeepLearningModule
from deep_learning.schema import DLInput
from module_registry.base import ModuleCategory, MaturityLabel

def test_dl_module_metadata():
    module = DeepLearningModule()
    metadata = module.metadata
    
    assert metadata.name == "deep_learning"
    assert metadata.version == "0.1.0"
    assert metadata.category == ModuleCategory.DEEP_LEARNING
    assert metadata.maturity == MaturityLabel.STANDARD

def test_dl_module_train_predict():
    module = DeepLearningModule()
    
    # Train
    X_train = np.random.rand(100, 10).tolist()
    y_train = np.random.rand(100).tolist()
    
    input_data = {
        "model_architecture": "mlp",
        "input_dim": 10,
        "epochs": 2,
        "batch_size": 16,
        "feature_data": {
            "train_features": X_train,
            "train_targets": y_train
        }
    }
    
    result = module.train(input_data)
    
    assert "model_architecture" in result
    assert result["model_architecture"] == "mlp"
    assert "loss_history" in result
    assert len(result["loss_history"]) == 2
    assert "metrics" in result
    assert "execution_time_ms" in result
    
    # Predict
    X_test = np.random.rand(20, 10).tolist()
    predict_data = {
        "model_architecture": "mlp",
        "input_dim": 10,
        "batch_size": 16,
        "feature_data": {
            "predict_features": X_test
        }
    }
    
    predictions = module.predict(predict_data)
    assert len(predictions) == 20
    
    # Explain
    explanation = module.explain(predict_data, predictions)
    assert "feature_importance" in explanation
    assert len(explanation["feature_importance"]) == 10
    
    # Evaluate
    evaluate_data = {
        "model_architecture": "mlp",
        "input_dim": 10,
        "batch_size": 16,
        "feature_data": {
            "test_features": X_test,
            "test_targets": np.random.rand(20).tolist()
        }
    }
    
    eval_metrics = module.evaluate(evaluate_data)
    assert "rmse" in eval_metrics
    assert "mse" in eval_metrics
    
    # Health check
    health = module.health_check()
    assert health["status"] == "healthy"

def test_dl_module_compliance_contract():
    module = DeepLearningModule()
    
    assert issubclass(module.input_schema, object)
    assert issubclass(module.output_schema, object)
    assert hasattr(module, "train")
    assert hasattr(module, "predict")
    assert hasattr(module, "explain")
    assert hasattr(module, "evaluate")
    assert hasattr(module, "health_check")
