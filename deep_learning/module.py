import time
import torch
import numpy as np
from typing import Dict, Any, Type
from pydantic import BaseModel
from module_registry.base import EIKAPModule, ModuleMetadata, ModuleCategory, MaturityLabel

from deep_learning.schema import DLInput, DLOutput
from deep_learning.models import TabularMLP, TabularResNet, CategoricalEmbeddingNet
from deep_learning.dataset import create_data_loaders, TabularDataset
from deep_learning.trainer import PyTorchTrainer
from torch.utils.data import DataLoader

class DeepLearningModule(EIKAPModule):
    def __init__(self):
        self._metadata = ModuleMetadata(
            name="deep_learning",
            version="0.1.0",
            description="Tabular Deep Learning & Neural Embedding Engine",
            category=ModuleCategory.DEEP_LEARNING,
            maturity=MaturityLabel.STANDARD,
            author="EIKAP Team",
            requires_gpu=torch.cuda.is_available()
        )
        self.trainer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    @property
    def metadata(self) -> ModuleMetadata:
        return self._metadata

    @property
    def input_schema(self) -> Type[BaseModel]:
        return DLInput

    @property
    def output_schema(self) -> Type[BaseModel]:
        return DLOutput

    def train(self, data: Any, **kwargs) -> Dict[str, Any]:
        start_time = time.time()
        
        input_data = self.validate_input(data)
        
        train_features = np.array(input_data.feature_data.get('train_features', []))
        train_targets = np.array(input_data.feature_data.get('train_targets', []))
        
        if len(train_features) == 0:
            train_features = np.random.rand(100, input_data.input_dim).astype(np.float32)
            train_targets = np.random.rand(100).astype(np.float32)
            
        train_loader, _ = create_data_loaders(
            train_features=train_features,
            train_targets=train_targets,
            batch_size=input_data.batch_size
        )
        
        if input_data.model_architecture == "mlp":
            model = TabularMLP(input_dim=input_data.input_dim, hidden_dims=[64, 32])
        elif input_data.model_architecture == "resnet":
            model = TabularResNet(input_dim=input_data.input_dim, hidden_dim=64)
        elif input_data.model_architecture == "embedding":
            # For simplicity, assuming a dummy categorical config if embedding requested without proper data
            model = CategoricalEmbeddingNet(num_dim=input_data.input_dim, embedding_sizes=[(10, 4)], hidden_dims=[64, 32])
        else:
            model = TabularMLP(input_dim=input_data.input_dim, hidden_dims=[64, 32])

        self.trainer = PyTorchTrainer(model=model, device=self.device)
        loss_history = self.trainer.fit(train_loader, epochs=input_data.epochs)
        
        metrics = {"final_loss": loss_history[-1] if loss_history else 0.0}
        
        execution_time_ms = (time.time() - start_time) * 1000
        
        return {
            "model_architecture": input_data.model_architecture,
            "loss_history": loss_history,
            "metrics": metrics,
            "execution_time_ms": execution_time_ms
        }

    def predict(self, input_data: Any, **kwargs) -> Any:
        if not self.trainer:
            raise ValueError("Model has not been trained yet.")
            
        data = self.validate_input(input_data)
        features = np.array(data.feature_data.get('predict_features', []))
        
        if len(features) == 0:
             features = np.random.rand(10, data.input_dim).astype(np.float32)
             
        dataset = TabularDataset(features)
        loader = DataLoader(dataset, batch_size=data.batch_size, shuffle=False)
        
        predictions = self.trainer.predict(loader)
        return predictions.tolist()

    def explain(self, input_data: Any, prediction: Any, **kwargs) -> Dict[str, Any]:
        # Placeholder for feature importance based on weights magnitude
        if not self.trainer:
             return {"importance": []}
             
        importance = []
        if isinstance(self.trainer.model, TabularMLP):
            with torch.no_grad():
                weights = self.trainer.model.network[0].weight.abs().mean(dim=0)
                importance = weights.cpu().numpy().tolist()
        return {"feature_importance": importance}

    def evaluate(self, test_data: Any, **kwargs) -> Dict[str, float]:
        if not self.trainer:
            raise ValueError("Model has not been trained.")
            
        data = self.validate_input(test_data)
        test_features = np.array(data.feature_data.get('test_features', []))
        test_targets = np.array(data.feature_data.get('test_targets', []))
        
        if len(test_features) == 0:
             return {"rmse": 0.0, "roc_auc": 0.0}
             
        dataset = TabularDataset(test_features, test_targets)
        loader = DataLoader(dataset, batch_size=data.batch_size, shuffle=False)
        
        loss = self.trainer.validate(loader)
        
        # Simplified evaluation metric return
        return {"rmse": float(np.sqrt(loss)), "mse": loss}

    def health_check(self) -> Dict[str, Any]:
        return {
            "pytorch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "device": self.device,
            "status": "healthy"
        }
