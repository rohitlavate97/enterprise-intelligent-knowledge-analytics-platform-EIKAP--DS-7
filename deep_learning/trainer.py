"""
Trainer module for EIKAP Deep Learning Phase.
"""

import copy
from pathlib import Path
from typing import Dict, List, Optional, Union
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from shared.logging import get_logger
from shared.exceptions import ModelTrainingError, ModelInferenceError

logger = get_logger(__name__)

class EarlyStopping:
    """
    Early stopping to stop the training when the loss does not improve after certain epochs.
    """
    def __init__(self, patience: int = 7, min_delta: float = 1e-4, mode: str = "min"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_loss = float('inf') if mode == "min" else -float('inf')
        self.early_stop = False
        self.best_model_weights: Optional[Dict[str, torch.Tensor]] = None

    def step(self, val_loss: float, model: Optional[nn.Module] = None) -> bool:
        if self.mode == "min":
            improvement = (self.best_loss - val_loss) > self.min_delta
        else:
            improvement = (val_loss - self.best_loss) > self.min_delta

        if improvement:
            self.best_loss = val_loss
            self.counter = 0
            if model is not None:
                self.best_model_weights = copy.deepcopy(model.state_dict())
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop

    def __call__(self, val_loss: float, model: Optional[nn.Module] = None) -> bool:
        """Callable alias for step."""
        return self.step(val_loss, model)

    def restore_best_weights(self, model: nn.Module) -> None:
        if self.best_model_weights is not None:
            model.load_state_dict(self.best_model_weights)
            logger.info("Restored best model weights from early stopping.")


class PyTorchTrainer:
    """
    A unified trainer for PyTorch models.
    """
    def __init__(
        self,
        model: nn.Module,
        criterion: Optional[nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        device: Optional[str] = None
    ):
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = model.to(self.device)
        self.criterion = criterion if criterion is not None else nn.MSELoss()
        self.optimizer = optimizer if optimizer is not None else torch.optim.Adam(self.model.parameters())
        logger.info(f"Initialized PyTorchTrainer on device: {self.device}")

    def train_epoch(self, train_loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        
        try:
            for batch in train_loader:
                if isinstance(batch, (tuple, list)) and len(batch) == 2:
                    X_batch, y_batch = batch
                    X_batch = X_batch.to(self.device)
                    y_batch = y_batch.to(self.device)
                else:
                    X_batch = batch.to(self.device)
                    y_batch = None
                
                self.optimizer.zero_grad()
                outputs = self.model(X_batch)
                
                if y_batch is not None:
                    # Unsqueeze target if 1D and outputs are 2D (batch_size, 1)
                    if outputs.ndim == 2 and y_batch.ndim == 1:
                        y_batch = y_batch.unsqueeze(1)
                    loss = self.criterion(outputs, y_batch)
                    loss.backward()
                    self.optimizer.step()
                    total_loss += loss.item() * X_batch.size(0)
            
            avg_loss = total_loss / len(train_loader.dataset) # type: ignore
            return avg_loss
        except Exception as e:
            logger.error(f"Error during training epoch: {e}")
            raise ModelTrainingError(f"Training epoch failed: {e}")

    def validate(self, val_loader: DataLoader) -> float:
        self.model.eval()
        total_loss = 0.0
        
        try:
            with torch.no_grad():
                for batch in val_loader:
                    if isinstance(batch, (tuple, list)) and len(batch) == 2:
                        X_batch, y_batch = batch
                        X_batch = X_batch.to(self.device)
                        y_batch = y_batch.to(self.device)
                    else:
                        X_batch = batch.to(self.device)
                        y_batch = None
                        
                    outputs = self.model(X_batch)
                    if y_batch is not None:
                        if outputs.ndim == 2 and y_batch.ndim == 1:
                            y_batch = y_batch.unsqueeze(1)
                        loss = self.criterion(outputs, y_batch)
                        total_loss += loss.item() * X_batch.size(0)
                        
            avg_loss = total_loss / len(val_loader.dataset) # type: ignore
            return avg_loss
        except Exception as e:
            logger.error(f"Error during validation: {e}")
            raise ModelTrainingError(f"Validation failed: {e}")

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 100,
        patience: int = 10,
        lr: float = 1e-3
    ) -> List[float]:
        """
        Full training loop returning list of epoch training losses.
        """
        logger.info(f"Starting training for {epochs} epochs with learning rate {lr}.")
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
            
        early_stopping = EarlyStopping(patience=patience)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=max(1, patience // 2)
        )
        
        history: List[float] = []
        
        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch(train_loader)
            history.append(train_loss)
            
            if val_loader is not None:
                val_loss = self.validate(val_loader)
                scheduler.step(val_loss)
                
                if early_stopping.step(val_loss, self.model):
                    logger.info(f"Early stopping triggered at epoch {epoch}.")
                    early_stopping.restore_best_weights(self.model)
                    break
                
        return history

    def predict(self, test_loader: DataLoader) -> np.ndarray:
        self.model.eval()
        predictions = []
        
        try:
            with torch.no_grad():
                for batch in test_loader:
                    if isinstance(batch, (tuple, list)):
                        X_batch = batch[0].to(self.device)
                    else:
                        X_batch = batch.to(self.device)
                        
                    outputs = self.model(X_batch)
                    predictions.append(outputs.cpu().numpy())
                    
            return np.vstack(predictions)
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            raise ModelInferenceError(f"Prediction failed: {e}")

    def save_checkpoint(self, filepath: Union[str, Path]) -> Path:
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            torch.save(self.model.state_dict(), path)
            logger.info(f"Saved model checkpoint to {path}")
            return path
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            raise ModelTrainingError(f"Save checkpoint failed: {e}")

    def load_checkpoint(self, filepath: Union[str, Path]) -> nn.Module:
        path = Path(filepath)
        if not path.exists():
            raise ModelTrainingError(f"Checkpoint not found: {path}")
            
        try:
            state_dict = torch.load(path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            logger.info(f"Loaded model checkpoint from {path}")
            return self.model
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            raise ModelTrainingError(f"Load checkpoint failed: {e}")
