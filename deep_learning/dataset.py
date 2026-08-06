"""
Dataset module for EIKAP Deep Learning Phase.
"""

from typing import Optional, Union, Tuple, Any
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from shared.logging import get_logger
from shared.exceptions import ModelTrainingError

logger = get_logger(__name__)

class TabularDataset(Dataset):
    """
    A PyTorch Dataset for tabular data supporting numerical, categorical, and target arrays.
    """
    def __init__(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        cat_cols: Optional[np.ndarray] = None,
        categorical_features: Optional[np.ndarray] = None
    ):
        try:
            self.X = torch.tensor(X, dtype=torch.float32)
            self.y = torch.tensor(y, dtype=torch.float32) if y is not None else None
            
            cat = categorical_features if categorical_features is not None else cat_cols
            self.cat_cols = torch.tensor(cat, dtype=torch.long) if cat is not None else None
            
            logger.debug(f"Initialized TabularDataset with {len(self.X)} samples.")
        except Exception as e:
            logger.error(f"Error initializing TabularDataset: {e}")
            raise ModelTrainingError(f"Dataset initialization failed: {e}")

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Any:
        if self.cat_cols is not None:
            features = (self.X[idx], self.cat_cols[idx])
        else:
            features = self.X[idx]
            
        if self.y is not None:
            return features, self.y[idx]
        return features


def create_data_loaders(
    *args,
    batch_size: int = 64,
    val_split: float = 0.2,
    random_state: int = 42,
    **kwargs
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """
    Creates PyTorch DataLoaders supporting positional or keyword signatures.
    """
    try:
        X_train = kwargs.get('train_features', None)
        y_train = kwargs.get('train_targets', None)
        X_val = kwargs.get('val_features', None)
        y_val = kwargs.get('val_targets', None)

        if X_train is None and len(args) > 0:
            X_train = args[0]
            if len(args) > 1:
                y_train = args[1]
            if len(args) > 2 and isinstance(args[2], np.ndarray):
                X_val = args[2]
            if len(args) > 3 and isinstance(args[3], np.ndarray):
                y_val = args[3]
            if len(args) > 4 and isinstance(args[4], int):
                batch_size = args[4]
            elif len(args) > 2 and isinstance(args[2], int):
                batch_size = args[2]

        bs = kwargs.get('batch_size', batch_size)

        if X_val is not None:
            train_ds = TabularDataset(X_train, y_train)
            val_ds = TabularDataset(X_val, y_val)
            train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True)
            val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False)
            return train_loader, val_loader

        if X_train is not None:
            num_samples = len(X_train)
            if val_split > 0.0:
                np.random.seed(random_state)
                indices = np.random.permutation(num_samples)
                split_idx = int(num_samples * (1 - val_split))
                
                t_idx, v_idx = indices[:split_idx], indices[split_idx:]
                
                t_X = X_train[t_idx]
                t_y = y_train[t_idx] if y_train is not None else None
                
                v_X = X_train[v_idx]
                v_y = y_train[v_idx] if y_train is not None else None
                
                train_dataset = TabularDataset(X=t_X, y=t_y)
                val_dataset = TabularDataset(X=v_X, y=v_y)
                
                train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True)
                val_loader = DataLoader(val_dataset, batch_size=bs, shuffle=False)
                return train_loader, val_loader
            else:
                train_dataset = TabularDataset(X=X_train, y=y_train)
                train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True)
                return train_loader, None

        raise ValueError("No input data provided to create_data_loaders")

    except Exception as e:
        logger.error(f"Error creating data loaders: {e}")
        raise ModelTrainingError(f"DataLoader creation failed: {e}")
