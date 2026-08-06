import torch
import torch.nn as nn
from typing import List
from shared.logging import get_logger

logger = get_logger(__name__)

class TabularMLP(nn.Module):
    """
    Multi-Layer Perceptron for tabular classification & regression.
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [128, 64, 32],
        output_dim: int = 1,
        dropout_rate: float = 0.2,
        use_batch_norm: bool = True,
        activation: str = "relu"
    ) -> None:
        super().__init__()
        logger.info(f"Initializing TabularMLP with input_dim={input_dim}, hidden_dims={hidden_dims}, output_dim={output_dim}")
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        if activation.lower() == "relu":
            act_layer = nn.ReLU
        elif activation.lower() == "leaky_relu":
            act_layer = nn.LeakyReLU
        elif activation.lower() == "elu":
            act_layer = nn.ELU
        else:
            raise ValueError(f"Unsupported activation function: {activation}")
            
        layers = []
        in_dim = input_dim
        
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(h_dim))
            layers.append(act_layer())
            layers.append(nn.Dropout(dropout_rate))
            in_dim = h_dim
            
        layers.append(nn.Linear(in_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the MLP.
        Args:
            x: Input tensor of shape (batch_size, input_dim)
        Returns:
            Output tensor of shape (batch_size, output_dim)
        """
        return self.network(x)
