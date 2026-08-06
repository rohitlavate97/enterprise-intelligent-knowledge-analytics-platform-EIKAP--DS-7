import torch
import torch.nn as nn
from shared.logging import get_logger

logger = get_logger(__name__)

class ResNetBlock(nn.Module):
    """
    Residual block for tabular data.
    """
    def __init__(self, dim: int, dropout_rate: float = 0.2) -> None:
        super().__init__()
        self.linear1 = nn.Linear(dim, dim)
        self.bn1 = nn.BatchNorm1d(dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.linear2 = nn.Linear(dim, dim)
        self.bn2 = nn.BatchNorm1d(dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the ResNet block.
        """
        residual = x
        
        out = self.linear1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        out = self.linear2(out)
        out = self.bn2(out)
        
        out = out + residual
        out = self.relu(out)
        return out


class TabularResNet(nn.Module):
    """
    ResNet architecture tailored for tabular data prediction.
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_blocks: int = 3,
        output_dim: int = 1,
        dropout_rate: float = 0.2
    ) -> None:
        super().__init__()
        logger.info(f"Initializing TabularResNet with input_dim={input_dim}, hidden_dim={hidden_dim}, num_blocks={num_blocks}, output_dim={output_dim}")
        
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )
        
        blocks = []
        for _ in range(num_blocks):
            blocks.append(ResNetBlock(hidden_dim, dropout_rate))
            
        self.blocks = nn.Sequential(*blocks)
        
        self.output_layer = nn.Sequential(
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the TabularResNet.
        """
        x = self.input_layer(x)
        x = self.blocks(x)
        x = self.output_layer(x)
        return x
