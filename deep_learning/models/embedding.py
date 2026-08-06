import torch
import torch.nn as nn
from typing import List, Tuple, Optional
from shared.logging import get_logger

logger = get_logger(__name__)

class CategoricalEmbeddingNet(nn.Module):
    """
    Neural Network combining Entity Embeddings for categorical features with continuous numerical features.
    """
    def __init__(
        self,
        num_numeric: int = 5,
        cat_dims: Optional[List[Tuple[int, int]]] = None,
        hidden_dims: List[int] = [128, 64],
        output_dim: int = 1,
        dropout_rate: float = 0.2,
        num_dim: Optional[int] = None,
        embedding_sizes: Optional[List[Tuple[int, int]]] = None,
        *args, **kwargs
    ) -> None:
        super().__init__()
        
        n_num = num_dim if num_dim is not None else num_numeric
        c_dims = embedding_sizes if embedding_sizes is not None else (cat_dims if cat_dims is not None else [(10, 4)])

        logger.info(f"Initializing CategoricalEmbeddingNet with num_numeric={n_num}, cat_dims={c_dims}")
        
        self.num_numeric = n_num
        self.cat_dims = c_dims
        
        # Create embedding layers for categorical features
        self.embeddings = nn.ModuleList([
            nn.Embedding(num_embeddings=num_classes, embedding_dim=emb_dim)
            for num_classes, emb_dim in c_dims
        ])
        
        # Calculate total dimension after concatenation
        total_emb_dim = sum(emb_dim for _, emb_dim in c_dims)
        in_dim = n_num + total_emb_dim
        
        # MLP backbone
        layers = []
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            in_dim = h_dim
            
        layers.append(nn.Linear(in_dim, output_dim))
        
        self.mlp = nn.Sequential(*layers)
        
    def forward(self, x_num: torch.Tensor, x_cat: Optional[torch.Tensor] = None) -> torch.Tensor:
        if x_cat is None and x_num.ndim == 2 and x_num.shape[1] > self.num_numeric:
            # Split if single tensor passed
            real_x_num = x_num[:, :self.num_numeric]
            real_x_cat = x_num[:, self.num_numeric:].long()
            return self.forward(real_x_num, real_x_cat)
            
        emb_outputs = []
        if x_cat is not None:
            for i, emb_layer in enumerate(self.embeddings):
                cat_col = x_cat[:, i].long()
                emb_out = emb_layer(cat_col)
                emb_outputs.append(emb_out)
            
        if emb_outputs:
            x_cat_emb = torch.cat(emb_outputs, dim=1)
            x_combined = torch.cat([x_num, x_cat_emb], dim=1)
        else:
            x_combined = x_num
            
        out = self.mlp(x_combined)
        return out
