import pytest
import torch

def test_mlp_forward_pass(mlp_model):
    batch_size = 16
    x = torch.randn(batch_size, 10)
    output = mlp_model(x)
    assert output.shape == (batch_size, 1)

def test_resnet_forward_pass(resnet_model):
    batch_size = 16
    x = torch.randn(batch_size, 10)
    output = resnet_model(x)
    assert output.shape == (batch_size, 1)

def test_embedding_forward_pass(embedding_model):
    batch_size = 16
    num_x = torch.randn(batch_size, 5)
    # 1 categorical feature with 10 categories
    cat_x = torch.randint(0, 10, (batch_size, 1))
    
    output = embedding_model(num_x, cat_x)
    assert output.shape == (batch_size, 1)
