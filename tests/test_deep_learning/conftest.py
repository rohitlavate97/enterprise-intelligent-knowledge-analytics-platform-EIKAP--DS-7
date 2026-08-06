import pytest
import numpy as np
from deep_learning.models import TabularMLP, TabularResNet, CategoricalEmbeddingNet

@pytest.fixture
def sample_tabular_data():
    np.random.seed(42)
    X_train = np.random.rand(100, 10).astype(np.float32)
    X_val = np.random.rand(20, 10).astype(np.float32)
    y_train = np.random.rand(100).astype(np.float32)
    y_val = np.random.rand(20).astype(np.float32)
    return X_train, X_val, y_train, y_val

@pytest.fixture
def sample_cat_data():
    np.random.seed(42)
    X_num = np.random.rand(100, 5).astype(np.float32)
    X_cat = np.random.randint(0, 10, size=(100, 1)).astype(np.int64)
    y = np.random.rand(100).astype(np.float32)
    return X_num, X_cat, y

@pytest.fixture
def mlp_model():
    return TabularMLP(input_dim=10, hidden_dims=[32, 16], output_dim=1)

@pytest.fixture
def resnet_model():
    return TabularResNet(input_dim=10, hidden_dim=32, num_blocks=2, output_dim=1)

@pytest.fixture
def embedding_model():
    return CategoricalEmbeddingNet(
        num_dim=5, 
        embedding_sizes=[(10, 4)], 
        hidden_dims=[32, 16], 
        output_dim=1
    )
