import pytest
import numpy as np
from deep_learning.dataset import TabularDataset, create_data_loaders

def test_tabular_dataset_len_getitem(sample_tabular_data, sample_cat_data):
    X_train, _, y_train, _ = sample_tabular_data
    dataset = TabularDataset(X_train, y_train)
    assert len(dataset) == 100
    
    features, target = dataset[0]
    assert features.shape == (10,)
    assert target.shape == ()
    
    # Test categorical
    X_num, X_cat, y = sample_cat_data
    cat_dataset = TabularDataset(X_num, y, categorical_features=X_cat)
    (f_num, f_cat), c_target = cat_dataset[0]
    
    assert f_num.shape == (5,)
    assert f_cat.shape == (1,)
    assert c_target.shape == ()

def test_create_data_loaders(sample_tabular_data):
    X_train, X_val, y_train, y_val = sample_tabular_data
    
    train_loader, val_loader = create_data_loaders(
        train_features=X_train,
        train_targets=y_train,
        val_features=X_val,
        val_targets=y_val,
        batch_size=16
    )
    
    assert train_loader is not None
    assert val_loader is not None
    
    # 100 items / 16 batch_size = 7 batches (6 of 16, 1 of 4)
    assert len(train_loader) == 7
    # 20 items / 16 batch_size = 2 batches
    assert len(val_loader) == 2
