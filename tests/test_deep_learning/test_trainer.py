import pytest
import tempfile
import os
import torch
from deep_learning.trainer import EarlyStopping, PyTorchTrainer
from deep_learning.dataset import create_data_loaders

def test_early_stopping():
    es = EarlyStopping(patience=2, min_delta=0.1)
    
    # Loss improves significantly
    assert not es(1.0)
    assert not es(0.8)
    
    # Loss improves but not by min_delta
    assert not es(0.75) 
    
    # Loss doesn't improve for patience epochs
    assert es(0.8)

def test_pytorch_trainer_fit_predict(mlp_model, sample_tabular_data):
    X_train, X_val, y_train, y_val = sample_tabular_data
    train_loader, val_loader = create_data_loaders(X_train, y_train, X_val, y_val, batch_size=16)
    
    trainer = PyTorchTrainer(model=mlp_model)
    loss_history = trainer.fit(train_loader, val_loader, epochs=2)
    
    assert len(loss_history) == 2
    
    preds = trainer.predict(val_loader)
    assert preds.squeeze().shape == (20,)

def test_checkpoint_save_load(mlp_model):
    trainer = PyTorchTrainer(model=mlp_model)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        checkpoint_path = os.path.join(temp_dir, "model.pt")
        
        # Save model
        trainer.save_checkpoint(checkpoint_path)
        assert os.path.exists(checkpoint_path)
        
        # Modify model weights slightly
        with torch.no_grad():
            for param in trainer.model.parameters():
                param.add_(1.0)
                
        # Load model
        trainer.load_checkpoint(checkpoint_path)
