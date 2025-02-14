from os import path as osp

from debcr.core.config_loader import load_config
from debcr.core.data_loader import load_dataset
from debcr.core.model.DeBCR import build_DeBCR
from debcr.core.training_loop import train_model

def train_from_paths(train_data_path, val_data_path, config_path="config/LM_2D_CARE_config.yaml"):
    """Train the model using train/validation dataset paths and config path."""
    
    config_dict = load_config(config_path)
    
    train_data = load_dataset(train_data_path, batch_size=config_dict.data["train_batch"], add_noise=config_dict.data["noise"])
    val_data = load_dataset(val_data_path, batch_size=config_dict.data["val_batch"], add_noise=config_dict.data["noise"])
    
    model = build_DeBCR()
    model_trained = train_model(model, train_data, val_data, config_dict.training)
    
    return model_trained

def train_from_arrays(train_data, val_data, config_dict):
    """Train the model using train/validation data and config dict."""

    model = build_DeBCR()
    model_trained = train_model(model, train_data, val_data, config_dict.training)
    
    return model_trained