import os
import glob
import numpy as np
from ._core import data
from ._core import model

def init(weights_path: str = None, ckpt_name: str = "ckpt-*"):
    
    init_model = model.build_and_compile()
    if weights_path is None:
        print('Initialized model - untrained')
        return init_model
    elif os.path.exists(weights_path) and os.path.isdir(weights_path):
        print(f'Weights path: {weights_path}')
        loaded_model, ckpt_path = model.restore_ckpt(init_model, weights_path, ckpt_name)
        print(f'Checkpoint loaded: {os.path.basename(ckpt_path)}')
        print('Initialized model - trained')
        return loaded_model
    else:
        raise ValueError(f'Non-existing weights path: {weights_path}')

def predict(eval_model, input_data: np.ndarray) -> np.ndarray:
    
    return model.predict_with_model(eval_model, input_data)