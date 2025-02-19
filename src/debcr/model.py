import os
import glob
import numpy as np
from ._core import data
from ._core import model
from ._core.inference import predict_with_model

def load(weights_path: str = None, ckpt_name: str = "ckpt-*"):
    
    eval_model = model.build_and_compile() # get model instance

    if weights_path is None:
        HERE_PATH = os.path.dirname(__file__)
        weights_path = os.path.join(HERE_PATH, "weights", "2D_denoising")
    
    eval_model = model.restore_ckpt(eval_model, weights_path, ckpt_name)
    
    return eval_model
    
#def _predict_numpy(input_data: np.ndarray, weights_path: str, ckpt_name: str = "ckpt-*") -> np.ndarray:
def predict(eval_model, input_data: np.ndarray) -> np.ndarray:
        
    # input_data = data.rescale_min_max(input_data) # already done in data.prepare()
    return predict_with_model(eval_model, input_data)

# for future extensions
#def predict(input_data: np.ndarray, weights_path: str = "/weights/2D_denoising", ckpt_name: str = "ckpt-*") -> np.ndarray:   
#    return _predict_numpy(input_data, weights_path, ckpt_name)
