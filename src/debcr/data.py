import os
import numpy as np
from ._core import data

#def _prepare_numpy(data_raw: np.ndarray, patch_size=128) -> np.ndarray:
def prepare(input_data: np.ndarray, patch_size: int = 128) -> np.ndarray:   
    data_patch = data.patchify(input_data, patch_size=patch_size)
    data_resc = data.rescale_min_max(data_patch)

    return data_resc

## for future extension
#def prepare(input_data: np.ndarray, patch_size: int = 128) -> np.ndarray:
#    return _prepare_numpy(input_data, patch_size)

def load(input_filepath: str) -> np.ndarray:
    
    input_fmt = data.get_format(input_filepath)
    data_loader = data.get_loader(input_fmt)
    input_data = data_loader(input_filepath)
    
    return input_data
