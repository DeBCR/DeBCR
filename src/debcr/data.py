from __future__ import annotations

from ._core import data as _data

# expose internal functions as API
from ._core.data import (
    crop_patches,
    stitch_patches,
    normalize,
)

def load(input_filepath: str) -> numpy.ndarray:
    
    input_fmt = _data.get_format(input_filepath)
    data_loader = _data.get_loader(input_fmt)
    input_data = data_loader(input_filepath)
    
    return input_data
