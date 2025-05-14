from __future__ import annotations

from ._core import data as _data

# expose internal functions as API
from ._core.data import (
    crop,
    stitch,
    normalize,
    show
)

def load(filepath: str) -> numpy.ndarray:
    
    input_fmt = _data.get_format(filepath)
    data_loader = _data.get_loader(input_fmt)
    data = data_loader(filepath)
    
    return data

def write(filepath: str, data):
    
    output_fmt = _data.get_format(filepath)
    data_writer = _data.get_writer(output_fmt)
    data_writer(filepath, data)