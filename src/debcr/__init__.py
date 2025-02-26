__version__ = "0.1.0"

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from . import data
from . import model

__all__ = [
    "data",
    "model"
]
