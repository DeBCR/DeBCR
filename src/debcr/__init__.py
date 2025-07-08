from importlib.metadata import version

__version__ = version("debcr")

from . import data
from . import model
from . import config

__all__ = [
     "data",
     "model",
     "config"
 ]