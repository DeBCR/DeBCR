import os
import numpy as np
import tifffile as tiff
import skimage.io as skio

# Registry for supported formats
FORMAT_REGISTRY = {}

def register_loader(*formats):
    """Decorator to register a new format loader."""
    def decorator(func):
        for fmt in formats:
            FORMAT_REGISTRY[fmt.lower()] = func
        return func
    return decorator

# TIFF Loader
@register_loader("tiff", "tif")
def load_tiff(file_path: str) -> np.ndarray:
    """Loads a TIFF or TIF image (stack)."""
    img_stack = tiff.imread(file_path)
    return img_stack

# PNG, JPG, JPEG Loader
@register_loader("png", "jpg", "jpeg")
def load_png_jpeg(file_path: str) -> np.ndarray:
    """Loads a PNG, JPG, or JPEG image (stack)."""
    img_stack = skio.imread(file_path)
    return img_stack

def get_loader(file_format):
    """Returns the correct loader function for the given format."""
    file_format = file_format.lower()
    if file_format not in FORMAT_REGISTRY:
        raise ValueError(f"Unsupported format: {file_format}. Available formats: {list(FORMAT_REGISTRY.keys())}")
    return FORMAT_REGISTRY[file_format]
