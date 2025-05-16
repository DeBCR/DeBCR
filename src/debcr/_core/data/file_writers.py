import os
import warnings

import numpy as np
import tifffile as tiff
import skimage.io as skio

# Registry for supported formats
FORMAT_REGISTRY = {}

def register_writer(*formats):
    """Decorator to register a new format writer."""
    def decorator(func):
        for fmt in formats:
            FORMAT_REGISTRY[fmt.lower()] = func
        return func
    return decorator

@register_writer("tiff", "tif")
def write_tiff(file_path: str, data: np.ndarray):
    """Writes a TIFF or TIF image (stack)."""
    tiff.imsave(file_path, data)

@register_writer("png", "jpg", "jpeg")
def write_png_jpeg(file_path: str, data: np.ndarray):
    """Writes a PNG, JPG, or JPEG image (stack)."""
    fmt = get_format(file_path)
    fmt = fmt.upper()
    # these formats require uint8 data, so input will be rescaled to [0,255]
    warnings.warn("The data will be rescaled to [0,255] and casted to 'uint8' due to the file format " + fmt);
    data_01 = (data - data.min()) / (data.max() - data.min() + 1e-16)
    data_01 = np.clip(data_01, 0, 1)
    data_uint = data.astype(np.uint8)
    
    # these formats do not natively support multi-arrays, so slices will be saved as individual images
    warnings.warn("The data slices will be saved as individual images due to the file format " + fmt);
    file_path_pref,file_format = os.path.splitext(file_path)
    for i in range(data_uint.shape[0]):
        slice_file_path = f'{file_path_pref}_{str(i+1).zfill(4)}{file_format}'
        skio.imsave(slice_file_path, data_uint[i])
    
@register_writer("npy")
def write_npy(file_path: str, data: np.ndarray):
    """Writes a single-array NPY file."""
    np.save(file_path, data)
    
@register_writer("npz")
def write_npz(file_path: str, data: dict):
    """Writes a multi-array NPZ archive."""
    np.savez(file_path, **data)

def get_writer(file_format):
    """Returns the correct writer function for the given format."""
    file_format = file_format.lower()
    if file_format not in FORMAT_REGISTRY:
        raise ValueError(f"Unsupported format: {file_format}. Available formats: {list(FORMAT_REGISTRY.keys())}")
    return FORMAT_REGISTRY[file_format]

def get_format(file_path: str) -> str:
    """Returns the file format for the given file path."""
    _,file_format = os.path.splitext(file_path)
    return file_format[1:] # truncate preceding dot

'''
def is_supported_format(file_format):
    """Returns True if the loader for the given format is implemented."""
    file_format = file_format.lower()
    is_supported = file_format in FORMAT_REGISTRY
    return is_supported
'''