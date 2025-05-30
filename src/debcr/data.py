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
    """
    Load data from a file into a NumPy array using a format-specific loader.

    The function automatically detects the file format based on its extension
    and uses the appropriate loader to read the data into a NumPy array.

    Supported file formats:
    - TIFF/TIF: `.tiff`, `.tif`
    - PNG/JPEG: `.png`, `.jpg`, `.jpeg`
    - NumPy array: `.npy`
    - NumPy archive: `.npz`

    Parameters
    ----------
    filepath : str
        Path to the input data file.

    Returns
    -------
    numpy.ndarray
        Loaded data as a NumPy array.

    Raises
    ------
    ValueError
        If the file format is unsupported or the file cannot be loaded.
    """
    
    input_fmt = _data.get_format(filepath)
    data_loader = _data.get_loader(input_fmt)
    data = data_loader(filepath)
    
    return data

def write(filepath: str, data):
    """
    Write data to a file using a format-specific writer.

    The file format is inferred from the file extension, and the appropriate
    writer is used to save the data. Supported formats and behaviors:

    Supported Output Formats
    ------------------------
    - `.tiff`, `.tif`:
        Saves the input array as a TIFF stack using `tifffile`.
    
    - `.png`, `.jpg`, `.jpeg`:
        Saves image slices as individual image files. These formats do not 
        support multi-dimensional stacks, so each slice (along axis 0) is saved 
        as a separate file (e.g., `image_0001.png`, `image_0002.png`, ...).
        Data is automatically rescaled to the range [0, 255] and cast to `uint8`.

    - `.npy`:
        Saves a single NumPy array to a `.npy` binary file.

    - `.npz`:
        Saves multiple arrays in a dictionary to a compressed `.npz` archive.

    Parameters
    ----------
    filepath : str
        Path to the output file. The file extension determines the format and
        behavior of the writer.

    data : numpy.ndarray or dict
        Data to be written:
        - For `.tif`, `.png`, `.jpg`, `.npy`: expects a NumPy array.
        - For `.npz`: expects a dictionary of NumPy arrays.

    Raises
    ------
    ValueError
        If the file format is unsupported or writing fails.
    """
    
    output_fmt = _data.get_format(filepath)
    data_writer = _data.get_writer(output_fmt)
    data_writer(filepath, data)