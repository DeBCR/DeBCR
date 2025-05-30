import numpy as np
    
def crop(data: np.ndarray, patch_size: int = 128, overlap = (0.5, 0.5), dry_run=False) -> np.ndarray:
    """
    Crop a 3D volume into overlapping 2D patches.

    The function extracts 2D patches from each slice (z-plane) of the input volume,
    with a specified patch size and overlap fraction. The patches are extracted
    along the spatial dimensions (x, y) of the volume.

    Parameters
    ----------
    data : np.ndarray
        3D input volume of shape (Z, X, Y), where:
        - Z is the number of slices (depth)
        - X and Y are spatial dimensions
    patch_size : int, optional
        Size of each square patch (patch_size x patch_size). Default is 128.
    overlap : tuple of float, optional
        Fractional overlap between patches along the X and Y dimensions.
        Values should be between 0 (no overlap) and less than 1.
        Default is (0.5, 0.5).
    dry_run : bool, optional
        If True, do not extract patches. Instead, return:
        - total number of patches
        - number of patches along (nx, ny) per slice

    Returns
    -------
    np.ndarray or tuple
        If `dry_run` is False: returns a 3D array of shape (N, patch_size, patch_size),
        where N is the total number of extracted patches.

        If `dry_run` is True: returns a tuple (n_patch, (nx, ny)), where:
        - n_patch: total number of patches
        - (nx, ny): number of patches per slice in X and Y directions
    
    """
    
    nz, sz_x, sz_y = data.shape
    over_x, over_y = overlap
    
    nx = int( (sz_x - patch_size*over_x) // ((1-over_x)*patch_size) )
    ny = int( (sz_y - patch_size*over_y) // ((1-over_y)*patch_size) )
    
    n_patch = nx * ny * nz
    
    if dry_run:
        return n_patch, (nx, ny) 
    
    patches = np.zeros((n_patch, patch_size, patch_size), dtype=data.dtype)
    
    for iz in range(nz):
        for ix in range(nx):
            ix_s = int(ix * (1-over_x) * patch_size)
            for iy in range(ny):    
                iy_s = int(iy * (1-over_y) * patch_size)
                patches[iz*(nx*ny) + ix*nx + iy] = data[iz, ix_s:ix_s+patch_size, iy_s:iy_s+patch_size]
                #patches[iz*nz:(iz+1)*nz] = data[:, ixs:ixs+patch_size, iys:iys+patch_size]
    
    return patches
    
def stitch(data: np.ndarray, patch_num: (int, int), overlap: (0.5, 0.5), use_cosine=True, dry_run=False) -> np.ndarray:
    """
    Reconstruct a 3D volume from overlapping 2D patches.

    This function reverses the operation of `crop()` by stitching patches back 
    into their original spatial arrangement. Overlapping regions are blended 
    using either a cosine (Hann) window or uniform averaging to reduce edge artifacts.

    Parameters
    ----------
    data : np.ndarray
        3D array of shape (N, patch_size, patch_size) containing the extracted 2D patches.
        The total number of patches N should equal `nx * ny * nz`.
    patch_num : tuple of int
        Number of patches along the X and Y dimensions per slice: (nx, ny).
    overlap : tuple of float, optional
        Fractional overlap used when cropping the patches. Should match the values
        used in `crop()`. Default is (0.5, 0.5).
    use_cosine : bool, optional
        If True, applies a cosine (Hann) window to blend overlapping patches. 
        If False, uses uniform averaging. Default is True.
    dry_run : bool, optional
        If True, returns only the expected output volume shape without performing stitching.

    Returns
    -------
    np.ndarray or tuple
        If `dry_run` is False: returns a reconstructed 3D volume of shape (Z, X, Y).
        If `dry_run` is True: returns a tuple `(nz, (sz_x, sz_y))`, where:
            - nz: number of slices
            - sz_x, sz_y: spatial dimensions of the reconstructed volume
    """
    
    patch_size = data[0].shape[0]
    nx, ny = patch_num
    over_x, over_y = overlap
    
    nz = data.shape[0] // (nx * ny)
    sz_x = int( nx*patch_size*(1-over_x) + patch_size*over_x )
    sz_y = int( ny*patch_size*(1-over_y) + patch_size*over_y )

    if dry_run:
        return nz, (sz_x, sz_y) 
    
    asmbl = np.zeros((nz, sz_x, sz_y), dtype=data.dtype)    

    # blend patches to avoid border artifacts
    # select blending approach: cosine (Hann) window or direct averaging
    if use_cosine:
        mask = _cosine_window(patch_size)
    else:
        mask = np.ones((patch_size, patch_size), dtype=np.float32)
    
    for iz in range(nz):
        asmbl_slice = np.zeros((sz_x, sz_y), dtype=data.dtype)
        asmbl_weight = np.zeros((sz_x, sz_y), dtype=data.dtype)
        for ix in range(nx):
            ix_s = int(ix * (1-over_x) * patch_size)
            for iy in range(ny):
                iy_s = int(iy * (1-over_y) * patch_size)
                asmbl_slice[ix_s:ix_s+patch_size, iy_s:iy_s+patch_size] += data[iz*(nx*ny) + ix*nx + iy] * mask
                asmbl_weight[ix_s:ix_s+patch_size, iy_s:iy_s+patch_size] += mask
        asmbl[iz] = asmbl_slice / (asmbl_weight + 1e-8)
    
    return asmbl

def _cosine_window(patch_size: int):
    window_1d = np.hanning(patch_size) # 1D cosine
    window_2d = np.outer(window_1d, window_1d) # make 2D window
    return window_2d

def normalize(data: np.ndarray, pmin=0.1, pmax=99.9, by_perc=True, vmin=None, vmax=None, per_slice=False, eps=1e-16, dtype=np.float32):
    """
    Normalize image data to the [0, 1] range using percentiles or fixed values.

    Supports global or per-slice normalization for 3D image stacks. The normalization 
    can be based on value percentiles (robust to outliers) or fixed min/max values.

    Parameters
    ----------
    data : np.ndarray
        Input data to normalize, typically a 3D array (e.g., (Z, X, Y)).
    pmin : float, optional
        Lower percentile for normalization if `by_perc` is True. Default is 0.1.
    pmax : float, optional
        Upper percentile for normalization if `by_perc` is True. Default is 99.9.
    by_perc : bool, optional
        If True, normalize using percentiles (`pmin`, `pmax`). If False, use absolute
        min/max values (`vmin`, `vmax`). Default is True.
    vmin : float or None, optional
        Minimum value for normalization if `by_perc` is False. Default is None.
    vmax : float or None, optional
        Maximum value for normalization if `by_perc` is False. Default is None.
    per_slice : bool, optional
        If True, normalize each slice (along axis 0) independently. Default is False.
    eps : float, optional
        Small constant to avoid division by zero. Default is 1e-16.
    dtype : np.dtype, optional
        Output data type. Default is np.float32.

    Returns
    -------
    np.ndarray
        Normalized data array of the same shape as input, with values in [0, 1].
    
    """
    
    get_minmax_fn = _get_minmax_perc if by_perc else _get_minmax_val
    if by_perc:    
        minmax_args = {'pmin': pmin, 'pmax': pmax}
    else:
        minmax_args = {'vmin': vmin, 'vmax': vmax}
        #minmax_args = {k: v for k, v in {'vmin': vmin, 'vmax': vmax}.items() if v is not None}
    
    data_norm = np.zeros(data.shape, dtype=dtype)
    if not per_slice:
        dmin, dmax = get_minmax_fn(data, **minmax_args)
        data_norm = (data - dmin) / (dmax - dmin + eps)
        data_norm = np.clip(data_norm, 0, 1)
    else:
        for idx, data_slice in enumerate(data):
            dmin, dmax = get_minmax_fn(data_slice, **minmax_args)
            data_norm[idx] = (data_slice - dmin) / (dmax - dmin + eps)
            data_norm[idx] = np.clip(data_norm[idx], 0, 1)
    return data_norm

def _get_minmax_val(data: np.ndarray, vmin, vmax):
    dmin = vmin if vmin is not None else data.min()
    dmax = vmax if vmax is not None else data.max()
    return dmin, dmax
    
def _get_minmax_perc(data: np.ndarray, pmin, pmax):
    return np.percentile(data, (pmin, pmax))

'''
def split_train_val(data, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):

    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1."

    # Calculate the split indices
    total_samples = data.shape[0]
    train_end = int(total_samples * train_ratio)
    val_end = train_end + int(total_samples * val_ratio)

    # Split the data
    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]

    return train_data, val_data, test_data
'''