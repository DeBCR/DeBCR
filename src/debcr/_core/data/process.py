import numpy as np

def crop(data: np.ndarray, patch_size: int = 128, overlap = (0.5, 0.5)) -> np.ndarray:

    nz, sz_x, sz_y = data.shape
    over_x, over_y = overlap
    
    nx = int( (sz_x - patch_size*over_x) // ((1-over_x)*patch_size) )
    ny = int( (sz_y - patch_size*over_y) // ((1-over_y)*patch_size) )
    
    n_patch = nx * ny * nz
    patches = np.zeros((n_patch, patch_size, patch_size), dtype=data.dtype)
    
    for iz in range(nz):
        for ix in range(nx):
            ix_s = int(ix * (1-over_x) * patch_size)
            for iy in range(ny):    
                iy_s = int(iy * (1-over_y) * patch_size)
                patches[iz*(nx*ny) + ix*nx + iy] = data[iz, ix_s:ix_s+patch_size, iy_s:iy_s+patch_size]
                #patches[iz*nz:(iz+1)*nz] = data[:, ixs:ixs+patch_size, iys:iys+patch_size]
    
    return patches, (nx, ny)

def stitch(data: np.ndarray, patch_num: (int, int), overlap: (0.5, 0.5), use_cosine=True) -> np.ndarray:

    patch_size = data[0].shape[0]
    nx, ny = patch_num
    over_x, over_y = overlap
    
    nz = data.shape[0] // (nx * ny)
    sz_x = int( nx*patch_size*(1-over_x) + patch_size*over_x )
    sz_y = int( ny*patch_size*(1-over_y) + patch_size*over_y )
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