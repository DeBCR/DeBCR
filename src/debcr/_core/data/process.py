import numpy as np

def crop_patches(data: np.ndarray, patch_size: int = 128) -> np.ndarray:

    nz, sz_x, sz_y = data.shape
    nx = sz_x // patch_size
    ny = sz_y // patch_size
    
    n_patch = nx * ny * nz
    patches = np.zeros((n_patch, patch_size, patch_size), dtype=data.dtype)
    
    iz = 0
    for ix in range(nx):
        for iy in range(ny):
            patches[iz*nz:(iz+1)*nz] = data[:, ix*patch_size:(ix+1)*patch_size, iy*patch_size:(iy+1)*patch_size]
            iz += 1
    
    return patches # np.expand_dims(patches_array, axis=-1)  # Ensure channel dimension

def stitch_patches(data: np.ndarray, nx: int, ny: int) -> np.ndarray:

    patch_size = data[0].shape[0]
    nz = data.shape[0] // (nx * ny)
    
    asmbl = np.zeros((nz, patch_size*nx, patch_size*ny), dtype=data.dtype)    
    
    iz = 0
    for ix in range(nx):
        for iy in range(ny):
            asmbl[:, ix*patch_size:(ix+1)*patch_size, iy*patch_size:(iy+1)*patch_size] = data[iz*nz:(iz+1)*nz]
            iz += 1
    
    return asmbl

def normalize(data: np.ndarray, pmin=0.1, pmax=99.9, by_perc=True, vmin=None, vmax=None, per_slice=False, eps=1e-16, dtype=np.float32):
        
    get_minmax_fn = get_minmax_perc if by_perc else get_minmax_val
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

def get_minmax_val(data: np.ndarray, vmin, vmax):
    dmin = vmin if vmin is not None else data.min()
    dmax = vmax if vmax is not None else data.max()
    return dmin, dmax
    
def get_minmax_perc(data: np.ndarray, pmin, pmax):
    return np.percentile(data, (pmin, pmax))

'''
def rescale_min_max(data, MIN=0, MAX=1) -> np.ndarray:
    # Rescale the whole stack
    if data[0].max() != 1:
        data_scale = []
        for stack in range(data.shape[0]):
            temp = data[stack, ...]
            temp_scale = np.interp(temp, (temp.min(), temp.max()), (MIN, MAX))
            data_scale.append(temp_scale.astype('float64'))
    else:
        data_scale = data
        
    return np.asarray(data_scale)

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