import os
import numpy as np

def patchify(data, patch_size=128) -> np.ndarray:
    num_patches_x = data.shape[1] // patch_size
    num_patches_y = data.shape[2] // patch_size

    patches = []
    for i in range(num_patches_x):
        for j in range(num_patches_y):
            patch = data[:, i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size]
            patches.append(patch)
    
    patches_array = np.array(patches)  # Shape: [num_patches, num_images, patch_size, patch_size]
    patches_array = np.concatenate(patches_array, axis=0)  # Flatten patch dimension
    return patches_array # np.expand_dims(patches_array, axis=-1)  # Ensure channel dimension

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