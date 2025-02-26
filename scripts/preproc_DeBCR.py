import os
import sys
import glob
import argparse
import numpy as np
#import random
from tifffile import imread
from natsort import natsorted
from skimage.io import imread as skimage_imread
import matplotlib.pyplot as plt

from debcr.utils.utils import rescale#, subShow 

# argparse formaters for argument help:
# - argparse.ArgumentDefaultsHelpFormatter - to show default value
# - argparse.RawTextHelpFormatter - to show multi-line help 
class MyFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawTextHelpFormatter):
    pass

def get_preproc_parser():
    parser = argparse.ArgumentParser(description=(
        "DeBCR: DL-based denoising, deconvolution and deblurring for light microscopy data.\n"
        "This is the subtool to prepare custom input data (TIF(F), PNG, JP(E)G) to train DeBCR model."
    ), formatter_class=MyFormatter)
    
    parser.add_argument("--input_path", type=str, default=None, required=True,
                        help="Path to the folder with input (corrupted) data to prepare for prediction or training.")
    parser.add_argument("--input_gt_path", type=str, default=None,
                        help=(
                            "Path to the folder with ground-truth data to prepare for training.\n"
                            "Skip this parameter if input data is prepared only for prediction."
                        ))
    parser.add_argument("--output_path", type=str, default=None, required=True,
                        help="Path to the output folder for the prepared data.")
    parser.add_argument("--split_ratio", type=float, nargs=3, default=[0.8, 0.1, 0.1], metavar=('train', 'val', 'test'),
                        help=(
                            "Ratio to split input data to train/val/test subsets. Provide as 3 space-separated floats.\n"
                            "Ignored if no --input_gt_path is set."
                        ))
    return parser

def read_images(input_path):
    """
    Reads image files (.tif, .tiff, .png, .jpg, .jpeg) from the specified folder,
    sorts them naturally, and returns an array of image data.

    Args:
        input_path (str): Path to the folder containing image files or to a single image.

    Returns:
        np.ndarray: Array of image data.
    """
    # Define valid file extensions
    valid_extensions = (".tif", ".tiff", ".png", ".jpg", ".jpeg")

    # List to store image arrays
    image_arrays = []
    
    # Traverse the folder and read images
    input_filepaths = [file for file in glob.glob(input_path) if file.lower().endswith(valid_extensions)]  # Check for valid extensions
    input_filepaths = natsorted(input_filepaths)
    
    for file_path in input_filepaths:
        
        # Read the image using appropriate method
        if file_path.lower().endswith((".tif", ".tiff")):
            image_array = imread(file_path)  # Use tifffile for TIFF images
        else:
            image_array = skimage_imread(file_path)  # Use skimage for other formats
        
        # Append image to the list
        image_arrays.append(image_array)

    # Convert the list of images to a NumPy array
    return np.concatenate(image_arrays)

def patchify(data, patch_size=128):
    arr = data

    # Calculate the number of patches in each dimension
    num_patches_x = arr.shape[1] // patch_size
    num_patches_y = arr.shape[2] // patch_size

    # Initialize an empty list to store the patches
    patches = []

    # Iterate over the array and extract patches
    for i in range(num_patches_x):
        for j in range(num_patches_y):
            patch = arr[:, i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size]
            patches.append(patch)
    patches_array = np.array(patches)
    
    return np.concatenate(patches_array, axis=0)

def split_dataset(data, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):

    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Split ratios must sum up to 1."

    # Calculate the split indices
    total_samples = data.shape[0]
    train_end = int(total_samples * train_ratio)
    val_end = train_end + int(total_samples * val_ratio)

    # Split the data
    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]

    return train_data, val_data, test_data

def preproc_subset(subset_path):
    # read in
    subset_raw = read_images(subset_path)
    
    # patchify as target size
    subset_patch = patchify(subset_raw)
    
    # rescale into [0, 1]
    subset_patch = rescale(subset_patch)
    return subset_patch
    
def preproc(args):
    
    low_path = args.input_path
    gt_path = args.input_gt_path

    # save the data
    output_path = args.output_path
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    if gt_path is None:
        # data is prepared for prediction
        low_raw_files = os.listdir(low_path)
        
        print('Patching input files...')
        for low_raw_file in low_raw_files:
            
            low_raw_filepath = os.path.join(low_path, low_raw_file)
            low_patch = preproc_subset(low_raw_filepath)
            
            low_file = os.path.splitext(low_raw_file)[0] + '.npz'
            low_npz_filepath = os.path.join(output_path, low_file)
            np.savez(low_npz_filepath, low=low_patch)
            print(low_npz_filepath)

        print('Done!')
        
    else:
        print('Patching input files...')
        
        # data is prepared for training
        low_path = os.path.join(low_path, '*')
        low_patch = preproc_subset(low_path)

        gt_path = os.path.join(gt_path, '*')
        gt_patch = preproc_subset(gt_path)
        
        # split the train/val/test
        ratio = args.split_ratio #[0.8, 0.1, 0.1]
        print('Split ratio: train={}, val={}, test={}'.format(*ratio))
        
        train_gt, val_gt, test_gt = split_dataset(gt_patch, ratio[0], ratio[1], ratio[2])
        train_low, val_low, test_low = split_dataset(low_patch, ratio[0], ratio[1], ratio[2])
    
        print('Data split: train={}, val={}, test={}'.format(train_low.shape[0], val_low.shape[0], test_low.shape[0]))

        for subset in ['train', 'val', 'test']:
            subset_gt = locals()[subset + '_gt']
            subset_low = locals()[subset + '_low']
            
            subset_path = os.path.join(output_path, subset)
            if not os.path.exists(subset_path):
                os.makedirs(subset_path)
            
            subset_filepath = os.path.join(subset_path, subset + '.npz')
            np.savez(subset_filepath, gt=subset_gt, low=subset_low)
            print(subset_filepath)

def main():
    parser = get_preproc_parser()
    
    # print help if no arguments passed
    if len(sys.argv)==1:
        parser.print_help(sys.stderr)
        sys.exit(1)
    
    args = parser.parse_args()
    
    preproc(args)

if __name__ == "__main__":
    main()