# metrics
import tensorflow as tf
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
    
def metrics(IMG1, IMG2):
    # input: IMG1, IMG2 - numpy arrays with shape [batch_size, height, width, channels]
    # return average psnr, ssim, rmse across all images
    
    batch_size = IMG1.shape[0]
    
    # Initialize accumulators
    psnr_accumulator = 0.0
    ssim_accumulator = 0.0
    rmse_accumulator = 0.0

    for i in range(batch_size):
        img1 = IMG1[i, ..., 0]  # Extract the single-channel image from the array
        img2 = IMG2[i, ..., 0]

        # PSNR
        psnr_value = peak_signal_noise_ratio(img1, img2, data_range=1)
        psnr_accumulator += psnr_value

        # SSIM
        ssim_value, _ = structural_similarity(img1, img2, data_range=1, full=True)
        ssim_accumulator += ssim_value

        # RMSE
        mse = np.mean((img1 - img2) ** 2)
        rmse_value = np.sqrt(mse)
        rmse_accumulator += rmse_value

    # Calculate average metrics
    avg_psnr = psnr_accumulator / batch_size
    avg_ssim = ssim_accumulator / batch_size
    avg_rmse = rmse_accumulator / batch_size

    return avg_psnr, avg_ssim, avg_rmse

def metrics_PSNR(y_true, y_pred):
    
    squared_difference = tf.square(y_true - y_pred)
    mse = tf.reduce_mean(squared_difference)

    # Define the maximum possible pixel value (1.0 for images in the range [0, 1])
    max_pixel_value = 1.0

    # Calculate the PSNR
    psnr = 10 * tf.math.log((max_pixel_value**2) / mse) / tf.math.log(10.0)
    
    return psnr

def metrics_func_mimo(y_true_list, y_pred_list):
    
    y_true, y_pred = y_true_list[0], y_pred_list[0]
    
    return metrics_PSNR(y_true, y_pred)
