import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import exposure
import glob
import random
import argparse

# visualization for two images
def subShow3(IMG1, IMG2, IMG3):
    
    color = 'inferno'
    
    plt.subplot(1,3,1)
    plt.imshow(IMG1, cmap=color)
    plt.axis('off')

    plt.subplot(1,3,2)
    plt.imshow(IMG2, cmap=color)
    plt.axis('off')

    plt.subplot(1,3,3)
    plt.imshow(IMG3, cmap=color)
    plt.axis('off')
    plt.show()

def subShow(IMG1, IMG2):
    
    color = 'inferno'
        
    plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(IMG1, cmap=color)
    plt.subplot(1,2,2)
    plt.imshow(IMG2, cmap=color)
    plt.show()
    plt.close()
     
def dict_to_namespace(config):
    """
    Converts a dictionary to a namespace object.
    """
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            # Recursively convert nested dictionaries
            new_value = dict_to_namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace

# multiple inputs for the model
def multi_input(w_img, o_img):
    
    w_0, o_0 = w_img, o_img
    w_2, o_2 = w_0[:, ::2, ::2, :], o_0[:, ::2, ::2, :]
    w_4, o_4 = w_0[:, ::4, ::4, :], o_0[:, ::4, ::4, :]
    
    return [w_0, w_2, w_4], [o_0, o_2, o_4]

def save_grid(pred_list, w_list, o_list, output_path, domain, eval_results, NUM=5, color='inferno'):
    num_images = NUM
    random_indices = random.sample(range(len(w_list)), num_images)
    
    # Format the title string using eval_results
    title_str = f'all test: PSNR: {eval_results[0]:.2f}, SSIM: {eval_results[1]:.4f}, RMSE: {eval_results[2]:.4f}'

    fig, axes = plt.subplots(3, num_images, figsize=(num_images * 3, 9))
    
    # Set the figure title
    fig.suptitle(title_str, fontsize=16)

    for i, idx in enumerate(random_indices):
        axes[0, i].imshow(w_list[idx], cmap=color)
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].axis('on')
            axes[0, i].set_ylabel('Input', fontsize=12, labelpad=10)
            axes[0, i].yaxis.set_ticks([])  
            axes[0, i].yaxis.set_ticklabels([])  
            axes[0, i].xaxis.set_ticks([])  
            axes[0, i].xaxis.set_ticklabels([]) 

    for i, idx in enumerate(random_indices):
        axes[1, i].imshow(pred_list[idx], cmap=color)
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].axis('on')
            axes[1, i].set_ylabel('Prediction', fontsize=12, labelpad=10)
            axes[1, i].yaxis.set_ticks([])  
            axes[1, i].yaxis.set_ticklabels([])  
            axes[1, i].xaxis.set_ticks([])  
            axes[1, i].xaxis.set_ticklabels([]) 

    for i, idx in enumerate(random_indices):
        axes[2, i].imshow(o_list[idx], cmap=color)
        axes[2, i].axis('off')
        if i == 0:
            axes[2, i].axis('on')
            axes[2, i].set_ylabel('Ground Truth', fontsize=12, labelpad=10)
            axes[2, i].yaxis.set_ticks([])  
            axes[2, i].yaxis.set_ticklabels([])  
            axes[2, i].xaxis.set_ticks([])  
            axes[2, i].xaxis.set_ticklabels([]) 
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust rect to make room for the title
    svg_filename = os.path.join(output_path, f"{str(domain)}.png")
    plt.savefig(svg_filename, bbox_inches='tight', pad_inches=0, dpi=96)
    plt.close()
    
def show_grid(pred_list, w_list, o_list, NUM=5, color='inferno'):
    num_images = NUM
    random_indices = random.sample(range(len(w_list)), num_images)

    fig, axes = plt.subplots(3, num_images, figsize=(num_images * 3, 9))
    
    for i, idx in enumerate(random_indices):
        axes[0, i].imshow(w_list[idx], cmap=color)
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].axis('on')
            axes[0, i].set_ylabel('Input', fontsize=12, labelpad=10)
            axes[0, i].yaxis.set_ticks([])  
            axes[0, i].yaxis.set_ticklabels([])  
            axes[0, i].xaxis.set_ticks([])  
            axes[0, i].xaxis.set_ticklabels([]) 

    for i, idx in enumerate(random_indices):
        axes[1, i].imshow(pred_list[idx], cmap=color)
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].axis('on')
            axes[1, i].set_ylabel('Prediction', fontsize=12, labelpad=10)
            axes[1, i].yaxis.set_ticks([])  
            axes[1, i].yaxis.set_ticklabels([])  
            axes[1, i].xaxis.set_ticks([])  
            axes[1, i].xaxis.set_ticklabels([]) 

    for i, idx in enumerate(random_indices):
        axes[2, i].imshow(o_list[idx], cmap=color)
        axes[2, i].axis('off')
        if i == 0:
            axes[2, i].axis('on')
            axes[2, i].set_ylabel('Ground Truth', fontsize=12, labelpad=10)
            axes[2, i].yaxis.set_ticks([])  
            axes[2, i].yaxis.set_ticklabels([])  
            axes[2, i].xaxis.set_ticks([])  
            axes[2, i].xaxis.set_ticklabels([]) 
    
    plt.tight_layout()
    plt.show()
    
def rescale(image_stack, MIN=0, MAX=1):
    # Rescale the whole stack
    if image_stack[0].max() != 1:
        image_scale = []
        for stack in range(image_stack.shape[0]):
            temp = image_stack[stack, ...]
            temp_scale = np.interp(temp, (temp.min(), temp.max()), (MIN, MAX))
            image_scale.append(temp_scale.astype('float64'))
    else:
        image_scale = image_stack
    return np.asarray(image_scale)

def clip_intensity(image_stack, LOW_PERC=2, HIGH_PERC=98):
    # Rescale the whole stack
    image_clip_intensity = []
    for stack in range(image_stack.shape[0]):
        temp = image_stack[stack, ...]
        p_low, p_high = np.percentile(temp, (LOW_PERC, HIGH_PERC))
        temp_clip_intensity = exposure.rescale_intensity(temp, in_range=(p_low, p_high))
        image_clip_intensity.append(temp_clip_intensity.astype('float64'))
    return np.asarray(image_clip_intensity)

# load databanks for both LM and EM
class DataGenerator:
    def __init__(self, data_dir, data_list, batch_size, noise):
        # Set the location of the data
        self.data_dir = data_dir
        self.data_list = data_list
        self.batch_size = batch_size
        self.noise = noise
        
    def _rescale(self, image_stack, MIN=0, MAX=1):
        # Rescale the whole stack
        if image_stack[0].max() != 1:
            image_scale = []
            for stack in range(image_stack.shape[0]):
                temp = image_stack[stack, ...]
                temp_scale = np.interp(temp, (temp.min(), temp.max()), (MIN, MAX))
                image_scale.append(temp_scale.astype('float64'))
        else:
            image_scale = image_stack
        return np.asarray(image_scale)
    
    def _norm_01(x):
        return np.nan_to_num((x - np.amin(x, axis=(1, 2, 3), keepdims=True)) / (
                np.amax(x, axis=(1, 2, 3), keepdims=True) - np.amin(x, axis=(1, 2, 3), keepdims=True)))
    
    def imageLoader(self):
        
            for index, dataset_name in enumerate(self.data_list):
                print('Loading dataset:', dataset_name)
                temp_dataset = np.load(os.path.join(self.data_dir,dataset_name))
                
                w_imgs, o_imgs = (
                    temp_dataset['low'],
                    temp_dataset['gt']
                )
                
                w_imgs, o_imgs = np.expand_dims(w_imgs, axis=3), np.expand_dims(o_imgs, axis=3)
                L = w_imgs.shape[0]
                batch_start = 0
                batch_end = self.batch_size
                
                while True:
                    
                    sample_indices = np.random.choice(w_imgs.shape[0], self.batch_size, replace=False)
                    w_img_temp, o_temp = w_imgs[sample_indices], o_imgs[sample_indices]
                    
                    # add noise
                    if self.noise:
                        gaussian_sigma, lambda_poisson=np.random.uniform(0.0, 0.05), np.random.uniform(0.0, 0.1)
                        gaussian_noise = np.random.normal(0, gaussian_sigma, w_img_temp.shape)
                        poisson_noise = np.random.poisson(lambda_poisson, w_img_temp.shape)
                        w_img_temp = w_img_temp + 0.5*gaussian_noise #+ 0.5*poisson_noise
                    
                    # Rescale into [0, 1]
                    w_img_temp = self._rescale(w_img_temp, MIN=0, MAX=1)
                    o_temp = self._rescale(o_temp, MIN=0, MAX=1)
                    
                    yield w_img_temp, o_temp
