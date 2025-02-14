import os
import random
import numpy as np
from natsort import natsorted

# load dataset to databank
def load_dataset(data_path, batch_size=16, add_noise=False): #config.data
    data_list = [data_path] if os.path.isfile(data_path) else natsorted(os.listdir(data_path))
    datagen_obj = DataGenerator(data_path, data_list, batch_size, add_noise)
    datagen_img = datagen_obj.imageLoader()
    return datagen_img

# generate databank
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
                        #poisson_noise = np.random.poisson(lambda_poisson, w_img_temp.shape)
                        w_img_temp = w_img_temp + 0.5*gaussian_noise #+ 0.5*poisson_noise
                    
                    # Rescale into [0, 1]
                    w_img_temp = self._rescale(w_img_temp, MIN=0, MAX=1)
                    o_temp = self._rescale(o_temp, MIN=0, MAX=1)
                    
                    yield w_img_temp, o_temp