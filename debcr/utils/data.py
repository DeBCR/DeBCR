import os 
import numpy as np
import matplotlib.pyplot as plt
import glob
import random
from natsort import natsorted
from debcr.utils.utils import *

def data_prep(BATCH= 64, path='./xxx/'):

    # define the data path 
    DATA_PATH = path 
    batch_size = BATCH
    
    # training
    train_data_dir = DATA_PATH + 'train/'
    train_data_list = natsorted(os.listdir(train_data_dir))  

    # validate
    val_data_dir = DATA_PATH + 'val/'
    val_data_list = natsorted(os.listdir(val_data_dir))  

    # data generator
    train_gen_class = DataGeneratorMix(train_data_dir, train_data_list,batch_size, True)
    train_img_datagen = train_gen_class.image_loader()
    
    val_gen_class = DataGeneratorMix(val_data_dir, val_data_list,batch_size=16, noise=True)
    val_img_datagen = val_gen_class.image_loader()
    
    
    return train_img_datagen, val_img_datagen