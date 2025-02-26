# unet for ddpm with time stamp

import tensorflow as tf
from keras.layers import *
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K
from tensorflow_addons.layers import InstanceNormalization


def tile_gamma(x):
    gamma = x[0]
    noisy_y = x[1]
    return tf.ones_like(noisy_y) * tf.expand_dims(tf.expand_dims(gamma, axis=-1), axis=-1)

def Unet_2d_ddpm(shape,previous_tensor=None,inputs=None,gamma_inp=None,full_model=False,noise_input=False,z_enc=False,out_filters=2):
    if inputs is None:
        inputs = Input(shape) 
    if gamma_inp is None:
        gamma_inp = Input((1,)) 

    if noise_input:
        noise_in = Input((128,))
        conv1 = Conv2D(64, 3, activation='softplus', padding='same')(inputs)
    else:
        conv1 = Conv2D(64, 3, activation='softplus', padding='same')(inputs) 

    tiled_gamma = Lambda(tile_gamma)([gamma_inp, inputs]) 
    conv_gamma = Conv2D(64, 7, activation='softplus', padding='same')(tiled_gamma)  
    conv1 = Add()([conv_gamma, conv1]) 
    
    conv1 = InstanceNormalization()(conv1)
    conv1 = Conv2D(64, 3, activation='softplus', padding='same')(conv1)
    conv1 = InstanceNormalization()(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    conv2 = (Conv2D(128, 3, activation='softplus', padding='same'))(pool1)
    conv2 = InstanceNormalization()(conv2)
    conv2 = Conv2D(128, 3, activation='softplus', padding='same')(conv2)
    conv2 = InstanceNormalization()(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    conv3 = Conv2D(256, 3, activation='softplus', padding='same')(pool2)
    conv3 = InstanceNormalization()(conv3)
    conv3 = Conv2D(256, 3, activation='softplus', padding='same')(conv3)
    conv3 = InstanceNormalization()(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    
    conv4 = Conv2D(512, 3, activation='softplus', padding='same')(pool3)
    conv4 = InstanceNormalization()(conv4)
    conv4 = Conv2D(512, 3, activation='softplus', padding='same')(conv4)
    conv4 = InstanceNormalization()(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(512, 3, activation='softplus', padding='same')(pool4)
    conv5 = InstanceNormalization()(conv5)
    conv5 = Conv2D(1024, 3, activation='softplus', padding='same')(conv5)
    conv5 = InstanceNormalization()(conv5)

    print(K.int_shape(conv5))
    if previous_tensor is not None:
        previous_tensor = Conv2DTranspose(1024, 3, strides=2, activation='softplus', padding='same')(previous_tensor)
        previous_tensor = Attention(1024)(previous_tensor)
        conv5 = Add()([conv5, previous_tensor])

    if z_enc is not False:
        gamma_vec = Concatenate()([z_enc, conv5])
    else:
        gamma_vec = conv5

    up6 = Conv2D(512,2,activation='softplus',padding='same')(UpSampling2D((2, 2))(gamma_vec))

    merge6 = Concatenate()([conv4, up6])
    conv6 = Conv2D(512, 3, activation='softplus', padding='same')(merge6)
    conv6 = InstanceNormalization()(conv6)
    conv6 = Conv2D(512, 3, activation='softplus', padding='same')(conv6)
    conv6 = InstanceNormalization()(conv6)

    up7 = Conv2D(256,2,activation='softplus',padding='same')(UpSampling2D((2, 2))(conv6))
    merge7 = Concatenate()([conv3, up7])
    conv7 = Conv2D(256, 3, activation='softplus', padding='same')(merge7)
    conv7 = InstanceNormalization()(conv7)
    conv7 = Conv2D(256, 3, activation='softplus', padding='same')(conv7)
    conv7 = InstanceNormalization()(conv7)

    up8 = Conv2D(128,2,activation='softplus',padding='same')(UpSampling2D((2, 2))(conv7))
    merge8 = Concatenate()([conv2, up8])
    conv8 = Conv2D(128, 3, activation='softplus', padding='same')(merge8)
    conv8 = InstanceNormalization()(conv8)
    conv8 = Conv2D(128, 3, activation='softplus', padding='same')(conv8)
    conv8 = InstanceNormalization()(conv8)

    up9 = Conv2D(64,2,activation='softplus',padding='same')(UpSampling2D((2, 2))(conv8))

    merge9 = Concatenate()([conv1, up9])
    conv9 = Conv2D(64, 3, activation='softplus', padding='same')(merge9)
    conv9 = InstanceNormalization()(conv9)
    conv9 = Conv2D(64, 3, activation='softplus', padding='same')(conv9)
    conv9 = InstanceNormalization()(conv9)
    conv9 = Conv2D(32, 3, activation='softplus', padding='same')(conv9)
    conv9 = InstanceNormalization()(conv9)
    
    conv10 = Conv2D(out_filters, 1, activation='linear')(conv9)
    
    outputs = conv10

    return outputs
