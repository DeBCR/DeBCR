from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, concatenate, Conv2DTranspose, BatchNormalization
from keras.layers import *
from tensorflow.keras.optimizers import Adam
from tensorflow_addons.layers import InstanceNormalization

def conv2d_block(input_tensor, n_filters, kernel_size=3, batchnorm=True):
    """Function to add 2D convolutional block"""
    # First layer
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer='he_normal', padding='same')(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    x = InstanceNormalization()(x)
    x = Activation('relu')(x)

    # Second layer
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer='he_normal', padding='same')(x)
    if batchnorm:
        x = BatchNormalization()(x)
    x = InstanceNormalization()(x)
    x = Activation('relu')(x)

    return x

def simple_unet_model(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS):
    """Function to build a simple U-Net model"""
    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    s = inputs

    # Contraction path
    c1 = conv2d_block(s, 16)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = conv2d_block(p1, 32)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = conv2d_block(p2, 64)
    p3 = MaxPooling2D((2, 2))(c3)

    c4 = conv2d_block(p3, 128)
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)

    c5 = conv2d_block(p4, 256)
    p5 = MaxPooling2D(pool_size=(2, 2))(c5)

    # Expand one layer, deepen the network
    c6 = conv2d_block(p5, 512)
    c6_hidden = InstanceNormalization()(c6)

    # Expansive path
    u7 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(c6_hidden)
    u7 = concatenate([u7, c5])
    c7 = conv2d_block(u7, 256)

    u8 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, c4])
    c8 = conv2d_block(u8, 128)

    u9 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, c3])
    c9 = conv2d_block(u9, 64)

    u10 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c9)
    u10 = concatenate([u10, c2])
    c10 = conv2d_block(u10, 32)

    u11 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c10)
    u11 = concatenate([u11, c1])
    c11 = conv2d_block(u11, 16)

    # Last layer, change the activation to sigmoid
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c11)

    model = Model(inputs=[inputs], outputs=[outputs])

    return model

# Example usage:
# model = simple_unet_model(256, 256, 1)
# model.summary()
