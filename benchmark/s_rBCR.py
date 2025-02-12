import keras
from keras.layers import Input, Conv2D, Dense, Reshape, AveragePooling2D, Concatenate, LocallyConnected2D
from keras.layers import *
from keras.models import Model
from util.loss_func import *
from util.metrics import *

def rdb_block(x, filters, k_size, strides, rdb_depth):
    """Residual Dense Block"""
    for _ in range(rdb_depth):
        y = x
        for _ in range(2):
            x = Conv2D(filters, k_size, strides=strides, padding='same', activation='relu')(x)
            x = Concatenate()([x, y])
        x = Conv2D(filters, (1, 1), padding='same')(x)
        x = keras.layers.add([x, y])
    return x

def RDB_LocallyConnected2D(x, growth_rate, stride=1, kernel_size=3, nb_layers=3):
    """RDB Locally Connected 2D"""
    for _ in range(nb_layers):
        dense1 = LocallyConnected2D(filters=growth_rate, kernel_size=1, strides=stride, activation='relu', padding='valid')(x)
        dense2 = LocallyConnected2D(filters=growth_rate, kernel_size=kernel_size, strides=stride, activation='relu', padding='same')(dense1)
        dense3 = LocallyConnected2D(filters=growth_rate, kernel_size=kernel_size, strides=stride, activation='relu', padding='same')(dense2)
        x = Concatenate(axis=-1)([x, dense3])
    return x

def residual_dense_block(input_tensor, filters, rdb_depth):
    """Residual Dense Block"""
    x = input_tensor
    for _ in range(rdb_depth):
        conv1 = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same', activation='relu')(x)
        conv2 = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same')(conv1)
        x = Concatenate()([x, conv2])
    x = Conv2D(filters=filters, kernel_size=1, strides=1, padding='same')(x)
    x = Concatenate()([x, input_tensor])
    return x

def BCR_RDN(input_shape, alpha, p, nb, K, L, L0, RDN=3):
    """BCR Residual Dense Network"""
    x_in = Input(input_shape)
    depth = RDN

    v = [None] * (L + 1)
    v[L] = x_in

    x = [None] * L
    for l in range(L-1, L0-1, -1):
        x[l] = rdb_block(v[l+1], filters=2*alpha, k_size=(2, 2), strides=(2, 2), rdb_depth=depth)
        v[l] = x[l][:, :, :, :alpha]

    u = [None] * (K + 1)
    u[0] = v[L0]
    for k in range(1, K+1):
        u[k] = Dense(units=alpha, activation='relu', name=f'dense_{k}')(u[k-1])
    u_final = u[K]

    for l in range(L0, L):
        yi = x[l]
        yi = rdb_block(v[l+1], filters=2*alpha, k_size=(2, 2), strides=(2, 2), rdb_depth=depth)

        zi = yi
        for _ in range(K):
            zi = RDB_LocallyConnected2D(zi, 2*alpha, kernel_size=1)

        Xi = Concatenate(axis=3)([zi[:, :, :, alpha:], u[l]])
        u[l+1] = rdb_block(Xi, filters=2*alpha, k_size=(1, 1), strides=(1, 1), rdb_depth=depth)

        ui = Reshape((2**(l+1), alpha, 3))(u[l+1])
        u_final = ui

    u_final = Conv2D(filters=128, kernel_size=(1, 1), strides=(1, 1), padding='same')(u_final)
    u_final = AveragePooling2D(pool_size=(1, 1), padding='valid')(u_final)
    y = u_final
    model = Model(inputs=x_in, outputs=y)
    return model

def inverse_RDN(alpha1, alpha2, w1, w2, Ncnn1, Ncnn2, RDN=3, input_shape=(128, 128, 128)):
    """Inverse Residual Dense Network"""
    rdb_depth = RDN
    x_inputs = Input(shape=input_shape)
    xi = x_inputs

    for index1 in range(Ncnn1):
        xi = Conv1D(alpha1, w1, activation='relu', padding='same', name=f'back_conv1d_{index1}')(xi)

    for _ in range(Ncnn2-1):
        xi = residual_dense_block(xi, filters=alpha2, rdb_depth=3)

    y_outputs = Conv2D(1, w2, padding='same', name='outputs')(xi)

    model = Model(inputs=x_inputs, outputs=y_outputs)
    return model

def model_s_rBCR(IMG_SHAPE=(128, 128, 1)):
    
    # Create the full model
    input_shape = IMG_SHAPE
    forward_model_RDN = BCR_RDN(input_shape=IMG_SHAPE, alpha=32, p=0.2, nb=3, K=12, L=12, L0=12, RDN=5) # 最后的参数标明residual的深度
    inverse_model_RDN = inverse_RDN(alpha1=32, alpha2=4, w1=5, w2=9, Ncnn1=6, Ncnn2=5, RDN=5)
    
    inputs = Input(shape=input_shape)
    x = inputs
    y = forward_model_RDN(x)
    z = inverse_model_RDN(y)

    # combine the forward and backward
    model = Model(inputs=inputs, outputs=z)
    model.compile(optimizer='adam', loss='mse', metrics=[metrics_PSNR]) 

    # print(model.input_shape, model.output_shape)
    # print(model.summary())
    
    return model