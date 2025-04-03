# multi-stage residual BCR

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Conv2D, AveragePooling2D, Reshape, Concatenate
from tensorflow.keras.layers import *
#from keras.layers.local import LocallyConnected2D
from tensorflow.keras.models import Model    

def rdb_block(x_in, filters, k_size, strides, rdb_depth):
    x = x_in
    for i in range(rdb_depth):
        y = x
        for j in range(2):
            x = Conv2D(filters, k_size, strides, padding='same', activation='relu', name=f'rdn_conv2d_{l}')(x)
            x = Concatenate()([x, y])
        x = Conv2D(filters, (1, 1), padding='same')(x)
        x = tf.keras.layers.add([x, y])
    return x

def RDB_LocallyConnected2D(x_in, growth_rate, stride=1, kernel_size=3, nb_layers=3):
    x = x_in
    for i in range(nb_layers):
        dense1 = LocallyConnected2D(filters=growth_rate, kernel_size=1, strides=stride, activation='relu', padding='valid')(x)
        dense2 = LocallyConnected2D(filters=growth_rate, kernel_size=kernel_size, strides=stride, activation='relu', padding='same')(dense1)
        dense3 = LocallyConnected2D(filters=growth_rate, kernel_size=kernel_size, strides=stride, activation='relu', padding='same')(dense2)
        x = Concatenate(axis=-1)([x, dense3])
    return x

def process_layer(v, x, L, L0, alpha, depth):
    for l in range(L-1, L0-1, -1):
        x[l] = rdb_block(v[l+1], filters=2*alpha, k_size=(2, 2), strides=(2, 2), rdb_depth=depth)
        v[l] = x[l][:, :, :, :alpha]

def process_upward(u, v, L0, K, alpha, name):
    u[0] = v[L0]
    for k in range(1, K+1):
        u[k] = Dense(units=alpha, activation='relu', name=f'dense_u{name}{k}')(u[k-1])
    return u[K]

def process_passthrough(x, v, u, K, L0, L, alpha, depth):
    for l in range(L0, L):
        yi = x[l]
        yi = rdb_block(v[l+1], filters=2*alpha, k_size=(2, 2), strides=(2, 2), rdb_depth=depth)

        zi = yi
        for k in range(K):
            zi = RDB_LocallyConnected2D(zi, 2*alpha, kernel_size=1)

        Xi = Concatenate(axis=3)([zi[:, :, :, alpha:], u[l]])

        u[l+1] = rdb_block(Xi, filters=2*alpha, k_size=(1, 1), strides=(1, 1), rdb_depth=depth)
        ui = Reshape((2**(l+1), alpha, 3))(u[l+1])
        u[K] = ui

def output_layer(u_final):
    u_final = Conv2D(filters=1, kernel_size=(1, 1), strides=(1, 1), padding='same')(u_final)
    u_final = AveragePooling2D(pool_size=(1, 1), padding='valid')(u_final)
    return u_final

class EB(tf.keras.layers.Layer):
    def __init__(self, out_plane=32):
        super(EB, self).__init__()
        self.main = tf.keras.Sequential([
            Conv2D(out_plane // 4, kernel_size=3, strides=1, padding='same', activation='relu'),
            Conv2D(out_plane // 2, kernel_size=1, strides=1, padding='same', activation='relu'),
            Conv2D(out_plane // 2, kernel_size=3, strides=1, padding='same', activation='relu'),
            Conv2D(out_plane - 3, kernel_size=1, strides=1, padding='same', activation='relu')
        ])

        self.conv = Conv2D(out_plane, kernel_size=1, strides=1, padding='same', activation=None)

    def call(self, x):
        x = Concatenate(axis=3)([x, self.main(x)])
        return self.conv(x)

class Fusion(tf.keras.Model):
    def __init__(self, in_channel, out_channel):
        super(Fusion, self).__init__()
        self.conv = tf.keras.Sequential([
            Conv2D(out_channel, kernel_size=(1, 1), strides=(1, 1), activation='relu', padding='valid'),
            Conv2D(out_channel, kernel_size=(3, 3), strides=(1, 1), activation=None, padding='same')
        ])

    def call(self, x1, x2):
        x = Concatenate(axis=-1)([x1, x2])
        return self.conv(x)

def BCR_RDN_mimo(input_shape, alpha, p, nb, K, L, L0, RDN=3):
    depth = RDN

    # Input layers
    x0 = Input(shape=input_shape, name='x0_input')
    x2 = Input(shape=(input_shape[0] // 2, input_shape[1] // 2, input_shape[2]), name='x2_input')
    x4 = Input(shape=(input_shape[0] // 4, input_shape[1] // 4, input_shape[2]), name='x4_input')

    # x_0
    v_0, x_0 = [None] * (L + 1), [None] * L
    v_0[L], x_0 = x0, [None] * L

    # x_2
    v_2, x_2 = [None] * (L + 1), [None] * L
    v_2[L], x_2 = x2, [None] * L
    
    # x_4
    v_4, x_4 = [None] * (L + 1), [None] * L
    v_4[L], x_4 = x4, [None] * L

    # Downward path - x_0
    process_layer(v_0, x_0, L, L0, alpha, depth)

    # Downward path - x_2
    process_layer(v_2, x_2, L, L0, alpha, depth)
    
    # Downward path - x_4
    process_layer(v_4, x_4, L, L0, alpha, depth)

    # Upward path - u_0
    u_0 = [None] * (K + 1)
    u_0_final = process_upward(u_0, v_0, L0, K, alpha, '0')

    # Upward path - u_2
    u_2 = [None] * (K + 1)
    u_2_final = process_upward(u_2, v_2, L0, K, alpha, '2')
    
    # Upward path - u_4
    u_4 = [None] * (K + 1)
    u_4_final = process_upward(u_4, v_4, L0, K, alpha, '4')
    
    # u fusion
    eb_layer_u_0 = EB()
    temp_x0 = tf.concat([x0, u_0_final], axis=3)
    post_x0 = eb_layer_u_0(temp_x0)
    
    temp_x2 = tf.concat([x2, u_2_final], axis=3)
    post_x2 = eb_layer_u_0(temp_x2)
    
    temp_x4 = tf.concat([x4, u_4_final], axis=3)
    post_x4 = eb_layer_u_0(temp_x4)

    # Passthrough layers - x_0
    process_passthrough(x_0, v_0, u_0, K, L0, L, alpha, depth)
    y_0 = u_0[K]

    # Passthrough layers - x_2
    process_passthrough(x_2, v_2, u_2, K, L0, L, alpha, depth)
    y_2 = u_2[K]
    
    # Passthrough layers - x_4
    process_passthrough(x_4, v_4, u_4, K, L0, L, alpha, depth)
    y_4 = u_4[K]

    # Fusion of y
    aff_layer = Fusion(in_channel=64, out_channel=32)
    y2_resized = tf.image.resize(y_2, size=(input_shape[0], input_shape[1]), method=tf.image.ResizeMethod.BILINEAR)
    y4_resized = tf.image.resize(y_4, size=(input_shape[0], input_shape[1]), method=tf.image.ResizeMethod.BILINEAR)
    y42_fusion = aff_layer(y_0, y2_resized)
    y20_fusion = aff_layer(y2_resized, y4_resized)
    y20_fusion = tf.image.resize(y20_fusion, size=(input_shape[0] // 2, input_shape[1] // 2), method=tf.image.ResizeMethod.BILINEAR)
    
    # Output layers - x_4
    y_4 = output_layer(y_4)
    
    # Output layers - x_2
    con_y_2 = tf.concat([y_2, y20_fusion], axis=3)
    y_2 = output_layer(con_y_2)
    
    # Output layers - x_0
    con_y_0 = tf.concat([y_0, y42_fusion], axis=3)
    y_0 = output_layer(con_y_0)

    model = tf.keras.Model(inputs=[x0, x2, x4], outputs=[y_0, y_2, y_4])

    return model

def residual_dense_block(input_tensor, filters, rdb_depth):
    x = input_tensor
    for i in range(rdb_depth):
        conv1 = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same', activation='relu')(x)
        conv2 = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same')(conv1)
        x = Concatenate()([x, conv2])
    x = Conv2D(filters=filters, kernel_size=1, strides=1, padding='same')(x)
    x = Concatenate()([x, input_tensor])
    return x

def inverse_process(xi, alpha1, alpha2, Ncnn1, Ncnn2, w1, name, rdb_depth):
    for index1 in range(Ncnn1):
        xi = Conv1D(alpha1, w1, activation='relu', padding='same', name=f'{name}_conv1d_{index1}')(xi)
        
    for index2 in range(Ncnn2-1):
        xi = residual_dense_block(xi, filters=alpha2, rdb_depth=3)
        
    return xi

def inverse_RDN_mimo(input_shape, alpha1, alpha2, w1, w2, Ncnn1, Ncnn2, RDN=3):
    rdb_depth = RDN
    # Input layers
    x0_inputs = Input(shape=input_shape, name='x0_input')
    x2_inputs = Input(shape=(input_shape[0] // 2, input_shape[1] // 2, input_shape[2]), name='x2_input')
    x4_inputs = Input(shape=(input_shape[0] // 4, input_shape[1] // 4, input_shape[2]), name='x4_input')
    
    aff_layer = Fusion(in_channel=80, out_channel=48)
    
    # Inverse the process for x4
    xi_4 = inverse_process(x4_inputs, alpha1, alpha2, Ncnn1, Ncnn2, w1, 'x4', rdb_depth)
    
    # Fusion of xi_4 and x2_inputs
    xi_4_resized = tf.image.resize(xi_4, size=(input_shape[0] // 2, input_shape[1] // 2), method=tf.image.ResizeMethod.BILINEAR)
    x42_fusion = aff_layer(x2_inputs, xi_4_resized)
    
    # Inverse the process for x2
    xi_2 = inverse_process(x42_fusion, alpha1, alpha2, Ncnn1, Ncnn2, w1, 'x2', rdb_depth)
    
    # Fusion of xi_2 and x0_inputs
    xi_2_resized = tf.image.resize(xi_2, size=(input_shape[0], input_shape[1]), method=tf.image.ResizeMethod.BILINEAR)
    x20_fusion = aff_layer(x0_inputs, xi_2_resized)
    
    # Inverse the process for x0 with fusion
    xi_0 = inverse_process(x20_fusion, alpha1, alpha2, Ncnn1, Ncnn2, w1, 'x0', rdb_depth)
    
    # Output layers with added inputs
    y4_outputs = Conv2D(1, w2, padding='same', name='ouputs_4')(xi_4 + x4_inputs)
    y2_outputs = Conv2D(1, w2, padding='same', name='ouputs_2')(xi_2 + x2_inputs)
    y0_outputs = Conv2D(1, w2, padding='same', name='ouput_0')(xi_0 + x0_inputs)

    model = Model(inputs=[x0_inputs, x2_inputs, x4_inputs], outputs=[y0_outputs, y2_outputs, y4_outputs])
    
    return model

def m_rBCR(input_shape=(128, 128, 1)):
    
    # Create the full model
    x0 = Input(shape=input_shape, name='x0_input')
    x2 = Input(shape=(input_shape[0] // 2, input_shape[1] // 2, input_shape[2]), name='x2_input')
    x4 = Input(shape=(input_shape[0] // 4, input_shape[1] // 4, input_shape[2]), name='x4_input')

    # initialize the model
    forward_model_RDN = BCR_RDN_mimo(input_shape=input_shape, alpha=32, p=0.2, nb=3, K=12, L=12, L0=12, RDN=7) 
    inverse_model_RDN = inverse_RDN_mimo(input_shape=input_shape, alpha1=32, alpha2=4, w1=5, w2=9, Ncnn1=6, Ncnn2=5, RDN=7)

    [y0, y2, y4] = forward_model_RDN([x0, x2, x4])
    [z0, z2, z4] = inverse_model_RDN([y0, y2, y4])

    model = Model(inputs=[x0, x2, x4], outputs=[z0, z2, z4])
    
    # print(model.input_shape, model.output_shape) # [(None, 128, 128, 1), (None, 64, 64, 1), (None, 32, 32, 1)] 
    # print(model.summary())
    
    return model