import tensorflow as tf
from keras.layers import Input, Concatenate, Lambda
from tensorflow.keras.models import Model
from models.unet_ddpm import Unet_2d_ddpm

def obtain_noisy_sample(x):
    x_0, gamma = x[0], x[1]
    gamma_vec = tf.expand_dims(tf.expand_dims(gamma, axis=-1), axis=-1)
    noise_sample = tf.random.normal(tf.shape(x_0))
    
    noisy = tf.sqrt(gamma_vec) * x_0 + tf.sqrt(1 - gamma_vec) * noise_sample
    return [noisy, noise_sample]

def ddpm_model(input_shape_noisy, out_channels):
    noisy_input = Input(input_shape_noisy)
    ref = Input(input_shape_noisy)

    c_input = Concatenate()([noisy_input, ref])  
    gamma_inp = Input((1,))
    noise_out = Unet_2d_ddpm(input_shape_noisy, inputs=c_input, gamma_inp=gamma_inp,
                             out_filters=out_channels, z_enc=False)  
    model_out = Model([noisy_input, ref, gamma_inp], noise_out)

    return model_out

def train_model(input_shape_condition, out_channels=1):
    ground_truth = Input(input_shape_condition) 
    dirty_img = Input(input_shape_condition)  
    gamma_inp = Input((1,))

    n_model = ddpm_model(input_shape_condition, out_channels)
    n_sample = Lambda(obtain_noisy_sample)([ground_truth, gamma_inp])

    noise_pred = n_model([n_sample[0], dirty_img, gamma_inp])
    delta_noise = noise_pred - n_sample[1]

    training_model = Model([ground_truth, dirty_img, gamma_inp], delta_noise)
    training_model.summary()

    return n_model, training_model