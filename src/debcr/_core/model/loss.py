# loss function 
import tensorflow as tf
import tensorflow.keras.backend as K

# the loss function
def loss_function(y_true, y_pred):
    # mse
    squared_diff = tf.square(y_true - y_pred)
    mse_loss = tf.reduce_mean(squared_diff)
    
    # fft
    label_fft1 = tf.signal.rfft2d(y_true)
    pred_fft1 = tf.signal.rfft2d(y_pred)
    fft_loss_raw = tf.math.reduce_mean(tf.math.abs(label_fft1 - pred_fft1), axis=(1,2,3))
    fft_loss = tf.math.reduce_mean(fft_loss_raw)

    total_loss = 1.0*mse_loss + 0.5*fft_loss
    
    return total_loss

def loss_function_mimo(y, y_pred):
    loss_0 = loss_function(y[0], y_pred[0])
    loss_2 = loss_function(y[1], y_pred[1])
    loss_4 = loss_function(y[2], y_pred[2])
    
    total_loss = loss_0 + loss_2 + loss_4
    
    return total_loss