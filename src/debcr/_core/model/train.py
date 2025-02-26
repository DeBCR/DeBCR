import random
import time

import tensorflow as tf

from .utils import multi_input
from .utils import setup_ckpt_manager
from .loss import loss_function_mimo
from .metrics import metrics_func_mimo
#from .show_utils import subShow3

def train_model(model, train_img_datagen, val_img_datagen, train_config): #config.training
    
    best_val_loss = float('inf')
    wait = 0
    
    checkpoint, checkpoint_manager = setup_ckpt_manager(model, train_config['ckpt_path']) # setup checkpoint manager
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=train_config['lr'])
    
    start_time = time.time()

    for step in range(train_config['NUM_STEPS']):
        w_train, o_train = train_img_datagen.__next__()
        w_train_list, o_train_list = multi_input(w_train), multi_input(o_train)

        with tf.GradientTape() as tape:
            # Forward pass
            predictions = model(w_train_list)
            
            # Calculate the loss manually
            loss = loss_function_mimo(o_train_list, predictions)
            metric = metrics_func_mimo(o_train_list, predictions)

        # Compute gradients
        gradients = tape.gradient(loss, model.trainable_variables)

        # Update the model's weights
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        if step % train_config['save_freq'] == 0:
            print(step, loss, metric)
            # Save the model weights using the Checkpoint
            checkpoint_manager.save()

        if step % train_config['val_freq'] == 0:
            w_eval, o_eval = val_img_datagen.__next__()
            w_eval_list, o_eval_list = multi_input(w_eval), multi_input(o_eval)
            val_predictions = model(w_eval_list)

            # Calculate the validation loss manually
            val_loss = loss_function_mimo(o_eval_list, val_predictions)
            val_metric = metrics_func_mimo(o_eval_list, val_predictions)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                wait = 0
                print('Validation best loss:', step, best_val_loss, val_metric)
                # Save the best model weights using the Checkpoint
                checkpoint_manager.save()
            else:
                wait += 1

            #if train_config['visual']:
            #    s_NUM = random.randint(0, predictions[0].shape[0] - 1)
            #    print('Objects:', s_NUM)
            #    subShow3(w_train[s_NUM], predictions[0][s_NUM], o_train[s_NUM])
            
            if wait >= train_config['patience']:
                print("Early stopping due to no improvement in validation loss.", step)
                # Save the early stopping model weights using the Checkpoint
                checkpoint_manager.save()
                break
                
    # Calculate and print the elapsed time
    elapsed_time = time.time() - start_time
    print("Elapsed time:", elapsed_time)
    return model