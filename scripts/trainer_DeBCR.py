import random
import time
import tensorflow as tf
from debcr.utils import utils  #  multi_input, loss_function_mimo, metrics_func_mimo

def train_model(config, model, multi_input, loss_function_mimo, metrics_func_mimo,  train_img_datagen,  val_img_datagen, visual=True):
    
    best_val_loss = float('inf')
    wait = 0

    optimizer = tf.keras.optimizers.Adam(learning_rate=config.training['lr'])
    checkpoint = tf.train.Checkpoint(model=model)
    checkpoint_manager = tf.train.CheckpointManager(checkpoint, config.training['ckpt_path'], max_to_keep=5)

    start_time = time.time()

    for step in range(config.training['NUM_STEPS']):
        w_train, o_train = train_img_datagen.__next__()
        w_train_list, o_train_list = multi_input(w_train, o_train)

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

        if step % config.training['save_freq'] == 0:
            print(step, loss, metric)
            # Save the model weights using the Checkpoint
            checkpoint_manager.save()

        if step % config.training['val_freq'] == 0:
            w_eval, o_eval = val_img_datagen.__next__()
            w_eval_list, o_eval_list = multi_input(w_eval, o_eval)
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

            s_NUM = random.randint(0, predictions[0].shape[0] - 1)
            print('Objects:', s_NUM)
            subShow3(w_train[s_NUM], predictions[0][s_NUM], o_train[s_NUM])

            if wait >= config.training['patience']:
                print("Early stopping due to no improvement in validation loss.", step)
                # Save the early stopping model weights using the Checkpoint
                checkpoint_manager.save()
                break
                
    # Calculate and print the elapsed time
    elapsed_time = time.time() - start_time
    print("Elapsed time:", elapsed_time)
    return model
