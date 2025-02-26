import os
import glob
import numpy as np
import tensorflow as tf

def build_and_compile():
    from .m_rBCR import m_rBCR
    from .loss import loss_function_mimo
    from .metrics import metrics_func_mimo
    
    model = m_rBCR()
    model.compile(optimizer='adam', loss=loss_function_mimo, metrics=[metrics_func_mimo]) 
    
    return model

def setup_ckpt_manager(model, checkpoint_dirpath: str):
    
    # setup checkpoint manager
    checkpoint = tf.train.Checkpoint(model=model)
    checkpoint_manager = tf.train.CheckpointManager(checkpoint, checkpoint_dirpath, max_to_keep=5)
    
    return checkpoint, checkpoint_manager

def restore_ckpt(model, ckpt_dirpath, ckpt_name):
    
    checkpoint = tf.train.Checkpoint(model=model)
    ckpt_path = get_ckpt_path(ckpt_dirpath, ckpt_name)
    status = checkpoint.restore(ckpt_path)#.expect_partial()
    #status.assert_consumed()
    return model, ckpt_path

def get_ckpt_path(ckpt_dirpath: str, ckpt_name: str):

    ckpt_filename = ckpt_name + '.index'
    ckpt_path_tmpl = os.path.join(ckpt_dirpath, ckpt_filename)
    ckpt_paths = sorted(glob.glob(ckpt_path_tmpl))
    ckpt_path,_ = os.path.splitext(ckpt_paths[-1]) # get the filename of the latest found checkpoint
    
    return ckpt_path

# get multiple inputs for the model
def multi_input(multi_data):
    
    data_l0 = multi_data
    data_l2 = data_l0[:, ::2, ::2, :]
    data_l4 = data_l0[:, ::4, ::4, :]
    
    return [data_l0, data_l2, data_l4]

    