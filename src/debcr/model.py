from __future__ import annotations

from ._core import model as _model

def init(weights_path: str = None, input_size: int = 128, ckpt_name: str = "ckpt-*"):
    
    init_model = _model.build_and_compile(input_shape = (input_size, input_size, 1))
    if weights_path is None:
        print('Initialized model - untrained')
        return init_model
    
    import os # delayed os-import
    if os.path.exists(weights_path) and os.path.isdir(weights_path):
        print(f'Weights path: {weights_path}')
        loaded_model, ckpt_path = _model.restore_ckpt(init_model, weights_path, ckpt_name)
        print(f'Checkpoint loaded: {os.path.basename(ckpt_path)}')
        print('Initialized model - trained')
        return loaded_model
    else:
        raise ValueError(f'Non-existing weights path: {weights_path}')

def predict(eval_model, input_data: numpy.ndarray, batch_size: int = 32) -> numpy.ndarray:
    
    print(f'Batch size: {batch_size}')
    return _model.predict_with_model(eval_model, input_data, batch_size)

def train(train_data, val_data, config: dict, init_model = None):
    
    if init_model is None:
        init_model = init()
    
    return _model.train_model(init_model, train_data, val_data, config)