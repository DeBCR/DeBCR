from __future__ import annotations

from ._core import model as _model

def init(weights_path: str = None, input_size: int = 128, ckpt_name: str = "ckpt-*"):
    """
    Initialize a model with optional weights restoration from a checkpoint.

    Builds and compiles a model with the specified input size. If a valid
    weights directory is provided, attempts to restore the latest checkpoint
    matching the given pattern.

    Parameters
    ----------
    weights_path : str, optional
        Path to a directory containing model checkpoints. If None, an untrained
        model is returned.
    input_size : int, optional
        Size of the input image (assumes square input). Default is 128.
    ckpt_name : str, optional
        Glob pattern for checkpoint file names. Default is "ckpt-*".

    Returns
    -------
    tf.keras.Model
        A compiled Keras model, either untrained or loaded from checkpoint.

    Raises
    ------
    ValueError
        If `weights_path` is provided but does not exist or is not a directory.
    """
    
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
    """
    Run model prediction on input data using a specified batch size.

    Uses the provided evaluation model to generate predictions on the input data.

    Parameters
    ----------
    eval_model : tf.keras.Model
        A compiled Keras model used for prediction.
    input_data : numpy.ndarray
        Input data for prediction, typically a 3D array (patch, height, width).
    batch_size : int, optional
        Number of samples per prediction batch. Default is 32.

    Returns
    -------
    numpy.ndarray
        The model's prediction output.
    """
    
    print(f'Batch size: {batch_size}')
    return _model.predict_with_model(eval_model, input_data, batch_size)

def train(train_data, val_data, config: dict, init_model = None):
    """
    Train a model using the provided training and validation data.

    If no initial model is provided, a new model is initialized. Training is 
    configured according to the parameters specified in the config dictionary.

    Parameters
    ----------
    train_data : tuple of numpy.ndarray
        Tuple `(low, gt)` where:
        - `low` is a NumPy array of pre-processed input data to restore.
        - `gt` is a NumPy array of corresponding pre-processed ground truth data.
    val_data : tuple of numpy.ndarray
        Tuple `(low, gt)` in the same format as `train_data`, used for validation.
    config : dict
        Configuration dictionary containing training parameters such as batch_size, learning_rate, etc.
    init_model : tf.keras.Model, optional
        A pre-initialized Keras model. If None, a new model is created using default settings.

    Returns
    -------
    tf.keras.Model
        A trained Keras model to be used for prediction.
    """
    
    if init_model is None:
        init_model = init()
    
    return _model.train_model(init_model, train_data, val_data, config)