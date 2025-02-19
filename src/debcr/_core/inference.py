import numpy as np
from .model.utils import multi_input

def predict_with_model(eval_model, input_data: np.ndarray) -> np.ndarray:

    input_data = np.expand_dims(input_data, axis=-1)
    input_data_list = multi_input(input_data)
    
    pred_test_list = eval_model.predict(input_data_list)

    pred_test_arr = np.asarray(pred_test_list[0]) 
    pred_test_arr = np.squeeze(pred_test_arr)

    return pred_test_arr

    