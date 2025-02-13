import numpy as np

# multiple inputs for the model
def multi_input(w_img, o_img):
    
    w_0, o_0 = w_img, o_img
    w_2, o_2 = w_0[:, ::2, ::2, :], o_0[:, ::2, ::2, :]
    w_4, o_4 = w_0[:, ::4, ::4, :], o_0[:, ::4, ::4, :]
    
    return [w_0, w_2, w_4], [o_0, o_2, o_4]

def rescale(image_stack, MIN=0, MAX=1):
    # Rescale the whole stack
    if image_stack[0].max() != 1:
        image_scale = []
        for stack in range(image_stack.shape[0]):
            temp = image_stack[stack, ...]
            temp_scale = np.interp(temp, (temp.min(), temp.max()), (MIN, MAX))
            image_scale.append(temp_scale.astype('float64'))
    else:
        image_scale = image_stack
    return np.asarray(image_scale)