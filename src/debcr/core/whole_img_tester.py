import numpy as np

# func for the whole image pred in 3D denoising
class PatchPred_3D_denoiser:
    def __init__(self, eval_model):
        self.eval_model = eval_model
        
    def multi_input(self, w_img, o_img):

        w_0, o_0 = w_img, o_img
        w_2, o_2 = w_0[:, ::2, ::2, :], o_0[:, ::2, ::2, :]
        w_4, o_4 = w_0[:, ::4, ::4, :], o_0[:, ::4, ::4, :]

        return [w_0, w_2, w_4], [o_0, o_2, o_4]
    
    def patchify(self, image, patch_size):
        patches = []
        height, width = image.shape[:2]
        patch_height, patch_width = patch_size

        for y in range(0, height, patch_height):
            for x in range(0, width, patch_width):
                patch = image[y:y+patch_height, x:x+patch_width]
                patches.append(patch)

        return patches

    def unpatchify(self, patches, image_shape):
        image = np.zeros(image_shape, dtype=patches[0].dtype)
        patch_height, patch_width = patches[0].shape

        num_cols = image_shape[1] // patch_width
        for i, patch in enumerate(patches):
            y = (i // num_cols) * patch_height
            x = (i % num_cols) * patch_width
            image[y:y+patch_height, x:x+patch_width] = patch

        return image
    
        
    def rescale(self, image_stack, MIN=0, MAX=1):
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

    def pred_one_patch(self, w_test_img, o_test_img):
        w_test_img = np.expand_dims(w_test_img, axis=0)
        o_test_img = np.expand_dims(o_test_img, axis=0)
        w_test_img = np.expand_dims(w_test_img, axis=3)
        o_test_img = np.expand_dims(o_test_img, axis=3)
        w_test_img = self.rescale(w_test_img)
        o_test_img = self.rescale(o_test_img)

        test_w_list, test_o_list = self.multi_input(w_test_img, o_test_img)

        pred_test_list = self.eval_model.predict(test_w_list)

        return pred_test_list[0]

    def pred_stack_patch(self, w_test_img, o_test_img):
        pred_stack_ori = []
        for NUM in range(w_test_img.shape[0]):
            temp_w, temp_o = w_test_img[NUM], o_test_img[NUM]
            patch_size = (128, 128)
            patches_w, patches_o = self.patchify(temp_w, patch_size), self.patchify(temp_o, patch_size)

            all_patches_pred = []
            all_mean = []
            for i in range(len(patches_w)):
                temp_patch_w, temp_patch_o = patches_w[i], patches_o[i]
                temp_patch_pred = self.pred_one_patch(temp_patch_w, temp_patch_o)
                temp_patch_pred = temp_patch_pred.squeeze()

                if temp_patch_pred.mean() > 0.15:  # the padding introduces patterns
                    temp_patch_pred = np.zeros_like(temp_patch_pred) 
                all_patches_pred.append(temp_patch_pred)

            re_temp_pred = self.unpatchify(all_patches_pred, temp_w.shape)
            pred_stack_ori.append(re_temp_pred)
        return np.asarray(pred_stack_ori)