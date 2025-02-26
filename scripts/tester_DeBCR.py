import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import argparse
import glob
import sys
import matplotlib.pyplot as plt
from natsort import natsorted

from debcr.utils import utils
from debcr.model import DeBCR
from debcr.utils import metrics
from debcr.utils import data
from debcr.utils import whole_img_tester

# argparse formaters for argument help:
# - argparse.ArgumentDefaultsHelpFormatter - to show default value
# - argparse.RawTextHelpFormatter - to show multi-line help 
class MyFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawTextHelpFormatter):
    pass

def get_test_parser():
    parser = argparse.ArgumentParser(description=(
        "DeBCR: DL-based denoising, deconvolution and deblurring for light microscopy data.\n"
        "This is the subtool to predict corrected data from corrupted one using trained DeBCR model."
    ), formatter_class=MyFormatter)
    
    parser.add_argument("--task_type", type=str, default='2D_denoising', choices=['2D_denoising','3D_denoising','bright_SR','confocal_SR'],
                        help="Task type to perform according to data nature.")
    parser.add_argument("--weights_path", type=str, default="./weights/TASK_TYPE/",
                        help="Path to the folder containing weights (checkpoints) of the trained DeBCR model.")
    parser.add_argument("--ckpt_name", type=str, default="ckpt-*",
                        help=("Filename (w/o file extension) of the checkpoint of choice (can be a wildcard as well).\n"
                        "If not provided, the latest (by sorted filename) checkpoint file will be used."))
    parser.add_argument("--input_path", type=str, default=None, help="Path to a single NPZ file or a folder with multiple NPZ files to process, should contain two subsets: \"low\" and \"gt\".")
    parser.add_argument("--save_fig", action="store_true", default=False, help="Flag to enable saving figures of the example test results.")
    parser.add_argument("--fig_path", type=str, default='./figures/', help="Path to save figures of the example test results.")
    parser.add_argument("--results_path", type=str, default='./results/', help="Path to save the test results.")
    parser.add_argument("--predict_3d", action="store_true", default=False, help="Flag to enable volumetric (3D) prediction.")
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID to be used.")
    return parser

def test(args):
    # lazy imports to offload showing help
    import numpy as np
    import tensorflow as tf
    import keras
    
    print('\ntask:', args.task_type)

    os.environ['CUDA_VISIBLE_DEVICES']=str(args.gpu_id)
    print('GPU(s): ', tf.config.list_physical_devices("GPU"))
    
    #####
    #config the model, load the weights
    weights_path = args.weights_path
    if "TASK_TYPE" in args.weights_path:
        weights_path = weights_path.replace("TASK_TYPE", args.task_type)
    print("weights: " + weights_path)
    
    ckpt_filename = args.ckpt_name + '.index'
    ckpt_path_tmpl = os.path.join(weights_path, ckpt_filename)
    ckpt_paths = sorted(glob.glob(ckpt_path_tmpl))
    ckpt_path,_ = os.path.splitext(ckpt_paths[-1]) # get the filename of the latest found checkpoint
    print("checkpoint: {}".format(os.path.basename(ckpt_path)))
    
    eval_model = model_DeBCR()
    eval_model.compile(optimizer='adam', loss=loss_function_mimo, metrics=[metrics_func_mimo]) 

    # reload the check point
    checkpoint = tf.train.Checkpoint(model=eval_model)
    checkpoint_manager = tf.train.CheckpointManager(checkpoint, weights_path, max_to_keep=5)

    # Restore the chosen checkpoint for testing
    status = checkpoint.restore(ckpt_path)
    #status.assert_consumed()
    
    ### Test dataset
    # Input directory for testing data
    print(f'Input path: {args.input_path}')
    if not os.path.exists(args.input_path):
        raise FileNotFoundError('No such file or directory!')    
    else:
        if os.path.isfile(args.input_path):    
            input_list = [args.input_path]
        elif os.path.isdir(args.input_path):
            input_list = [args.input_path + os.sep + filename for filename in natsorted(os.listdir(args.input_path))]

    results_path = args.results_path
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    
    if not args.predict_3d and args.task_type in ['2D_denoising', '3D_denoising', 'bright_SR', 'confocal_SR']:

        for idx in range(len(input_list)):
            input_filepath = input_list[idx]
            test_raw = np.load(input_filepath)
            w_test_img, o_test_img = test_raw['low'], test_raw['gt']
            w_test_img, o_test_img = np.expand_dims(w_test_img, axis=3), np.expand_dims(o_test_img, axis=3)
            w_test_img, o_test_img = rescale(w_test_img), rescale(o_test_img)
            
            test_w_list, test_o_list = multi_input(w_test_img, o_test_img)
            pred_test_list = eval_model.predict(test_w_list) 
    
            # evaluation by the metrics
            pred_test_norm, test_o_norm = rescale(pred_test_list[0]), rescale(test_o_list[0])
            psnr_value, ssim_value, rmse = metrics(pred_test_norm, test_o_norm)
            print('Test performance(PSNR, SSIM, RMSE)', psnr_value.round(2), ssim_value.round(2), rmse.round(2))

            pred_test_arr = np.asarray(pred_test_list[0]) 
            pred_test_arr = np.squeeze(pred_test_arr)

            file_suffix = f'_{idx}' if idx > 0 else ''
            np.savez(results_path + os.sep + f'results{file_suffix}.npz', pred=pred_test_arr)
            
            # save the results as fig
            if args.save_fig:
                # save a random fig under path
                fig_path = args.fig_path
            
                if not os.path.exists(fig_path):
                    os.makedirs(fig_path)
            
                eval_results = [psnr_value, ssim_value, rmse]
                save_grid(pred_test_list[0], test_w_list[0], test_o_list[0], fig_path, f'{args.task_type}{file_suffix}', eval_results)
                print('Example results figure saved at', fig_path)
    
    elif args.predict_3d and args.task_type == '3D_denoising':
        
        print('Predict volume was requested.')

        for idx in range(len(input_list)):
            input_filepath = input_list[idx]
            test_raw = np.load(testset_filepath)
            w_test_img, o_test_img = test_raw['low'], test_raw['gt'] # (45, 1024, 512)
        
            patch_predictor = PatchPred_3D_denoiser(eval_model)
            # pred_result = patch_predictor.pred_one_patch(w_test_img, o_test_img) 
            
            stacked_pred_result = patch_predictor.pred_stack_patch(w_test_img, o_test_img) # whole xz
            file_suffix = f'_{idx}' if idx > 0 else ''
            np.savez(results_path + os.sep + f'results{file_suffix}.npz', pred=stacked_pred_result)
            
            # save the results as fig
            if args.save_fig:
                # save the xz
                NUM = 511
                re_temp_pred, temp_w, temp_o = stacked_pred_result[:, NUM, :], w_test_img[:, NUM, :], o_test_img[:, NUM, :]
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                axes[0].imshow(re_temp_pred, cmap='inferno')
                axes[0].set_title('Predicted')
                axes[0].axis('off')
                axes[1].imshow(temp_w, cmap='inferno')
                axes[1].set_title('Input')
                axes[1].axis('off')
                axes[2].imshow(temp_o, cmap='inferno')
                axes[2].set_title('Ground Truth')
                axes[2].axis('off')
        
                plt.savefig(args.fig_path + os.sep + f'{args.task_type}{file_suffix}_yz.svg', format='svg')
                plt.close(fig)
    else:
        print('--predict_3d is not available for {} task type!'.format(args.task_type))

def main():
    parser = get_test_parser()
    
    # print help if no arguments passed
    if len(sys.argv)==1:
        parser.print_help(sys.stderr)
        sys.exit(1)
    
    args = parser.parse_args()
    
    test(args)

if __name__ == "__main__":
    main()