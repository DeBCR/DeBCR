import torch
from torchvision.transforms import functional as F
from data import valid_dataloader
from utils import Adder
import os
from skimage.metrics import peak_signal_noise_ratio


def _valid(model, args, ep, steps_per_epoch, input_img, label_img):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model.eval()
    psnr_adder = Adder()
    input_img, label_img = torch.from_numpy(input_img), torch.from_numpy(label_img)
    input_img = input_img.to(device, dtype=torch.float)
    label_img = label_img.to(device, dtype=torch.float) # [None, 3, 128, 128]

    with torch.no_grad():
        print('Start deconv Evaluation')
        for idx in range(steps_per_epoch):
            
            # input_img, label_img = torch.from_numpy(input_img), torch.from_numpy(label_img)
            # input_img = input_img.to(device, dtype=torch.float)
            # label_img = label_img.to(device, dtype=torch.float)
            
            if not os.path.exists(os.path.join(args.result_dir, '%d' % (ep))):
                os.mkdir(os.path.join(args.result_dir, '%d' % (ep)))
    
            # print('validaton形状：', input_img.shape, label_img.shape)  # [None, 3, 128, 128]
            pred = model(input_img)

            pred_clip = torch.clamp(pred[2], 0, 1)
            p_numpy = pred_clip.squeeze(0).cpu().numpy()
            label_numpy = label_img.squeeze(0).cpu().numpy()

            psnr = peak_signal_noise_ratio(p_numpy, label_numpy, data_range=1)

            psnr_adder(psnr)
            print('\r%03d'%idx, end=' ')

    print('\n')
    model.train()
    return psnr_adder.average()
