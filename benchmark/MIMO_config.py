import argparse

parser = argparse.ArgumentParser()

# Directories
parser.add_argument('--model_name', default='MIMO-UNet', choices=['MIMO-UNet', 'MIMO-UNetPlus'], type=str)
parser.add_argument('--data_dir', type=str, default='/bigdata/casus/MLID/RuiLi/Data/LM/deepNuclei/imn_bio_v2/')
parser.add_argument('--mode', default='train', choices=['train', 'test'], type=str)

# Train
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--learning_rate', type=float, default=1e-4)
parser.add_argument('--weight_decay', type=float, default=0)
parser.add_argument('--num_epoch', type=int, default=3000)
parser.add_argument('--print_freq', type=int, default=100)
parser.add_argument('--num_worker', type=int, default=8)
parser.add_argument('--save_freq', type=int, default=100)
parser.add_argument('--valid_freq', type=int, default=100) 
parser.add_argument('--resume', type=str, default='')
parser.add_argument('--gamma', type=float, default=0.5)
parser.add_argument('--lr_steps', type=list, default=[(x+1) * 500 for x in range(3000//500)])

# Test
parser.add_argument('--test_model', type=str, default='weights/Best.pkl') #MIMO-UNet.pkl
parser.add_argument('--save_image', type=bool, default=False, choices=[True, False])

# save
parser.add_argument('--model_save_dir', type=str, default='./weights/')
parser.add_argument('--result_dir', type=str, default='./results/')

# args = parser.parse_args()
# args.model_save_dir = os.path.join('results/', args.model_name, 'weights/')
# args.result_dir = os.path.join('results/', args.model_name, 'result_image/')