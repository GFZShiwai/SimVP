import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

import argparse
from exp import Exp



# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

# os.environ["CUDA_VISIBLE_DEVICES"] = "2, 3"

import warnings
warnings.filterwarnings('ignore')

def create_parser():
    parser = argparse.ArgumentParser()
    # Set-up parameters
    parser.add_argument('--device', default='cuda', type=str, help='Name of device to use for tensor computations (cuda/cpu)')
    parser.add_argument('--res_dir', default='./results_original', type=str)
    parser.add_argument('--ex_name', default='Debug', type=str)
    parser.add_argument('--use_gpu', default=True, type=bool)
    parser.add_argument('--gpu', default="0,1", type=str)
    parser.add_argument('--seed', default=1, type=int)

    # dataset parameters
    parser.add_argument('--batch_size', default=2, type=int, help='Batch size')
    parser.add_argument('--val_batch_size', default=2, type=int, help='Batch size')
    parser.add_argument('--data_root', default='/workspace/xly-FramePrediction/SimVP/data/')
    parser.add_argument('--dataname', default='uav', choices=['mmnist', 'taxibj', 'uav'])
    parser.add_argument('--num_workers', default=4, type=int)

    # model parameters
    parser.add_argument('--in_shape', default=[10, 3, 64, 64], type=int,nargs='*') # [10, 1, 64, 64] for mmnist, [4, 2, 32, 32] for taxibj  
    parser.add_argument('--hid_S', default=1024, type=int)
    parser.add_argument('--hid_T', default=256, type=int)
    parser.add_argument('--N_S', default=4, type=int)
    parser.add_argument('--N_T', default=8, type=int)
    parser.add_argument('--groups', default=4, type=int)

    # Training parameters
    parser.add_argument('--epochs', default=2000, type=int)
    parser.add_argument('--log_step', default=1, type=int)
    parser.add_argument('--lr', default=0.01, type=float, help='Learning rate')
    return parser




if __name__ == '__main__':
    args = create_parser().parse_args()
    config = args.__dict__
    exp = Exp(args)
    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>  start <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
    # exp.train(args)
    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>> testing <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
    mse = exp.test(args)