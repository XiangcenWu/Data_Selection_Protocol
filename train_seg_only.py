import torch


from monai.networks.nets.swin_unetr import SwinUNETR
from monai.networks.nets import UNet
import glob
import torch
import os
from training_helpers import *

from monai.losses import  DiceLoss

from data_preparation.data_dir_shuffle import read_data_list
from data_preparation.generate_data_loader import create_data_loader

#################################################
import argparse
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
parser = argparse.ArgumentParser()
parser.add_argument('device', type=str, help='device to calculate')
# parser.add_argument('num_patch', type=int, help='Sequence Length')
parser.add_argument('batch_size', type=int, help='Sequence Length')
# parser.add_argument('class_idx', type=int, help='class index')
# parser.add_argument('num_selection', type=int, help='num of selection (smaller than Sequence Length)')
parser.add_argument('nickname', type=str, help='saved stuff nickname')
args = parser.parse_args()


device = args.device
##############################

data_list = read_data_list('/home/xiangcen/Selecting_performance-representative_Validation/data/data_dir_pancreas_h5.txt')
train_list, test_list = data_list[:50], data_list[50:]


seg_loader = create_data_loader(train_list, batch_size=args.batch_size)
test_loader = create_data_loader(test_list, batch_size=args.batch_size)


#############################
seg_loss_function = DiceLoss(sigmoid=True)


Seg_model  =SwinUNETR((96, 96, 64), 1, 1).to(device)


##############################
seg_optimizer = torch.optim.AdamW(Seg_model.parameters(), lr=1e-4)



seg, test= [], []
for e in range(100):
    print("This is epoch: ", e)
    train_loss = train_seg_net_h5(Seg_model, seg_loader, seg_optimizer, seg_loss_function, device)

    test_loss = test_seg_net_h5(Seg_model, test_loader, device)
    print(train_loss, test_loss)

    seg.append(train_loss)
    test.append(test_loss)
    seg_t = torch.tensor(seg)
    test_t = torch.tensor(test)
    torch.save(seg_t, './results/seg_results/'+args.nickname+'.t')
    torch.save(test_t, './results/seg_results/test_'+args.nickname+'.t')


    torch.save(Seg_model.state_dict(), './models/seg_model_only_'+args.nickname+'.ptm')
    print('model saved!')

