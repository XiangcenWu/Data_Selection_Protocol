import torch


from monai.networks.nets.swin_unetr import SwinUNETR
import torch
from training_helpers import *


from monai.losses import DiceLoss
from Selection_Model import *
# from mmd_loss import MMD

from data_preparation.data_dir_shuffle import read_data_list
from data_preparation.generate_data_loader import create_data_loader



from selection_protocol import inference_all_data_accuracy
from mmd_loss import *
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
parser.add_argument('sequence_length', type=int, help='Sequence Length')
# parser.add_argument('class_idx', type=int, help='class index')
# parser.add_argument('num_selection', type=int, help='num of selection (smaller than Sequence Length)')
parser.add_argument('nickname', type=str, help='saved stuff nickname')
args = parser.parse_args()


device = args.device




##############################

data_list = read_data_list('/home/xiangcen/xiaoyao/prostate_training/data_preparation/data_dir_shuffled_obturator.txt')

support_set_list, holdout_test_list = data_list[:469], data_list[469:]
support_set_list, query_set_list = support_set_list[:189], support_set_list[189:]
print(len(support_set_list), len(query_set_list), len(holdout_test_list))
test_loader = create_data_loader(holdout_test_list, batch_size=args.sequence_length, shuffle=True, drop_last=True)
selection_loader = create_data_loader(query_set_list, batch_size=args.sequence_length, shuffle=True, drop_last=True)


#############################
seg_loss_function = DiceLoss(sigmoid=True)


Seg_model = SwinUNETR((96, 96, 64), 1, 1)
Seg_model.load_state_dict(torch.load('/home/xiangcen/xiaoyao/prostate_training/train_seg_results/seg_model_only_obturator.ptm', map_location=args.device))
Seg_model.to(device)

Sel_model = SelectionUNet((96, 96, 64), 4096, encoder_drop = 0, transformer_drop=0)
# Sel_model.load_state_dict(torch.load('/home/xiangcen/xiaoyao/prostate_training/train_sel_results/sel_model_only_prostate_v3_yphsd.ptm', map_location=device))
Sel_model.to(device)
##############################
sel_optimizer = torch.optim.AdamW(Sel_model.parameters(), lr=1e-6)


query_set_true_performance = inference_all_data_accuracy(query_set_list, Seg_model, 1, device).view(-1, 1)
query_set_true_mean = query_set_true_performance.mean()
query_set_true_std = query_set_true_performance.std()


train, test = [], []
for e in range(200):
    print("This is epoch: ", e)
    train_sel_loss = train_sel_net_h5(
        sel_model=Sel_model,
        sel_loader=selection_loader,
        sel_optimizer=sel_optimizer,
        query_set_true_mean = query_set_true_mean,
        query_set_true_std = query_set_true_std,
        sel_loss_function=custom_wa_difference_loss_std,
        seg_model=Seg_model,
        device=args.device,
    )
    test_sel_loss = test_sel_net_h5(
        sel_model=Sel_model, 
        sel_loader=test_loader,
        query_set_true_mean = query_set_true_mean,
        query_set_true_std = query_set_true_std,
        sel_loss_function=custom_wa_difference_loss_std,
        seg_model=Seg_model,
        device=args.device,
    )
    
    train.append(train_sel_loss)
    test.append(test_sel_loss)
    
    sel_t = torch.tensor(train)
    test_t = torch.tensor(test)
    torch.save(sel_t, './train_sel_results/sel_'+args.nickname+'.t')
    torch.save(test_t, './train_sel_results/test_'+args.nickname+'.t')


    torch.save(Sel_model.state_dict(), './train_sel_results/sel_model_only_'+args.nickname+'.ptm')
    print('model saved!')
