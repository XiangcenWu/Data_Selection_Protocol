import torch
import sys
import os
# Get the current script's directory
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory by going one level up
parent_dir = os.path.dirname(current_dir)
# Add the parent directory to sys.path
sys.path.append(parent_dir)


from monai.networks.nets.swin_unetr import SwinUNETR
import torch

import os
import scipy
from training_helpers import *


from monai.losses import DiceFocalLoss, DiceLoss
from Selection_Model import *

import random
from monai.losses import DiceLoss

from data_preparation.data_dir_shuffle import read_data_list
from selection_protocol import *



from monai.transforms import (
    Compose
)

import h5py



device = 'cuda:0'
##############################

data_list = read_data_list('./data_preparation/data_dir_shuffled_glad.txt')

support_set_list, holdout_test_list = data_list[:469], data_list[469:]
support_set_list, query_set_list = support_set_list[:189], support_set_list[189:]
print(len(support_set_list), len(query_set_list), len(holdout_test_list))

t = 0.9
for i in range(5):
    random.shuffle(holdout_test_list)


    Sel_model = SelectionUNet((96, 96, 64), 4096, encoder_drop = 0, transformer_drop=0)
    Sel_model.load_state_dict(torch.load('/home/xiangcen/xiaoyao/prostate_training/train_sel_results/sel_model_only_prostate_8_v2_yphsd.ptm', map_location=device))
    Sel_model.to(device)
    Sel_model.eval()

    Seg_model = SwinUNETR((96, 96, 64), 1, 1)
    Seg_model.load_state_dict(torch.load('/home/xiangcen/xiaoyao/prostate_training/train_seg_results/seg_model_only_yphsd.ptm', map_location=device))
    Seg_model.to(device)
    Seg_model.eval()

    parent_set = holdout_test_list[:60]
    cousin_set = [i for i in holdout_test_list if i not in parent_set]


    ########################################
    selected_val_set = []
    for i in range(300):
        sample = random.sample(parent_set, 8)
        loader = create_data_loader(sample, 8, shuffle=False)
        batch = next(iter(loader))
        img, label = batch['image'].to(device), batch['label'].to(device)
        with torch.no_grad():
            # output = Seg_model(img)
            # accuracy = dice_metric(output, label, post=True, mean=False)
            sel_output = Sel_model(torch.cat([img, label], dim=1))
            indexes_top = torch.topk(sel_output, 1, dim=0).indices.flatten().item()
            scores_top = torch.topk(sel_output, 1, dim=0).values.flatten().item()
            if scores_top > t:
                # print('accept')
            
                selected_data = [sample[indexes_top]]
                selected_dice = inference_all_data_accuracy(selected_data, Seg_model, 1, device)
                # print(indexes_top, scores_top, selected_dice)
                
                
                if sample[indexes_top] not in selected_val_set:
                    selected_val_set.append(sample[indexes_top])
    print(len(selected_val_set))
    #############################################
    del Sel_model

    selected_train_list = [i for i in parent_set if i not in selected_val_set]
    print(inference_all_data_accuracy(selected_val_set, Seg_model, 1, device))
    print(inference_all_data_accuracy(parent_set, Seg_model, 1, device).mean().item())
    del Seg_model


    # # train from scratch
    Seg_model = SwinUNETR((96, 96, 96), 1, 1)
    Seg_model.load_state_dict(torch.load('/home/xiangcen/xiaoyao/prostate_training/train_seg_results/seg_model_only_yphsd.ptm', map_location=device))
    Seg_model.to(device)
    selected_optimizer = torch.optim.Adam(Seg_model.parameters(), lr=1e-3)


    train_loader = create_data_loader(selected_train_list, 2, drop_last=True, shuffle=True)
    seg_loss_function = DiceLoss(sigmoid=True)
    for e in range(30):
        # print("This is epoch: ", e)
        train_loss = train_seg_net_h5(Seg_model, train_loader, selected_optimizer, seg_loss_function, device)
        # print(train_loss)



    selected_val_scores = inference_all_data_accuracy(selected_val_set, Seg_model, 1, device)
    # query_set_list_scores
    cousin_set_list_scores = inference_all_data_accuracy(cousin_set, Seg_model, 1, device)

    print("This is the result for selected")
    print('mean: ', selected_val_scores.mean().item(), cousin_set_list_scores.mean().item())
    print('std: ', selected_val_scores.std().item(), cousin_set_list_scores.std().item())




















    ####random
    ###############################
    ###############################
    ###############################
    ###############################
    ###############################
    
    random_val_list = random.sample(parent_set, len(selected_val_set))
    random_train_list = [i for i in parent_set if i not in random_val_list]

    del Seg_model
    Seg_model = SwinUNETR((96, 96, 96), 1, 1)
    Seg_model.load_state_dict(torch.load('/home/xiangcen/xiaoyao/prostate_training/train_seg_results/seg_model_only_yphsd.ptm', map_location=device))
    Seg_model.to(device)
    selected_optimizer = torch.optim.Adam(Seg_model.parameters(), lr=1e-3)


    train_loader = create_data_loader(random_train_list, 2, drop_last=True, shuffle=True)
    seg_loss_function = DiceLoss(sigmoid=True)
    for e in range(30):
        # print("This is epoch: ", e)
        train_loss = train_seg_net_h5(Seg_model, train_loader, selected_optimizer, seg_loss_function, device)
        # print(train_loss)


    # random_val_list = [holdout_test_list[50:56][-2], holdout_test_list[0:6][-2]]  
    # random_train_list = [i for i in parent_set if i not in random_val_list]
    random_val_scores = inference_all_data_accuracy(random_val_list, Seg_model, 1, device)
    # query_set_list_scores
    cousin_set_list_scores = inference_all_data_accuracy(cousin_set, Seg_model, 1, device)

    print("This is the result for random sample")
    print('mean: ', random_val_scores.mean().item(), cousin_set_list_scores.mean().item())
    print('std: ', random_val_scores.std().item(), cousin_set_list_scores.std().item())



    ###############################
    ###############################
    ###############################
    ###############################
    ###############################