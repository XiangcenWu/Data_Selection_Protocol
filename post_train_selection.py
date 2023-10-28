import torch


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



device = 'cpu'
##############################

data_list = read_data_list('./data_preparation/data_dir_shuffled.txt')

support_set_list, holdout_test_list = data_list[:529], data_list[529:]
support_set_list, query_set_list = support_set_list[:189], support_set_list[189:]
print(len(support_set_list), len(query_set_list), len(holdout_test_list))

# random.shuffle(holdout_test_list)


Sel_model = SelectionUNet((96, 96, 64), 4096, encoder_drop = 0, transformer_drop=0)
Sel_model.load_state_dict(torch.load('/home/xiangcen/xiaoyao/prostate_training/train_sel_results/sel_model_only_prostate_v4.ptm', map_location=device))
Sel_model.to(device)


Seg_model = SwinUNETR((96, 96, 96), 1, 1)
Seg_model.load_state_dict(torch.load('/home/xiangcen/xiaoyao/prostate_training/train_seg_results/seg_model_only_prostate.ptm', map_location=device))
Seg_model.to(device)



holdout_test_true_accuracies = inference_all_data_accuracy(holdout_test_list, Seg_model, 1, device)
#####
Seg_model.eval()
Sel_model.eval()


selected_val = []
for i in range(10):
    sample = holdout_test_list[0+i*6:6+i*6]
    loader = create_data_loader(sample, 6, shuffle=False)
    batch = next(iter(loader))
    img, label = batch['image'].to(device), batch['label'].to(device)
    with torch.no_grad():
        # output = Seg_model(img)
        # accuracy = dice_metric(output, label, post=True, mean=False)
        sel_output = Sel_model(torch.cat([img, label], dim=1))
        indexes_top = torch.topk(sel_output, 1, dim=0).indices.flatten().tolist()

        
        

        for i in indexes_top:
            selected_val.append(sample[i])




selected_true_accuracy = inference_all_data_accuracy(selected_val, Seg_model, 1, device)
######
random_selected_val = random.sample(holdout_test_list, 6)
r_selected_true_accuracy = inference_all_data_accuracy(random_selected_val, Seg_model, 1, device)


selected_train_list = [i for i in holdout_test_list if i not in selected_val]
random_train_list = [i for i in holdout_test_list if i not in random_selected_val]



print(holdout_test_true_accuracies.mean().item(), selected_true_accuracy.mean().item(), r_selected_true_accuracy.mean().item())
print(holdout_test_true_accuracies.std().item(), selected_true_accuracy.std().item(),  r_selected_true_accuracy.std().item())

print(scipy.stats.ttest_ind(holdout_test_true_accuracies, selected_true_accuracy))
print(scipy.stats.ttest_ind(holdout_test_true_accuracies, r_selected_true_accuracy))

















# adaption




# selected_train_list = [i for i in holdout_test_list if i not in selected_val]
# random_train_list = [i for i in holdout_test_list if i not in random_selected_val]
# adaption_batch = 2

# # create the loader for both random and select
# selected_loader = create_data_loader(selected_train_list, adaption_batch)
# r_selected_loader = create_data_loader(random_train_list, adaption_batch)

# # create loss function for both s and r_s
# seg_loss_function = DiceLoss(sigmoid=True)




# # adapt for select
# Seg_model = SwinUNETR((96, 96, 96), 1, 1)
# Seg_model.load_state_dict(torch.load('/home/xiangcen/xiaoyao/prostate_training/train_seg_results/seg_model_only_prostate.ptm', map_location=device))
# Seg_model.to(device)

# selected_optimizer = torch.optim.Adam(Seg_model.parameters(), lr=1e-5)

# for e in range(30):
#     # print("This is epoch: ", e)
#     train_loss = train_seg_net_h5(Seg_model, selected_loader, selected_optimizer, seg_loss_function, device)
#     # print(train_loss)


# selected_val_scores = inference_all_data_accuracy(selected_val, Seg_model, 1, device)
# # query_set_list_scores
# query_set_list_scores = inference_all_data_accuracy(query_set_list, Seg_model, 1, device)

# print("This is the result for selected train (with adaption)")
# print('mean: ', query_set_list_scores.mean().item(), selected_val_scores.mean().item())
# print('std: ', query_set_list_scores.std().item(), selected_val_scores.std().item())
# print('ttest: ', scipy.stats.ttest_ind(query_set_list_scores, selected_val_scores))




# # adapt for random_select
# Seg_model = SwinUNETR((96, 96, 96), 1, 1)
# Seg_model.load_state_dict(torch.load('/home/xiangcen/xiaoyao/prostate_training/train_seg_results/seg_model_only_prostate.ptm', map_location=device))
# Seg_model.to(device)

# selected_optimizer = torch.optim.Adam(Seg_model.parameters(), lr=1e-6)

# for e in range(30):
#     # print("This is epoch: ", e)
#     train_loss = train_seg_net_h5(Seg_model, r_selected_loader, selected_optimizer, seg_loss_function, device)
#     # print(train_loss)


# r_selected_val_scores = inference_all_data_accuracy(random_selected_val, Seg_model, 1, device)
# # query_set_list_scores
# query_set_list_scores = inference_all_data_accuracy(query_set_list, Seg_model, 1, device)

# print("This is the result for random selection (with adaption)")
# print('mean: ', query_set_list_scores.mean().item(), r_selected_val_scores.mean().item())
# print('std: ', query_set_list_scores.std().item(), r_selected_val_scores.std().item())
# print('ttest: ', scipy.stats.ttest_ind(query_set_list_scores, r_selected_val_scores))
