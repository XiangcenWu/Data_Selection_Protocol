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

device = 'cuda:0'
nick_name = 'pretrain_msd'


train_dir = []
for i in os.listdir('/home/xiangcen/xiaoyao/prostate_training/data_prostate_msd_h5'):
    train_dir.append('/home/xiangcen/xiaoyao/prostate_training/data_prostate_msd_h5/'+i)
    

print(train_dir)

seg_loader = create_data_loader(train_dir, batch_size=3)



#############################
seg_loss_function = DiceLoss(sigmoid=True)


Seg_model = model  =SwinUNETR((96, 96, 64), 1, 1).to(device)


##############################
seg_optimizer = torch.optim.AdamW(Seg_model.parameters(), lr=1e-3)




for e in range(30):
    print("This is epoch: ", e)
    train_loss = train_seg_net_h5(Seg_model, seg_loader, seg_optimizer, seg_loss_function, device)

    print(train_loss)

    


    torch.save(Seg_model.state_dict(), './train_seg_results/seg_model_only_'+nick_name+'.ptm')
    print('model saved!')


