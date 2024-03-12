import h5py
import torch

import os
# import matplotlib.pyplot as plt
import numpy as np
from monai.transforms import Resized, Compose, LoadImaged, Spacingd, EnsureChannelFirstd, Orientationd, ScaleIntensityRanged, CropForegroundd, SpatialCropd, CenterSpatialCropd, SpatialPadd
import nibabel as nib


class Load_File(object):
    """load the file dir dict and convert to actualy file dir"""
    
    def load_file(self, input_dict):
        print(input_dict)
    
        data_dict = {
            'image': torch.tensor(nib.load(input_dict['image']).get_fdata()).unsqueeze(0), 
            'label': convert_label(torch.tensor(nib.load(input_dict['label']).get_fdata())).unsqueeze(0)
        }
        
        return data_dict

    def __call__(self, input_dict):
        return self.load_file(input_dict)


data_reader = Compose(
    [
        Load_File(),
        # EnsureChannelFirstd(keys=["image", "label"]),
        # Orientationd(keys=["image", "label"], axcodes="RAS"),
        # Spacingd(
        #     keys=["image", "label"],
        #     pixdim=(2.0, 2.0, 3.0),
        #     mode=("bilinear", "nearest"),
        # ),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=-125,
            a_max=512,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        Resized(keys=["image", "label"], spatial_size=(128, 128, 64)),
        
        CenterSpatialCropd(keys=["image", "label"], roi_size=(96, 96, 64)),
        # SpatialCropd(keys=["image", "label"], roi_center=(40, 32, 32) , roi_size=(96, 64)),
        # SpatialPadd(keys=['image', 'label'], spatial_size=(96, 96, 96))
    ]
)

# data_reader = Load_File()



def convert_label(input_label):
    input_label_5 = (input_label != 0).float()
    return input_label_5


def convert_h5(img_dir, label_dir, des_dir):
    for img_name in os.listdir(img_dir):
        # print(img_name)
        # make sure all data are CT data
        if img_name.endswith('.nii.gz'):
            image_index = img_name[9:12]
            img_file_dir = os.path.join(img_dir, 'pancreas_' + image_index + '.nii.gz')
            label_file_dir = os.path.join(label_dir, 'pancreas_' + image_index + '.nii.gz')

            dir_dict = {
                'image' : img_file_dir,
                'label' : label_file_dir
            }
            

            
            loaded_dict = data_reader(dir_dict)
            
            with h5py.File(os.path.join(des_dir, image_index + '.h5'), 'w') as hf:
                hf.create_dataset('image', data=loaded_dict['image'])
                hf.create_dataset('label', data=(loaded_dict['label'] > 0.5).float())



convert_h5('/home/xiangcen/Selecting_performance-representative_Validation/data/Task07_Pancreas/imagesTr', 
           '/home/xiangcen/Selecting_performance-representative_Validation/data/Task07_Pancreas/labelsTr',
           '/home/xiangcen/Selecting_performance-representative_Validation/data/pancreas_h5')