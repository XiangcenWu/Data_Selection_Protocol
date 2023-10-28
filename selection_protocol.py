import torch
from data_preparation.generate_data_loader import create_data_loader
from training_helpers import dice_metric
import random
##############################################
# select based on selection net


# 把所有的数据的预测dice值都存到一个list里面
def pred_all_data_accuracy(data_dir, sel_model, sequence_length, device):
    sel_model.eval()
    loader = create_data_loader(data_dir, sequence_length, True, True)
    list = []
    for batch in loader:
        img = batch['image'].to(device)
        with torch.no_grad():
            accuracy = sel_model(img)
            list.append(accuracy.flatten())
    all = torch.stack(list, dim=0).flatten()
    return all


def inference_all_data_accuracy(data_dir, seg_model, sequence_length, device):
    seg_model.eval()
    loader = create_data_loader(data_dir, sequence_length, True, False)
    list = []
    for batch in loader:
        img, label = batch['image'].to(device), batch['label'].to(device)
        with torch.no_grad():
            output = seg_model(img)
            accuracy = dice_metric(output, label, True, False)
            
            list.append(accuracy.flatten())
    all = torch.stack(list, dim=0).flatten()
    return all


# 把传进来的数据dirlist变成dicelist
def pred_list_data_accuracy(data_dir, sel_model, device):
    sel_model.eval()
    loader = create_data_loader(data_dir, len(data_dir), True, False)
    for batch in loader:
        img = batch['image'].to(device)
        with torch.no_grad():
            accuracy = sel_model(img)

    return accuracy.flatten()



def inference_list_data_accuracy(data_dir, seg_model, device):
    seg_model.eval()
    loader = create_data_loader(data_dir, len(data_dir), True, False)
    for batch in loader:
        img, label = batch['image'].to(device), batch['label'].to(device)
        with torch.no_grad():
            output = seg_model(img)
            accuracy = dice_metric(output, label, True, False)

    return accuracy.flatten()



def selection_baseon_selection_net(all_data_dir, sel_model, sequence_length, num_of_sequence_to_select, device):
    l = []
    all_data_accuracy = pred_all_data_accuracy(all_data_dir, sel_model, 1, device)
    for i in range(300):
        data_dir = random.sample(all_data_dir, sequence_length)
        accuracy = pred_list_data_accuracy(data_dir, sel_model, device)
        std = torch.abs(torch.std(accuracy) - torch.std(all_data_accuracy))
        mean = torch.abs(torch.mean(accuracy) - torch.mean(all_data_accuracy))
        l.append({'data_dir':data_dir, 'std':std.item(), 'mean': mean.item()})
    newlist = sorted(l, key = lambda d: d['mean'])
    top_mean = newlist[:20]
    newlist = sorted(top_mean, key = lambda d: d['std'])
    elements = newlist[:num_of_sequence_to_select]
    newlist =[]
    for item in elements:
        newlist.append(item['data_dir'])
    return [x  for sublist in newlist for x in sublist]
######################################################



########################################
# validate the selection by segmentation net
def calculate_mean_std(data_dir, seg_model, sequence_length, device):
    loader = create_data_loader(data_dir, sequence_length, True, True)
    l = []
    for batch in loader:
        img, label = batch['image'].to(device), batch['label'].to(device)
        with torch.no_grad():
            output = seg_model(img)
            true_accuracy = dice_metric(output, label, post=True, mean=False)
            l.append(true_accuracy)
    result = torch.stack(l).flatten()
    return torch.mean(result), torch.std(result)



