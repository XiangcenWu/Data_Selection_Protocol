import torch
from monai.transforms import AsDiscrete
from monai.metrics import DiceMetric
from data_preparation.data_dir_shuffle import read_data_list





th = AsDiscrete(threshold=0.5)
def post_process(output):
    output = torch.sigmoid(output)
    return th(output)
dm = DiceMetric(reduction='mean')
def dice_metric(y_pred, y_true, post=False, mean=False):
    """Calculate the dice score (accuracy) give the prediction from a trained segmentation network
    

    Args:
        y_pred (torch.tensor): output from the segmentation network 
        y_true (torch.tensor): ground truth of the output
        post (bool, optional): whether to do the post process (threshold, simoid etc...) 
                               before calculate the dice. Defaults to False.
        mean (bool, optional): calculate the mean of a batch of dice score. Defaults to False.

    Returns:
        torch.tensor: dice score of a single number or a batch of numbers
    """
    if post:
        y_pred = post_process(y_pred)


    return dm(y_pred, y_true).mean() if mean else dm(y_pred, y_true)


def train_seg_net_h5(
        seg_model, 
        seg_loader,
        seg_optimizer,
        seg_loss_function,
        device='cpu',
    ):
    # remember the loader should drop the last batch to prevent differenct sequence number in the last batch
    seg_model.train()
    
    step = 0.
    loss_a = 0.
    for batch in seg_loader:

        img, label = batch["image"].to(device), batch["label"].to(device)
        # forward pass and calculate the selection


        # forward pass of selected data
        output = seg_model(img)
        
        loss = seg_loss_function(output, label)

        

        loss.backward()
        seg_optimizer.step()
        seg_optimizer.zero_grad()

        loss_a += loss.item()
        step += 1.
    loss_of_this_epoch = loss_a / step

    return loss_of_this_epoch


def test_seg_net_h5(seg_model, test_loader, device):


    seg_model.eval()
    performance_a = 0.
    step = 0.

    for batch in test_loader:

        img, label = batch["image"].to(device), batch["label"].to(device)


        with torch.no_grad():
            seg_output = seg_model(img)
            # print(seg_output.shape, seg_output[0, 0, 100, 100, :10])
            # seg_output = post_process(seg_output)
            # print(seg_output.shape, seg_output[0, 0, 100, 100, :10])
            performance = dice_metric(seg_output, label, post=True, mean=True)
            # print(performance)


            performance_a += performance.item()
            step += 1.

    performance_of_this_epoch = performance_a / step

    return performance_of_this_epoch




def train_sel_net_h5(
        sel_model,
        sel_loader,
        sel_optimizer,
        query_set_true_mean,
        query_set_true_std,
        sel_loss_function,
        seg_model,
        device='cpu',
    ):
    seg_model.eval()
    sel_model.train()
    
    
    step_e = 0.
    loss_e = 0.
    for batch in sel_loader:
        img, label = batch["image"].to(device), batch["label"].to(device)
        
        with torch.no_grad():
            output = seg_model(img)
            accuracy = dice_metric(output, label, post=True, mean=False)
            # print(accuracy)
            
        sel_output = sel_model(torch.cat([img, label], dim=1))
        
        # print('true mean: ', query_set_true_mean)
        # print('train true accuracy', accuracy)
        
        # print('train sel output', sel_output)
        
        
        loss = sel_loss_function(sel_output, accuracy, 2, query_set_true_mean, query_set_true_std)
        
        
        # print('train loss:', loss)
        
        loss.backward()
        if torch.rand(1).item() > 0.75:
            sel_optimizer.step()
            sel_optimizer.zero_grad()
            
        
        
        
        
        loss_e += loss.item()
        step_e += 1
        
        
    # print(sel_output)
    # print(accuracy)
    return loss_e / step_e
        
        
def test_sel_net_h5(
        sel_model, 
        sel_loader,
        query_set_true_mean,
        query_set_true_std,
        sel_loss_function,
        seg_model,
        device='cpu',
    ):
    seg_model.eval()
    sel_model.eval()
    
    
    step_e = 0.
    loss_e = 0.
    for batch in sel_loader:
        img, label = batch["image"].to(device), batch["label"].to(device)
        with torch.no_grad():
            output = seg_model(img)
            accuracy = dice_metric(output, label, post=True, mean=False)
            # print(accuracy)
            
            sel_output = sel_model(torch.cat([img, label], dim=1))
            
            print('test true accuracy', accuracy)
            
            print('test sel output', sel_output)
            
            loss = sel_loss_function(sel_output, accuracy, 2, query_set_true_mean, query_set_true_std)


        loss_e += loss.item()
        step_e += 1
    return loss_e / step_e



# def convert_data_list(original_list, new_data_dir):
#     l = read_data_list(original_list)
#     empty = []

#     for i in l:
#         image_index = i[-9:-4]
#         new_path = os.path.join(new_data_dir, image_index + '.h5')
#         print(new_path)
#         empty.append(new_path)
#     return empty


# if __name__ == "__main__":
#     convert_data_list('/home/xiangcen/xiaoyao/prostate_training/data_preparation/data_dir_shuffled.txt', '/home/xiangcen/xiaoyao/prostate_training/data_prostate_ta_h5')

