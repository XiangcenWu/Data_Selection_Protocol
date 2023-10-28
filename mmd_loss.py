import torch
from torch import nn
from selection_protocol import inference_all_data_accuracy

class RBF(nn.Module):

    def __init__(self, n_kernels=5, mul_factor=2.0, bandwidth=None, device='cpu'):
        super().__init__()
        self.bandwidth_multipliers = mul_factor ** (torch.arange(n_kernels, device=device) - n_kernels // 2)
        self.bandwidth = bandwidth

    def get_bandwidth(self, L2_distances):
        if self.bandwidth is None:
            n_samples = L2_distances.shape[0]
            return L2_distances.data.sum() / (n_samples ** 2 - n_samples)

        return self.bandwidth

    def forward(self, X):
        L2_distances = torch.cdist(X, X) ** 2
        return torch.exp(-L2_distances[None, ...] / (self.get_bandwidth(L2_distances) * self.bandwidth_multipliers)[:, None, None]).sum(dim=0)


class MMDLoss(nn.Module):

    def __init__(self, kernel=RBF()):
        super().__init__()
        self.kernel = kernel

    def forward(self, X, Y):
        K = self.kernel(torch.vstack([X, Y]))

        X_size = X.shape[0]
        XX = K[:X_size, :X_size].mean()
        XY = K[:X_size, X_size:].mean()
        YY = K[X_size:, X_size:].mean()
        return XX - 2 * XY + YY

# mmd = MMDLoss(kernel=RBF(device='cuda:0'))
sm = torch.nn.Softmax(dim=0)





class Select_Max(torch.autograd.Function):
    # Simple case where everything goes well
    @staticmethod
    def forward(ctx, x):
        # This time we save the output
        top_k_indices = torch.topk(x, 1, dim=0).indices.flatten().tolist()
        result = torch.zeros_like(x, dtype=torch.float)
        for i in top_k_indices:
            result[i, 0] = 1.
        ctx.save_for_backward(result)
        return result

    @staticmethod
    def backward(ctx, grad_out):
        result, = ctx.saved_tensors
        return result * grad_out
    


def custom_mmd_loss(scores, true_accuracy, number_choice, true_performance):
    # (8, 1) should be input size
    
    # select the top k indices
    top_k_indices = torch.topk(scores, number_choice, dim=0).indices.flatten().tolist()
    # transfer the model output scores into 0 and 1 vectors
    activation_vector = Select_Max.apply(scores)
    
    selected_true_performance = activation_vector * true_accuracy
    selected_true_performance = selected_true_performance[top_k_indices]
    
    return torch.abs(selected_true_performance.mean() - true_performance.mean())




def custom_wa_difference_loss(scores, true_accuracy, number_choice, true_mean, true_std):
    
    # top_k_indices = torch.topk(scores, number_choice, dim=0).indices.flatten().tolist()

    tad = torch.abs(true_accuracy - true_mean)
    
    return (scores * tad).mean() 



def weighted_std(scores, true_accuracy, true_mean):
    difference_sqrt = (true_accuracy - true_mean)**2
    return torch.sqrt(    (scores * difference_sqrt).sum() / scores.sum()     )
    


def custom_wa_difference_loss_std(scores, true_accuracy, number_choice, true_mean, true_std):
    
    
    
    
    tad = torch.abs(true_accuracy - true_mean)
    
    
    mean_loss =  (scores * tad).sum() / scores.sum()
    std_loss = torch.abs( weighted_std(scores, true_accuracy, true_mean) - true_std  )
    return mean_loss + 10*std_loss

