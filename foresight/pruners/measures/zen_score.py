import torch, math

from . import measure
from ..p_utils import AvgrageMeter

@measure('zen_score', bn=True, mode='param')
def compute_zen_score(net, device, inputs, targets, mode, loss_fn, split_data=1, skip_grad=False):
    alpha = 0.01
    p = 1
    eps = 1e-6
    zen_score = AvgrageMeter()
    log_mean_std_pre_layer = []
    net.zero_grad()
    N = inputs.shape[0]
    with torch.no_grad():
        for sp in range(split_data):
            st=sp*N//split_data
            en=(sp+1)*N//split_data

            epsilon = torch.zeros_like(inputs[st:en]).normal_()
            epsilon_ = torch.zeros_like(inputs[st:en]).normal_()
            zen_score.update((net.forward(epsilon + alpha*epsilon_, pre_GAP=True) - net.forward(epsilon, pre_GAP=True)).norm(p=p).detach().cpu().item(), en - st)

    for layer in net.modules():
        if isinstance(layer, torch.nn.BatchNorm2d):
            log_mean_std_pre_layer.append(math.log(layer.running_var.mean().sqrt().detach().cpu().item() + eps))

    return math.log(zen_score.avg + eps) + sum(log_mean_std_pre_layer)
