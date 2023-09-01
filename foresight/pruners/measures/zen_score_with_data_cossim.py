import torch, math
import torch.nn.functional as F

from . import measure
from ..p_utils import AvgrageMeter

@measure('zen_score_with_data_cossim', bn=True, mode='param')
def compute_zen_score_with_data_cossim(net, device, inputs, targets, mode, loss_fn, split_data=1, skip_grad=False):
    alpha = 0.01
    zen_score_with_data_cossim = AvgrageMeter()
    net.zero_grad()
    N = inputs.shape[0]
    with torch.no_grad():
        for sp in range(split_data):
            st=sp*N//split_data
            en=(sp+1)*N//split_data

            epsilon_ = torch.zeros_like(inputs[st:en]).normal_()
            out1 = net.forward(inputs[st:en], pre_GAP=True)
            out2 = net.forward(inputs[st:en] + alpha*epsilon_, pre_GAP=True)
            cos_sim = F.cosine_similarity(out1, out2).abs_()
            zen_score_with_data_cossim.update(torch.sum(cos_sim).item(), en - st)

    return zen_score_with_data_cossim.avg
