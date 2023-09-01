import torch

from . import measure
from ..p_utils import get_layer_metric_array, sum_arr


@measure('plain', bn=True, mode='param')
def compute_plain_per_weight(net, device, inputs, targets, mode, loss_fn, split_data=1):

    net.zero_grad()
    N = inputs.shape[0]
    for sp in range(split_data):
        st=sp*N//split_data
        en=(sp+1)*N//split_data

        outputs = net.forward(inputs[st:en])
        loss = loss_fn(outputs, targets[st:en])
        loss.backward()

    # select the gradients that we want to use for search/prune
    def plain(layer):
        if layer.weight.grad is not None:
            return (layer.weight.grad / split_data) * layer.weight
        else:
            return None #torch.zeros_like(layer.weight)

    grads_abs = get_layer_metric_array(net, plain, mode)
    return sum_arr(grads_abs)
