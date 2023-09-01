import torch

import copy

from . import measure
from ..p_utils import get_layer_metric_array, sum_arr

@measure('grad_norm', bn=True, mode='param')
def get_grad_norm_arr(net, device, inputs, targets, mode, loss_fn, split_data=1, skip_grad=False):
    net.zero_grad()
    N = inputs.shape[0]
    for sp in range(split_data):
        st=sp*N//split_data
        en=(sp+1)*N//split_data

        outputs = net.forward(inputs[st:en])
        loss = loss_fn(outputs, targets[st:en])
        loss.backward()

    grad_norm_arr = get_layer_metric_array(net, lambda l: (l.weight.grad/split_data).norm() if l.weight.grad is not None else None, mode=mode)

    return sum_arr(grad_norm_arr)
