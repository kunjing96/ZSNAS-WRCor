import torch, numpy as np

from . import measure
from ..p_utils import mean_arr


def get_batch_jacobian(net, x):
    net.zero_grad()
    x.requires_grad_(True)
    y = net(x)
    y.backward(torch.ones_like(y))
    jacob = (x.grad).abs_()

    return jacob.detach()


@measure('resp_sens', bn=True, mode='param')
def compute_resp_sens(net, device, inputs, targets, mode, loss_fn, split_data=1):
    jacobs  = []
    N = inputs.shape[0]
    for sp in range(split_data):
        net.zero_grad()

        st=sp*N//split_data
        en=(sp+1)*N//split_data

        jacobs_batch = get_batch_jacobian(net, inputs[st:en])
        jacobs.append(jacobs_batch.reshape(jacobs_batch.size(0), -1))

    return mean_arr(jacobs)
