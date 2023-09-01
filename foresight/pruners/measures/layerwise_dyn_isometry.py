import torch

from . import measure
from ..p_utils import get_layer_metric_array, sum_arr


@measure('layerwise_dyn_isometry', copy_net=False, mode='param')
def get_layerwise_dyn_isometry(net, device, inputs, targets, mode, split_data=1, loss_fn=None):

    def func(layer):
        v = torch.reshape(layer.weight, (layer.weight.size(0), -1))
        isometry_penalty = torch.norm(torch.mm(v, v.t()) - torch.eye(v.size(0)), p=2)
        return -isometry_penalty

    layerwise_dyn_isometry = get_layer_metric_array(net, func, mode=mode)

    return sum_arr(layerwise_dyn_isometry)

'''
import torch, numpy as np

from . import measure


def eval_score(modules):
    s_layerwise = []
    for module in modules:
        jacobs = torch.cat(module.jacobs, 0)
        _, s, _ = torch.svd(jacobs)
        s_layerwise.append(s.cpu().numpy())
    s_batch = np.concatenate(s_layerwise, axis=-1)
    v = np.prod(s_batch, axis=-1)
    k = 1e-5
    score = np.mean(np.log(v + k) + 1./(v + k))
    return score


@measure('layerwise_dyn_isometry', bn=True, mode='param')
def compute_layerwise_dyn_isometry(net, device, inputs, targets, mode, loss_fn, split_data=1):

    def jacob_forward_hook(module, inp, out):
        if not hasattr(module, 'jacobs'): module.jacobs = []
        try:
            jacob = torch.autograd.functional.jacobian(lambda inp: out, inp)
            jacob = jacob.reshape(out.size(0), int(np.prod(list(out.size())[1:])), inp.size(0), int(np.prod(list(inp.size())[1:])))
            jacob = torch.sum(jacob, 2).detach()
            module.jacobs.append(jacob)
        except:
            pass

    modules = net.cells if hasattr(net, 'cells') else net.layers if hasattr(net, 'layers') else None
    if modules is not None:
        for module in modules:
            module.register_forward_hook(jacob_forward_hook)
    else:
        raise ValueError('The provided net has no cells or layers attribute set.')

    net.zero_grad()
    inputs.requires_grad_(True)
    N = inputs.shape[0]
    for sp in range(split_data):
        st=sp*N//split_data
        en=(sp+1)*N//split_data

        net(inputs[st:en])

    try:
        layerwise_dyn_isometry = eval_score(modules)
    except Exception as e:
        print(e)
        layerwise_dyn_isometry = np.nan

    return layerwise_dyn_isometry
'''
