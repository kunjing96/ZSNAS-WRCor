import torch

from . import measure
from ..p_utils import get_layer_metric_array, sum_arr


@measure('synflow', bn=False, mode='param')
@measure('synflow_bn', bn=True, mode='param')
def compute_synflow_per_weight(net, device, inputs, targets, mode, split_data=1, loss_fn=None):

    device = inputs.device

    #convert params to their abs. Keep sign for converting it back.
    @torch.no_grad()
    def linearize(net):
        signs = {}
        for name, param in net.state_dict().items():
            signs[name] = torch.sign(param)
            param.abs_()
        return signs

    #convert to orig values
    @torch.no_grad()
    def nonlinearize(net, signs):
        for name, param in net.state_dict().items():
            if 'weight_mask' not in name:
                param.mul_(signs[name])

    # keep signs of all params
    signs = linearize(net)
    
    # Compute gradients with input of 1s 
    net.zero_grad()
    net.double()
    input_dim = list(inputs[0,:].shape)
    inputs = torch.ones([1] + input_dim).double().to(device)
    output = net.forward(inputs)
    torch.sum(output).backward() 

    # select the gradients that we want to use for search/prune
    def synflow(layer):
        if layer.weight.grad is not None:
            return torch.abs(layer.weight * layer.weight.grad)
        else:
            return None # torch.zeros_like(layer.weight)

    grads_abs = get_layer_metric_array(net, synflow, mode)

    # apply signs of all params
    nonlinearize(net, signs)

    return sum_arr(grads_abs)


