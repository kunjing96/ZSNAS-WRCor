from . import measure
from ..p_utils import get_layer_metric_array, sum_arr


@measure('l2_norm', copy_net=False, mode='param')
def get_l2_norm_array(net, device, inputs, targets, mode, split_data=1, loss_fn=None):
    l2_norm = get_layer_metric_array(net, lambda l: l.weight.norm(), mode=mode)
    return sum_arr(l2_norm)
