import torch, math

from . import measure


def _get_weight_amplitude(layer):
    """ With this function we approximate the weight
        amplitude of a layer. A layer could consist of multiple 
        sub layers (e.g. a vgg block with multiple layers).
        Therefore, we take the mean amplitude of each weight in 
        trainable_weights.
        param: layer - Layer for which the amplitude should be known
        return: Single floating point value of the max. weight amplitude of some sub layers
    """
    if layer.parameters() is None:
        return 0.0

    ret = []
    for p in layer.parameters():
        ret.append(p.abs().max().detach().cpu().item())
    return max(ret)


def _calculate_bundles(X, Y, num_classes):
    """ Calculates all bundles of X and calculates the bundle entropy using 
        the label information from Y. Iterates over X and therefore the 
        complexity is O(X) if calculate_bundle can be fully vectorized.
    """    

    # Create array for each feature and assign a unique bundlenumber...
    bundle = [0] * X.size(0)
    bundle_entropy = 0
    for i in range(X.size(0)):
        if bundle[i] == 0:
            bundle, bundle_entropy = _calculate_single_bundle(i, bundle, X, Y, bundle_entropy, num_classes)

    num_bundles = max(bundle)
    return num_bundles, bundle_entropy / X.size(0)
    

def _calculate_single_bundle(i, bundle, X, Y, bundle_entropy, num_classes):
    """ This function calculates a bundle which contains all x which are similar
        than X[i] using vectorization such that only O(|X|) is needed to calculate
        bundles for one layer. The idea is to use equality operators, mask 
        all others out, set the bundle and reuse this information vector 
        for the next samples. As a vector contains all equivalent samples
        also the bundle entropy at time step t can immediately be calculated.
    """
    dim_X = X.size(1)
    next_bundle_id = max(bundle) + 1.0
    x = X[i]

    # Ignore all that are already bundleed (bundle > 0)
    zero_out = [float(x<=0) for x in bundle]
    # Set bundle id for current x[i]
    bundle[i] += next_bundle_id * zero_out[i]
    # Also ignore current x[i]
    zero_out = [float(x<=0) for x in bundle]
    # Get all equivalent components (single component only of possible ndim inputs)
    equal_components = (x == X).float()
    # All components must be equivalent, therefore check (using the sum and dim_X) whether this is the case
    num_equal_components = torch.sum(equal_components, -1)
    same_bundle = (num_equal_components >= dim_X).int()
    # And bundle all components
    for j in range(same_bundle.size(0)):
        bundle[j] += same_bundle[j].item() * next_bundle_id * zero_out[j]

    # Calculate the bundle entropy for the current bundle (same_bundle) using the entropy
    bundle_only = (Y.int()*same_bundle.int()).int()
    bundle_class_prob = torch.bincount(bundle_only, minlength=num_classes).float()
    bundle_class_prob /= torch.sum(bundle_class_prob)
    bundle_size = torch.sum(same_bundle).float().item()
    entropy = -torch.sum(bundle_class_prob * (bundle_class_prob+1e-5).log(), -1).item()
    bundle_entropy += max(0.0, entropy) * bundle_size

    return bundle, bundle_entropy


@measure('bundle_entropy', bn=True, mode='param')
def compute_bundle_entropy(net, device, inputs, targets, mode, loss_fn, split_data=1, all_layers=True):
    """ Given a dataset inputs and a net, this function returns
        foreach a^l the number of bundles and the bundle entropy at time 
        step t. 
        
        Limitation: This implementation is currently only for a single
        GPU. I.e. you can train your net with multiple GPUs, and evaluate 
        cb with a single gpu.
        
        returns: [[num_bundles_1, bundle_entropy_1], ... [num_bundles_L, bundle_entropy_L]]
    """

    def cb_forward_hook(module, inp, out):
        if not hasattr(module, 'cb'): module.cb = []
        module.cb.append(out.detach())

    modules = net.cells if hasattr(net, 'cells') else net.layers if hasattr(net, 'layers') else net.block_list if hasattr(net, 'block_list') else None
    if modules is not None:
        for module in modules:
            module.register_forward_hook(cb_forward_hook)
    else:
        raise ValueError('The provided net has no cells or layers attribute set.')

    N = inputs.shape[0]
    num_classes = net.classifier.out_features if hasattr(net, 'classifier') else net.num_classes
    layer_eval = 0 if all_layers else len(modules)-1
    A, Y = [], []

    net.zero_grad()
    net.eval()
    with torch.no_grad():
        for sp in range(split_data):
            st=sp*N//split_data
            en=(sp+1)*N//split_data

            net(inputs[st:en])

    for module in modules:
        A.append(torch.cat(module.cb, 0))
    
    Y = targets

    # If needed we return the conflicts for each layer. For evaluation only 
    # a^{(L)} is needed, but e.g. to use it with auto-tune all conflicting 
    # layers are needed. Therefore if all layers are evaluated the complexity
    # is O(L * |X|)
    res = []
    train_lr = 1e-3
    weights_amplitude = _get_weight_amplitude(net)
    for i, a in enumerate(A):
        # As written in the paper we not directly compare a_i and a_j in order 
        # to consider also floating point representations during backpropagation
        # Instead of doing an inequallity check using \gamma, we do an equality
        # check after subtracting the values from the maximum weights which 
        # is equivalent. Note that this is not possible if gamma should be 
        # larger than the floating point resolution.
        if i >= layer_eval:
            equality_check = weights_amplitude - a * train_lr * (1.0 / N)
            equality_check = equality_check.reshape(equality_check.size(0), -1)
            num_bundle, bundle_entropy = _calculate_bundles(equality_check, Y, num_classes)
            res.append(bundle_entropy)

    return - sum(res) / len(res)
