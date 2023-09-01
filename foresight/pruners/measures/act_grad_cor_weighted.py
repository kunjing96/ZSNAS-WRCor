import torch, numpy as np

from . import measure


@measure('act_grad_cor_weighted', bn=True, mode='param')
def compute_act_grad_cor_weighted(net, device, inputs, targets, mode, loss_fn, split_data=1):

    def get_counting_forward_hook(weight):
        def counting_forward_hook(module, inp, out):
            try:
                if isinstance(inp, tuple):
                    inp = inp[0]
                inp = inp.view(inp.size(0), -1)
                act_corrs = np.corrcoef(inp.detach().cpu().numpy())
                if np.sum(np.isnan(act_corrs)) == 0:
                    net.K = net.K + weight * (np.abs(act_corrs))
                    #net.N = net.N + weight
            except Exception as e:
                print(e)
        return counting_forward_hook

    def get_counting_backward_hook(weight):
        def counting_backward_hook(module, grad_input, grad_output):
            try:
                if isinstance(grad_input, tuple):
                    grad_input = grad_input[0]
                grad_input = grad_input.view(grad_input.size(0), -1)
                grad_corrs = np.corrcoef(grad_input.detach().cpu().numpy())
                if np.sum(np.isnan(grad_corrs)) == 0:
                    net.K = net.K + weight * (np.abs(grad_corrs))
                    #net.N = net.N + weight
            except Exception as e:
                print(e)
        return counting_backward_hook

    def hooklogdet(K):
        s, ld = np.linalg.slogdet(K)
        return ld

    modules = net.cells if hasattr(net, 'cells') else net.layers if hasattr(net, 'layers') else net.block_list if hasattr(net, 'block_list') else None
    for i, module in enumerate(modules):
        for name, m in module.named_modules():
            if 'ReLU' in str(type(m)):
            #if isinstance(m, torch.nn.ReLU):
                m.register_forward_hook(get_counting_forward_hook(2**i))
                m.register_backward_hook(get_counting_backward_hook(2**i))
    if hasattr(net, 'lastact'): # for nasbench201
        for name, m in net.lastact.named_modules():
            if 'ReLU' in str(type(m)):
            #if isinstance(m, torch.nn.ReLU):
                m.register_forward_hook(get_counting_forward_hook(2**len(modules)))
                m.register_backward_hook(get_counting_backward_hook(2**len(modules)))

    s = []
    N = inputs.shape[0]
    for sp in range(split_data):
        net.zero_grad()
        net.K = np.zeros((N//split_data, N//split_data))
        #net.N = 0

        st=sp*N//split_data
        en=(sp+1)*N//split_data

        outputs = net(inputs[st:en])
        loss = loss_fn(outputs, targets[st:en])
        loss.backward()
        #if net.N != 0:
        #    s.append(hooklogdet(net.K / net.N))
        s.append(hooklogdet(net.K))
    #act_grad_cor_weighted = np.mean(s)
    act_grad_cor_weighted = np.prod(s)

    return act_grad_cor_weighted
