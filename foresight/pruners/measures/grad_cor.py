import torch, numpy as np

from . import measure


@measure('grad_cor', bn=True, mode='param')
def compute_grad_cor(net, device, inputs, targets, mode, loss_fn, split_data=1):

    def counting_forward_hook(module, inp, out):
        try:
            if isinstance(inp, tuple):
                inp = inp[0]
            inp = inp.view(inp.size(0), -1)
            act_corrs = np.corrcoef(inp.detach().cpu().numpy())
            if np.sum(np.isnan(act_corrs)) == 0:
                net.K = net.K + np.abs(act_corrs)
                #net.N = net.N + 1
        except Exception as e:
            print(e)

    def counting_backward_hook(module, grad_input, grad_output):
        try:
            if isinstance(grad_input, tuple):
                grad_input = grad_input[0]
            grad_input = grad_input.view(grad_input.size(0), -1)
            grad_corrs = np.corrcoef(grad_input.detach().cpu().numpy())
            if np.sum(np.isnan(grad_corrs)) == 0:
                net.K = net.K + np.abs(grad_corrs)
                #net.N = net.N + 1
        except Exception as e:
            print(e)

    def hooklogdet(K):
        s, ld = np.linalg.slogdet(K)
        return ld

    for name, module in net.named_modules():
        if 'ReLU' in str(type(module)):
        #if isinstance(module, torch.nn.ReLU):
            #module.register_forward_hook(counting_forward_hook)
            module.register_backward_hook(counting_backward_hook)

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
    #grad_cor = np.mean(s)
    grad_cor = np.prod(s)

    return grad_cor
