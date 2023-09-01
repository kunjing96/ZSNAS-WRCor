import torch

from . import measure


def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

@measure('ssl_score', bn=True, mode='param')
def compute_ssl_score(net, device, inputs, targets, mode, loss_fn, split_data=1, skip_grad=False):
    alpha = 0.01
    lambd = 0.005
    z1s = []
    z2s = []
    net.zero_grad()
    N = inputs.shape[0]
    with torch.no_grad():
        for sp in range(split_data):
            st=sp*N//split_data
            en=(sp+1)*N//split_data

            epsilon1 = torch.zeros_like(inputs[st:en]).normal_()
            epsilon2 = torch.zeros_like(inputs[st:en]).normal_()
            y1 = inputs[st:en] + alpha*epsilon1
            y2 = inputs[st:en] + alpha*epsilon2
            z1 = net.forward(y1, pre_GAP=True).reshape(en-st, -1)
            z2 = net.forward(y2, pre_GAP=True).reshape(en-st, -1)
            z1s.append(z1)
            z2s.append(z2)
    z1s = torch.cat(z1s, 0)
    z2s = torch.cat(z2s, 0)
    local_bn = torch.nn.BatchNorm1d(z1s.size(-1), affine=False).to(device)
    c = local_bn(z1s).T @ local_bn(z2s)
    c.div_(N)
    on_diag  = torch.diagonal(c).add_(-1).pow_(2).sum()
    off_diag = off_diagonal(c).pow_(2).sum()
    ssl_score = (on_diag + lambd * off_diag).detach().cpu().item()

    return -ssl_score
