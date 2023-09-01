import torch, numpy as np

from . import measure


def get_batch_jacobian(net, x, y):
    net.zero_grad()
    x.requires_grad_(True)
    jacob = torch.autograd.functional.jacobian(net, x).detach()
    in_dim  = int( np.prod(list(x.size())[1:]) )
    out_dim = int( net.classifier.out_features ) if hasattr(net, 'classifier') else int( net.num_classes )
    #jacob = jacob.view(y.size(0), out_dim, x.size(0), in_dim).mean(dim=(0, 2))
    jacob = torch.sum(jacob.view(y.size(0), out_dim, x.size(0), in_dim), 2)
    return jacob


def eval_score(jacobs, labels):
    _, s, _ = torch.svd(jacobs)
    #v = torch.prod(s).item()
    #v = torch.prod(s, -1).cpu().numpy()
    v = torch.log(s).mean(-1).cpu().numpy()
    k = 1e-5
    #score = - np.mean(np.log(v + k) + 1./(v + k))
    score = - np.mean(v + 1./(np.e**v))
    return score


@measure('dyn_isometry', bn=True, mode='param')
def compute_dyn_isometry(net, device, inputs, targets, mode, loss_fn, split_data=1):
    #jacobs = None
    jacobs = []
    N = inputs.shape[0]
    for sp in range(split_data):
        net.zero_grad()

        st=sp*N//split_data
        en=(sp+1)*N//split_data

        jacobs_batch = get_batch_jacobian(net, inputs[st:en], targets[st:en])
        #if jacobs is None:
        #    jacobs = jacobs_batch
        #else:
        #    jacobs = jacobs + jacobs_batch
        jacobs.append(jacobs_batch)
    #jacobs  = jacobs / split_data
    jacobs  = torch.cat(jacobs, 0)
    targets = targets.detach()

    try:
        dyn_isometry = eval_score(jacobs, targets)
    except Exception as e:
        print(e)
        dyn_isometry = np.nan

    return dyn_isometry
