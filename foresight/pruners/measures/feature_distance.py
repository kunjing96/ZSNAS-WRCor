import torch, numpy as np

from . import measure


def pairwise_distance(x, p=2):
    batch = x.shape[0]
    ret = np.zeros((batch, batch))
    for i in range(batch):
        for j in range(batch):
            ret[i,j] = ret[j,i] = np.linalg.norm(x[i] - x[j], ord=p)
    return ret


@measure('feature_distance', bn=True, mode='param')
def compute_feature_distance(net, device, inputs, targets, mode, loss_fn, split_data=1):

    def hooklogdet(out, targets, num_classes):
        idx_c_list = []
        mean_c_list = []
        dist_c_list = []
        for c in range(num_classes):
            idx_c = np.where(targets==c)[0]
            if idx_c.size == 0: continue
            idx_c_list.append(idx_c)
            out_c = out[idx_c]
            mean_c = np.mean(out_c, 0)
            dist_c = np.mean(pairwise_distance(out_c, p=2))
            mean_c_list.append(mean_c)
            dist_c_list.append(dist_c)
        mean_c_list = np.vstack(mean_c_list)
        dist = np.mean(pairwise_distance(mean_c_list, p=2)) / np.mean(dist_c_list)
        return dist

    outs = []
    net.zero_grad()
    N = inputs.shape[0]
    with torch.no_grad():
        for sp in range(split_data):
            st=sp*N//split_data
            en=(sp+1)*N//split_data

            out = net.forward(inputs[st:en], pre_GAP=True)
            out = out.reshape(out.size(0), -1)
            outs.append(out)
    outs = torch.cat(outs, 0)
    num_classes = net.classifier.out_features if hasattr(net, 'classifier') else net.num_classes
    feature_distance = hooklogdet(outs.detach().cpu().numpy(), targets.cpu().numpy(), num_classes)

    return feature_distance
