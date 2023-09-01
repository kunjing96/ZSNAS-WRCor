import torch, numpy as np

from . import measure


def get_batch_jacobian(net, x):
    net.zero_grad()
    x.requires_grad_(True)
    y = net(x)
    y.backward(torch.ones_like(y))
    jacob = x.grad

    return jacob.detach()


def eval_score_perclass(jacobs, labels):
    k = 1e-5
    n_classes = len(np.unique(labels))
    per_class={}
    for i, label in enumerate(labels):
        if label in per_class:
            per_class[label] = np.vstack((per_class[label],jacobs[i]))
        else:
            per_class[label] = jacobs[i]

    ind_corr_matrix_score = {}
    for c in per_class.keys():
        s = 0
        try:
            corrs = np.corrcoef(per_class[c])

            s = np.sum(np.log(abs(corrs)+k))
            if n_classes > 100:
                s /= len(corrs)
        except: # defensive programming
            continue

        ind_corr_matrix_score[c] = s

    # per class-corr matrix A and B
    score = 0
    ind_corr_matrix_score_keys = ind_corr_matrix_score.keys()
    if n_classes <= 100:
        for c in ind_corr_matrix_score_keys:
            # B)
            score += np.absolute(ind_corr_matrix_score[c])
    else: 
        for c in ind_corr_matrix_score_keys:
            # A)
            for cj in ind_corr_matrix_score_keys:
                score += np.absolute(ind_corr_matrix_score[c]-ind_corr_matrix_score[cj])

        # should divide by number of classes seen
        score /= len(ind_corr_matrix_score_keys)

    return score


@measure('epe_score', bn=True, mode='param')
def compute_epe_score(net, device, inputs, targets, mode, loss_fn, split_data=1):
    jacobs  = []
    N = inputs.shape[0]
    for sp in range(split_data):
        net.zero_grad()

        st=sp*N//split_data
        en=(sp+1)*N//split_data

        jacobs_batch = get_batch_jacobian(net, inputs[st:en])
        jacobs.append(jacobs_batch.reshape(jacobs_batch.size(0), -1).cpu().numpy())
    jacobs  = np.concatenate(jacobs,  axis=0)
    targets = targets.detach().cpu().numpy()

    try:
        epe_score = eval_score_perclass(jacobs, targets)
    except Exception as e:
        print(e)
        epe_score = np.nan

    return epe_score
