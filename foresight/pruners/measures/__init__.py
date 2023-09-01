available_measures = []
_measure_impls = {}


def measure(name, bn=True, copy_net=True, force_clean=True, **impl_args):
    def make_impl(func):
        def measure_impl(net_orig, device, *args, **kwargs):
            if copy_net:
                net = net_orig.get_prunable_copy(bn=bn).to(device)
            else:
                net = net_orig
            ret = func(net, device, *args, **kwargs, **impl_args)
            if copy_net and force_clean:
                import gc
                import torch
                del net
                torch.cuda.empty_cache()
                gc.collect()
            return ret

        global _measure_impls
        if name in _measure_impls:
            raise KeyError(f'Duplicated measure! {name}')
        available_measures.append(name)
        _measure_impls[name] = measure_impl
        return func
    return make_impl


def calc_measure(name, net, device, *args, **kwargs):
    return _measure_impls[name](net, device, *args, **kwargs)


def load_all():
    from . import plain
    from . import grad_norm
    from . import l2_norm
    from . import snip
    from . import grasp
    from . import fisher
    from . import jacob_cor
    from . import jacob_cor_logdet
    from . import epe_score
    from . import synflow
    from . import phi_score
    from . import phi_score_with_data
    from . import zen_score
    from . import zen_score_with_data
    from . import zen_score_cossim
    from . import zen_score_with_data_cossim
    from . import ssl_score
    from . import bundle_entropy
    from . import dyn_isometry
    from . import layerwise_dyn_isometry
    from . import feature_distance
    from . import resp_sens
    from . import act_hamming
    from . import grad_hamming
    from . import act_grad_hamming
    from . import act_cor
    from . import grad_cor
    from . import act_grad_cor
    from . import act_grad_cor_weighted


# TODO: should we do that by default?
load_all()
