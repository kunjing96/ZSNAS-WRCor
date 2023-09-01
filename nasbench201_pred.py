import pickle
import torch
import argparse
import random

from foresight.models import get_nasbench201_model
from foresight.pruners import *
from foresight.dataset import *
from foresight.weight_initializers import init_net

def get_num_classes(args):
    return 100 if args.dataset == 'cifar100' else 10 if args.dataset == 'cifar10' else 120

def parse_arguments():
    parser = argparse.ArgumentParser(description='Zero-cost Metrics for NAS-Bench-201')
    parser.add_argument('--api_loc', default='data/NAS-Bench-201-v1_1-096897.pth', type=str, help='path to API')
    parser.add_argument('--outdir', default='./', type=str, help='output directory')
    parser.add_argument('--init_w_type', type=str, default='none', help='weight initialization (before pruning) type [none, xavier, kaiming, zero, N(0,1)]')
    parser.add_argument('--init_b_type', type=str, default='none', help='bias initialization (before pruning) type [none, xavier, kaiming, zero, N(0,1)]')
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--dataset', type=str, default='cifar10', help='dataset to use [cifar10, cifar100, ImageNet16-120]')
    parser.add_argument('--gpu', type=int, default=0, help='GPU index to work on')
    parser.add_argument('--num_data_workers', type=int, default=0, help='number of workers for dataloaders')
    parser.add_argument('--dataload', type=str, default='random', help='random or grasp supported')
    parser.add_argument('--dataload_info', type=int, default=1, help='number of batches to use for random dataload or number of samples per class for grasp dataload')
    parser.add_argument('--seed', type=int, default=42, help='pytorch manual seed')
    parser.add_argument('--start', type=int, default=0, help='start index')
    parser.add_argument('--end', type=int, default=0, help='end index')
    parser.add_argument('--write_freq', type=int, default=1, help='frequency of write to file')
    parser.add_argument('--available_measures', type=str, default=None, nargs='+', help='available measures')
    args = parser.parse_args()
    args.device = torch.device("cuda:"+str(args.gpu) if torch.cuda.is_available() else "cpu")
    return args


if __name__ == '__main__':
    args = parse_arguments()
    
    from nas_201_api import NASBench201API as API
    api = API(args.api_loc)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    args.end = len(api) if args.end == 0 else args.end
    print(f'Running models {args.start} to {args.end} out of {len(api)}')
    
    train_loader, val_loader = get_dataloaders(args.batch_size, args.batch_size, args.dataset, args.num_data_workers)

    cached_res = []
    avg_times = {}
    pre='cf' if 'cifar' in args.dataset else 'im'
    pfn=f'nb2_{pre}{get_num_classes(args)}_initw{args.init_w_type}_initb{args.init_b_type}_bsz{args.batch_size}_seed{args.seed}_start{args.start}_end{args.end}.p'
    op = os.path.join(args.outdir,pfn)

    print('outfile =',op)

    #loop over nasbench201 archs

    for i, arch_str in enumerate(api):

        if i < args.start:
            continue
        if i >= args.end:
            break
        print(f'idx = {i}')

        res = {'i':i, 'arch':arch_str}

        # model
        config = api.get_net_config(i, args.dataset)
        net = get_nasbench201_model(config)
        net.to(args.device)

        init_net(net, args.init_w_type, args.init_b_type)

        measures, times = predictive.find_measures(net, 
                                                   train_loader, 
                                                   (args.dataload, args.dataload_info, get_num_classes(args)),
                                                   args.device,
                                                   measure_names=args.available_measures)
        res['logmeasures'] = measures
        
        print(res)
        cached_res.append(res)

        for tk,tv in times.items():
            if tk not in avg_times.keys():
                avg_times[tk] = tv
            else:
                avg_times[tk] = (avg_times[tk] * (i - args.start) + tv) / (i - args.start + 1)

        #write to file
        if i % args.write_freq == 0 or i == len(api)-1 or i == 10:
            print(f'writing {len(cached_res)} results to {op}')
            pf=open(op, 'ab')
            for cr in cached_res:
                pickle.dump(cr, pf)
            pf.close()
            cached_res = []

    print('Time cost: {}'.format(avg_times))
