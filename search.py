import os, time, random, argparse, collections, copy, logging, PIL
import torch
from torch.distributions import Categorical, Bernoulli
import numpy as np
from tqdm import tqdm
from functools import cmp_to_key
from nasbench import api
from nas_201_api import NASBench201API as API

from foresight.models import get_nasbench101_model
from foresight.models import get_nasbench201_model
from foresight.models import get_MobileNetV2_model
from foresight.models.MobileNetV2_ops.global_utils import smart_round
from foresight.pruners import *
from foresight.dataset import *
from foresight.weight_initializers import init_net


def get_num_classes(dataset):
    return 100 if dataset == 'cifar100' else 10 if dataset == 'cifar10' else 120 if dataset == 'ImageNet16-120' else 1000


def find_nth(s, c, n):
    if n < 0: return -1
    r = s.find(c)
    while n > 0 and r >= 0:
        r = s.find(c, r + 1)
        n -= 1
    return r


def prepare_logger(args):
    args = copy.deepcopy(args)
    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(args.save_dir, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)
    logging.info('Main Function with logger : {:}'.format(logging))
    logging.info('Arguments : -------------------------------')
    for name, value in args._get_kwargs():
        logging.info('{:16} : {:}'.format(name, value))
    logging.info("Python  Version  : {:}".format(sys.version.replace('\n', ' ')))
    logging.info("Pillow  Version  : {:}".format(PIL.__version__))
    logging.info("PyTorch Version  : {:}".format(torch.__version__))
    logging.info("cuDNN   Version  : {:}".format(torch.backends.cudnn.version()))
    logging.info("CUDA available   : {:}".format(torch.cuda.is_available()))
    logging.info("CUDA GPU numbers : {:}".format(torch.cuda.device_count()))
    logging.info("CUDA_VISIBLE_DEVICES : {:}".format(os.environ['CUDA_VISIBLE_DEVICES'] if 'CUDA_VISIBLE_DEVICES' in os.environ else 'None'))


def parse_arguments():
    parser = argparse.ArgumentParser(description='Neural Architecture Search')
    parser.add_argument('--search_space', default=None, type=str, choices=['nasbench101', 'nasbench201', 'MobileNetV2'], help='search space')
    parser.add_argument('--search_algo', default=None, type=str, choices=['random', 'evolution', 'rl'], help='search algorithm')
    parser.add_argument('--outdir', default='./', type=str, help='output directory')
    parser.add_argument('--init_w_type', type=str, default='none', help='weight initialization (before pruning) type [none, xavier, kaiming, zero, N(0,1)]')
    parser.add_argument('--init_b_type', type=str, default='none', help='bias initialization (before pruning) type [none, xavier, kaiming, zero, N(0,1)]')
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--dataset', type=str, default='cifar10', help='dataset to use [cifar10, cifar100, ImageNet16-120, ImageNet1k]')
    parser.add_argument('--gpu', type=int, default=0, help='GPU index to work on')
    parser.add_argument('--num_data_workers', type=int, default=0, help='number of workers for dataloaders')
    parser.add_argument('--dataload', type=str, default='random', help='random or grasp supported')
    parser.add_argument('--dataload_info', type=int, default=1, help='number of batches to use for random dataload or number of samples per class for grasp dataload')
    parser.add_argument('--seed', type=int, default=42, help='pytorch manual seed')
    parser.add_argument('--write_freq', type=int, default=100, help='frequency of write to file')
    parser.add_argument('--measures', type=str, default=None, nargs='+', help='available measures')
    parser.add_argument('--N', default=1000, type=int, help='the number of searched archs')
    parser.add_argument('--population_size', default=64, type=int)
    parser.add_argument('--tournament_size', default=10, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--EMA_momentum', default=0.9, type=float)
    parser.add_argument('--flops_limit', default=600e6, type=float)
    args = parser.parse_args()
    args.device = torch.device("cuda:"+str(args.gpu) if torch.cuda.is_available() else "cpu")
    if args.search_space == 'nasbench101':
        args.api_loc = './data/nasbench_full.tfrecord'
    elif args.search_space == 'nasbench201':
        args.api_loc = './data/NAS-Bench-201-v1_1-096897.pth'
    else:
        args.api_loc = None
    args.save_dir = os.path.join(args.outdir, f'{args.search_space}_{args.search_algo}_N{args.N}_{args.measures}_{args.dataset}_initw{args.init_w_type}_initb{args.init_b_type}_bsz{args.batch_size}_seed{args.seed}')
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    return args


class ExponentialMovingAverage(object):
    """Class that maintains an exponential moving average."""

    def __init__(self, momentum):
        self._numerator = 0
        self._denominator = 0
        self._momentum = momentum

    def update(self, value):
        self._numerator = (
            self._momentum * self._numerator + (1 - self._momentum) * value
        )
        self._denominator = self._momentum * self._denominator + (1 - self._momentum)

    def value(self):
        """Return the current value of the moving average"""
        return self._numerator / self._denominator


class NAS(object):

    def __init__(self, N, search_space, measures, dataset, batch_size, num_data_workers, dataload, dataload_info, init_w_type, init_b_type, flops_limit, api_loc=None, device='cpu'):
        self.N = N
        self.search_space = search_space
        self.dataset = dataset
        self.dataload = dataload
        self.dataload_info = dataload_info
        self.init_w_type = init_w_type
        self.init_b_type = init_b_type
        self.flops_limit = flops_limit
        self.device = device
        self.measures = [m_str[1:] for m_str in measures]
        self.sign = {m_str: -1 if m_str[0] == '-' else 1 for m_str in self.measures}
        if self.search_space == 'nasbench101':
            self.nasbench = api.NASBench(api_loc)
            self.candidates = list(self.nasbench.fixed_statistics.items())
            self.candidates = [(candidate[0], candidate[1]['module_adjacency'], candidate[1]['module_operations']) for candidate in self.candidates]
        elif self.search_space == 'nasbench201':
            self.nasbench = API(api_loc)
            self.candidates = list(enumerate(self.nasbench))
        elif self.search_space == 'MobileNetV2':
            self.c_range = [range(16, 49), range(16, 49), range(16, 49), range(32, 65), range(48, 97), range(80, 257), range(80, 257), range(128, 513), range(128, 1537)]
            self.b_range = [range(16, 49), range(16, 49), range(16, 49), range(32, 65), range(48, 97), range(80, 257), range(80, 257)]
            self.l_range = [range(1, 3), range(2, 4), range(2, 4), range(2, 5), range(2, 7), range(2, 7), range(1, 3)]
            self.e_range = [1, 2, 4, 6]
            self.k_range = [3, 5, 7]
            self.r_range = range(192, 321)
            self.candidates = None
        else:
            raise ValueError('There is no {:} search space.'.format(self.search_space))
        self.train_loader, self.val_loader = get_dataloaders(batch_size, batch_size, self.dataset, num_data_workers)

    def sample_arch(self):
        if self.candidates is not None:
            sampled_arch = random.choice(self.candidates)
        else:
            while True:
                c = [smart_round(random.choice(cr), base=8) for cr in self.c_range]
                b = [smart_round(random.choice(br), base=8) for br in self.b_range]
                l = [random.choice(lr) for lr in self.l_range]
                e = [random.choice(self.e_range) for _ in range(7)]
                k = [random.choice(self.k_range) for _ in range(7)]
                s_idx = random.sample(range(7), 4)
                s = [2 if i in s_idx else 1 for i in range(7)]
                input_resolution = smart_round(random.choice(self.r_range), base=32)
                arch_str = f'SuperConvK3BNRELU(3,{c[0]},2,1)SuperResIDWE{e[0]}K{k[0]}({c[0]},{c[1]},{s[0]},{b[0]},{l[0]})SuperResIDWE{e[1]}K{k[1]}({c[1]},{c[2]},{s[1]},{b[1]},{l[1]})SuperResIDWE{e[2]}K{k[2]}({c[2]},{c[3]},{s[2]},{b[2]},{l[2]})SuperResIDWE{e[3]}K{k[3]}({c[3]},{c[4]},{s[3]},{b[3]},{l[3]})SuperResIDWE{e[4]}K{k[4]}({c[4]},{c[5]},{s[4]},{b[4]},{l[4]})SuperResIDWE{e[5]}K{k[5]}({c[5]},{c[6]},{s[5]},{b[5]},{l[5]})SuperResIDWE{e[6]}K{k[6]}({c[6]},{c[7]},{s[6]},{b[6]},{l[6]})SuperConvK1BNRELU({c[7]},{c[8]},1,1)'
                sampled_arch = (arch_str, input_resolution)

                num_classes = get_num_classes(self.dataset)
                plainnet_struct, input_resolution = sampled_arch[0], sampled_arch[1]
                net = get_MobileNetV2_model(plainnet_struct, num_classes=num_classes, input_resolution=input_resolution)
                if net.get_FLOPs() <= self.flops_limit:
                    break
        return sampled_arch

    def eval_arch(self, arch):
        total_eval_time = 0
        start_time = time.time()
        if self.search_space == 'nasbench101':
            num_classes = get_num_classes(self.dataset)
            spec = api.ModelSpec(matrix=arch[1], ops=arch[2])
            net = get_nasbench101_model(spec, num_classes=num_classes)
        elif self.search_space == 'nasbench201':
            config = self.nasbench.get_net_config(arch[0], self.dataset)
            net = get_nasbench201_model(config)
        elif self.search_space == 'MobileNetV2':
            num_classes = get_num_classes(self.dataset)
            plainnet_struct, input_resolution = arch[0], arch[1]
            net = get_MobileNetV2_model(plainnet_struct, num_classes=num_classes, input_resolution=input_resolution)
        else:
            raise ValueError('There is no {:} search space.'.format(self.search_space))
        net.to(self.device)
        init_net(net, self.init_w_type, self.init_b_type)
        if 'acc' in self.sign.keys():
            self.measures.remove('acc')
        measures, _ = predictive.find_measures(net, 
                                               self.train_loader, 
                                               (self.dataload, self.dataload_info, get_num_classes(self.dataset)),
                                               self.device,
                                               measure_names=self.measures)
        total_eval_time += (time.time() - start_time)
        if 'acc' in self.sign.keys():
            if self.search_space == 'nasbench101':
                info = self.nasbench.query(spec, epochs=12)
                measures['acc'] = info['validation_accuracy']
                total_eval_time += info['training_time']
            elif self.search_space == 'nasbench201':
                dataset = self.dataset if self.dataset != 'cifar10' else 'cifar10-valid'
                info = self.nasbench.get_more_info(arch[0], dataset, iepoch=None, hp="12", is_random=True)
                measures['acc'] = info['valid-accuracy']
                total_eval_time += (info["train-all-time"] + info["valid-per-time"])
            else:
                raise ValueError('Arch in {:} search space have to be trained or this space does not exist.'.format(self.search_space))
            self.measures.append('acc')
        for i, m in enumerate(self.measures):
            measures[m] = self.sign[m] * measures[m]
        return measures, total_eval_time

    def cmp(self, x, y):
        ret = []
        for k in self.measures:
            ret.append(x[1][k] - y[1][k])
        ret = np.array(ret)
        if sum(ret<0)>len(self.measures)/2: return -1
        elif sum(ret>0)>len(self.measures)/2: return 1
        else: return 0


class Random_NAS(NAS):

    def __init__(self, N, search_space, measures, dataset, batch_size, num_data_workers, dataload, dataload_info, init_w_type, init_b_type, flops_limit, api_loc=None, device='cpu'):
        super(Random_NAS, self).__init__(N, search_space, measures, dataset, batch_size, num_data_workers, dataload, dataload_info, init_w_type, init_b_type, flops_limit, api_loc=api_loc, device=device)

    def run(self):
        total_eval_time = 0
        best = None
        history  = []
        for i in tqdm(range(self.N)):
            arch     = self.sample_arch()
            measures, eval_time = self.eval_arch(arch)
            total_eval_time += eval_time
            cur = (arch, measures)
            history.append(cur)
            best = cur if best is None else max([cur, best], key=cmp_to_key(self.cmp))
        return best, history, total_eval_time


class RL_NAS(NAS):

    def __init__(self, N, search_space, measures, dataset, batch_size, num_data_workers, dataload, dataload_info, init_w_type, init_b_type, flops_limit, lr, EMA_momentum, api_loc=None, device='cpu'):
        super(RL_NAS, self).__init__(N, search_space, measures, dataset, batch_size, num_data_workers, dataload, dataload_info, init_w_type, init_b_type, flops_limit, api_loc=api_loc, device=device)
        if self.search_space == 'nasbench101':
            self.available_ops = ['conv3x3-bn-relu', 'conv1x1-bn-relu', 'maxpool3x3']
            self.op_parameters = torch.nn.Parameter(1e-3 * torch.randn(5, len(self.available_ops)))
            self.edge_parameters = torch.nn.Parameter(1e-3 * torch.randn(7, 7))
            self.optimizer = torch.optim.Adam([self.op_parameters, self.edge_parameters], lr=lr)
            self.baseline = ExponentialMovingAverage(EMA_momentum)
        elif self.search_space == 'nasbench201':
            self.available_ops = ['none', 'skip_connect', 'nor_conv_1x1', 'nor_conv_3x3', 'avg_pool_3x3']
            self.arch_parameters = torch.nn.Parameter(1e-3 * torch.randn(sum(range(4)), len(self.available_ops)))
            self.optimizer = torch.optim.Adam([self.arch_parameters], lr=lr)
            self.baseline = ExponentialMovingAverage(EMA_momentum)
        elif self.search_space == 'MobileNetV2':
            self.c_parameters = [torch.nn.Parameter(1e-3 * torch.randn((max(cr)-min(cr)) // 8 + 1)) for cr in self.c_range]
            self.b_parameters = [torch.nn.Parameter(1e-3 * torch.randn((max(br)-min(br)) // 8 + 1)) for br in self.b_range]
            self.l_parameters = [torch.nn.Parameter(1e-3 * torch.randn((max(lr)-min(lr)) // 1 + 1)) for lr in self.l_range]
            self.e_parameters = [torch.nn.Parameter(1e-3 * torch.randn(len(self.e_range))) for _ in range(7)]
            self.k_parameters = [torch.nn.Parameter(1e-3 * torch.randn(len(self.k_range))) for _ in range(7)]
            self.s_parameters = [torch.nn.Parameter(1e-3 * torch.randn(7)) for _ in range(4)]
            self.r_parameters = [torch.nn.Parameter(1e-3 * torch.randn((max(self.r_range)-min(self.r_range)) // 32 + 1))]
            self.optimizer = torch.optim.Adam(self.c_parameters + self.b_parameters + self.l_parameters + self.e_parameters + self.k_parameters + self.s_parameters + self.r_parameters, lr=lr)
            self.baseline = ExponentialMovingAverage(EMA_momentum)
        else:
            raise ValueError('There is no {:} search space.'.format(self.search_space))

    def select_action(self):
        if self.search_space == 'nasbench101':
            while True:
                op_probs = torch.nn.functional.softmax(self.op_parameters, dim=-1)
                m_op = Categorical(op_probs)
                ops = m_op.sample()
                ops_ = ['input'] + [self.available_ops[op] for op in ops] + ['output']
                matrix_probs = torch.nn.functional.sigmoid(self.edge_parameters).triu(1)
                m_mat = Bernoulli(matrix_probs)
                matrix = m_mat.sample()
                matrix_ = np.int8(matrix.cpu().numpy())
                spec = api.ModelSpec(matrix=matrix_, ops=ops_)
                if self.nasbench.is_valid(spec):
                    break
            return (m_op.log_prob(ops), m_mat.log_prob(matrix)), (ops_, matrix_)
        elif self.search_space == 'nasbench201':
            probs = torch.nn.functional.softmax(self.arch_parameters, dim=-1)
            m = Categorical(probs)
            action = m.sample()
            return m.log_prob(action), action.cpu().tolist()
        elif self.search_space == 'MobileNetV2':
            log_prob = []
            action = {}
            c_probs = [torch.nn.functional.softmax(cp, dim=-1) for cp in self.c_parameters]
            m_c = [Categorical(cp) for cp in c_probs]
            c = [m.sample() for m in m_c]
            c_ = [x.item() for x in c]
            log_prob += [m.log_prob(a) for m, a in zip(m_c, c)]
            action['c'] = c_
            b_probs = [torch.nn.functional.softmax(bp, dim=-1) for bp in self.b_parameters]
            m_b = [Categorical(bp) for bp in b_probs]
            b = [m.sample() for m in m_b]
            b_ = [x.item() for x in b]
            log_prob += [m.log_prob(a) for m, a in zip(m_b, b)]
            action['b'] = b_
            l_probs = [torch.nn.functional.softmax(lp, dim=-1) for lp in self.l_parameters]
            m_l = [Categorical(lp) for lp in l_probs]
            l = [m.sample() for m in m_l]
            l_ = [x.item() for x in l]
            log_prob += [m.log_prob(a) for m, a in zip(m_l, l)]
            action['l'] = l_
            e_probs = [torch.nn.functional.softmax(ep, dim=-1) for ep in self.e_parameters]
            m_e = [Categorical(ep) for ep in e_probs]
            e = [m.sample() for m in m_e]
            e_ = [x.item() for x in e]
            log_prob += [m.log_prob(a) for m, a in zip(m_e, e)]
            action['e'] = e_
            k_probs = [torch.nn.functional.softmax(kp, dim=-1) for kp in self.k_parameters]
            m_k = [Categorical(kp) for kp in k_probs]
            k = [m.sample() for m in m_k]
            k_ = [x.item() for x in k]
            log_prob += [m.log_prob(a) for m, a in zip(m_k, k)]
            action['k'] = k_
            r_probs = torch.nn.functional.softmax(self.r_parameters[0], dim=-1)
            m_r = Categorical(r_probs)
            r = m_r.sample()
            r_ = r.item()
            log_prob += [m_r.log_prob(r)]
            action['r'] = r_
            while(1):
                s_probs = [torch.nn.functional.softmax(sp, dim=-1) for sp in self.s_parameters]
                m_s = [Categorical(sp) for sp in s_probs]
                s = [m.sample() for m in m_s]
                s_ = [x.item() for x in s]
                if len(set(s_)) == len(s_):
                    break
            log_prob += [m.log_prob(a) for m, a in zip(m_s, s)]
            action['s'] = s_
            return log_prob, action
        else:
            raise ValueError('There is no {:} search space.'.format(self.search_space))

    def generate_arch(self, actions):
        if self.search_space == 'nasbench101':
            spec = api.ModelSpec(matrix=actions[1], ops=actions[0])
            spec_hash = spec.hash_spec(self.available_ops)
            return (spec_hash, actions[1], actions[0])
        elif self.search_space == 'nasbench201':
            spec = [self.available_ops[action] for action in actions]
            arch_str = '|{:}~0|+|{:}~0|{:}~1|+|{:}~0|{:}~1|{:}~2|'.format(*spec)
            i = self.nasbench.query_index_by_arch(arch_str)
            return (i, arch_str)
        elif self.search_space == 'MobileNetV2':
            c = [int(min(cr)+c*8) for c, cr in zip(actions['c'], self.c_range)]
            b = [int(min(br)+b*8) for b, br in zip(actions['b'], self.b_range)]
            l = [int(min(lr)+l) for l, lr in zip(actions['l'], self.l_range)]
            e = [int(self.e_range[e]) for e in actions['e']]
            k = [int(self.k_range[k]) for k in actions['k']]
            s = [2 if i in actions['s'] else 1 for i in range(7)]
            input_resolution = int(min(self.r_range)+actions['r']*32)
            arch_str = f'SuperConvK3BNRELU(3,{c[0]},2,1)SuperResIDWE{e[0]}K{k[0]}({c[0]},{c[1]},{s[0]},{b[0]},{l[0]})SuperResIDWE{e[1]}K{k[1]}({c[1]},{c[2]},{s[1]},{b[1]},{l[1]})SuperResIDWE{e[2]}K{k[2]}({c[2]},{c[3]},{s[2]},{b[2]},{l[2]})SuperResIDWE{e[3]}K{k[3]}({c[3]},{c[4]},{s[3]},{b[3]},{l[3]})SuperResIDWE{e[4]}K{k[4]}({c[4]},{c[5]},{s[4]},{b[4]},{l[4]})SuperResIDWE{e[5]}K{k[5]}({c[5]},{c[6]},{s[5]},{b[5]},{l[5]})SuperResIDWE{e[6]}K{k[6]}({c[6]},{c[7]},{s[6]},{b[6]},{l[6]})SuperConvK1BNRELU({c[7]},{c[8]},1,1)'
            return (arch_str, input_resolution)
        else:
            raise ValueError('There is no {:} search space.'.format(self.search_space))

    def select_generate(self):
        while True:
            log_prob, action = self.select_action()
            arch = self.generate_arch(action)
            if self.search_space == 'MobileNetV2':
                num_classes = get_num_classes(self.dataset)
                plainnet_struct, input_resolution = arch[0], arch[1]
                net = get_MobileNetV2_model(plainnet_struct, num_classes=num_classes, input_resolution=input_resolution)
                if net.get_FLOPs() <= self.flops_limit:
                    break
        return log_prob, arch

    def run(self):
        runs = []
        for _ in range(100):
            arch     = self.sample_arch()
            measures, _ = self.eval_arch(arch)
            run = [measures[k] for k in self.measures]
            if not (np.isnan(sum(run)) or np.isinf(sum(run))):
                runs.append(run)
        runs = np.array(runs)
        measures_mean = runs.mean(0).tolist()
        measures_std = runs.std(0).tolist()

        total_eval_time = 0
        best = None
        history  = []
        for i in tqdm(range(self.N)):
            log_prob, arch = self.select_generate()
            measures, eval_time = self.eval_arch(arch)
            total_eval_time += eval_time
            cur = (arch, measures)
            history.append(cur)
            best = cur if best is None else max([cur, best], key=cmp_to_key(self.cmp))

            reward = sum([(measures[k]-measures_mean[j])/measures_std[j] for j, k in enumerate(self.measures)])
            if not (np.isnan(reward) or np.isinf(reward)):
                self.baseline.update(reward)
                if self.search_space == 'nasbench101':
                    policy_loss = (-log_prob[0] * (reward - self.baseline.value())).sum() + (-log_prob[1] * (reward - self.baseline.value())).sum()
                elif self.search_space == 'nasbench201':
                    policy_loss = (-log_prob * (reward - self.baseline.value())).sum()
                elif self.search_space == 'MobileNetV2':
                    policy_loss = sum([(-x * (reward - self.baseline.value())).sum() for x in log_prob])
                else:
                    raise ValueError('There is no {:} search space.'.format(self.search_space))
                self.optimizer.zero_grad()
                policy_loss.backward()
                self.optimizer.step()
            else:
                print('No updating!')

        return best, history, total_eval_time


class Evolved_NAS(NAS):

    def __init__(self, N, search_space, measures, population_size, tournament_size, dataset, batch_size, num_data_workers, dataload, dataload_info, init_w_type, init_b_type, flops_limit, api_loc=None, device='cpu'):
        super(Evolved_NAS, self).__init__(N, search_space, measures, dataset, batch_size, num_data_workers, dataload, dataload_info, init_w_type, init_b_type, flops_limit, api_loc=api_loc, device=device)
        self.population_size = population_size
        self.tournament_size = tournament_size

    def mutate(self, parent, p):
        if self.search_space == 'nasbench101':
            if random.random() < p:
                while True:
                    available_ops = ['conv3x3-bn-relu', 'conv1x1-bn-relu', 'maxpool3x3']
                    old_matrix, old_ops = parent[1], parent[2]
                    idx_to_change = random.randrange(len(old_ops[1:-1])) + 1
                    entry_to_change = old_ops[idx_to_change]
                    possible_entries = [x for x in available_ops if x != entry_to_change]
                    new_entry = random.choice(possible_entries)
                    new_ops = copy.deepcopy(old_ops)
                    new_ops[idx_to_change] = new_entry
                    idx_to_change = random.randrange(sum(range(1, len(old_matrix))))
                    new_matrix = copy.deepcopy(old_matrix)
                    num_node = len(old_matrix)
                    idx_to_ij = {int(i*(num_node-1)-i*(i-1)/2+(j-i-1)): (i, j) for i in range(num_node) for j in range(i+1, num_node)}
                    i, j = idx_to_ij[idx_to_change]
                    new_matrix[i][j] = random.choice([0, 1])
                    new_spec = api.ModelSpec(matrix=new_matrix, ops=new_ops)
                    if self.nasbench.is_valid(new_spec):
                        spec_hash = new_spec.hash_spec(available_ops)
                        child = (spec_hash, new_matrix, new_ops)
                        break
            else:
                child = parent
        elif self.search_space == 'nasbench201':
            if random.random() < p:
                available_ops = ['none', 'skip_connect', 'nor_conv_1x1', 'nor_conv_3x3', 'avg_pool_3x3']
                nodes = parent[1].split('+')
                nodes = [node[1:-1].split('|') for node in nodes]
                nodes = [[op_and_input.split('~')[0]  for op_and_input in node] for node in nodes]
                old_spec = [op for node in nodes for op in node]
                idx_to_change = random.randrange(len(old_spec))
                entry_to_change = old_spec[idx_to_change]
                possible_entries = [x for x in available_ops if x != entry_to_change]
                new_entry = random.choice(possible_entries)
                new_spec = copy.deepcopy(old_spec)
                new_spec[idx_to_change] = new_entry
                arch_str = '|{:}~0|+|{:}~0|{:}~1|+|{:}~0|{:}~1|{:}~2|'.format(*new_spec)
                i = self.nasbench.query_index_by_arch(arch_str)
                child = (i, arch_str)
            else:
                child = parent
        elif self.search_space == 'MobileNetV2':
            while True:
                old_arch_str = parent[0]
                idx_to_change = random.choice(range(7))
                idx_begin_to_pre_change = find_nth(old_arch_str, ')', idx_to_change - 1) + 1
                idx_end_to_pre_change   = find_nth(old_arch_str, ')', idx_to_change)     + 1
                idx_begin_to_change = find_nth(old_arch_str, ')', idx_to_change)     + 1
                idx_end_to_change   = find_nth(old_arch_str, ')', idx_to_change + 1) + 1
                old_pre_h = old_arch_str[idx_begin_to_pre_change: idx_end_to_pre_change]
                old_h     = old_arch_str[idx_begin_to_change: idx_end_to_change]
                old_e = eval(old_h[old_h.find('E') + 1: old_h.find('K')])
                old_k = eval(old_h[old_h.find('K') + 1: old_h.find('(')])
                old_c = eval(old_h[old_h.find('(') + 1: find_nth(old_h, ',', 0)])
                out_c = eval(old_h[find_nth(old_h, ',', 0) + 1: find_nth(old_h, ',', 1)])
                old_s = eval(old_h[find_nth(old_h, ',', 1) + 1: find_nth(old_h, ',', 2)])
                old_b = eval(old_h[find_nth(old_h, ',', 2) + 1: find_nth(old_h, ',', 3)])
                old_l = eval(old_h[find_nth(old_h, ',', 3) + 1: old_h.find(')')])
                if random.random() < p:
                    while(1):
                        new_e = random.choice(self.e_range)
                        new_k = random.choice(self.k_range)
                        if old_e != new_e or old_k != new_k: break
                else:
                    new_e = old_e
                    new_k = old_k
                new_c = smart_round(random.choice(self.c_range[idx_to_change]), base=8)
                new_b = smart_round(random.choice(self.b_range[idx_to_change]), base=8)
                new_l = random.choice(self.l_range[idx_to_change])
                new_pre_h = old_pre_h[:find_nth(old_pre_h, ',', 0) + 1] + f'{new_c}' + old_pre_h[find_nth(old_pre_h, ',', 1):]
                new_h = f'SuperResIDWE{new_e}K{new_k}({new_c},{out_c},{old_s},{new_b},{new_l})'
                new_input_resolution = smart_round(random.choice(self.r_range), base=32)
                new_arch_str = old_arch_str[:idx_begin_to_pre_change] + new_pre_h + new_h + old_arch_str[idx_end_to_change:]
                child = (new_arch_str, new_input_resolution)

                num_classes = get_num_classes(self.dataset)
                plainnet_struct, input_resolution = child[0], child[1]
                net = get_MobileNetV2_model(plainnet_struct, num_classes=num_classes, input_resolution=input_resolution)
                if net.get_FLOPs() <= self.flops_limit:
                    break
        else:
            raise ValueError('There is no {:} search space.'.format(self.search_space))
        return child

    def run(self):
        total_eval_time = 0
        best = None
        history  = []
        population = collections.deque()
        for i in range(self.population_size):
            arch     = self.sample_arch()
            measures, eval_time = self.eval_arch(arch)
            total_eval_time += eval_time
            cur = (arch, measures)
            population.append(cur)
            history.append(cur)
            best = cur if best is None else max([cur, best], key=cmp_to_key(self.cmp))
        for i in tqdm(range(self.N)):
            '''if i < self.N / 2:
                lambd = (self.N - 2*i) / self.N
                p = lambd + 0.1 * (1 - lambd)
            else:
                p = 0.1'''
            p = 1.0
            samples  = random.sample(population, self.tournament_size)
            parent   = max(samples, key=cmp_to_key(self.cmp))
            child    = self.mutate(parent[0], p)
            measures, eval_time = self.eval_arch(child)
            total_eval_time += eval_time
            cur = (child, measures)
            population.append(cur)
            history.append(cur)
            population.popleft()
            best = cur if best is None else max([cur, best], key=cmp_to_key(self.cmp))
        return best, history, total_eval_time


if __name__ == '__main__':
    args = parse_arguments()
    prepare_logger(args)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    if args.search_algo == 'random':
        nas = Random_NAS(args.N, args.search_space, args.measures, args.dataset, args.batch_size, args.num_data_workers, args.dataload, args.dataload_info, args.init_w_type, args.init_b_type, args.flops_limit, api_loc=args.api_loc, device=args.device)
    elif args.search_algo == 'evolution':
        nas = Evolved_NAS(args.N, args.search_space, args.measures, args.population_size, args.tournament_size, args.dataset, args.batch_size, args.num_data_workers, args.dataload, args.dataload_info, args.init_w_type, args.init_b_type, args.flops_limit, api_loc=args.api_loc, device=args.device)
    elif args.search_algo == 'rl':
        nas = RL_NAS(args.N, args.search_space, args.measures, args.dataset, args.batch_size, args.num_data_workers, args.dataload, args.dataload_info, args.init_w_type, args.init_b_type, args.flops_limit, args.lr, args.EMA_momentum, api_loc=args.api_loc, device=args.device)
    else:
        raise ValueError('There is no {:} algorithm.'.format(self.search_algo))

    begin_time = time.time()
    best, history, total_eval_time = nas.run()
    end_time = time.time()

    logging.info('The search history:')
    for i, h in enumerate(history):
        logging.info('{:5d} {:}'.format(i, h))
    logging.info('\n' + '-' * 100)
    logging.info('The best architectures:\n{:}'.format(best))
    logging.info(
        '{:} : search {:} architectures, total cost {:.1f} s, total evalution cost {:.1f} s, the best is {:}, the measures is {:}.'.format(
            args.search_algo, args.N, end_time - begin_time, total_eval_time, best[0], best[1]
        )
    )
    if args.search_space == 'nasbench101':
        logging.info('{:}\n{:}'.format(nas.nasbench.fixed_statistics[best[0][0]], nas.nasbench.computed_statistics[best[0][0]][108]))
    elif args.search_space == 'nasbench201':
        logging.info('{:}'.format(nas.nasbench.query_by_arch(best[0][1], "200")))
    elif args.search_space == 'MobileNetV2':
        logging.info('Train and test MobileNetV2 model using train_MobileNetV2.py and test_MobileNetV2.py!')
    else:
        raise ValueError('There is no {:} search space.'.format(self.search_space))

    torch.save({'best': best, 'history': history}, os.path.join(args.save_dir, 'history.ckp'))
