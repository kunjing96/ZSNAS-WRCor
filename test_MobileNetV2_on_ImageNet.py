import os, sys, shutil, time, random, argparse, collections, copy, logging, PIL
import torch
import numpy as np
from collections import OrderedDict

import genotypes
from foresight.dataset import get_dataloaders
from foresight.models import get_MobileNetV2_model
from foresight.weight_initializers import init_net
from train_MobileNetV2_on_ImageNet import *


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

    if args.checkpoint:
        logging.info('Loading checkpoint {:} ...'.format(args.checkpoint))
        _, net_state_dict, _, _, _ = load_checkpoint(args.checkpoint)
    else:
        raise ValueError('Checkpoint is None!')

    logging.info('---------Genotype---------')
    genotype = eval('genotypes.{:}'.format(args.arch))
    logging.info(genotype)
    net = get_MobileNetV2_model(genotype[0], num_classes=1000, input_resolution=genotype[1], no_reslink=False, no_BN=False, use_se=True, dropout=dropout)
    if net_state_dict:
        net.load_state_dict(net_state_dict)
    else:
        raise ValueError('net_state_dict is None!')
    logging.info('param size = {:}'.format(net.get_model_size()))
    logging.info('flops      = {:}'.format(net.get_FLOPs()))
    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        net = torch.nn.DataParallel(net)
        net = net.to(args.device)
    else:
        net = net.to(args.device)
    logging.info('--------------------------') 

    criterion = torch.nn.CrossEntropyLoss()
    criterion = criterion.to(args.device)

    train_loader, val_loader = get_dataloaders(args.batch_size, args.batch_size, 'ImageNet1k', args.num_data_workers)

    start_time = time.time()
    valid_acc_top1, valid_acc_top5, valid_obj = infer(val_loader, net, criterion, args.report_freq)
    logging.info('Valid_acc_top1: {:}'.format(valid_acc_top1))
    logging.info('Valid_acc_top5: {:}'.format(valid_acc_top5))
    duration = time.time() - start_time
    logging.info('Time: {:}s.'.format(duration))
