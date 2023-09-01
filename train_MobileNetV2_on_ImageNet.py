import os, sys, shutil, time, random, argparse, collections, copy, logging, PIL
import torch
import torchvision
import numpy as np
from collections import OrderedDict

import genotypes
from foresight.dataset import get_dataloaders
from foresight.models import get_MobileNetV2_model
from foresight.weight_initializers import init_net


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
    parser = argparse.ArgumentParser("training imagenet")
    parser.add_argument('--arch', type=str, default=None, help='which architecture to use')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size')
    parser.add_argument('--learning_rate', type=float, default=0.1, help='init learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--weight_decay', type=float, default=4e-5, help='weight decay')
    parser.add_argument('--epochs', type=int, default=480, help='num of training epochs')
    parser.add_argument('--dropout', type=float, default=0, help='drop path probability')
    parser.add_argument('--kd_type', type=str, default='ce', choices=['ce', 'mse'], help='type of knowledge distilltion')
    parser.add_argument('--kd_ratio', type=float, default=1.0, help='ratio of knowledge distilltion')
    parser.add_argument('--init_w_type', type=str, default='none', help='weight initialization (before pruning) type [none, xavier, kaiming, zero, N(0,1)]')
    parser.add_argument('--init_b_type', type=str, default='none', help='bias initialization (before pruning) type [none, xavier, kaiming, zero, N(0,1)]')
    parser.add_argument('--grad_clip', type=float, default=5., help='gradient clipping')
    parser.add_argument('--label_smooth', type=float, default=0.1, help='label smoothing')
    parser.add_argument('--lr_scheduler', type=str, default='cosine', help='lr scheduler, linear or cosine')
    parser.add_argument('--gpu', type=int, default=0, help='GPU index to work on')
    parser.add_argument('--num_data_workers', type=int, default=32, help='number of workers for dataloaders')
    parser.add_argument('--seed', type=int, default=42, help='pytorch manual seed')
    parser.add_argument('--write_freq', type=int, default=100, help='frequency of write to file')
    parser.add_argument('--outdir', default='./', type=str, help='output directory')
    parser.add_argument('--checkpoint', default=None, type=str, help='checkpoint directory')
    args = parser.parse_args()
    args.device = torch.device("cuda:"+str(args.gpu) if torch.cuda.is_available() else "cpu")
    args.save_dir = os.path.join(args.outdir, f'train_ImageNet1k_seed{args.seed}_arch{args.arch}_initw{args.init_w_type}_initb{args.init_b_type}_bsz{args.batch_size}_e{args.epochs}')
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    return args


class AvgrageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0/batch_size))
    return res


class CrossEntropyLabelSmooth(torch.nn.Module):

    def __init__(self, num_classes, epsilon):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.logsoftmax = torch.nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (-targets * log_probs).mean(0).sum()
        return loss


def mixup_data(x, y, alpha=1.0, use_cuda=True):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def adjust_lr(optimizer, epoch):
    # Smaller slope for the last 5 epochs because lr * 1/250 is relatively large
    if args.epochs -  epoch > 5:
        lr = args.learning_rate * (args.epochs - 5 - epoch) / (args.epochs - 5)
    else:
        lr = args.learning_rate * (args.epochs - epoch) / ((args.epochs - 5) * 5)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr        


def save_checkpoint(state, is_best, save):
    filename = os.path.join(save, 'checkpoint.pth.tar')
    torch.save(state, filename)
    if is_best:
        best_filename = os.path.join(save, 'model_best.pth.tar')
        shutil.copyfile(filename, best_filename)


def load_checkpoint(save):
    filename = os.path.join(save, 'checkpoint.pth.tar')
    state = torch.load(filename)
    new_state_dict = OrderedDict()
    for k, v in state['state_dict'].items():
        if 'module.' in k[:7]:
            name = k[7:] # remove `module.`
        else:
            name = k
        new_state_dict[name] = v
    return state['epoch'], new_state_dict, state['best_acc_top1'], state['best_acc_top5'], state['optimizer']


def cross_entropy_loss_with_logits(logits, guides):
    soft_target = torch.nn.functional.softmax(guides, dim=1)
    logsoftmax = torch.nn.LogSoftmax()
    return torch.mean(torch.sum(- soft_target * logsoftmax(logits), 1))


def train(train_loader, net, criterion, optimizer, teacher, kd_type, kd_ratio, grad_clip, write_freq):
    objs = AvgrageMeter()
    top1 = AvgrageMeter()
    top5 = AvgrageMeter()
    batch_time = AvgrageMeter()
    net.train()

    for step, (input, target) in enumerate(train_loader):
        input  = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        input, target_a, target_b, lam = mixup_data(input, target, 0.4, True)

        b_start = time.time()
        optimizer.zero_grad()
        logits = net(input)
        loss = mixup_criterion(criterion, logits, target_a, target_b, lam)

        if teacher is not None:
            teacher.train()
            with torch.no_grad():
                guides = teacher(input).detach()
            if kd_type == 'ce':
                kd_loss = cross_entropy_loss_with_logits(logits, guides)
            elif kd_type == 'mse':
                kd_loss = F.mse_loss(logits, guides)
            else:
                raise ValueError('There is no {:} type of kd.'.format(kd_type))
            loss += kd_ratio * kd_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), grad_clip)
        optimizer.step()
        batch_time.update(time.time() - b_start)
        prec1, prec5 = accuracy(logits, target, topk=(1, 5))
        n = input.size(0)
        objs.update(loss.data.item(), n)
        top1.update(prec1.data.item(), n) 
        top5.update(prec5.data.item(), n)

        if step % write_freq == 0:
            end_time = time.time()
            if step == 0:
                duration = 0
                start_time = time.time()
            else:
                duration = int(round(end_time - start_time))
                start_time = time.time()
            logging.info('TRAIN Step: {:03d} Objs: {:e} R1: {:f} R5: {:f} Duration: {:d}s BTime: {:.3f}s'.format(step, objs.avg, top1.avg, top5.avg, duration, batch_time.avg))

    return top1.avg, objs.avg


def infer(val_loader, net, criterion, write_freq):
    objs = AvgrageMeter()
    top1 = AvgrageMeter()
    top5 = AvgrageMeter()
    net.eval()

    for step, (input, target) in enumerate(val_loader):
        input  = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        with torch.no_grad():
            logits = net(input)
            loss = criterion(logits, target)

        prec1, prec5 = accuracy(logits, target, topk=(1, 5))
        n = input.size(0)
        objs.update(loss.data.item(), n)
        top1.update(prec1.data.item(), n)
        top5.update(prec5.data.item(), n)

        if step % write_freq == 0:
            end_time = time.time()
            if step == 0:
                duration = 0
                start_time = time.time()
            else:
                duration = int(round(end_time - start_time))
                start_time = time.time()
            logging.info('VALID Step: {:03d} Objs: {:e} R1: {:f} R5: {:f} Duration: {:d}s'.format(step, objs.avg, top1.avg, top5.avg, duration))

    return top1.avg, top5.avg, objs.avg


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
        start_epoch, net_state_dict, best_acc_top1, best_acc_top5, optimizer_state_dict = load_checkpoint(args.checkpoint)
    else:
        start_epoch, best_acc_top1, best_acc_top5 = 0, 0, 0
        net_state_dict, optimizer_state_dict = None, None

    logging.info('---------Genotype---------')
    genotype = eval('genotypes.{:}'.format(args.arch))
    logging.info(genotype)
    dropout = None if args.dropout <= 0 else args.dropout
    net = get_MobileNetV2_model(genotype[0], num_classes=1000, input_resolution=genotype[1], no_reslink=False, no_BN=False, use_se=True, dropout=dropout)
    if net_state_dict:
        net.load_state_dict(net_state_dict)
    #if args.kd_ratio > 0:
        #net.block_list[2].register_forward_hook(lambda m, i, o: net.features = [o.detach()])
        #net.block_list[4].register_forward_hook(lambda m, i, o: net.features.append(o.detach()))
        #net.block_list[6].register_forward_hook(lambda m, i, o: net.features.append(o.detach()))
        #net.block_list[7].register_forward_hook(lambda m, i, o: net.features.append(o.detach()))
        #net.fc_linear.register_forward_hook(lambda m, i, o: net.features.append(o.detach()))
    logging.info('param size = {:}'.format(net.get_model_size()))
    logging.info('flops      = {:}'.format(net.get_FLOPs()))
    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        net = torch.nn.DataParallel(net)
        net = net.cuda()
    else:
        net = net.cuda()
    logging.info('--------------------------')

    if args.kd_ratio > 0:
        teacher = torchvision.models.resnet152(pretrained=True)
        if num_gpus > 1:
            teacher = torch.nn.DataParallel(teacher)
            teacher = teacher.cuda()
        else:
            teacher = teacher.cuda()
        #teacher.layer1.register_forward_hook(lambda m, i, o: teacher.features = [o.detach()])
        #teacher.layer2.register_forward_hook(lambda m, i, o: teacher.features.append(o.detach()))
        #teacher.layer3.register_forward_hook(lambda m, i, o: teacher.features.append(o.detach()))
        #teacher.layer4.register_forward_hook(lambda m, i, o: teacher.features.append(o.detach()))
        #teacher.fc.register_forward_hook(lambda m, i, o: teacher.features.append(o.detach()))
    else:
        teacher = None

    optimizer = torch.optim.SGD(net.parameters(), args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    if optimizer_state_dict:
        optimizer.load_state_dict(optimizer_state_dict)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs))
    for epoch in range(start_epoch):
        if args.lr_scheduler == 'cosine':
            scheduler.step()

    criterion = torch.nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    criterion_smooth = CrossEntropyLabelSmooth(1000, args.label_smooth)
    criterion_smooth = criterion_smooth.cuda()

    train_loader, val_loader = get_dataloaders(args.batch_size, args.batch_size, 'ImageNet1k', args.num_data_workers)

    for epoch in range(start_epoch, args.epochs):
        if args.lr_scheduler == 'cosine':
            scheduler.step()
            current_lr = scheduler.get_lr()[0]
        elif args.lr_scheduler == 'linear':
            current_lr = adjust_lr(optimizer, epoch)
        else:
            logging.info('Wrong lr type, exit')
            sys.exit(1)
        logging.info('Epoch: {:} lr {:}'.format(epoch, current_lr))

        epoch_start = time.time()
        train_acc, train_obj = train(train_loader, net, criterion_smooth, optimizer, teacher, args.kd_type, args.kd_ratio, args.grad_clip, args.write_freq)
        logging.info('Train_acc: {:}'.format(train_acc))

        valid_acc_top1, valid_acc_top5, valid_obj = infer(val_loader, net, criterion, args.write_freq)
        logging.info('Valid_acc_top1: {:}'.format(valid_acc_top1))
        logging.info('Valid_acc_top5: {:}'.format(valid_acc_top5))
        epoch_duration = time.time() - epoch_start
        logging.info('Epoch time: {:}s.'.format(epoch_duration))

        is_best = False
        if valid_acc_top5 > best_acc_top5:
            best_acc_top5 = valid_acc_top5
        if valid_acc_top1 > best_acc_top1:
            best_acc_top1 = valid_acc_top1
            is_best = True
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': net.state_dict(),
            'best_acc_top1': best_acc_top1,
            'best_acc_top5': best_acc_top5,
            'optimizer' : optimizer.state_dict(),
            }, is_best, args.save_dir)
        if is_best:
            logging.info('Saving the best checkpoint.')
