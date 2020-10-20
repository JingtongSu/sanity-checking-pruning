# ========== Thanks https://github.com/Eric-mingjie/rethinking-network-pruning ============
# ========== we adopt the code from the above link and did modifications ============
# ========== the comments as #=== === were added by us, while the comments as # were the original one ============

from __future__ import print_function

import argparse
import math
import os
import random
import shutil
import time
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from tensorboardX import SummaryWriter
import models as models

from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig
from utils.misc import get_zero_param
from pruner.GraSP import GraSP
from pruner.SNIP import SNIP
from pruner.SmartRatio import SmartRatio

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch CIFAR10/100/TinyImagenet Training')
# Datasets
parser.add_argument('-d', '--dataset', default='cifar10', type=str)
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
# Optimization options
parser.add_argument('--epochs', default=160, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--train-batch', default=64, type=int, metavar='N',
                    help='train batchsize')
parser.add_argument('--test-batch', default=50, type=int, metavar='N',
                    help='test batchsize')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--drop', '--dropout', default=0, type=float,
                    metavar='Dropout', help='Dropout ratio')
parser.add_argument('--schedule', type=int, nargs='+', default=[80, 120],
                        help='Decrease learning rate at these epochs.')
parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
# Checkpoints
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--model', default='', type=str, metavar='PATH',
                    help='path to the initialization checkpoint (default: none)')
# Architecture
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet20',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('--depth', type=int, default=29, help='Model depth.')
# Miscs
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')

parser.add_argument('--save_dir', default='results/', type=str)
#Device options
parser.add_argument('--gpu-id', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')

# ========== these attributes were added by us in order to meet the needs of our experiment ============

# ========== the following 3 attributes have str/float type ============
# ========== for --writerdir, you should name it as Yourfolder/Expname, then set your tensorboard path to Yourfolder/ ============ 
parser.add_argument('--writerdir',default = 'InitExp/', type = str)
# ========== the linear_keep_ratio attribute should be used together with the smart_ratio attribute ============
parser.add_argument('--linear_keep_ratio', type=float, default=0.3, help='smart ratio: linear keep ratio')
# ========== the init_prune_ratio attribute should use together with the smart_ratio/GraSP/SNIP attribute ============
parser.add_argument('--init_prune_ratio', type=float, default=0.98, help='init pruning ratio')

# ========== the following attributes have INT type, but actually they are BOOLEAN: zero or NONZERO ============
parser.add_argument('--rearrange',type = int, default = 0,help = 'rearrange the masks')
parser.add_argument('--shuffle_unmasked_weights',default = 0, type = int)
parser.add_argument('--smart_ratio',default = 0, type = int,help = 'using smart ratio')
parser.add_argument('--GraSP', type=int, default=0, help='Using GraSP')
parser.add_argument('--SNIP', type=int, default=0, help='Using SNIP')
parser.add_argument('--randLabel',type=int, default=0,help = 'Using randLabel Dataset for GraSP/SNIP')
parser.add_argument('--shufflePixel',type=int, default=0,help = 'Using shufflePixel AND RANDLABEL Dataset for GraSP/SNIP')
parser.add_argument('--hybrid',type=int, default=0,help = 'the Hybrid Method, should use with Smart Ratio')
parser.add_argument('--linear_decay',type=int, default=0,help = 'Ablation: Using Linear Decay,should use with Smart Ratio')
parser.add_argument('--ascend',type=int, default=0,help = 'Ablation: Using Ascend Smart Ratio')
parser.add_argument('--uniform',type=int, default=0,help = 'Ablation: Using Balance Keep_Ratio')
parser.add_argument('--cubic',type=int, default=0,help = 'Ablation: Using Cubic Keep_Ratio')

# ========== Can use this BOOLEAN attribute to read in the model in and Run it on the Trainloader to see ACC ============
parser.add_argument('--print_output',default = 0, type = int)

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

# Validate dataset
assert args.dataset == 'cifar10' or args.dataset == 'cifar100' or args.dataset == 'tinyimagenet', 'Dataset can only be cifar10 or cifar100 or tinyimagenet.'


gpu_id = args.gpu_id
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
use_cuda = torch.cuda.is_available()

# Random seed
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 100000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
if use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)

best_acc = 0  # best test accuracy


class CIFAR10RandomLabels(datasets.CIFAR10):
    """CIFAR10 dataset, with support for randomly corrupt labels.
    Params
    ------
    corrupt_prob: float
      Default 1.0. The probability of a label being replaced with
      random label.
    num_classes: int
      Default 10. The number of classes in the dataset.
    """

    def __init__(self, corrupt_prob=1.0, num_classes=10, **kwargs):
        super(CIFAR10RandomLabels, self).__init__(**kwargs)
        self.n_classes = num_classes
        if corrupt_prob > 0:
            self.corrupt_labels(corrupt_prob)

    def corrupt_labels(self, corrupt_prob):
        labels = np.array(self.targets)
        np.random.seed(12345)
        mask = np.random.rand(len(labels)) <= corrupt_prob
        rnd_labels = np.random.choice(self.n_classes, mask.sum())
        labels[mask] = rnd_labels
        # we need to explicitly cast the labels from npy.int64 to
        # builtin int type, otherwise pytorch will fail...
        targets = [int(x) for x in labels]
        self.targets = targets
        
        if args.shufflePixel != 0:
            print('********************* DEBUG PRINT : ADDITION : SHUFFLE PIXEL ************************')
            xs = torch.tensor(self.data)
            Size = xs.size()
            # e.g. for CIFAR10, is 50000 * 32 * 32 * 3
            xs = xs.reshape(Size[0],-1)
            for i in range(Size[0]):
                xs[i] = xs[i][torch.randperm(xs[i].nelement())]
            xs = xs.reshape(Size)
            xs = xs.numpy()
            self.data = xs

class CIFAR100RandomLabels(datasets.CIFAR100):
    """CIFAR100 dataset, with support for randomly corrupt labels.
    Params
    ------
    corrupt_prob: float
      Default 1.0. The probability of a label being replaced with
      random label.
    num_classes: int
      Default 100. The number of classes in the dataset.
    """

    def __init__(self, corrupt_prob=1.0, num_classes=100, **kwargs):
        super(CIFAR10RandomLabels, self).__init__(**kwargs)
        self.n_classes = num_classes
        if corrupt_prob > 0:
            self.corrupt_labels(corrupt_prob)

    def corrupt_labels(self, corrupt_prob):
        labels = np.array(self.targets)
        np.random.seed(12345)
        mask = np.random.rand(len(labels)) <= corrupt_prob
        rnd_labels = np.random.choice(self.n_classes, mask.sum())
        labels[mask] = rnd_labels
        # we need to explicitly cast the labels from npy.int64 to
        # builtin int type, otherwise pytorch will fail...
        targets = [int(x) for x in labels]
        self.targets = targets
        
        if args.shufflePixel != 0:
            print('********************* DEBUG PRINT : ADDITION : SHUFFLE PIXEL ************************')
            xs = torch.tensor(self.data)
            Size = xs.size()
            # e.g. for CIFAR100, is 50000 * 32 * 32 * 3
            xs = xs.reshape(Size[0],-1)
            for i in range(Size[0]):
                xs[i] = xs[i][torch.randperm(xs[i].nelement())]
            xs = xs.reshape(Size)
            xs = xs.numpy()
            self.data = xs

def main():
    global best_acc
    start_epoch = args.start_epoch  # start from epoch 0 or last checkpoint epoch
    if args.print_output == 0:
        writer = SummaryWriter(args.writerdir) 
        os.makedirs(args.save_dir, exist_ok=True)

    # Data
    # ========== The following preprocessing procedure is adopted from https://github.com/alecwangcq/GraSP ============
    print('==> Preparing dataset %s' % args.dataset)
    if args.dataset == 'cifar10':
        transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

        transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
        dataloader = datasets.CIFAR10
        num_classes = 10
    elif args.dataset == 'cifar100':
        dataloader = datasets.CIFAR100
        num_classes = 100
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
    elif args.dataset == 'tinyimagenet':
        args.schedule = [150,225]
        num_classes = 200
        tiny_mean = [0.48024578664982126, 0.44807218089384643, 0.3975477478649648]
        tiny_std = [0.2769864069088257, 0.26906448510256, 0.282081906210584]
        transform_train = transforms.Compose([
            transforms.RandomCrop(64, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(tiny_mean, tiny_std)])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(tiny_mean, tiny_std)])
        args.workers = 16
        args.epochs = 300
    
        


    if args.dataset != 'tinyimagenet':
        trainset = dataloader(root='./data', train=True, download=True, transform=transform_train)
    else:
        trainset = datasets.ImageFolder('./data' + '/tiny_imagenet/train', transform=transform_train)
    
    trainloader = data.DataLoader(trainset, batch_size=args.train_batch, shuffle=True, num_workers=args.workers)
    
    if args.dataset != 'tinyimagenet':
        testset = dataloader(root='./data', train=False, download=False, transform=transform_test)
        testloader = data.DataLoader(testset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)
    else:
        testset = datasets.ImageFolder('./data' + '/tiny_imagenet/val', transform=transform_test)
        testloader = data.DataLoader(testset, batch_size=args.test_batch, shuffle=False,
                                             num_workers=args.workers)
    
    # Model
    print("==> creating model '{}'".format(args.arch))
    if args.arch.endswith('resnet'):
        model = models.__dict__[args.arch](
                    num_classes=num_classes,
                    depth=args.depth,
                )
        model_ref = models.__dict__[args.arch](
                    num_classes=num_classes,
                    depth=args.depth,
                )
    else:
        model = models.__dict__[args.arch](num_classes=num_classes)
        model_ref = models.__dict__[args.arch](num_classes=num_classes)
        
    model.cuda()
    model_ref.cuda()

    cudnn.benchmark = True
    print('    Total Conv and Linear Params: %.2fM' % (sum(p.weight.data.numel() for p in model.modules() if isinstance(p,nn.Linear) or isinstance(p,nn.Conv2d))/1000000.0))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay) # default is 0.001

    # Resume
    if args.dataset == 'cifar10':
        title = 'cifar-10-' + args.arch
    elif args.dataset == 'cifar100':
        title = 'cifar-100-' + args.arch
    else:
        title = 'tinyimagenet' + args.arch
    if args.resume:
        # Load checkpoint.
        print('==> Getting reference model from checkpoint..')
        assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
        checkpoint = torch.load(args.resume, map_location='cpu')
        start_epoch = args.start_epoch
        model_ref.load_state_dict(checkpoint['state_dict'])
        if args.randLabel != 0 or args.shufflePixel != 0:
            assert args.dataset == 'cifar10' or args.dataset == 'cifar100','randLabel/shufflePixel can only be used together with cifar10/100.'
            print('###################### DEBUG PRINT : USING RANDLABEL TO CALCULATE ####################')
            if args.datset == 'cifar10':
                trainset = CIFAR10RandomLabels(root='./data', train=True, download=True, transform=transform_train)
            else:
                trainset = CIFAR100RandomLabels(root='./data', train=True, download=True, transform=transform_train)
            trainloader = data.DataLoader(trainset, batch_size=args.train_batch, shuffle=True, num_workers=args.workers)
    if args.print_output == 0:
        logger = Logger(os.path.join(args.save_dir, 'log_scratch.txt'), title=title)
        logger.set_names(['Learning Rate', 'Train Loss', 'Valid Loss', 'Train Acc.', 'Valid Acc.'])

    # set some weights to zero, according to model_ref ---------------------------------
    if args.model:
        print('==> Loading init model from %s'%args.model)
        checkpoint = torch.load(args.model, map_location='cpu')
        model.load_state_dict(checkpoint['state_dict'])
 
    # ========== the following code is the implementation of Smart Ratio ============
    if args.smart_ratio != 0:
        print("################### DEBUG PRINT : USING SMART RATIO ###################")
        masks = SmartRatio(model,args.init_prune_ratio,'cuda',args)
        
    # ========== the following code is the implementation of GraSP ============
    if args.GraSP != 0:
        print("################### DEBUG PRINT : USING GraSP ###################")
        # ========== If use ResNet56, there will be risk to meet the CUDA OUT OF MEMORY ERROR ============
        samples_per_class = 10
        num_iters = 1
        if args.arch == 'resnet' and args.depth > 32:
            samples_per_class = 1
            num_iters = 10
        if args.dataset == 'tinyimagenet':
            samples_per_class = 1
            num_iters = 10
        if args.randLabel != 0 or args.shufflePixel != 0:
            assert args.dataset == 'cifar10' or args.dataset == 'cifar100','randLabel/shufflePixel can only be used together with cifar10/100.'
            print('###################### DEBUG PRINT : USING RANDLABEL TO CALCULATE ####################')
            if args.dataset == 'cifar10':
                randset = CIFAR10RandomLabels(root='./data', train=True, download=True, transform=transform_train)
            else:
                randset = CIFAR100RandomLabels(root='./data', train=True, download=True, transform=transform_train)
            randloader = data.DataLoader(randset, batch_size=args.train_batch, shuffle=True, num_workers=args.workers)
            masks = GraSP(model, args.init_prune_ratio, randloader, 'cuda',num_classes,samples_per_class,num_iters)
        else:
            masks = GraSP(model, args.init_prune_ratio, trainloader, 'cuda',num_classes,samples_per_class,num_iters)
        
    # ========== the following code is the implementation of SNIP ============
    if args.SNIP != 0:
        print("################### DEBUG PRINT : USING SNIP ###################")
        # ========== If use ResNet56, there will be risk to meet the CUDA OUT OF MEMORY ERROR ============
        samples_per_class = 10
        num_iters = 1
        if args.arch == 'resnet' and args.depth > 32:
            samples_per_class = 1
            num_iters = 10
        if args.dataset == 'tinyimagenet':
            samples_per_class = 1
            num_iters = 10
        if args.randLabel != 0 or args.shufflePixel != 0:
            assert args.dataset == 'cifar10' or args.dataset == 'cifar100','randLabel/shufflePixel can only be used together with cifar10/100.'
            print('###################### DEBUG PRINT : USING RANDLABEL TO CALCULATE ####################')
            if args.dataset == 'cifar10':
                randset = CIFAR10RandomLabels(root='./data', train=True, download=True, transform=transform_train)
            else:
                randset = CIFAR100RandomLabels(root='./data', train=True, download=True, transform=transform_train)
            randloader = data.DataLoader(randset, batch_size=args.train_batch, shuffle=True, num_workers=args.workers)
            masks = SNIP(model, args.init_prune_ratio, randloader, 'cuda',num_classes,samples_per_class,num_iters)
        else:
            masks = SNIP(model, args.init_prune_ratio, trainloader, 'cuda',num_classes,samples_per_class,num_iters)
    
    CNT = 0
    
    for m,m_ref in zip(model.modules(),model_ref.modules()):
                
        if isinstance(m, nn.Conv2d) or isinstance(m,nn.Linear):
            if isinstance(m,nn.Conv2d):
                TYPE = "Conv"
            else:
                TYPE = "Linear"
            weight_copy = m_ref.weight.data.abs().clone()
            # DEFAULT : generate the masks from model_ref, i.e. the LT method
            mask = weight_copy.gt(0).float().cuda()
            
            # Else : generate the masks using the Smart Ratio / GraSP / SNIP
            
            # ========== set the Smart Ratio / GraSP / SNIP masks ============
            if args.smart_ratio != 0:
                mask = masks[CNT]
            elif args.GraSP != 0:
                mask = masks[m]
            elif args.SNIP != 0:
                mask = masks[m]
            CNT += 1
            total = mask.numel()
            
            # ========== print the keep-ratio and #para, #remained ============
            remained = int(torch.sum(mask))
            keep_ratio = remained/total
            print("LAYER %d(%s) : KEEP_RATIO = %.6f    NUM_PARA = %d    REMAINED_PARA = %d" % (CNT,TYPE,keep_ratio*100,total,remained))    
            
            # ========== rearrange the masks (if stated) ============
            # ========== note that this operation will also change the weight retained ============
            if args.rearrange != 0:
                print("################### DEBUG PRINT : REARRANGE ###################")
                mask = mask.view(-1)[torch.randperm(mask.nelement())].view(mask.size())
            
            # ========== set the pruned weights to 0 ============
            m.weight.data.mul_(mask)
            
            # ========== Ablation study: Shuffle Weights ============
            # ========== shuffle the unmasked weights (if stated) ============
            # ========== we keep the arch but change the position of the weight ============
            if args.shuffle_unmasked_weights != 0:
                print("################### DEBUG PRINT : SHUFFLE UNMASKED WEIGHTS ###################")
                Size = mask.size()
                mask = mask.view(-1)
                m.weight.data = m.weight.data.view(-1)
                non_zero = int(sum(mask).item())
                value,idx = torch.topk(mask,non_zero)
                rand_idx = idx.view(-1)[torch.randperm(idx.nelement())].view(idx.size())
                
                m.weight.data[rand_idx] = m.weight.data[idx]
                
                mask = mask.view(Size)
                m.weight.data = m.weight.data.view(Size)
                
    # ========== print the training acc and RETURN (if stated) ============           
    if args.print_output != 0:
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            # measure data loading time
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

            # compute output
            outputs = model(inputs)
            print(outputs)
            return
            loss = criterion(outputs, targets)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
            losses.update(loss.data.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))
            top5.update(prec5.item(), inputs.size(0))
            
        print("Train acc : {}".format(top1.avg))
        return
            
            
    # Train and val
    for epoch in range(start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)
        num_parameters = 0
        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, state['lr']))
        
        # ========== calculate #ZERO paras (including zero Conv weights and zero Linear weights) ============ 
        ZERO_parameters = get_zero_param(model)
        print('Zero parameters: {}'.format(ZERO_parameters))
        
        # ========== calculate #paras (including Conv weights and Linear weights) ============ 
        for m in model.modules():
            if isinstance(m,nn.Conv2d) or isinstance(m,nn.Linear):
                num_parameters += m.weight.data.numel()
                
        # ========== print the #weights information at every epoch to make sure the pruning pipeline is executed ============
        print('Parameters: {}'.format(num_parameters))
        print('Overall Pruning Ratio : {}'.format(float(ZERO_parameters)/float(num_parameters)))
        train_loss, train_acc = train(trainloader, model, criterion, optimizer, epoch, use_cuda)
        test_loss, test_acc = test(testloader, model, criterion, epoch, use_cuda)
        
        # ========== write the scalar to tensorboard ============ 
        writer.add_scalar('train_loss', train_loss,epoch)
        writer.add_scalar('test_loss',test_loss,epoch)
        writer.add_scalar('train_acc', train_acc,epoch)
        writer.add_scalar('test_acc', test_acc,epoch)
        
        # append logger file
        logger.append([state['lr'], train_loss, test_loss, train_acc, test_acc])

        # save model
        is_best = test_acc > best_acc
        best_acc = max(test_acc, best_acc)
        save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'acc': test_acc,
                'best_acc': best_acc,
                'optimizer' : optimizer.state_dict(),
            }, is_best, checkpoint=args.save_dir)
    logger.close()
    writer.close()
    print('Best acc:')
    print(best_acc)



def train(trainloader, model, criterion, optimizer, epoch, use_cuda):
    # switch to train mode
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    bar = Bar('Processing', max=len(trainloader))
    print(args)
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

        # compute output

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.data.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()

        for k, m in enumerate(model.modules()):
            # print(k, m)
            if isinstance(m, nn.Conv2d) or isinstance(m,nn.Linear):
                weight_copy = m.weight.data.abs().clone()
                mask = weight_copy.gt(0).float().cuda()
                m.weight.grad.data.mul_(mask)
        optimizer.step()


        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                    batch=batch_idx + 1,
                    size=len(trainloader),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg,
                    top1=top1.avg,
                    top5=top5.avg,
                    )
        bar.next()
    bar.finish()
    print("Train acc : {}".format(top1.avg))
    return (losses.avg, top1.avg)

def test(testloader, model, criterion, epoch, use_cuda):
    global best_acc

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    bar = Bar('Processing', max=len(testloader))
    for batch_idx, (inputs, targets) in enumerate(testloader):
        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = torch.autograd.Variable(inputs, volatile=True), torch.autograd.Variable(targets)

        # compute output
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.data.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                    batch=batch_idx + 1,
                    size=len(testloader),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg,
                    top1=top1.avg,
                    top5=top5.avg,
                    )
        bar.next()
    bar.finish()
    print("Test acc : {}".format(top1.avg))
    return (losses.avg, top1.avg)

def save_checkpoint(state, is_best, checkpoint, filename='scratch.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)

def adjust_learning_rate(optimizer, epoch):
    global state
    if epoch in args.schedule:
        state['lr'] *= args.gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] = state['lr']

if __name__ == '__main__':
    main()
