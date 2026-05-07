#!/usr/bin/env python

# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import argparse
import builtins
import math
import os
import random
import shutil
import time
import warnings

import moco.builder
import moco.loader
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from model.randstainna import RandStainNA
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from albumentations import (RandomRotate90, GridDistortion, HueSaturationValue, ISONoise, GaussNoise,
                            RandomGamma, RandomBrightnessContrast)
from PIL import Image
from PIL import ImageFilter
from RandAugment import RandAugment
from torch.utils.data import DataLoader, ConcatDataset

model_names = sorted(
    name
    for name in models.__dict__
    if name.islower() and not name.startswith("__") and callable(models.__dict__[name])
)

parser = argparse.ArgumentParser(description="PyTorch ImageNet Training")
parser.add_argument("--data",default = './DATA/GDPH-CRC-HE-MS/S01', metavar="DIR", help="path to dataset")
parser.add_argument("--root",default = './DATA/GDPH-CRC-HE-MS/S01', metavar="DIR", help="path to dataset")
parser.add_argument("--source_domain",default = '/S01', type=str)
parser.add_argument("--files", default=['preprocess//S01_LAB.yaml', 'preprocess//S01_HSV.yaml', 'preprocess//S01_HED.yaml'])

parser.add_argument(
    "-a",
    "--arch",
    metavar="ARCH",
    default="resnet18", 
    choices=model_names,
    help="model architecture: " + " | ".join(model_names) + " (default: resnet50)",
)
parser.add_argument(
    "-j",
    "--workers",
    default=16,
    type=int,
    metavar="N",
    help="number of data loading workers (default: 32)",
)
parser.add_argument(
    "--epochs", default=100, type=int, metavar="N", help="number of total epochs to run"
)
parser.add_argument(
    "--start-epoch",
    default=0,
    type=int,
    metavar="N",
    help="manual epoch number (useful on restarts)",
)
parser.add_argument(
    "-b",
    "--batch-size",
    default=256,
    type=int,
    metavar="N",
    help="mini-batch size (default: 256), this is the total "
    "batch size of all GPUs on the current node when "
    "using Data Parallel or Distributed Data Parallel",
)
parser.add_argument(
    "--lr",
    "--learning-rate",
    default=0.03,
    type=float,
    metavar="LR",
    help="initial learning rate",
    dest="lr",
)
parser.add_argument(
    "--schedule",
    default=[120, 160],
    nargs="*",
    type=int,
    help="learning rate schedule (when to drop lr by 10x)",
)
parser.add_argument(
    "--momentum", default=0.9, type=float, metavar="M", help="momentum of SGD solver"
)
parser.add_argument(
    "--wd",
    "--weight-decay",
    default=1e-4,
    type=float,
    metavar="W",
    help="weight decay (default: 1e-4)",
    dest="weight_decay",
)
parser.add_argument(
    "-p",
    "--print_freq",
    default=350,
    type=int,
    metavar="N",
    help="print frequency (default: 10)",
)
parser.add_argument(
    "--resume",
    default="",
    type=str,
    metavar="PATH",
    help="path to latest checkpoint (default: none)",
)
parser.add_argument(
    "--world-size",
    default=-1,
    type=int,
    help="number of nodes for distributed training",
)
parser.add_argument(
    "--rank", default=-1, type=int, help="node rank for distributed training"
)
parser.add_argument(
    "--dist-url",
    default="tcp://224.66.41.62:23456",
    type=str,
    help="url used to set up distributed training",
)
parser.add_argument(
    "--dist-backend", default="nccl", type=str, help="distributed backend"
)
parser.add_argument(
    "--seed", default=None, type=int, help="seed for initializing training. "
)
parser.add_argument("--gpu", default=0, type=int, help="GPU id to use.")
parser.add_argument(
    "--multiprocessing-distributed",
    default = False,
    action="store_true",
    help="Use multi-processing distributed training to launch "
    "N processes per node, which has N GPUs. This is the "
    "fastest way to use PyTorch for either single node or "
    "multi node data parallel training",
)

# moco specific configs:
parser.add_argument(
    "--moco-dim", default=128, type=int, help="feature dimension (default: 128)"
)
parser.add_argument(
    "--moco-k",
    default=65536,
    type=int,
    help="queue size; number of negative keys (default: 65536)",
)
parser.add_argument(
    "--moco-m",
    default=0.999,
    type=float,
    help="moco momentum of updating key encoder (default: 0.999)",
)
parser.add_argument(
    "--moco-t", default=0.07, type=float, help="softmax temperature (default: 0.07)"
)

# options for moco v2
parser.add_argument("--mlp", default=True, action="store_true", help="use mlp head")
parser.add_argument("--cos", default=True, action="store_true", help="use cosine lr schedule")
parser.add_argument("--randstainna", default=True, help='whether to use randstainna')



def main():
    args = parser.parse_args()
    args.files=['preprocess/'+args.source_domain+'_LAB.yaml',
                'preprocess/'+args.source_domain+'_HSV.yaml',
                'preprocess/'+args.source_domain+'_HED.yaml']
    args.experiment_name = args.source_domain
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn(
            "You have chosen to seed training. "
            "This will turn on the CUDNN deterministic setting, "
            "which can slow down your training considerably! "
            "You may see unexpected behavior when restarting "
            "from checkpoints."
        )

    ngpus_per_node = 1

    main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu


    print("Use GPU: {} for training".format(args.gpu))

    # create model
    print("=> creating model '{}'".format(args.arch))
    model = moco.builder.MoCo(
        models.__dict__[args.arch],
        args.moco_dim,
        args.moco_k,
        args.moco_m,
        args.moco_t,
        args.mlp, 
    )
    print(model)

    torch.cuda.set_device(args.gpu)
    model = model.cuda(args.gpu)
    # comment out the following line for debugging
    # raise NotImplementedError("Only DistributedDataParallel is supported.")

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    optimizer = torch.optim.SGD(
        model.parameters(),
        args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    # optionally resume from a checkpoint
    ## 加预训练参数
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = "cuda:{}".format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint["epoch"]
            model.load_state_dict(checkpoint["state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            print(
                "=> loaded checkpoint '{}' (epoch {})".format(
                    args.resume, checkpoint["epoch"]
                )
            )
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    dataset_paths = args.root.split(':')
    # dataset_paths = args.data
    # cudnn.benchmark = False
    # Data loading code
    # traindir = args.data
    # k不进行randstainna的变换,我觉得甚至可以不进行colorjitter。不行的话还是和q1一样
    augmentation_k = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
        transforms.RandomApply(
            [transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8  # not strengthened
        ),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([moco.loader.GaussianBlur([0.1, 2.0])], p=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Randstain
    augmentation_q1 = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
        transforms.RandomApply(
            [transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8  # not strengthened
        ),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([moco.loader.GaussianBlur([0.1, 2.0])], p=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Texture
    augmentation_q2 = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(),
        PILtoNumpy(),
        ApplyOnKey(func=RandomRotate90(p=1.0), key='image'),

        ApplyOnKey(func=RandomGamma(p=0.3, gamma_limit=(80, 120)), key='image'),
        ApplyOnKey(func=GridDistortion(p=0.3, num_steps=5, distort_limit=(-0.3, 0.3)), key='image'),
        ApplyOnKey(func=ISONoise(p=0.3, intensity=(0.1, 0.5), color_shift=(0.01, 0.05)), key='image'),
        ApplyOnKey(func=GaussNoise(p=0.3, var_limit=(10.0, 50.0)), key='image'),
        NumpyToPIL(),
        transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    dataset_paths = args.root.split(':')
    # dataset_paths=args.data
    train_dataset = []
    augmentation_q2.transforms.insert(0, RandAugment(3, 30))
    for i, p in enumerate(dataset_paths):
        # Create dataset
        # logger.debug("Load dataset {} from: {}".format(i, p))
        dataset_train_ = datasets.ImageFolder(
            p, moco.loader.ThreeCropsTransform(augmentation_q1,augmentation_q2,augmentation_k)
        )
        # Remap dataset labels to current dataset index

        # Limit size of dataset to the expected number of samples
        train_dataset.append(dataset_train_)

    train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        ConcatDataset(train_dataset),
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        num_workers=args.workers,
        pin_memory=True,
        sampler=train_sampler,
        drop_last=True,
    )
    train(train_loader, model, criterion, optimizer, args, augmentation_q2,augmentation_k)

def train(train_loader, model, criterion, optimizer, args,augmentation_q2,augmentation_k):
    
    best_loss = np.inf
    for epoch in tqdm(range(args.start_epoch, args.epochs)):
        # if epoch == 150:
        #     stage = 2  # 从第49个epoch开始,stage的值是0,1,2
        #     number = 0   # 从第49个epoch开始,number0-4三个循环
        #     dataset = train_loader.dataset
        #     dataset.transform = moco.loader.ThreeCropsTransform((transforms.Compose([
        #     transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
        #     RandStainNA(yaml_file=args.files[stage], ran=number, std_hyper=-0.4, probability=1.0, distribution='normal', is_train=True), # ran是第几个阶段
        #     transforms.RandomGrayscale(p=0.2),
        #     transforms.RandomApply(
        #         [transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8  # not strengthened
        #     ),
        #     # transforms.RandomApply([transforms.GaussianBlur([.1, 2.])], p=0.5),
        #     transforms.RandomHorizontalFlip(),
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),        
        #     ])),                           
        #     augmentation_q2,
        #     augmentation_k)  
        if epoch>48 and (epoch+1) % 10 == 0 and epoch != 199:
            if (epoch+1-50) % 50 == 0 and epoch != 199:
                stage = (epoch+1-50) // 50  # 从第49个epoch开始,stage的值是0,1,2
            number = (epoch+1-50-stage*50)//10   # 从第49个epoch开始,number0-4三个循环
            dataset = train_loader.dataset
            dataset.transform = moco.loader.ThreeCropsTransform((transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
            RandStainNA(yaml_file=args.files[stage], ran=number, std_hyper=-0.4, probability=1.0, distribution='normal', is_train=True), # ran是第几个阶段
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply(
                [transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8  # not strengthened
            ),
            # transforms.RandomApply([transforms.GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),        
            ])),
            augmentation_q2,
            augmentation_k)                                                                                                                                                                
        
        adjust_learning_rate(optimizer, epoch, args)
        # train for one epoch
        batch_time = AverageMeter("Time", ":6.3f")
        data_time = AverageMeter("Data", ":6.3f")
        losses = AverageMeter("Loss", ":.4e")
        top1_1 = AverageMeter("Acc@1", ":6.2f")
        top1_2 = AverageMeter("Acc@1", ":6.2f")
        # top5 = AverageMeter("Acc@5", ":6.2f")
        progress = ProgressMeter(
            len(train_loader),
            [batch_time, data_time, losses, top1_1, top1_2],
            prefix="Epoch: [{}]".format(epoch),
        )

        # switch to train mode
        model.train()

        end = time.time()
        for i, (images, _) in enumerate(train_loader):
            # measure data loading time
            data_time.update(time.time() - end)

            if args.gpu is not None:
                images[0] = images[0].cuda(args.gpu, non_blocking=True) # q1
                images[1] = images[1].cuda(args.gpu, non_blocking=True) # q2
                images[2] = images[2].cuda(args.gpu, non_blocking=True) # k

            # compute output
            output1, output2, target1, target2 = model(im_q1=images[0], im_q2=images[1], im_k=images[2])
            loss1 = criterion(output1, target1)
            loss2 = criterion(output2, target2)
            loss = loss1+loss2
            # acc1/acc5 are (K+1)-way contrast classifier accuracy
            # measure accuracy and record loss
            acc1_1, acc5_1 = accuracy(output1, target1, topk=(1, 5))
            acc1_2, acc5_2 = accuracy(output2, target2, topk=(1, 5))
            losses.update(loss.item(), images[0].size(0))
            top1_1.update(acc1_1[0], images[0].size(0))
            top1_2.update(acc1_2[0], images[1].size(0))
            # top5.update(acc5[0], images[0].size(0))

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)
            if loss < best_loss:
                save_checkpoint(
                    {
                        "epoch": epoch + 1,
                        "arch": args.arch,
                        "state_dict": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                    },
                    is_best = False,
                    filename="./checkpoints/Phase1_"+args.source_domain+".pth.tar"
                )
            if (epoch+1) % 50 == 0:
                save_checkpoint(
                    {
                        "epoch": epoch + 1,
                        "arch": args.arch,
                        "state_dict": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                    },
                    is_best = False,
                    filename='./checkpoints/'+args.experiment_name + "_{:04d}.pth.tar".format(epoch),
                    )


def save_checkpoint(state, is_best, filename="checkpoint.pth.tar"):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, "randstainna_HSV_best.pth.tar")


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)

class ApplyOnKey:

    def __init__(self, func, key):
        self.func = func
        self.key = key

    def __call__(self, x):
        data = {self.key: x}
        return self.func(**data)[self.key]

    def __repr__(self):
        return self.func.__repr__()


class PILtoNumpy:
    def __call__(self, x):
        return np.array(x)

    def __repr__(self):
        return self.__class__.__name__ + '()'
    
class NumpyToPIL:
    def __call__(self, x):
        return Image.fromarray(x.astype(np.uint8))

    def __repr__(self):
        return self.__class__.__name__ + '()'
    
class ProgressMeter:
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print("\t".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"
    
class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

    def __repr__(self):
        return self.__class__.__name__ + '(sigma={})'.format(self.sigma)


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    if args.cos:  # cosine lr schedule
        lr *= 0.5 * (1.0 + math.cos(math.pi * epoch / args.epochs))
    else:  # stepwise lr schedule
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.0
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == "__main__":
    import torch.multiprocessing
    torch.multiprocessing.set_sharing_strategy('file_system')
    main()
