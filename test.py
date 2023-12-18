#!/usr/bin/env python

import os
import dill 
import argparse
import numpy as np
import torch as ch
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from robustness import model_utils, datasets, train, defaults
from robustness.train import make_optimizer_and_schedule
from robustness.tools import helpers
from robustness.tools.helpers import AverageMeter
from cox.utils import Parameters
import cox.store
from tqdm import tqdm
from data_utils import *
from scipy.spatial.distance import pdist, squareform


def cal_acc_per_class(outputs, labels, class_correct, class_total, class_pre, class_detail):

    _, predicted = ch.max(outputs.data, 1)
    c = (predicted == labels).squeeze()
    for i in range(labels.size(0)):
        label = int(labels[i].item())
        pre = int(predicted[i].item())
        class_correct[label] += c[i].item()
        class_total[label] += 1
        class_pre[pre] += 1
        class_detail[pre][label] += 1
    return class_correct, class_total, class_pre, class_detail

def eval_model(test_loader, model):
    class_num = 10
    class_correct = { i:0 for i in range(class_num) }
    class_total = { i:0 for i in range(class_num) }
    class_pre =  { i:0 for i in range(class_num) }
    class_detail = dict()
    for i in range(class_num):
        class_detail[i] = [0 for j in range(class_num)]
    class_fea = { i:[] for i in range(class_num)}
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    activate = []
    y = []
    pred = []
    criterion = ch.nn.CrossEntropyLoss()
    model = model.eval()
    test_iterator = tqdm(enumerate(test_loader), total=len(test_loader))
    # with ch.no_grad():
    for i, (inp, target) in test_iterator:
        target = ch.LongTensor(target)
        target = target.cuda()
        inp = inp.cuda()
        (output,features), _ = model(inp, target=target, with_latent=True)

        loss = criterion(output, target)

        model_logits = output
        _, pred_label = ch.max(model_logits.data, 1)
        maxk = min(5, model_logits.shape[-1])
        prec1, prec5 = helpers.accuracy(model_logits, target, topk=(1, maxk))
        prec1, prec5 = prec1[0], prec5[0]

        class_correct,class_total,class_pre,class_detail = cal_acc_per_class(model_logits, target, class_correct, class_total,class_pre, class_detail)

        losses.update(loss.item(), inp.size(0))
        top1.update(prec1, inp.size(0))
        top5.update(prec5, inp.size(0))

        top1_acc = top1.avg
        top5_acc = top5.avg


        # features = features.cpu().detach()
        # activate.append(features)
        # y.append(target.cpu())
        # pred.append(pred_label.cpu())

        desc = ('Test: Loss {loss.avg:.4f} | '
                'NatPrec1 {top1_acc:.3f} | NatPrec5 {top5_acc:.3f} | '.format( 
                loss=losses, top1_acc=top1_acc, top5_acc=top5_acc))
        test_iterator.set_description(desc)
        test_iterator.refresh()

    print(class_pre)
    class_acc = []
    class_norm = []
    for c in class_correct:
        class_acc.append(round(class_correct[c] / class_total[c],2))
    for i in range(class_num):
        print(class_detail[i])
    print(class_acc)
    # fea = ch.cat(activate).numpy()
    # pred = ch.cat(pred).numpy()
    # print(fea.shape, pred.shape)
    print("=> Test Acc is:{}".format(top1.avg))
    # np.save('/data/zhiyu/research/f_exp/features/cifar10/resnet18_activate_'+_type+'.npy',fea)
    # np.save('/data/zhiyu/research/imRecog/freq_pred',pred)
    # print('save ok!')

       

    
parser = argparse.ArgumentParser(description='train configs')
parser.add_argument('--dataset' ,help='the used dataset')
parser.add_argument('--arch' , default='resnet18', help='the used model')
parser.add_argument('--freq', default=None, help='frequecy')
parser.add_argument('--r', default='4', help='radius of frequecy domain')
parser.add_argument('--gpu', default="0", help='choose gpu to train')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

DATA_PATH_DICT = {'cifar10': '/data/cifar','cifar100':'/data/cifar','resIN':'/data/ILSVRC2012'}
DATA_NAME_DICT = {'cifar10':'CIFAR', 'cifar100':'CIFAR100','resIN':'RestrictedImageNet'}
DEVICES_ID = None
if len(args.gpu)>1: 
    DEVICES_ID = [int(id) for id in args.gpu.split(',')]
# configs
ARCH = args.arch
FREQ  = args.freq
RADIUS = args.r
DATA = args.dataset
dataset_function = getattr(datasets, DATA_NAME_DICT[DATA])
dataset = dataset_function(DATA_PATH_DICT[DATA])


# DATA loader
print("preparing data {}...".format(DATA))
train_loader, test_loader = get_dataloader(DATA, FREQ, RADIUS)
print('data loader is ok :)')

# prepare model
print("preparing model {}...".format(ARCH))
model_kwargs = {
    'arch': ARCH,
    'dataset': dataset,
    'resume_path':'/data/zhiyu/research/f_exp/'+DATA+'/'+ARCH+'_baseline/checkpoint.pt.best'
}
model, _ = model_utils.make_and_restore_model(**model_kwargs)
print('model is ok :)')

eval_model(test_loader,model)
