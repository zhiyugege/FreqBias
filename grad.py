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
import random
import matplotlib.pyplot as plt
import seaborn as sns

def azimuthalAverage(image, center=None):
   
    y, x = np.indices(image.shape)

    if not center:
        center = np.array([(x.max()-x.min())/2.0, (y.max()-y.min())/2.0])

    r = np.hypot(x - center[0], y - center[1])
    ind = np.argsort(r.flat)
    r_sorted = r.flat[ind]
    i_sorted = image.flat[ind]
    r_int = r_sorted.astype(int)
    deltar = r_int[1:] - r_int[:-1]  # Assumes all radii represented
    rind = np.where(deltar)[0] # location of changed radius
    nr = rind[1:] - rind[:-1]        # number of radius bin
    
    # Cumulative sum to figure out sums for each radius bin
    csim = np.cumsum(i_sorted, dtype=float)
    tbin = csim[rind[1:]] - csim[rind[:-1]]
    radial_prof = tbin / nr
    return radial_prof

def cal_power(grad):
    epsilon = 1e-8
    psd1D_total = []
    grad = grad.transpose(0,2,3,1)
    for i in range(len(grad)):
        data = grad[i]
        data = np.sum(data,axis=2)
        f = np.fft.fft2(data)
        fshift = np.fft.fftshift(f)

        magnitude_spectrum = np.abs(fshift)
        psd1D = azimuthalAverage(magnitude_spectrum)
        psd1D = (psd1D-np.min(psd1D))/(np.max(psd1D)-np.min(psd1D))
        psd1D_total.append(psd1D)
        # psd1D_total[i,:] = psd1D
    psd1D_total = np.array(psd1D_total)
    _dim = psd1D_total.shape[1]
    psd1D_org_mean = np.zeros(_dim)
    psd1D_org_std = np.zeros(_dim)
    for x in range(_dim):
        psd1D_org_mean[x] = np.mean(psd1D_total[:,x])
        psd1D_org_std[x] = np.std(psd1D_total[:,x])

    return psd1D_org_mean,psd1D_org_std

def eval_model(test_loader, model):


    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    grad_arr = []
    criterion = ch.nn.CrossEntropyLoss()
    model = model.eval()
    test_iterator = tqdm(enumerate(test_loader), total=len(test_loader))
    # with ch.no_grad():
    
    for i, (inp, target) in test_iterator:
        target = target.cuda()
        inp = inp.cuda().requires_grad_(True)
        output,features = model(inp, target=target, with_latent=True,with_image=False)

        loss = criterion(output, target)
        loss.backward()

        model_logits = output
        value, pred_label = ch.max(model_logits.data, 1)
        maxk = min(5, model_logits.shape[-1])
        prec1, prec5 = helpers.accuracy(model_logits, target, topk=(1, maxk))
        prec1, prec5 = prec1[0], prec5[0]

        class_correct,class_total,class_pre,class_detail = cal_acc_per_class(model_logits, target, class_correct, class_total,class_pre, class_detail)

        losses.update(loss.item(), inp.size(0))
        top1.update(prec1, inp.size(0))
        top5.update(prec5, inp.size(0))

        top1_acc = top1.avg
        top5_acc = top5.avg

        grad = inp.grad.cpu().detach()
        grad_arr.append(grad)
       

        desc = ('Test: Loss {loss.avg:.4f} | '
                'NatPrec1 {top1_acc:.3f} | NatPrec5 {top5_acc:.3f} | '.format( 
                loss=losses, top1_acc=top1_acc, top5_acc=top5_acc))
        test_iterator.set_description(desc)
        test_iterator.refresh()

    grad_arr = ch.cat(grad_arr).numpy()
    print(grad_arr.shape)
    return grad_arr
  
       


    
parser = argparse.ArgumentParser(description='train configs')
parser.add_argument('--dataset' ,help='the used dataset')
parser.add_argument('--arch' , default='resnet18', help='the used model')
parser.add_argument('--gpu', help='choose gpu to train')
args = parser.parse_args()



os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
DATA_PATH_DICT = {'cifar10': '/data/zhiyu/cifar10','resIN':'/data/zhiyu/dddd/data/ILSVRC2012/'}
DATA_NAME_DICT = {'cifar10':'CIFAR', 'cifar100':'CIFAR100','resIN':'RestrictedImageNet'}

DATA = args.dataset 
ARCH = args.arch

dataset_function = getattr(datasets, DATA_NAME_DICT[DATA])
dataset = dataset_function(DATA_PATH_DICT[DATA])

print("preparing data {}...".format(args.dataset))
_, test_loader = dataset.make_loaders(workers=8, batch_size=128)

print('=> dataset is ok:)')

grad_save = {'mean':[],'std':[]}
_index = [i for i in range(0,100)]
# _index = [i for i in range(0,200,5)]+[199]
for k in _index:
    model_kwargs = {
    'arch': ARCH,
    'dataset': dataset,

    'resume_path':'/data/zhiyu/research/f_exp/'+DATA+'/'+ARCH+'_baseline/'+str(k)+'_checkpoint.pt'
    }

    model, checkpoint = model_utils.make_and_restore_model(**model_kwargs)
    print('=> model is ok:)')  
    grad_epoch = eval_model(test_loader,model)
    grad_mean, grad_std = cal_power(grad_epoch)
    grad_save['mean'].append(grad_mean)
    grad_save['std'].append(grad_std)

grad_save['mean'] = np.array(grad_save['mean'])
grad_save['std'] = np.array(grad_save['std'])

np.save('/data/zhiyu/research/imRecog/grad/imagenet/resnet50_inp_grad_power_epo100',grad_save)

mean_grad = grad_save['mean']
_mean_grad = (mean_grad-np.min(mean_grad))/(np.max(mean_grad)-np.min(mean_grad))

plt.figure(figsize=(10,8),dpi=200)
xticklabels = [0, 4, 9, 14, 19]
xticks, idx = [], 0
for i in range(20):
    if xticklabels[idx] == i:
        xticks.append(i)
        idx += 1
    else:
        xticks.append(' ')
ax = sns.heatmap(_mean_grad,annot=False,cbar=True,vmin=0, vmax=1,yticklabels=yticks, xticklabels=xticks) 
ax.invert_yaxis()
cbar = ax.collections[0].colorbar
cbar.set_ticks([0, 1])
cbar.ax.set_yticklabels(['0.0', '1.0'], size=30)
cbar.ax.tick_params(labelsize=25)
plt.xlabel('Frequency bands', fontsize=30)
plt.ylabel('Training epochs', fontsize=30)
plt.tick_params(labelsize=30)  
plt.show()

