# -*- encoding: utf-8 -*-
'''
-----------------------------------
    File name:   toy_dataset.py
    Time:        2022/09/22 14:46:28
    Editor:      ZhiyuGeGe & Figo
-----------------------------------
'''

import os
import torch
import torch.nn as nn
import torchvision as tv
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils import min_max_normalize, visual_spectral_density
from robustness.datasets import*
from robustness.tools.helpers import AverageMeter


class toy_kernel(nn.Module):
    def __init__(self, number, kernel_size, channels):
        """
        Args:
            :: kernel_size: the size of convolutional kernel
            :: channels: the channels of an image
        Return:
            :: the fake image with has the same shape of x (B x C x H x W)
        """
        super(toy_kernel, self).__init__()

        if not kernel_size % 2 == 1: 
            raise ValueError("Expected kernel size is odd, but got {}".format(kernel_size))
        
        self.kernel_size = kernel_size
        self.padding = int((kernel_size - 1) / 2)
        self.kernels = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=number, kernel_size=self.kernel_size, stride=1, padding=self.padding),
            nn.Conv2d(in_channels=number, out_channels=channels, kernel_size=self.kernel_size, stride=1, padding=self.padding),
        )
    
    def forward(self, x):
        return self.kernels(x)


def model_loop(loop_type, model, loader, image_density, opt, epoch, writer=None):
    """
    Args:
        :: loop_type: the type of loop ("train", "val")
        :: model: a set of kernels which change the spectral density of input image 
        :: loader: the image loader  (data.DataLoader)
        :: image_density: the target density function (specificly double peak density)
        :: opt: the optimizer
        :: epoch: the current epoch
    Return:
        :: loss and spectral density
    """
    model.train() if loop_type == "train" else model.eval()
    
    """ ----------- loss function ----------- """
    spectral_criterion = nn.MSELoss().cuda()                                                                                # spectrum-align loss
    image_criterion = nn.MSELoss().cuda()                                                                                   # image classify loss

    iterator = tqdm(enumerate(loader), total=len(loader))  # dataloader
    lr = opt.param_groups[0]['lr'] if loop_type == "train" else 0
    losses = AverageMeter()

    for i, (inp, _) in iterator:
        true_sd = torch.tensor(
            image_density, 
            dtype=torch.float32
        ).unsqueeze(dim=0).repeat(inp.shape[0], 1).cuda()                                                                   # get real spectral density of input image
        noise = min_max_normalize(torch.rand_like(inp, dtype=torch.float32)) * inp.mean() / 10
        real_image = inp + noise                                                                                            # real input image + random noise
        fake_image = model(real_image.cuda())
        fake_spectral_density = visual_spectral_density(fake_image)
        
        if loop_type == "train":
            """ ---------------------- calculate the loss ---------------------- """
            spectral_loss = spectral_criterion(fake_spectral_density.cuda(), true_sd.cuda())                                # spectral density loss (L2-distance)
            #!In order to keep the similarity between ori_image and generated image, we add image similarity loss
            image_loss = image_criterion(fake_image.cuda(), real_image.cuda())
            loss_rate = [spectral_loss / (spectral_loss + image_loss), image_loss / (spectral_loss + image_loss)]
            loss = (spectral_loss + image_loss).requires_grad_(True)  # loss part
            losses.update(loss.item(), inp.size(0))
            
            """ ----------------- update the parameters ----------------- """
            opt.zero_grad()
            loss.backward()
            opt.step()
        else: loss = torch.tensor([0.0], dtype=torch.float32)

        if loop_type == "val": desc = ('{} Epoch:{} || Loss {loss:.4f} || lr: {lr:.4f}'.format(loop_type, epoch, loss=loss.item(), lr=lr))
        else: desc = ('{} Epoch:{} || Loss {:.4f} || lr: {:.4f} || Loss Rate: {:.4f}-{:.4f}'.format(loop_type, epoch, loss.item(), lr, loss_rate[0], loss_rate[1]))
        iterator.set_description(desc)
        iterator.refresh()
    
    """ --------- plot on tensorboard --------- """
    if (loop_type == "val") and (writer is not None):
        writer.add_scalar('_'.join(['nat {}'.format(loop_type), 'loss']), losses.avg, epoch)
        loss_rate_writer = {'spectral-loss': loss_rate[0], 'similarity-loss': loss_rate[1]}
        writer.add_scalars("Loss rate", loss_rate_writer, epoch)

    if loop_type == "train": 
        return losses.avg, loss_rate
    else: 
        return fake_spectral_density, fake_image


def show_fake_image(image: torch.Tensor, epoch: int, filename: str):
    """
    Args:
        :: image: A batch of image, with shape of (B, C, H, W)
        :: epoch: The current epoch
        :: filename: The path of saving figure
    """
    plt.figure(figsize=(16, 8))
    for idx in range(10):
        image_show = image[idx].permute(1, 2, 0).cpu().detach().numpy()
        plt.subplot(2, 5, idx+1)
        image_show = min_max_normalize(image_show)
        plt.imshow(image_show)
    plt.title("The generated image of epoch{}".format(epoch))
    plt.savefig(filename)
    print("=> Finished drawing the picture!")


def make_toy_data_kernel(args, model, loader, image_density, store):
    """
    Args:
        :: loader: the dataloader of real image
        :: image_density: the desired spectral density
        :: args: (epochs, lr, momentum, weight_decay, step_lr, step_lr_gamma, out_dir)
    Return:
        :: the kernels which repaire the spectral density (Note: it maybe a list of kernels)
    """
    LOGS_SCHEMA = {'epoch':int, 'loss': float, 'loss_rate': float}
    writer = store.tensorboard if store else None
    if store is not None: store.add_table('logs', LOGS_SCHEMA)

    if not os.path.exists(args.out_dir): os.mkdir(args.out_dir)
    exp_folder_path = args.out_dir
    img_folder_path = os.path.join(exp_folder_path, "Img/")
    if not os.path.exists(img_folder_path): os.mkdir(img_folder_path)
    
    optim = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    schedule = torch.optim.lr_scheduler.StepLR(optim, step_size=args.step_lr, gamma=args.step_lr_gamma)
    model = torch.nn.DataParallel(model).cuda()
    
    for epoch in range(args.epochs):
        
        def plot_spectral_density(sd: torch.Tensor, epoch: int, filename: str):
            """ ---- spectral density test ---- 
            Args:
                :: sd: A batch of image energy density, with shape of (B, 1)
                :: epoch: The current epoch
                :: filename: The path of saving figure
            """
            plt.figure(figsize=(8, 8))
            sd = sd.cpu().detach().numpy()
            plt.plot(sd.mean(axis=0), label="test epoch:{}".format(epoch))
            plt.legend()
            plt.grid()
            plt.savefig(filename)
            print("=> Finished drawing the picture!")
        
        """ ---- save the checkpoints ---- """
        def save_checkpoint(filename):
            ckpt_save_path = os.path.join(exp_folder_path, filename)
            torch.save(checkpoints, ckpt_save_path)
        
        """ ---- Train stage ---- """
        loss, loss_rate = model_loop("train", model, loader, image_density, optim, epoch, writer)
    
        checkpoints = {
            'model':model.state_dict(),
            'optimizer':optim.state_dict(),
            'schedule':(schedule and schedule.state_dict()),
            'epoch': epoch+1,
            'loss': loss,
            'loss_rate': loss_rate
        }
        save_checkpoint("latest.pth")

        """ ---- Val stage ---- """
        if epoch % 4 == 0: 
            with torch.no_grad():
                spectral_density, image = model_loop("val", model,  loader, image_density, optim, epoch)
            plot_spectral_density(spectral_density, epoch, filename=img_folder_path+"/Sp-epoch{}.png".format(epoch))
            show_fake_image(image, epoch, filename=img_folder_path+"/Img-epoch{}.png".format(epoch))
            save_checkpoint("Epoch{}.pth".format(epoch))

        if schedule: schedule.step()
    print("=> Kernels have been finished!")
