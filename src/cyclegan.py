import argparse
import os
import numpy as np
import math
import itertools
import datetime
import time

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

from GANmodels import *
from SEGmodel import *
from datasets import *
from utils import *

import torch.nn as nn
import torch.nn.functional as F
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=0, help='epoch to start training from')
parser.add_argument('--n_epochs', type=int, default=200, help='number of epochs of training')
parser.add_argument('--dataset_name', type=str, default="monet2photo", help='name of the dataset')
parser.add_argument('--batch_size', type=int, default=1, help='size of the batches')
parser.add_argument('--lr', type=float, default=0.0002, help='adam: learning rate')
parser.add_argument('--b1', type=float, default=0.5, help='adam: decay of first order momentum of gradient')
parser.add_argument('--b2', type=float, default=0.999, help='adam: decay of first order momentum of gradient')
parser.add_argument('--decay_epoch', type=int, default=100, help='epoch from which to start lr decay')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--img_height', type=int, default=256, help='size of image height')
parser.add_argument('--img_width', type=int, default=256, help='size of image width')
parser.add_argument('--channels', type=int, default=3, help='number of image channels')
parser.add_argument('--nclasses', type=int, default=8, help='number of classes in segmentation network')
parser.add_argument('--sample_interval', type=int, default=100, help='interval between sampling images from generators')
parser.add_argument('--checkpoint_interval', type=int, default=-1, help='interval between saving model checkpoints')
parser.add_argument('--n_residual_blocks', type=int, default=9, help='number of residual blocks in generator')
opt = parser.parse_args()
print(opt)

# Create sample and checkpoint directories
os.makedirs('images/%s' % opt.dataset_name, exist_ok=True)
os.makedirs('saved_models/%s' % opt.dataset_name, exist_ok=True)

# Losses
criterion_GAN = torch.nn.MSELoss()
criterion_cycle = torch.nn.L1Loss()
criterion_identity = torch.nn.L1Loss()
criterion_dice = dice_loss()
criterion_ce = torch.nn.CrossEntropyLoss()

cuda = True if torch.cuda.is_available() else False

# Calculate output of image discriminator (PatchGAN)
patch = (1, opt.img_height // 2**4, opt.img_width // 2**4)

# Initialize generator and discriminator
G_MRI_CT = GeneratorResNet(res_blocks=opt.n_residual_blocks)
G_CT_MRI = GeneratorResNet(res_blocks=opt.n_residual_blocks)
D_MRI = Discriminator()
D_CT = Discriminator()
D_SEG_MRI = FCDenseNet57(opt.mri_nclasses, opt.channels)
D_SEG_CT = FCDenseNet57(opt.ct_nclasses, opt.channels)


if cuda:
    G_MRI_CT = G_MRI_CT.cuda()
    G_CT_MRI = G_CT_MRI.cuda()
    D_MRI = D_MRI.cuda()
    D_CT = D_CT.cuda()
    D_SEG_MRI = D_SEG_MRI.cuda()
    D_SEG_CT = D_SEG_CT.cuda()
    criterion_GAN.cuda()
    criterion_cycle.cuda()
    criterion_identity.cuda()

if opt.epoch != 0:
    # Load pretrained models
    G_MRI_CT.load_state_dict(torch.load('saved_models/%s/G_MRI_CT_%d.pth' % (opt.dataset_name, opt.epoch)))
    G_CT_MRI.load_state_dict(torch.load('saved_models/%s/G_CT_MRI_%d.pth' % (opt.dataset_name, opt.epoch)))
    D_MRI.load_state_dict(torch.load('saved_models/%s/D_MRI_%d.pth' % (opt.dataset_name, opt.epoch)))
    D_CT.load_state_dict(torch.load('saved_models/%s/D_CT_%d.pth' % (opt.dataset_name, opt.epoch)))
    D_SEG_CT.load_state_dict(torch.load('saved_models/%s/D_SEG_CT_%d.pth' % (opt.dataset_name, opt.epoch)))
    D_SEG_MRI.load_state_dict(torch.load('saved_models/%s/D_SEG_MRI_%d.pth' % (opt.dataset_name, opt.epoch)))
else:
    # Initialize weights
    G_MRI_CT.apply(weights_init_normal)
    G_CT_MRI.apply(weights_init_normal)
    D_MRI.apply(weights_init_normal)
    D_CT.apply(weights_init_normal)
    D_SEG_CT.apply(weights_init_normal)
    D_SEG_MRI.apply(weights_init_normal)

# Loss weights
lambda_cyc = 10
lambda_id = 0.5 * lambda_cyc

# Optimizers
optimizer_G = torch.optim.Adam(itertools.chain(G_MRI_CT.parameters(), G_CT_MRI.parameters()),
                                lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D_MRI = torch.optim.Adam(D_MRI.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D_CT = torch.optim.Adam(D_CT.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D_SEG_MRI = torch.optim.Adam(D_SEG_MRI.parametrs(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D_SEG_CT = torch.optim.Adam(D_SEG_CT.parametrs(), lr=opt.lr, betas=(opt.b1, opt.b2))


# Learning rate update schedulers
lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
lr_scheduler_D_MRI = torch.optim.lr_scheduler.LambdaLR(optimizer_D_MRI, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
lr_scheduler_D_CT = torch.optim.lr_scheduler.LambdaLR(optimizer_D_CT, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
lr_scheduler_D_SEG_MRI = torch.optim.lr_scheduler.LambdaLR(optimizer_D_SEG_MRI, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
lr_scheduler_D_SEG_CT = torch.optim.lr_scheduler.LambdaLR(optimizer_D_SEG_CT, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)


Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

# Buffers of previously generated samples
fake_MRI_buffer = ReplayBuffer()
fake_CT_buffer = ReplayBuffer()


# Image transformations
transforms_ = [ transforms.Resize(int(opt.img_height*1.12), Image.BICUBIC),
                transforms.RandomCrop((opt.img_height, opt.img_width)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) ]


# TODO:


# Training data loader
dataloader = DataLoader(ImageDataset("../../data/%s" % opt.dataset_name, transforms_=transforms_, unaligned=True),
                        batch_size=opt.batch_size, shuffle=True, num_workers=opt.n_cpu)
# Test data loader
val_dataloader = DataLoader(ImageDataset("../../data/%s" % opt.dataset_name, transforms_=transforms_, unaligned=True, mode='test'),
                        batch_size=5, shuffle=True, num_workers=1)



def sample_images(batches_done):
    """Saves a generated sample from the test set"""
    imgs = next(iter(val_dataloader))
    real_MRI = Variable(imgs['MRI'].type(Tensor))
    fake_CT = G_MRI_CT(real_MRI)
    real_CT = Variable(imgs['CT'].type(Tensor))
    fake_MRI = G_CT_MRI(real_CT)
    img_sample = torch.cat((real_MRI.data, fake_CT.data,
                            real_CT.data, fake_MRI.data), 0)
    save_image(img_sample, 'images/%s/%s.png' % (opt.dataset_name, batches_done), nrow=5, normalize=True)


# ----------
#  Training
# ----------

prev_time = time.time()
for epoch in range(opt.epoch, opt.n_epochs):
    for i, batch in enumerate(dataloader):

        # Set model input
        real_MRI = Variable(batch['MRI'].type(Tensor))
        real_CT = Variable(batch['CT'].type(Tensor))

        # Adversarial ground truths
        valid = Variable(Tensor(np.ones((real_MRI.size(0), *patch))), requires_grad=False)
        fake = Variable(Tensor(np.zeros((real_MRI.size(0), *patch))), requires_grad=False)

        # CT MRI Segmentation Ground Truths
        CT_GT = 
        CT_MRI_GT = 

        MRI_GT = 
        MRI_CT_GT = 

        # TODO:

        # ------------------
        #  Train Generators
        # ------------------

        optimizer_G.zero_grad()

        # Identity loss
        loss_id_MRI= criterion_identity(G_CT_MRI(real_MRI), real_MRI)
        loss_id_CT = criterion_identity(G_MRI_CT(real_CT), real_CT)

        loss_identity = (loss_id_MRI+ loss_id_CT) / 2.0

        # GAN loss
        fake_CT = G_MRI_CT(real_MRI)
        loss_GAN_MRI_CT = criterion_GAN(D_CT(fake_CT), valid)

        fake_MRI = G_CT_MRI(real_CT)
        loss_GAN_CT_MRI = criterion_GAN(D_MRI(fake_MRI), valid)

        loss_GAN = (loss_GAN_MRI_CT + loss_GAN_CT_MRI) / 2


        # Cycle loss
        recov_MRI = G_CT_MRI(fake_CT)
        loss_cycle_MRI = criterion_cycle(recov_MRI, real_MRI)
        recov_CT = G_MRI_CT(fake_MRI)
        loss_cycle_CT = criterion_cycle(recov_CT, real_CT)

        loss_cycle = (loss_cycle_MRI + loss_cycle_CT) / 2

        # Total loss
        loss_G =    loss_GAN + \
                    lambda_cyc * loss_cycle + \
                    lambda_id * loss_identity


        loss_G.backward()
        optimizer_G.step()

        # --------------------------
        #  Train Segmentation Net CT
        # --------------------------

        optimizer_D_SEG_CT.zero_grad()

        real_CT_segmentation = D_SEG_CT(real_CT)
        fake_CT_segmentation = D_SEG_CT(fake_CT)

        # dice loss
        loss_dice_real_CT = criterion_dice.compute(real_CT_segmentation, CT_GT)
        loss_dice_fake_CT = criterion_dice.compute(fake_CT_segmentation, MRI_CT_GT)

        # CE loss
        loss_ce_real_CT = criterion_ce(real_CT_segmentation, CT_GT) 
        loss_ce_fake_CT = criterion_ce(fake_CT_segmentation, MRI_CT_GT)

        loss_SEG_CT = (loss_dice_real_CT + loss_dice_fake_CT + loss_ce_real_CT + loss_ce_fake_CT) / 4.0

        loss_SEG_CT.backward()
        optimizer_D_SEG_CT.step()

        # ---------------------------
        #  Train Segmentation Net MRI
        # ----------------------------

        optimizer_D_SEG_MRI.zero_grad()
        real_MRI_segmentation = D_SEG_MRI(real_MRI)
        fake_MRI_segmentation = D_SEG_MRI(fake_MRI)

        # dice loss
        loss_dice_real_MRI = criterion_dice.compute(real_MRI_segmentation, CT_GT)
        loss_dice_fake_MRI = criterion_dice.compute(fake_MRI_segmentation, CT_MRI_GT)

        # CE loss
        loss_ce_real_MRI = criterion_ce(real_MRI_segmentation, MRI_GT)
        loss_ce_fake_MRI = criterion_ce(fake_MRI_segmentation, CT_MRI_GT)
        
        loss_SEG_MRI = (loss_dice_real_MRI + loss_dice_fake_MRI + loss_ce_real_MRI + loss_ce_fake_MRI) / 4.0
        
        loss_SEG_MRI.backward()
        optimizer_D_SEG_MRI.step()

        # -----------------------
        #  Train Discriminator A
        # -----------------------

        optimizer_D_MRI.zero_grad()

        # Real loss
        loss_real = criterion_GAN(D_MRI(real_MRI), valid)

        # Fake loss (on batch of previously generated samples)
        fake_MRI_ = fake_MRI_buffer.push_and_pop(fake_MRI)
        loss_fake = criterion_GAN(D_MRI(fake_MRI_.detach()), fake)
        
        # Total loss
        loss_D_MRI = (loss_real + loss_fake) / 2

        loss_D_MRI.backward()
        optimizer_D_MRI.step()

        # -----------------------
        #  Train Discriminator B
        # -----------------------

        optimizer_D_CT.zero_grad()

        # Real loss
        loss_real = criterion_GAN(D_CT(real_CT), valid)
        # Fake loss (on batch of previously generated samples)
        fake_CT_ = fake_CT_buffer.push_and_pop(fake_CT)
        loss_fake = criterion_GAN(D_CT(fake_CT_.detach()), fake)
        # Total loss
        loss_D_CT = (loss_real + loss_fake) / 2

        loss_D_CT.backward()
        optimizer_D_CT.step()

        loss_D = (loss_D_MRI + loss_D_CT) / 2

        
        # --------------
        #  Log Progress
        # --------------

        # Determine approximate time left
        batches_done = epoch * len(dataloader) + i
        batches_left = opt.n_epochs * len(dataloader) - batches_done
        time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
        prev_time = time.time()

        # Print log
        sys.stdout.write("\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, adv: %f, cycle: %f, identity: %f] ETA: %s" %
                                                        (epoch, opt.n_epochs,
                                                        i, len(dataloader),
                                                        loss_D.item(), loss_G.item(),
                                                        loss_GAN.item(), loss_cycle.item(),
                                                        loss_identity.item(), time_left))

        # If at sample interval save image
        if batches_done % opt.sample_interval == 0:
            sample_images(batches_done)


    # Update learning rates
    lr_scheduler_G.step()
    lr_scheduler_D_MRI.step()
    lr_scheduler_D_CT.step()

    if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0:
        # Save model checkpoints
        torch.save(G_MRI_CT.state_dict(), 'saved_models/%s/G_MRI_CT_%d.pth' % (opt.dataset_name, epoch))
        torch.save(G_CT_MRI.state_dict(), 'saved_models/%s/G_CT_MRI_%d.pth' % (opt.dataset_name, epoch))
        torch.save(D_MRI.state_dict(), 'saved_models/%s/D_MRI_%d.pth' % (opt.dataset_name, epoch))
        torch.save(D_CT.state_dict(), 'saved_models/%s/D_CT_%d.pth' % (opt.dataset_name, epoch))
