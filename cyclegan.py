import argparse
import os
import numpy as np
import pandas as pd
import math
import itertools
import datetime
import time

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

from models import *
from datasets import *
from utils import *

import torch.nn as nn
import torch.nn.functional as F
import torch

from tensorboardX import SummaryWriter

import json
import pytorch_ssim

writer = SummaryWriter()


path = './data' ##path for the DATA

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=0, help='epoch to start training from')
parser.add_argument('--n_epochs', type=int, default=3001, help='number of epochs of training')
parser.add_argument('--dataset_name', type=str, default="try1", help='name of the dataset')
parser.add_argument('--batch_size', type=int, default=4, help='size of the batches')
parser.add_argument('--lr', type=float, default=0.0002, help='adam: learning rate')
parser.add_argument('--b1', type=float, default=0.5, help='adam: decay of first order momentum of gradient')
parser.add_argument('--b2', type=float, default=0.999, help='adam: decay of first order momentum of gradient')
parser.add_argument('--decay_epoch', type=int, default= 100, help='epoch from which to start lr decay')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--img_height', type=int, default=256, help='size of image height')
parser.add_argument('--img_width', type=int, default=256, help='size of image width')
parser.add_argument('--channels', type=int, default=1, help='number of image channels')
parser.add_argument('--sample_interval', type=int, default= 100, help='interval between sampling images from generators')
parser.add_argument('--checkpoint_interval', type=int, default= 50, help='interval between saving model checkpoints')
parser.add_argument('--n_residual_blocks', type=int, default=9, help='number of residual blocks in generator')
opt = parser.parse_args()
print(opt)

# Create sample and checkpoint directories
os.makedirs('images/%s' % opt.dataset_name, exist_ok=True)
os.makedirs('saved_models/%s' % opt.dataset_name, exist_ok=True)
os.makedirs('saved_logs/%s' % opt.dataset_name, exist_ok=True)

###------------
# Info JSON dump
###--------------
info_dict = {
            'epoch': opt.epoch,
            'n_epochs' :opt.n_epochs,
            'dataset_name': opt.dataset_name,
            'batch_size': opt.batch_size,
            'lr': opt.lr,
            'b1': opt.b1,
            'b2':opt.b2,
            'decay_epoch': opt.decay_epoch,
            'n_cpu': opt.n_cpu,
            'img_height': opt.img_height,
            'img_width': opt.img_width,
            'channels': opt.channels,
            # 'sample_interval': opt.sample_interval,
            'checkpoint_interval': opt.checkpoint_interval,
            'n_residual_blocks': opt.n_residual_blocks
} 

with open('saved_logs/%s/config.txt' % opt.dataset_name,'w') as outFile:
    json.dump(info_dict, outFile)


# Losses
criterion_GAN = torch.nn.MSELoss()
criterion_cycle = torch.nn.L1Loss()
criterion_identity = torch.nn.L1Loss()
ssim_loss = pytorch_ssim.SSIM()     ## Structural Similarity


cuda = True if torch.cuda.is_available() else False

# Calculate output of image discriminator (PatchGAN)
patch = (1, opt.img_height // 2**4, opt.img_width // 2**4)

# Initialize generator and discriminator
G_AB = GeneratorResNet(in_channels=opt.channels, out_channels=opt.channels, res_blocks=opt.n_residual_blocks)
G_BA = GeneratorResNet(in_channels=opt.channels, out_channels=opt.channels, res_blocks=opt.n_residual_blocks)
D_A = Discriminator(in_channels=opt.channels)
D_B = Discriminator(in_channels=opt.channels)

if cuda:
    G_AB = G_AB.cuda()
    G_BA = G_BA.cuda()
    D_A = D_A.cuda()
    D_B = D_B.cuda()
    criterion_GAN.cuda()
    criterion_cycle.cuda()
    criterion_identity.cuda()

if opt.epoch != 0:
    # Load pretrained models
    print('Loading Pretrained Model')
    G_AB.load_state_dict(torch.load('saved_models/%s/G_AB_%d.pth' % (opt.dataset_name, opt.epoch)))
    G_BA.load_state_dict(torch.load('saved_models/%s/G_BA_%d.pth' % (opt.dataset_name, opt.epoch)))
    D_A.load_state_dict(torch.load('saved_models/%s/D_A_%d.pth' % (opt.dataset_name, opt.epoch)))
    D_B.load_state_dict(torch.load('saved_models/%s/D_B_%d.pth' % (opt.dataset_name, opt.epoch)))
else:
    # Initialize weights
    G_AB.apply(weights_init_normal)
    G_BA.apply(weights_init_normal)
    D_A.apply(weights_init_normal)
    D_B.apply(weights_init_normal)

# Loss weights
lambda_cyc = 9.67
lambda_id = 0.5 * lambda_cyc
lambda_ssim = 0.5

# Optimizers
optimizer_G = torch.optim.Adam(itertools.chain(G_AB.parameters(), G_BA.parameters()),
                                lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D_A = torch.optim.Adam(D_A.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D_B = torch.optim.Adam(D_B.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

# Learning rate update schedulers
lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(optimizer_D_A, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(optimizer_D_B, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)

Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

# Buffers of previously generated samples
fake_A_buffer = ReplayBuffer()
fake_B_buffer = ReplayBuffer()

# Image transformations
# transforms_ = [transforms.Resize((128,128), Image.BICUBIC),
#                 transforms.ToTensor(),
#                 transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) ]

# Image transformations
transforms_ = [ transforms.Resize(int(opt.img_height*1.12), Image.BICUBIC),
                transforms.RandomRotation(180),
                transforms.RandomCrop((opt.img_height, opt.img_width)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor()]
                # transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) ]

# Training data loader
dataloader = DataLoader(ImageDataset(path, transforms_=transforms_, unaligned=True),
                        batch_size=opt.batch_size, shuffle=True, num_workers=opt.n_cpu)
# Test data loader
val_dataloader = DataLoader(ImageDataset(path, transforms_=transforms_, unaligned=True, mode='test'),
                        batch_size=6, shuffle=True, num_workers=1)


def sample_images(batches_done):
    """Saves a generated sample from the test set"""
    imgs = next(iter(val_dataloader))
    real_A = Variable(imgs['A'].type(Tensor))
    fake_B = G_AB(real_A)
    real_B = Variable(imgs['B'].type(Tensor))
    fake_A = G_BA(real_B)
    
    recov_A = G_BA(fake_B)
    recov_B = G_AB(fake_A)

    img_sample = torch.cat((real_A.data, fake_B.data, recov_A.data,
                            real_B.data, fake_A.data, recov_B.data), 0)
    save_image(img_sample, 'images/%s/%s_N.png' % (opt.dataset_name, batches_done), nrow=5, normalize=True)
    save_image(img_sample, 'images/%s/%s.png' % (opt.dataset_name, batches_done), nrow=5, normalize=False)



#--------
#  Logs
#--------

time_list = []
loss_identity_list = []
loss_GAN_list = []
loss_cycle_list = []
loss_G_net_list = []
loss_D_A_list = []
loss_D_B_list = []
loss_D_net_list = []
epoch_list = []
batch_list = []


# ----------
#  Training
# ----------
iteration = 0


prev_time = time.time()
for epoch in range(opt.epoch, opt.n_epochs):
    for i, batch in enumerate(dataloader):
        # Set model input
        real_A = Variable(batch['A'].type(Tensor))
        real_B = Variable(batch['B'].type(Tensor))

        # Adversarial ground truths
        valid = Variable(Tensor(np.ones((real_A.size(0), *patch))), requires_grad=False)
        fake = Variable(Tensor(np.zeros((real_A.size(0), *patch))), requires_grad=False)

        # ------------------
        #  Train Generators
        # ------------------

        optimizer_G.zero_grad()

        # Identity loss
        loss_id_A = criterion_identity(G_BA(real_A), real_A)
        loss_id_B = criterion_identity(G_AB(real_B), real_B)

        loss_identity = (loss_id_A + loss_id_B) / 2



        # GAN loss
        fake_B = G_AB(real_A)
        loss_GAN_AB = criterion_GAN(D_B(fake_B), valid)
        fake_A = G_BA(real_B)
        loss_GAN_BA = criterion_GAN(D_A(fake_A), valid)

        loss_GAN = (loss_GAN_AB + loss_GAN_BA) / 2

        # Cycle loss
        recov_A = G_BA(fake_B)
        loss_cycle_A = criterion_cycle(recov_A, real_A)
        recov_B = G_AB(fake_A)
        loss_cycle_B = criterion_cycle(recov_B, real_B)

        loss_cycle = (loss_cycle_A + loss_cycle_B) / 2

        # SSIM loss
        ssim_out_AB = -ssim_loss(real_A, fake_B)
        ssim_out_BA = -ssim_loss(real_B, fake_A)

        loss_ssim = (ssim_out_AB + ssim_out_BA ) / 2


        # Total loss
        loss_G =    loss_GAN + \
                    lambda_cyc * loss_cycle + \
                    lambda_id * loss_identity + \
                    lambda_ssim * loss_ssim


        loss_G.backward()
        optimizer_G.step()

        # -----------------------
        #  Train Discriminator A
        # -----------------------

        optimizer_D_A.zero_grad()

        # Real loss
        loss_D_A_real = criterion_GAN(D_A(real_A), valid)
        # Fake loss (on batch of previously generated samples)
        fake_A_ = fake_A_buffer.push_and_pop(fake_A)
        loss_D_A_fake = criterion_GAN(D_A(fake_A_.detach()), fake)
        # Total loss
        loss_D_A = (loss_D_A_real + loss_D_A_fake) / 2

        loss_D_A.backward()
        optimizer_D_A.step()

        # -----------------------
        #  Train Discriminator B
        # -----------------------

        optimizer_D_B.zero_grad()

        # Real loss
        loss_D_B_real = criterion_GAN(D_B(real_B), valid)
        # Fake loss (on batch of previously generated samples)
        fake_B_ = fake_B_buffer.push_and_pop(fake_B)
        loss_D_B_fake = criterion_GAN(D_B(fake_B_.detach()), fake)
        # Total loss
        loss_D_B = (loss_D_B_real + loss_D_B_fake) / 2

        loss_D_B.backward()
        optimizer_D_B.step()

        loss_D = (loss_D_A + loss_D_B) / 2
    
        # -------------
        # Tensorboard Log information
        # ------------
        writer.add_scalar('generator/loss_id_A', loss_id_A, int(iteration))
        writer.add_scalar('generator/loss_id_B', loss_id_B, int(iteration))
        writer.add_scalar('generator/loss_identity', loss_identity, int(iteration))
        writer.add_scalar('generator/loss_GAN_AB', loss_GAN_AB, int(iteration))
        writer.add_scalar('generator/loss_GAN_BA', loss_GAN_BA, int(iteration))
        writer.add_scalar('generator/loss_GAN', loss_GAN, int(iteration))
        writer.add_scalar('generator/loss_cycle_A', loss_cycle_A, int(iteration))
        writer.add_scalar('generator/loss_cycle_B', loss_cycle_B, int(iteration))
        writer.add_scalar('generator/loss_cycle', loss_cycle, int(iteration))
        writer.add_scalar('generator/loss_ssim', loss_ssim, int(iteration))
        writer.add_scalar('generator/loss', loss_G, int(iteration))


        writer.add_scalar('discriminatorA/loss_real', loss_D_A_real, int(iteration))
        writer.add_scalar('discriminatorA/loss_fake', loss_D_A_fake, int(iteration))
        writer.add_scalar('discriminatorA/loss_D', loss_D_A, int(iteration))
        
        writer.add_scalar('discriminatorB/loss_real', loss_D_B_real, int(iteration))
        writer.add_scalar('discriminatorB/loss_fake', loss_D_B_fake, int(iteration))
        writer.add_scalar('discriminatorB/loss_D', loss_D_B, int(iteration))

        writer.add_scalar('discriminator/loss', loss_D, int(iteration))

        iteration += 1

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
        
        #update Logs
        time_list.append(time.strftime('%X %x'))
        epoch_list.append(epoch)
        batch_list.append(i)

        loss_identity_list.append(loss_identity.item())
        loss_GAN_list.append(loss_GAN.item())
        loss_cycle_list.append(loss_cycle.item())
        loss_G_net_list.append(loss_G.item())
        loss_D_A_list.append(loss_D_A.item())
        loss_D_B_list.append(loss_D_B.item())
        loss_D_net_list.append(loss_D.item())


    # Update learning rates
    lr_scheduler_G.step()
    lr_scheduler_D_A.step()
    lr_scheduler_D_B.step()

    if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0:
        with torch.set_grad_enabled(False):
            sample_images(epoch)

        save_logs = pd.DataFrame()
        save_logs['time'] = time_list
        save_logs['epoch'] = epoch_list
        save_logs['batch'] = batch_list
        save_logs['loss_identity'] = loss_identity_list
        save_logs['loss_GAN'] = loss_GAN_list
        save_logs['loss_cycle'] = loss_cycle_list
        save_logs['loss_G_net'] = loss_G_net_list
        save_logs['loss_D_A'] = loss_D_A_list
        save_logs['loss_D_B'] = loss_D_B_list
        save_logs['loss_D_net'] = loss_D_net_list
        save_logs.to_csv('saved_logs/%s/training_logs_%s.csv' % (opt.dataset_name, epoch), index = False)
        del save_logs
        # Save model checkpoints
        torch.save(G_AB.state_dict(), 'saved_models/%s/G_AB_%d.pth' % (opt.dataset_name, epoch))
        torch.save(G_BA.state_dict(), 'saved_models/%s/G_BA_%d.pth' % (opt.dataset_name, epoch))
        torch.save(D_A.state_dict(), 'saved_models/%s/D_A_%d.pth' % (opt.dataset_name, epoch))
        torch.save(D_B.state_dict(), 'saved_models/%s/D_B_%d.pth' % (opt.dataset_name, epoch))


save_logs = pd.DataFrame()
save_logs['time'] = time_list
save_logs['epoch'] = epoch_list
save_logs['batch'] = batch_list
save_logs['loss_identity'] = loss_identity_list
save_logs['loss_GAN'] = loss_GAN_list
save_logs['loss_cycle'] = loss_cycle_list
save_logs['loss_G_net'] = loss_G_net_list
save_logs['loss_D_A'] = loss_D_A_list
save_logs['loss_D_B'] = loss_D_B_list
save_logs['loss_D_net'] = loss_D_net_list
save_logs.to_csv('saved_logs/%s/training_logs.csv' % opt.dataset_name, index = False)

#############################
### Saving the last Model ###
#############################

# Save model checkpoints
torch.save(G_AB.state_dict(), 'saved_models/%s/G_AB_%d.pth' % (opt.dataset_name, epoch))
torch.save(G_BA.state_dict(), 'saved_models/%s/G_BA_%d.pth' % (opt.dataset_name, epoch))
torch.save(D_A.state_dict(), 'saved_models/%s/D_A_%d.pth' % (opt.dataset_name, epoch))
torch.save(D_B.state_dict(), 'saved_models/%s/D_B_%d.pth' % (opt.dataset_name, epoch))



### for log usage
writer.export_scalars_to_json("./all_scalars.json")
writer.close()
