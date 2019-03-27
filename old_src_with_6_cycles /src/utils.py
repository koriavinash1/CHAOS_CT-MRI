import random
import time
import datetime
import sys

from torch.autograd import Variable
import torch
import numpy as np

from torchvision.utils import save_image

class dice_loss(object):
    def __init__(self):
        pass

    def background(self, data):
        return data == 0

    def liver(self, data):
        return data == 1

    def right_kidney(self, data):
        return data == 2

    def left_kidney(self, data):
        return data == 3

    def spleen(self, data):
        return data == 4

    def compute(self,prediction,ground_truth):
        masks = [self.background, 
                self.liver, 
                self.right_kidney,
                self.left_kidney,
                self.spleen]
        masks_weight = [1.0, 1.0, 1.0, 1.0, 1.0]
        pred = torch.exp(prediction)
        p = np.uint8(np.argmax(pred.data.cpu().numpy(), axis=1))
        gt = np.uint8(ground_truth.data.cpu().numpy())
        scores = np.array([2.* wt *np.sum(func(p)*func(gt))/(np.sum(func(p)) + np.sum(func(gt))+1e-3) \
                    for func, wt in zip(masks, masks_weight)])
        dice_score = np.mean(scores)
        return 1.0 - dice_score

class ReplayBuffer():
    def __init__(self, max_size=50):
        assert (max_size > 0), 'Empty buffer or trying to create a black hole. Be careful.'
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        to_return = []
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0,1) > 0.5:
                    i = random.randint(0, self.max_size-1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return Variable(torch.cat(to_return))

class LambdaLR():
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert ((n_epochs - decay_start_epoch) > 0), "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch)/(self.n_epochs - self.decay_start_epoch)
