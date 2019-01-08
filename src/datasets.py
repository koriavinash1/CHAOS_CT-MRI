import glob
import random
import os
import pandas as pd
import pydicom

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

class ImageDataset(Dataset):
    def __init__(self, csv_root, transforms = None, mode='train'):
        """
        """
        self.transform = transforms
        self.unaligned = unaligned

        self.files_MRI = pd.read_csv(os.path.join(csv_root, mode + '_MRI.csv'))['dcm_pth'].as_matrix()
        self.files_MRI_GT = pd.read_csv(os.path.join(csv_root, mode + '_MRI.csv'))['gt_pth'].as_matrix()

        self.files_CT  = pd.read_csv(os.path.join(csv_root, mode + '_CT.csv'))['dcm_pth'].as_matrix()
        self.files_CT_GT = pd.read_csv(os.path.join(csv_root, mode + '_CT.csv'))['gt_pth'].as_matrix()

    def applyTransform(self, image, mask):
        """
        """
        # Resize
        resize = transforms.Resize(size=transforms['out_size'])
        image = resize(image)
        mask = resize(mask)

        # Random crop
        i, j, h, w = transforms.RandomCrop.get_params(
            image, output_size=transforms['out_size'])
        image = TF.crop(image, i, j, h, w)
        mask = TF.crop(mask, i, j, h, w)

        # Random horizontal flipping
        if random.random() > 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        # Random vertical flipping
        if random.random() > 0.5:
            image = TF.vflip(image)
            mask = TF.vflip(mask)
            
        # Transform to tensor
        image = TF.to_tensor(image)
        mask = TF.to_tensor(mask)
        return image, mask

    def convertMRIGT(self, image):
        """
        """
        actual = [80, 160, 240, 255]
        replace = [1, 2, 3, 4]

        for ai, ri in zip(actual, replace):
            image[image == ai] = ri

        return image

    def convertCTGT(self, image):
        """
        """
        image[image == 255] = 1
        return image

    def convertMRICT_GT(self, image):
        """
        """
        image[image != 1] = 0
        return image

    def __getitem__(self, index):
        """
        """
        item_MRI = pydicom.dcmread(self.files_MRI[index]) 
        item_CT = pydicom.dcmread(self.files_CT[index])
        item_MRI_GT = self.convertMRIGT(Image.open(self.files_MRI_GT[index]).convert('L'))
        item_CT_GT = self.convertCTGT(Image.open(self.files_CT_GT[index]).convert('L'))
        
        item_CT_MRI_GT = item_CT_GT
        item_MRI_CT_GT = self.convertMRICT_GT(item_MRI_GT)        
        return {'MRI': item_MRI, 'CT': item_CT, \
                'MRI_GT': item_MRI_GT, 'CT_GT': item_CT_GT, \
                'MRI_CT_GT': item_MRI_CT_GT, 'CT_MRI_GT': item_CT_MRI_GT}

    def __len__(self):
        return max(len(self.files_MRI), len(self.files_CT))
