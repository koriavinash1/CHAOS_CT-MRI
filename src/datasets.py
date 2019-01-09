import glob
import random
import os
import numpy as np
import pandas as pd
import pydicom

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

class ImageDataset(Dataset):
    def __init__(self, csv_root, transforms_ = None, mode='train'):
        """
        """
        self.transforms_ = transforms_
        self.MRI_df = pd.read_csv(os.path.join(csv_root, mode + '_MRI.csv'))
        self.CT_df = pd.read_csv(os.path.join(csv_root, mode + '_CT.csv'))

        if len(self.MRI_df) > len(self.CT_df):
            buffer = self.CT_df.sample(n = len(self.MRI_df) - len(self.CT_df))
            self.CT_df = pd.concat([self.CT_df, buffer])
        elif len(self.CT_df) > len(self.MRI_df):
            buffer = self.MRI_df.sample(n = len(self.CT_df) - len(self.MRI_df))
            self.MRI_df = pd.concat([self.MRI_df, buffer])

        print ("[INFO] MRI df: {}, CT df: {}".format(len(self.MRI_df), len(self.CT_df)))
        self.files_MRI = self.MRI_df['dcm_path'].as_matrix()
        self.files_MRI_GT = self.MRI_df['gt_path'].as_matrix()

        self.files_CT  = self.CT_df['dcm_path'].as_matrix()
        self.files_CT_GT = self.CT_df['gt_path'].as_matrix()

    def applyTransform(self, image, mask):
        """
        """
        # Resize
        resize_img = transforms.Resize(size=self.transforms_['out_size'], interpolation = 2)
        resize_mask = transforms.Resize(size=self.transforms_['out_size'], interpolation = 0)
        print ("=======")
        image = resize_img(image)
        mask = resize_mask(mask)

        # Random horizontal flipping
        if random.random() > 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        # Random vertical flipping
        if random.random() > 0.5:
            image = TF.vflip(image)
            mask = TF.vflip(mask)
            
        # # Transform to tensor
        # image = TF.to_tensor(image)
        # mask = TF.to_tensor(mask)
        return image, mask

    def convertMRIGT(self, image):
        """
        """
        actual = [80, 160, 240, 255]
        replace = [1, 2, 3, 4]
        image = np.array(image)
        for ai, ri in zip(actual, replace):
            image[image == ai] = ri

        return Image.fromarray(image)

    def convertCTGT(self, image):
        """
        """
        image = np.array(image)
        image[image == 255] = 1
        return Image.fromarray(image)

    def convertMRICT_GT(self, image):
        """
        """
        image[image != 1] = 0
        return image

    def __getitem__(self, index):
        """
        """
        item_MRI_GT = self.convertMRIGT(Image.open(self.files_MRI_GT[index]).convert('L'))
        item_MRI = np.array(pydicom.dcmread(self.files_MRI[index]).pixel_array).reshape(item_MRI_GT.size[0], item_MRI_GT.size[1])
        
        item_CT_GT = self.convertCTGT(Image.open(self.files_CT_GT[index]).convert('L'))
        item_CT = np.array(pydicom.dcmread(self.files_CT[index]).pixel_array).reshape(item_CT_GT.size[0], item_CT_GT.size[1])

        # print (item_MRI.shape, item_MRI_GT.size, item_MRI_GT.size, item_MRI.dtype)
        # print (item_CT_GT.size, item_CT_GT.dtype, item_CT.shape, item_CT.dtype)        
        item_CT, item_MRI = Image.fromarray(item_CT), Image.fromarray(item_MRI)

        item_CT, item_CT_GT = self.applyTransform(item_CT, item_CT_GT)
        item_MRI, item_MRI_GT = self.applyTransform(item_MRI, item_MRI_GT)

        item_CT_GT = np.uint8(item_CT_GT)
        item_MRI_GT = np.uint8(item_MRI_GT)
        item_CT_MRI_GT = np.uint8(item_CT_GT)
        item_MRI_CT_GT = np.uint8(self.convertMRICT_GT(item_MRI_GT))

        return {'MRI': item_MRI, 'CT': item_CT, \
                'MRI_GT': item_MRI_GT, 'CT_GT': item_CT_GT, \
                'MRI_CT_GT': item_MRI_CT_GT, 'CT_MRI_GT': item_CT_MRI_GT}

    def __len__(self):
        return max(len(self.files_MRI), len(self.files_CT))
