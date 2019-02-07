import torch
from torch.utils import data
import skimage
from skimage import io, transform
from glob import glob
import os
import numpy as np
from PIL import Image 

class ISICKerasDataset(data.Dataset):
    def __init__(self, dataset_dir, data_type='train', transform=None):
        self.img_files = glob(os.path.join(dataset_dir, data_type, 'image/*'))
        self.mask_files = glob(os.path.join(dataset_dir, data_type, 'label/*'))
        self.mask_dir = os.path.join(dataset_dir, data_type, 'label/')
        
        self.transform = transform
    
    def __len__(self):
        ''' Only return the num of train img 
        '''
        return len(self.img_files)

    def __getitem__(self, idx):
        img_file = self.img_files[idx]
        mask_file = img_file.split('/')[-1]
        img_id = mask_file.split('.')[0]
        mask_file = img_id + '_segmentation.png'
        mask_file = os.path.join(self.mask_dir, mask_file)
    
        # mask_file = self.mask_files[idx]
        # Image shaoe (256,256,3) for rgb [0,255] uint8
        img = io.imread(img_file)
        # img = img/127.5 - 1
        # Image shape (256,256) for gray [0,255] uint8
        mask = io.imread(mask_file)
        mask = np.resize(mask, (256,256,1)).astype(np.uint8)
        if self.transform is not None:
            combine = np.concatenate((img, mask), axis=2)
            combine = Image.fromarray(combine)
            combine = self.transform(combine)
            img = combine[:3,:,:]
            mask = combine[3:,:,:]
        return img, mask

class ISBI2018Dataset(data.Dataset):
    def __init__(self, dataset_dir, data_type='Training Set', transform=None):
        self.img_files = glob(os.path.join(dataset_dir, 'Original Images', data_type, '*'))
        # self.mask_files = glob(os.path.join(dataset_dir, 'All Segmentation Groundtruths', data_type, '*'))
        self.mask_dir = os.path.join(dataset_dir, 'All Segmentation Groundtruths', data_type, 'Optic Disc/')
        self.transform = transform
    
    def __len__(self):
        ''' Only return the num of train img 
        '''
        return len(self.img_files)

    def __getitem__(self, idx):
        img_file = self.img_files[idx]
        mask_file = img_file.split('/')[-1]
        img_id = mask_file.split('.')[0]
        mask_file = img_id + '_OD.png'
        mask_file = os.path.join(self.mask_dir, mask_file)
    
        # mask_file = self.mask_files[idx]
        # Image shaoe (256,256,3) for rgb [0,255] uint8
        img = io.imread(img_file)
        img = transform.resize(img, (512,512,3))
        img = img.astype(np.uint8)
        # img = img/127.5 - 1
        # Image shape (256,256) for gray [0,255] uint8
        mask = io.imread(mask_file)
        mask = transform.resize(mask, (512,512))
        mask = np.resize(mask, (512,512,1)).astype(np.uint8)
        if self.transform is not None:
            combine = np.concatenate((img, mask), axis=2)
            combine = Image.fromarray(combine)
            combine = self.transform(combine)
            img = combine[:3,:,:]
            mask = combine[3:,:,:]

        return img, mask


def imsave(img):
    npimg = img.numpy()
    img = np.transpose(npimg, (1,2,0))
    img = skimage.img_as_ubyte(img)
    skimage.io.imsave('train_img_example.jpg', img)


 