import torch
from torchvision import transforms
from dataset.dataloader import ISICKerasDataset
from torch.utils.data import DataLoader
import numpy as np
import os

transformations = transforms.Compose([transforms.RandomHorizontalFlip(), 
                                      transforms.RandomVerticalFlip(), 
                                      transforms.ToTensor()])
eval_transformations = transforms.Compose([transforms.ToTensor()])

dataset_dir = 'data/ISICKeras/'

train_dataset = ISICKerasDataset(dataset_dir, data_type='train', transform=transformations)
train_datasetLoader = DataLoader(train_dataset, shuffle=True, batch_size=1, num_workers=0)

img, mask = train_dataset[0]
print('test')