from model.U_net import Unet
from utils.tools import set_logger, print_current_losses
import logging
import datetime
from torchvision import transforms
from dataset.dataloader import ISICKerasDataset
from torch.utils.data import DataLoader
import torch
from torch.nn import init
import numpy as np

transformations = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(), 
                                    transforms.ToTensor()])
eval_transformations = transforms.Compose([transforms.ToTensor()])

if __name__ == "__main__":
    set_logger('train_Unet.log')
    dataset_dir = '../data/ISICKeras/'
    batch_size = 2
    gpu_ids = ['1']
    device = torch.device('cuda:{}'.format(gpu_ids[0])) if torch.cuda.is_available() else torch.device('cpu')
    train_dataset = ISICKerasDataset(dataset_dir, data_type='train', transform=transformations)
    train_datasetLoder = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=0)

    # train_img = train_dataset[0]

    eval_dataset = ISICKerasDataset(dataset_dir, data_type='eval', transform=eval_transformations)
    eval_datasetLoader = DataLoader(eval_dataset, shuffle=False, batch_size=batch_size, num_workers=0)

    # eval_img = eval_dataset[0]

    dataset_size = len(train_datasetLoder)
    eval_dataset_size = len(eval_datasetLoader)
    model = Unet(3, 1)
    def init_normal(m):
        if type(m) == torch.nn.Conv2d:
            init.normal_(m.weight.data, 0.0, 0.02)
    model.apply(init_normal)
    model.to(device)

    max_iou = 0
    total_steps = 0

    train_start_time = datetime.datetime.now()

    for epoch in range(1, 200):
        epoch_start_time = datetime.datetime.now()
        epoch_iter = 0
        eval_epoch_iter = 0
        eval_iou_stack = 0

        model.train()
        for i, data in enumerate(train_datasetLoder):
            total_steps += batch_size
            epoch_iter += batch_size
            model.set_input(data[0], data[1])
            model.optimize_params()

            # train_losses
            if total_steps % 2 == 0: # print frequency
                print_time = datetime.datetime.now()
                losses = model.get_current_losses()
                iter_time = print_time - epoch_start_time
                print_current_losses(epoch, epoch_iter, losses, iter_time)

        eval_start_time = datetime.datetime.now()
        losses = {}
        model.eval()
        for i, data in enumerate(eval_datasetLoader):
            model.set_input(data[0], data[1])
            model.eval_iou()
            eval_iou_stack += model.accu_iou()
            eval_epoch_iter += batch_size

        losses['eval_iou'] = eval_iou_stack / eval_dataset_size
        iter_time = datetime.datetime.now() - eval_start_time
        print_current_losses(epoch, eval_epoch_iter, losses, iter_time)

        if losses['eval_iou'] > max_iou:
            max_iou = losses['eval_iou']
            print('\033[92mBest eval IoU updated: {}\033[0m'.format(max_iou))
            logging.info('Best IoU is [{}] at [{}] epoch'.format(losses['eval_iou'], epoch))
            torch.save(model.state_dict(), '../model_weights/ISIC_RGB_Pytorch_Unet_epoch_{}.pth'.format(epoch))