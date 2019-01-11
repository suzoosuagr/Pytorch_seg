import datetime
import logging
import numpy as np
from utils.tools import set_logger, print_current_losses
from utils.options import Options
import torchvision 
from torchvision import transforms
from dataset.dataloader import ISICKerasDataset
from model.pix2pix import Pix2Pix
from torch.utils.data import DataLoader



transformations = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(), 
                                    transforms.ToTensor()])
eval_transformations = transforms.Compose([transforms.ToTensor()])

if __name__ == "__main__":
    set_logger('train.log')
    # prepare params for training
    opt = Options()
    args = opt.opts()
    args = opt.params(args)
    
    # Create datasetloader
    train_dataset = ISICKerasDataset(args.dataset_dir, data_type='train', transform=transformations)
    train_datasetLoder = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size, num_workers=0)

    # train_img = train_dataset[0]

    eval_dataset = ISICKerasDataset(args.dataset_dir, data_type='eval', transform=eval_transformations)
    eval_datasetLoader = DataLoader(eval_dataset, shuffle=False, batch_size=args.batch_size, num_workers=0)

    # eval_img = eval_dataset[0]

    dataset_size = len(train_datasetLoder)
    eval_dataset_size = len(eval_datasetLoader)
    model = Pix2Pix(input_nc=3, output_nc=1, ngf=64, isTrain=True, args=args)
    # print(model)
    max_iou = 0
    total_steps = 0

    train_start_time = datetime.datetime.now()
    for epoch in range(args.epoch_count, args.niter + args.niter_decay + 1):
        epoch_start_time = datetime.datetime.now()
        epoch_iter = 0
        eval_epoch_iter = 0
        eval_iou_stack = 0

        # data ==> [img_A, img_B]
        for i, data in enumerate(train_datasetLoder): # 1037
            total_steps += args.batch_size
            epoch_iter += args.batch_size
            # Train the model 
            model.set_input(data[0], data[1])
            model.optimize_parameters()

            # show the train_losses 
            if total_steps % args.print_freq == 0:
                print_time = datetime.datetime.now()
                losses = model.get_current_losses()
                losses['train_iou'] = model.get_current_iou()
                iter_time = print_time - epoch_start_time
                print_current_losses(epoch, epoch_iter, losses, iter_time)
        
        # For Eval
        eval_start_time = datetime.datetime.now()
        losses = {}
        for i, data in enumerate(eval_datasetLoader):
            model.set_input(data[0], data[1])
            model.eval()
            eval_iou_stack += model.get_current_iou()
            eval_epoch_iter += args.batch_size
        losses['eval_iou'] = eval_iou_stack / eval_dataset_size
        iter_time = datetime.datetime.now() - eval_start_time
        print_current_losses(epoch, eval_epoch_iter, losses, iter_time)

        # The max_iou is the eval iou
        if losses['eval_iou'] > max_iou:
            max_iou = losses['eval_iou']
            print('\033[92mBest eval IoU updated: {}\033[0m'.format(max_iou))
            logging.info('Best IoU is [{}] at [{}] epoch'.format(losses['eval_iou'], epoch))
            model.save_networks(epoch)

    train_time = datetime.datetime.now() - train_start_time
    logging.info('Finished {}'.format(train_time))

