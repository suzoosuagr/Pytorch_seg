import argparse

class Options():
    def __init__(self):
        self.initialized=False
        self.parser=argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    def opts(self):
        self.parser.add_argument('--dataset_dir', default='../data/ISICKeras/', help='The dir of dataset')
        self.parser.add_argument('--logger_dir', default='../pytorch_logger.log', help='The dir of logger')
        self.parser.add_argument('--weights_dir', default='../../model_weights/ISIC_gray_inputs_cGAN_pix2pix_pytorch/', help='The dir of trained weights')
        self.parser.add_argument('--summary_dir', default='../experiments/ISIC_gray_inputs_cGAN_pix2pix_pytorch/', help='The dir for summary(tensorboard)')
        self.initialized=True
        return self.parser.parse_args()
    
    def params(self, args):
        args.save_dir = '../model_weights/'
        args.save_filename = 'CGAN_Pytorch'

        args.batch_size = 2
        args.epochs = 200
        args.no_lsgan = False
        args.lr = 0.0002
        args.ngf = 64
        args.ndf = 64
        args.gpu_ids = []
        args.lambda_L1 = 100

        args.epoch_count = 1
        args.niter = 100
        args.niter_decay = 100
        args.print_freq = 2 # when debug print_freq = 2 for train print_freq = 100

        return args

