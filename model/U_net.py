import torch
import os
import torch.nn as nn
from torch.nn import init
import logging
import functools

class double_conv(nn.Module):
    ''' Conv => Batch_Norm => ReLU => Conv2d => Batch_Norm => ReLU
    '''
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class inconv(nn.Module):
    ''' input conv layer
        let input 3 channels image to 64 channels
        The oly difference between `inconv` and `down` is maxpool layer 
    '''
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x

class down(nn.Module):
    ''' normal down path 
        MaxPool2d => double_conv
    '''
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            double_conv(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x

class up(nn.Module):
    ''' up path
        conv_transpose => double_conv
    '''
    def __init__(self, in_ch, out_ch, Transpose=False):
        super(up, self).__init__()

        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        if Transpose:
            self.up = nn.ConvTranspose2d(in_ch, in_ch//2, 2, stride=2)
        else:
            # self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                    nn.Conv2d(in_ch, in_ch//2, kernel_size=1, padding=0),
                                    nn.ReLU(inplace=True))
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2):
        ''' 
            conv output shape = (input_shape - Filter_shape + 2 * padding)/stride + 1
        '''

        x1 = self.up(x1)

        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = nn.functional.pad(x1, (diffX // 2, diffX - diffX//2,
                                    diffY // 2, diffY - diffY//2))

        x = torch.cat([x2,x1], dim=1)
        x = self.conv(x)
        return x

class outconv(nn.Module):
    ''' Output conv layer
        1conv
        shrink output channel to out_ch
    '''
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(2, out_ch, 1)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class Unet(nn.Module):
    def __init__(self, in_ch, out_ch, gpu_ids=[]):
        super(Unet, self).__init__()
        self.display_names = ['loss', 'matrix_iou']
        self.gpu_ids = gpu_ids
        self.bce_loss = nn.BCELoss()
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if torch.cuda.is_available() else torch.device('cpu')
        self.inc = inconv(in_ch, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.drop3 = nn.Dropout2d(0.5)
        self.down4 = down(512, 1024)
        self.drop4 = nn.Dropout2d(0.5)
        self.up1 = up(1024, 512, True)
        self.up2 = up(512, 256, True)
        self.up3 = up(256, 128, True)
        self.up4 = up(128, 64, True)
        self.outc = outconv(64, 1)
        # self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0005)

    def forward(self):
        x1 = self.inc(self.x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x4 = self.drop3(x4)
        x5 = self.down4(x4)
        x5 = self.drop4(x5)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        self.pred_y = nn.functional.sigmoid(x)

    def set_input(self, x, y):
        self.x = x.to(self.device)
        self.y = y.to(self.device)

    def optimize_params(self):
        self.forward()
        self._bce_iou_loss()
        _ = self.accu_iou()
        self.zero_grad()
        self.loss.backward()
        self.optimizer.step()

    def accu_iou(self):
        # B is the mask pred, A is the malanoma 
        y_pred = (self.pred_y > 0.5) * 1.0
        y_true = (self.y > 0.5) * 1.0
        pred_flat = y_pred.view(y_pred.numel())
        true_flat = y_true.view(y_true.numel())

        intersection = float(torch.sum(pred_flat * true_flat)) + 1e-7
        denominator = float(torch.sum(pred_flat + true_flat)) - intersection + 1e-7

        self.matrix_iou = intersection/denominator
        return self.matrix_iou

    def _bce_iou_loss(self):
        y_pred = self.pred_y
        y_true = self.y
        pred_flat = y_pred.view(y_pred.numel())
        true_flat = y_true.view(y_true.numel())

        intersection = torch.sum(pred_flat * true_flat) + 1e-7
        denominator = torch.sum(pred_flat + true_flat) - intersection + 1e-7
        iou = torch.div(intersection, denominator)
        bce_loss = self.bce_loss(pred_flat, true_flat)
        self.loss = bce_loss - iou + 1
        
    def get_current_losses(self):
        errors_ret = {}
        for name in self.display_names:
            if isinstance(name, str):
                errors_ret[name] = float(getattr(self, name))
        return errors_ret
        
    def eval_iou(self):
        with torch.no_grad():
            self.forward()




