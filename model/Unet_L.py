import torch
import os
import torch.nn as nn
from torch.nn import init
import numpy as np
import logging
import functools
import skimage

# def save_img(img, name):
#     npimg = img.numpy()
#     npimg = np.reshape(npimg, (256,256))
#     max_scale = np.absolute(npimg)
#     max_scale = np.max(max_scale)
#     npimg = npimg/max_scale
#     npimg = skimage.img_as_ubyte(npimg)
#     skimage.io.imsave('./results/'+ name + '.jpg', npimg)

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
        self.conv.apply(self.init_weights)
    
    def forward(self, x):
        x = self.conv(x)
        return x

    @staticmethod
    def init_weights(m):
        if type(m) == nn.Conv2d:
            init.xavier_normal(m.weight)
            init.constant(m.bias,0)

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
            self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
        else:
            # self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                    nn.Conv2d(in_ch, out_ch, kernel_size=1, padding=0),
                                    nn.ReLU(inplace=True))
        self.conv = double_conv(in_ch+3, out_ch)
        self.up.apply(self.init_weights)

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

    @staticmethod
    def init_weights(m):
        if type(m) == nn.Conv2d:
            init.xavier_normal(m.weight)
            init.constant(m.bias,0)

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
        self.conv.apply(self.init_weights)

    def forward(self, x):
        x = self.conv(x)
        return x

    @staticmethod
    def init_weights(m):
        if type(m) == nn.Conv2d:
            init.xavier_normal(m.weight)
            init.constant(m.bias,0)

class laplayer(nn.Module):
    def __init__(self):
        super(laplayer, self).__init__()
        lap_filter = np.array([[0, 1, 0],
                                [1, -4,1],
                                [0, 1, 0]])
        self.laplayer = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=3//2)
        self.laplayer.weight.data.copy_(torch.from_numpy(lap_filter))
        self.laplayer.bias.data.copy_(torch.from_numpy(np.array([0.0])))
        for param in self.laplayer.parameters():
            param.requires_grad = False

    def forward(self, img):
        img_r = img[:,0:1,:,:]
        img_g = img[:,1:2,:,:]
        img_b = img[:,2:3,:,:]

        lap_r = self.laplayer(img_r)
        lap_g = self.laplayer(img_g)
        lap_b = self.laplayer(img_b)
        lap = torch.cat((lap_r, lap_g, lap_b), 1)
        return lap

class sobel_layer(nn.Module):
    def __init__(self):
        super(sobel_layer, self).__init__()
        sobel_x_filter = np.array(
            [[1, 0, -1],
             [2, 0, -2],
             [1, 0, -1]]
        )
        sobel_y_filter = np.array(
            [[ 1,  2,  1],
             [ 0,  0,  0],
             [-1, -2, -1]]
        )
        self.sobel_x = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=3//2)
        self.sobel_x.weight.data.copy_(torch.from_numpy(sobel_x_filter))
        self.sobel_x.bias.data.copy_(torch.from_numpy(np.array([0.0])))
        for param in self.sobel_x.parameters():
            param.requires_grad = False

        self.sobel_y = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=3//2)
        self.sobel_y.weight.data.copy_(torch.from_numpy(sobel_y_filter))
        self.sobel_y.bias.data.copy_(torch.from_numpy(np.array([0.0])))
        for param in self.sobel_y.parameters():
            param.requires_grad = False

    def forward(self, img):
        img_r = img[:,0:1,:,:]
        img_g = img[:,1:2,:,:]
        img_b = img[:,2:3,:,:]

        sobel_r_x = self.sobel_x(img_r)
        sobel_g_x = self.sobel_x(img_g)
        sobel_b_x = self.sobel_x(img_b)

        sobel_r_y = self.sobel_y(img_r)
        sobel_g_y = self.sobel_y(img_g)
        sobel_b_y = self.sobel_y(img_b)

        sobel = torch.cat((sobel_r_x, sobel_r_y, sobel_g_x, sobel_g_y, sobel_b_x, sobel_b_y), 1)
        return sobel

class Unet_L(nn.Module):
    def __init__(self, in_ch, out_ch, gpu_ids=[]):
        super(Unet_L, self).__init__()
        self.loss_stack = 0
        self.matrix_iou_stack = 0
        self.stack_count = 0
        self.display_names = ['loss_stack', 'matrix_iou_stack']
        self.gpu_ids = gpu_ids
        self.bce_loss = nn.BCELoss()
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if torch.cuda.is_available() else torch.device('cpu')
        self.inc = inconv(in_ch, 64)
        # self.lap = laplayer()
        self.sobel = sobel_layer()
        self.max_pool = nn.MaxPool2d(2)
        self.down1 = down(64, 128)
        # print(list(self.down1.parameters()))
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.drop3 = nn.Dropout2d(0.5)
        self.down4 = down(512, 1024)
        self.drop4 = nn.Dropout2d(0.5)
        self.up1 = up(1024, 512, False)
        self.up2 = up(512, 256, False)
        self.up3 = up(256, 128, False)
        self.up4 = up(128, 64, False)
        self.outc = outconv(64, 1)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        # self.optimizer = torch.optim.SGD(self.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0005)

    def forward(self):
        lap1 = self.sobel(self.x)
        lap2 = self.max_pool(lap1)
        lap3 = self.max_pool(lap2)
        lap4 = self.max_pool(lap3)

        x1 = self.inc(self.x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x4 = self.drop3(x4)
        x5 = self.down4(x4)
        x5 = self.drop4(x5)

        x4 = torch.cat((x4, lap4), 1)
        x = self.up1(x5, x4)
        x3 = torch.cat((x3, lap3), 1)
        x = self.up2(x, x3)
        x2 = torch.cat((x2, lap2), 1)
        x = self.up3(x, x2)
        x1 = torch.cat((x1, lap1), 1)
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
        self.stack_count += 1
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
        self.matrix_iou_stack += self.matrix_iou
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
        self.loss_stack += self.loss
        
    def get_current_losses(self):
        errors_ret = {}
        for name in self.display_names:
            if isinstance(name, str):
                errors_ret[name] = float(getattr(self, name)) / self.stack_count
        self.loss_stack = 0
        self.matrix_iou_stack = 0
        self.stack_count = 0
        return errors_ret
        
    def eval_iou(self):
        with torch.no_grad():
            self.forward()
            self._bce_iou_loss()
            _ = self.accu_iou()
            self.stack_count += 1




