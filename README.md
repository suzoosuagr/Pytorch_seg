# Unet_Pytorch

## Dataset
 ISIC2018 Task1 training data 
> ISIC2018 Task1 training data  [ISIC2018](https://challenge2018.isic-archive.com/task1/)

Directory Structure:
```
../data/ISICKeras/  
                    |-- eval
                    |   |-- image
                    |   |   |-- ISIC_0000039.jpg
                    |   |   |-- ...
                    |   |
                    |   |-- label
                    |       |-- ISIC_0000039_segmentation.png
                    |       |-- ...
                    |
                    |-- train
                        |-- image
                        |   |-- ISIC_0000000.jpg
                        |   |-- ...
                        |   
                        |-- label
                            |-- ISIC_0000000_segmentation.png
                            |-- ...

```
## Pix2Pix (working)
**train**:
```bash
$ CUDA_VISIBLE_DEVICES=0 python train.py
```
## Unet (working)
**train**
```bash
$ CUDA_VISIBLE_DEVICES=0, python train_unet.py
```

## Problem
keras版本的Unet 训练 在40 epoch左右就可以达到 **eval_iou 0.74**
但是torch版本的效果却不好114epoch后也只有**eval_iou0.38**
两个版本之间的区别是torch版本我有加batch_norm层(参考的https://github.com/milesial/Pytorch-UNet)