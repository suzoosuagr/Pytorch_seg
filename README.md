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
