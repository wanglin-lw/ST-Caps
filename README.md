# STVD

This repository contains the reference code for the paper *ST-Caps: Towards Scene Text Video Captioning with Hierarchical Contrastive Learning.*

## E-TVCaps Dataset
ST-Caps Dataset can be download [here](https://taocaption.oss-cn-hangzhou.aliyuncs.com/TaVD_data/Tao_caption.csv).

## Environment

### Requirements

- Ubuntu 18.04
- CUDA 11.4
- Nvidia Geforce GTX 3090

### Setup

Clone the repository and setup python environment

```
pip install --upgrade pip
pip install -r requirements.txt
```

## Prepare Data

To run the code,  features for the ST-Caps dataset are needed. Please download the features file and place it in the STVD_data folder.

- [ST-Caps_resnet152.hdf5](https://taocaption.oss-cn-hangzhou.aliyuncs.com/TaVD_data/Tao_resnet152.hdf5)
- [ST-Caps_region.hdf5](https://taocaption.oss-cn-hangzhou.aliyuncs.com/TaVD_data/Tao_region.hdf5)
- [ST-Caps_geometry.hdf5](https://taocaption.oss-cn-hangzhou.aliyuncs.com/TaVD_data/Tao_geometry.hdf5)
- [ST-Caps_semantic.hdf5](https://taocaption.oss-cn-hangzhou.aliyuncs.com/TaVD_data/Tao_semantic.hdf5)
- [ST-Caps_shot.hdf5](https://taocaption.oss-cn-hangzhou.aliyuncs.com/TaVD_data/Tao_shot.hdf5)
- [ST-Caps_ocr_vector_512.hdf5](https://taocaption.oss-cn-hangzhou.aliyuncs.com/TaVD_data/Tao_ocr_vector_512.hdf5)

## Evaluation

To reproduce the results reported in our paper, download the pretrained model file [STVD.pth](https://taocaption.oss-cn-hangzhou.aliyuncs.com/TaVD_data/TaVD.pth) and place it in the STVD_data folder.

Run `python test.py` 

## Training procedure

Run `python train.py --exp_name STVD --batch_size 32 --head 8 ` 
