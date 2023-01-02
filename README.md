# TAVD

This repository contains the reference code for the paper *Towards Text-Aware Video Captioning with Hierarchical Modal Attention.*

## E-TVCaps Dataset
E-TVCaps Dataset can be download [here](https://taocaption.oss-cn-hangzhou.aliyuncs.com/TaVD_data/Tao_caption.csv).

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

To run the code,  features for the E-TVCaps dataset are needed. Please download the features file and place it in the Tao_data folder.

- [E-TVCaps_resnet152.hdf5](https://taocaption.oss-cn-hangzhou.aliyuncs.com/TaVD_data/Tao_resnet152.hdf5)
- [E-TVCaps_region.hdf5](https://taocaption.oss-cn-hangzhou.aliyuncs.com/TaVD_data/Tao_region.hdf5)
- [E-TVCaps_geometry.hdf5](https://taocaption.oss-cn-hangzhou.aliyuncs.com/TaVD_data/Tao_geometry.hdf5)
- [E-TVCaps_semantic.hdf5](https://taocaption.oss-cn-hangzhou.aliyuncs.com/TaVD_data/Tao_semantic.hdf5)
- [E-TVCaps_shot.hdf5](https://taocaption.oss-cn-hangzhou.aliyuncs.com/TaVD_data/Tao_shot.hdf5)
- [E-TVCaps_ocr_vector_512.hdf5](https://taocaption.oss-cn-hangzhou.aliyuncs.com/TaVD_data/Tao_ocr_vector_512.hdf5)

## Evaluation

To reproduce the results reported in our paper, download the pretrained model file [TaVD.pth](https://taocaption.oss-cn-hangzhou.aliyuncs.com/TaVD_data/TaVD.pth) and place it in the TaVD_data folder.

Run `python test.py` 

## Training procedure

Run `python train.py --exp_name TaVD --batch_size 32 --head 8 --features_path ../TaVD/Tao_resnet152.hdf5 ` 
