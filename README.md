# TAVD

This repository contains the reference code for the paper *Tao caption： Twords Text-aware Oriented Video Captioning*

## Tao-Caption Dataset
Tao-Caption Dataset can be download [here](https://taocaption.oss-cn-hangzhou.aliyuncs.com/TaVD_data/Tao_caption.csv).

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

To run the code,  features for the Tao-caption dataset are needed. Please download the features file and place it in the Tao_data folder.

- [Tao_resnet152.hdf5](https://taocaption.oss-cn-hangzhou.aliyuncs.com/TaVD_data/Tao_resnet152.hdf5)
- [Tao_region.hdf5](https://taocaption.oss-cn-hangzhou.aliyuncs.com/TaVD_data/Tao_region.hdf5)
- [Tao_geometry.hdf5](https://taocaption.oss-cn-hangzhou.aliyuncs.com/TaVD_data/Tao_geometry.hdf5)
- [Tao_semantic.hdf5](https://taocaption.oss-cn-hangzhou.aliyuncs.com/TaVD_data/Tao_semantic.hdf5)
- [Tao_shot.hdf5](https://taocaption.oss-cn-hangzhou.aliyuncs.com/TaVD_data/Tao_shot.hdf5)
- [Tao_ocr_vector_512.hdf5](https://taocaption.oss-cn-hangzhou.aliyuncs.com/TaVD_data/Tao_ocr_vector_512.hdf5)

## Evaluation

To reproduce the results reported in our paper, download the pretrained model file [TaVD.pth](https://taocaption.oss-cn-hangzhou.aliyuncs.com/TaVD_data/TaVD.pth) and place it in the TaVD_data folder.

Run `python test.py` 

## Training procedure

Run `python train.py --exp_name TaVD --batch_size 32 --head 8 --features_path ../TaVD/Tao_resnet152.hdf5 ` 
