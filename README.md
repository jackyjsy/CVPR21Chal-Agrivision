# CVPR 2021 Challenge on Agriculture Vision
This repo contains the code to reproduce our results in CVPR21 Challenge on Agriculture-Vision. We ranked 4th in the supervised track.

By [Songyao Jiang](https://www.songyaojiang.com/), [Bin Sun](https://github.com/Sun1992/), and [Yun Fu](http://www1.ece.neu.edu/~yunfu/), from [Smile Lab @ Northeastern University](https://web.northeastern.edu/smilelab/)

## Introduction
The first model is modified MSCG-Net, please see [README.md](MSCG-Net/README.md) to train and test the model. The second model is modified MSCG-Net, please see [README.md](Deeplabv3_Ensemble/Readme.txt) to train and test the model. The results of the above models are assembled together to improve the overall mIoU using the ensemble code in Deeplabv3_Ensemble. We used ensemble results from two models to be our final submitted results in the challenge

## Pretrained models
[Google Drive](https://drive.google.com/drive/folders/1hwGQ_aQbLREs2srYm9ktPTJbnFexcTc6?usp=sharing)

## Code structure

```
├── MSCGNet                # Model 1
├── Deeplabv3_Ensemble	   # Model 2 and ensemble
└── challenge_report       # Detailed report submitted

```
## Results Summary
| Model      | Backbone | #Params | mIoU |
| ---------  | -------- | ------- | ---- |
| MSCG-Net   | ResNet-101 | 31M | 0.464 |
| DeepLabv3  | ResNet-101 | 60M | 0.494 |
| Ensemble   | N/A | 91M | 0.507 |


---

## [MSCG-Net for Semantic Segmentation](MSCG-Net)
## Introduce
This model is modified from MSCG-Net models (MSCG-Net-50 and MSCG-Net-101) for semantic segmentation in [Agriculture-Vision Challenge and Workshop](https://www.agriculture-vision.com/agriculture-vision-2021/prize-challenge-2021) (CVPR 2021). 

### Pretrained model
[https://drive.google.com/file/d/1oW503NxUfwANfKQZ8zT3gG_XDWSuwwsQ/view?usp=sharing](https://drive.google.com/file/d/1oW503NxUfwANfKQZ8zT3gG_XDWSuwwsQ/view?usp=sharing)


## [DeepLabv3 with Ensemble](Deeplabv3_Ensemble)
### Introduce
This repository contains code modified from Deeplabv3 for the CVPR 2021 Challenge on Agriculture Vision. This folder also contains ensemble code to obtain our final results.

### Pretrained models
[https://drive.google.com/drive/folders/1VnPKVErUHEjbCe5ailsSvhXCjZWnw0qH?usp=sharing](https://drive.google.com/drive/folders/1VnPKVErUHEjbCe5ailsSvhXCjZWnw0qH?usp=sharing)




