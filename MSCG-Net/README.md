# MSCG-Net for Semantic Segmentation
## Introduce
This repository contains code modified from MSCG-Net models (MSCG-Net-50 and MSCG-Net-101) for semantic segmentation in [Agriculture-Vision Challenge and Workshop](https://www.agriculture-vision.com/agriculture-vision-2021/prize-challenge-2021) (CVPR 2021). Original readme file can be found [here](MSCGNET-README.MD)

## Code structure

```
├── config		# config code
├── data		# dataset loader and pre-processing code
├── tools		# train and test code, ckpt and model_load
├── lib			# model block, loss, utils code, etc
└── ckpt 		# output check point, trained weights, log files, etc

```

## Environments

- python 3.7
- pytorch 1.7.1
- opencv
- tensorboardx
- albumentations
- pretrainedmodels
- others (see requirements.txt)

## Pretrained model
[https://drive.google.com/file/d/1oW503NxUfwANfKQZ8zT3gG_XDWSuwwsQ/view?usp=sharing](https://drive.google.com/file/d/1oW503NxUfwANfKQZ8zT3gG_XDWSuwwsQ/view?usp=sharing)

## Dataset prepare

1. change DATASET_ROOT to your dataset path in ./data/AgricultureVision/pre_process.py
```
DATASET_ROOT = '/your/path/to/Agriculture-Vision'
```

2. keep the dataset structure as the same with the official structure shown as below
```
Agriculture-Vision
|-- train
|   |-- masks
|   |-- labels
|   |-- boundaries
|   |-- images
|   |   |-- nir
|   |   |-- rgb
|-- val
|   |-- masks
|   |-- labels
|   |-- boundaries
|   |-- images
|   |   |-- nir
|   |   |-- rgb
|-- test
|   |-- boundaries
|   |-- images
|   |   |-- nir
|   |   |-- rgb
|   |-- masks
```

## Train with all available GPUs

```
python train_R101.py
```

## Test

```
python test_submission.py
```

#### Trained weights for R101 (save to ./ckpt/R101_baseline before run test_submission)
[ckpt](https://drive.google.com/drive/folders/1RisJyMAqxawGebnky8C35a5KhfuFcaGD?usp=sharing)

## Results Summary
mIoU: 0.464251


