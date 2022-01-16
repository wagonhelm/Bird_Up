# YOLOv3 Implementation in PyTorch (WIP)

## Required Package
- Python3
- PyTorch >= 0.4.1
- OpenCV
- Imgaug (https://github.com/aleju/imgaug)
- Pillow-SIMD (https://github.com/uploadcare/pillow-simd)
## Pretrained Weights & Raccoon Dataset Weights
https://drive.google.com/drive/folders/1Fh1mTowJR1KuVhCTRAnEL28UpKUyD7lx?usp=sharing (1)


## How to use?
Training (Network configuration is defined in config-raccoon.py)
```
python3 train.py config-raccoon.py
```
Predict (Download racccoon/best.pth from above link and put in trainingLogs/raccoonDataset/best.pth)
```
python3 detect.py config-raccoon.py datasets/raccoon/raccoon-10.jpg
```

## Available Backbones
- Darknet53
- Darknet21 (WIP)
- NASNetAMobile

## TODO:
- Training visualization using TensorboardX
- MAP evaluation
- User friendly documents
- More backbone networks
- CPU only Inference

# Credits: https://github.com/sapjunior/objdetection-pytorch
