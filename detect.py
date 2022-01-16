import sys
import os
import importlib
import cv2
import numpy as np
import time
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import torch.optim as optim

from objdetection.models.yolo import YOLONet
from objdetection.utils.yoloBoxCoder import YOLOBoxCoder, YOLOBoxPostProcess
from objdetection.loss.yoloLoss import YOLOLoss

if len(sys.argv) != 3:
    print("Usage: python3 detect.py config.py imageFile")
    sys.exit()
if not os.path.isfile(sys.argv[1]):
    print("Configuration file not found!")
    sys.exit()
trainingConfig = importlib.import_module(sys.argv[1][:-3]).TRAINING_PARAMS

#### Define Network ####
numClass = len(trainingConfig['labels'])
labels = trainingConfig['labels']
net = YOLONet(numClass=numClass, backboneName=trainingConfig['backbone'],
              backboneWeightFile=trainingConfig['backbonePretrained'])
saveModel = torch.load(trainingConfig['workingDirectory']+'/best.pth')
net.load_state_dict(saveModel['state_dict'])
net.eval()
if torch.cuda.is_available():
    net.cuda()
boxCoder = YOLOBoxCoder(anchors=trainingConfig['anchors'], numClass=numClass,
                        inputImageSize=trainingConfig['trainImageSize'], inputFeatMapSizes=net.outputFeatMapSizes)
torchTransform = [transforms.ToTensor()]
if trainingConfig['backbone'] == 'nasnetamobile':
    torchTransform.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))

#### Open Single Image ####
inputImage = cv2.imread(sys.argv[2])

oriImageHeight, oriImageWidth, _ = inputImage.shape
inputImagePrep = cv2.resize(cv2.cvtColor(inputImage,cv2.COLOR_BGR2RGB), trainingConfig['trainImageSize'])
inputImagePrep = transforms.Compose(torchTransform)(inputImagePrep).unsqueeze(0)
print('==== Network Loaded ====')

startInfer = time.time()
#### Network Forward ####
with torch.no_grad():
    outputs = net(inputImagePrep.cuda())


#### Single Image Decode ####
boxes = boxCoder.decode(outputs)
boxes = YOLOBoxPostProcess(boxes, numClass = numClass,objectThreshold=trainingConfig['detectionThreshold'],nmsThreshold=trainingConfig['nmsTheshold'])[0]
endInfer = time.time()
print('Detection finish in',endInfer-startInfer,'s')

re = cv2.resize(inputImage,trainingConfig['trainImageSize'])

if boxes is not None:
    trainH, trainW = trainingConfig['trainImageSize']
    ratioH, ratioW = oriImageHeight/trainH , oriImageWidth/trainW
    for x1,y1,x2,y2,conf,cls_pred in boxes:
        x1,y1,x2,y2 = int(x1*ratioW),int(y1*ratioH),int(x2*ratioW),int(y2*ratioH)
        print(x1,y1,x2,y2,conf,labels[int(cls_pred)])


        (text_width, text_height) = cv2.getTextSize(labels[int(cls_pred)]+':'+str(round(conf,3)),  cv2.FONT_HERSHEY_PLAIN, fontScale=1, thickness=1)[0]
        cv2.rectangle(inputImage,(x1,y1),(x2,y2),(0,255,0),2)
        cv2.rectangle(inputImage,(x1,y1),(x1+text_width-2,y1-text_height-2),(0,255,0),cv2.FILLED)
        cv2.putText(inputImage,labels[int(cls_pred)]+':'+str(round(conf,3)),(x1,y1), cv2.FONT_HERSHEY_PLAIN , 1,(0,0,0),1,cv2.LINE_AA)


       

cv2.imshow('Detection Output',inputImage)
cv2.waitKey(0)
