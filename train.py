import sys
import os
import importlib
import cv2
import numpy as np
from tqdm import tqdm
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import torch.optim as optim
import imgaug as ia
from imgaug import augmenters as iaa

from objdetection.dataset.listDataset import ListDataset
from objdetection.models.yolo import YOLONet
from objdetection.utils.yoloBoxCoder import YOLOBoxCoder, YOLOBoxPostProcess
from objdetection.loss.yoloLoss import YOLOLoss

if len(sys.argv) != 2:
    print("Usage: python3 train.py config.py")
    sys.exit()
if not os.path.isfile(sys.argv[1]):
    print("Configuration file not found!")
    sys.exit()
trainingConfig = importlib.import_module(sys.argv[1][:-3]).TRAINING_PARAMS

#### Define Network ####
numClass = len(trainingConfig['labels'])
net = YOLONet(numClass=numClass, backboneName=trainingConfig['backbone'],
              backboneWeightFile=trainingConfig['backbonePretrained'])
if trainingConfig['lr']['freezeBackbone']:
    for layerParam in net.backbone.parameters():
        layerParam.requires_grad = False

if torch.cuda.is_available():
    net.cuda()

boxCoder = YOLOBoxCoder(anchors=trainingConfig['anchors'], numClass=numClass,
                        inputImageSize=trainingConfig['trainImageSize'], inputFeatMapSizes=net.outputFeatMapSizes)

yoloLoss = YOLOLoss(numClass=numClass)
optimizer = optim.SGD(net.parameters(), lr=0.001,momentum=0.9,weight_decay=5e-4)

#### Dataset Transformer and Loader ####
imgAugTransform = iaa.Sequential([
    iaa.Scale({"height": trainingConfig['trainImageSize']
               [0], "width": trainingConfig['trainImageSize'][1]}),
    iaa.Multiply((0.6, 1.6)),
    iaa.AffineCv2(
        rotate=(-5, 5),
        scale=(0.8, 1.2),
        shear=(-15, 15)
    ),
    iaa.Fliplr(0.5),
    iaa.AddToHueAndSaturation((-20, 20), per_channel=True),
    iaa.GaussianBlur(sigma=(0, 3.0)),
    iaa.CoarseDropout(p=0.05, size_percent=0.5),
])

torchTransform = [transforms.ToTensor()]
if trainingConfig['backbone'] == 'nasnetamobile':
    torchTransform.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))

def testTransformer(image, boxes):
    oriImageHeight, oriImageWidth, _ = image.shape
    image = cv2.resize(image, trainingConfig['trainImageSize'])

    boxes = boxes / \
        torch.tensor([oriImageWidth, oriImageHeight,
                      oriImageWidth, oriImageHeight, 1.0])
    boxes[:, 0] = boxes[:, 0] + boxes[:, 2]/2
    boxes[:, 1] = boxes[:, 1] + boxes[:, 3]/2

    image = transforms.Compose(torchTransform)(image)

    encodeBoxes, objMask, noObjMask = boxCoder.encode(boxes)

    return image, (encodeBoxes, objMask, noObjMask)


def trainTransformer(image, boxes):
    oriImageHeight, oriImageWidth, _ = image.shape

    imgAugTransformDet = imgAugTransform.to_deterministic()
    xyxyboxes = boxes.numpy().astype(np.float)
    xyxyboxes[:, 2] += boxes[:, 0]
    xyxyboxes[:, 3] += boxes[:, 1]
    imgAugBboxList = [ia.BoundingBox(
        x1=box[0], y1=box[1], x2=box[2], y2=box[3], label=box[4].item()) for box in xyxyboxes]
    augBoxes = imgAugTransformDet.augment_bounding_boxes(
        [ia.BoundingBoxesOnImage(imgAugBboxList, shape=image.shape)])[0].cut_out_of_image()
    augmentedBoxes = np.zeros((len(augBoxes.bounding_boxes),5),dtype=np.float32)
    augmentedBoxes = boxes.clone()
    #print(len(augBoxes.bounding_boxes),boxes.shape)
    for boxIdx, augBox in enumerate(augBoxes.bounding_boxes):
        augmentedBoxes[boxIdx, 0] = augBox.x1 + (augBox.x2-augBox.x1)/2
        augmentedBoxes[boxIdx, 1] = augBox.y1 + (augBox.y2-augBox.y1)/2
        augmentedBoxes[boxIdx, 2] = augBox.x2-augBox.x1
        augmentedBoxes[boxIdx, 3] = augBox.y2-augBox.y1
        augmentedBoxes[boxIdx, 4] = augBox.label

    # Normalized
    augmentedBoxes = augmentedBoxes / torch.tensor([trainingConfig['trainImageSize'][0], trainingConfig['trainImageSize']
                                                    [1], trainingConfig['trainImageSize'][0], trainingConfig['trainImageSize'][1], 1.0])

    image = imgAugTransformDet.augment_image(image)

    image = transforms.Compose(torchTransform)(image)

    encodeBoxes, objMask, noObjMask = boxCoder.encode(augmentedBoxes)
    return image, (encodeBoxes, objMask, noObjMask)


trainset = ListDataset(
    root=trainingConfig['imageDirectory'], listFile=trainingConfig['trainListFile'], transform=trainTransformer)
trainsetLoader = data.DataLoader(
    trainset, batch_size=trainingConfig['batchSize'], shuffle=True, num_workers=trainingConfig['numDataloaderThread'], pin_memory=True)

testset = ListDataset(
    root=trainingConfig['imageDirectory'], listFile=trainingConfig['valListFile'], transform=testTransformer)
testsetLoader = data.DataLoader(
    testset, batch_size=trainingConfig['batchSize'], shuffle=False, num_workers=trainingConfig['numDataloaderThread'], pin_memory=True)




print('==== Network Summary ====')
print(net)
print('==== Dataset Summary ====')
print('Total training images :',len(trainset))
print('Total testing images :',len(testset))
print('=========================')

def launchEpoch(net, dataloader, mode='train'):
    if mode == 'train':
        net.train()
    else:
        net.eval()
    
    epochLoss = 0
    accuX = 0
    accuY = 0
    accuW = 0
    accuH = 0
    accuObj = 0
    accuNoObj = 0
    accuClassConf = 0

    progressBar = tqdm(enumerate(dataloader),total=len(dataloader))
    for batchIdx, (images, data) in progressBar:
        targetBoxes, objMask , noObjMask = data
        out = net(images.cuda())
        loss, lossX , lossY , lossW , lossH , lossObjConf , loossNoObjConf,  lossClassConf = yoloLoss(out,targetBoxes.cuda(), objMask.cuda(), noObjMask.cuda())
        optimizer.zero_grad()
        loss.backward()

        epochLoss+=loss.item()
        accuX += lossX.item() 
        accuY += lossY.item()
        accuW += lossW.item()
        accuH += lossH.item()
        accuObj += lossObjConf.item()
        accuNoObj += loossNoObjConf.item()
        accuClassConf += lossClassConf.item()

        progressBar.set_description("totalLoss: %f | locLoss: %f | objLoss: %f | noObjLoss: %f | classConfLoss: %f" 
        % ( epochLoss/(batchIdx+1), (accuX+accuY+accuW+accuH)/(batchIdx+1), accuObj/(batchIdx+1) , accuNoObj/(batchIdx+1),  accuClassConf/(batchIdx+1)))
        
        optimizer.step()
    epochTotalLoss = epochLoss/len(dataloader)
    return epochTotalLoss


bestLoss = float('inf')
bestEpoch = None
startEpoch = 0

#### Create working directory ###
if not os.path.exists(trainingConfig['workingDirectory']):
    os.makedirs(trainingConfig['workingDirectory'])
print('Training Log and Model will be saved at:',trainingConfig['workingDirectory'])
input('Press enter to start training...')
for epoch in range(trainingConfig['epochs']):
    print('Epoch',epoch,'/',trainingConfig['epochs'])

    ### Train ###
    if trainingConfig['lr']['freezeBackbone'] == False and epoch == trainingConfig['thawBackboneEpoch']:
        for layerParam in net.backbone.parameters():
            layerParam.requires_grad = True
    

    launchEpoch(net, trainsetLoader, mode='train')

    ### Test ###
    epochTestLoss = launchEpoch(net, testsetLoader, mode='test')
    if epochTestLoss < bestLoss:
        print('**Test Loss decrease (%f)==>(%f). Saving Model...' % (bestLoss,epochTestLoss))
        bestLoss = epochTestLoss
        bestEpoch = epoch

        torch.save({
                    'state_dict': net.state_dict(),
                    'epoch': bestEpoch,
                    },trainingConfig['workingDirectory']+'/best.pth')
        
    else:
        print('**Last best epoch: %d | best loss: %f' % (bestEpoch,bestLoss))
    
    if (epoch+1) % 50 == 0:
        print('Checkpoint. Saving Model...')
        torch.save({
                    'state_dict': net.state_dict(),
                    'epoch': epoch,
                    'optimizer': optimizer.state_dict(),
                    },trainingConfig['workingDirectory']+'/epoch'+str(epoch+1)+'.pth')

