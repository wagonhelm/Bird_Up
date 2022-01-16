import torch
import math
import numpy as np
from .box import nms, bboxIOU


class YOLOBoxCoder():
    def __init__(self, anchors, numClass, inputFeatMapSizes, inputImageSize=(412, 412)):
        self.anchors = anchors
        self.numClass = numClass
        self.inputFeatMapSizes = inputFeatMapSizes

        self.numAnchors = [len(anchor) for anchor in self.anchors]
        self.scaledAnchors = []

        self.inputImageHeight, self.inputImageWidth = inputImageSize[0], inputImageSize[1]

        # Ratio between inputImageSize and inputFeatMapSize
        self.heightRatio = []
        self.widthRatio = []

        self.anchorBoxes, self.widthHeightRatios = self._getDefaultAnchor()
    
    def _getDefaultAnchor(self):
        anchorBoxes = torch.empty((0,4),requires_grad = False)
        widthHeightRatios = torch.empty((0,4),requires_grad = False)

        #### Anchor Generation ####
        for featMapIdx, inputFeatMapSize in enumerate(self.inputFeatMapSizes):
            inputFeatMapHeight, inputFeatMapWidth = inputFeatMapSize

            heightRatio = self.inputImageHeight / inputFeatMapHeight
            widthRatio = self.inputImageWidth / inputFeatMapWidth
            widthRatioTensor = torch.tensor([widthRatio] * inputFeatMapWidth * inputFeatMapHeight * self.numAnchors[featMapIdx])
            heightRatioTensor = torch.tensor([heightRatio] * inputFeatMapWidth * inputFeatMapHeight * self.numAnchors[featMapIdx])
            widthHeightRatio = torch.stack((widthRatioTensor,heightRatioTensor), dim=1).repeat(1,2)
            widthHeightRatios = torch.cat((widthHeightRatios,widthHeightRatio), dim=0)

            # Scale from anchor in inputImageSize world to inputFeatMapSize
            scaledAnchors = [(anchorWidth / widthRatio, anchorHeight / heightRatio)
                             for anchorWidth, anchorHeight in self.anchors[featMapIdx]]
            self.scaledAnchors.append(scaledAnchors)

            # Grid offset equal to inputFeatMapSize
            gridX = torch.linspace(0, inputFeatMapWidth-1, inputFeatMapWidth).view(
                inputFeatMapWidth, 1).repeat(inputFeatMapWidth, self.numAnchors[featMapIdx]).flatten()
            gridY = torch.linspace(0, inputFeatMapHeight-1, inputFeatMapHeight).view(
                inputFeatMapHeight, 1).repeat(1,inputFeatMapHeight * self.numAnchors[featMapIdx]).flatten()

            # Anchor width hiehgt for each grid location
            anchorW = torch.tensor(scaledAnchors).index_select(1, torch.tensor([0])).flatten()
            anchorH = torch.tensor(scaledAnchors).index_select(1, torch.tensor([1])).flatten()
            
            anchorW = anchorW.repeat(inputFeatMapHeight * inputFeatMapWidth).flatten()
            anchorH = anchorH.repeat(inputFeatMapHeight * inputFeatMapWidth).flatten()

            anchorBoxes = torch.cat((anchorBoxes, torch.stack((gridX,gridY,anchorW,anchorH),dim=1)), dim=0)
            
        return anchorBoxes.cuda(), widthHeightRatios.cuda()

    def encode(self, boxes):
        
        target = torch.zeros((self.anchorBoxes.shape[0], 5 + self.numClass), requires_grad=False) # x,y,w,h,objMask,
        objMask = torch.zeros((self.anchorBoxes.shape[0]), requires_grad=False, dtype=torch.uint8)
        noObjMask = torch.ones((self.anchorBoxes.shape[0]), requires_grad=False, dtype=torch.uint8)

        for box in boxes:

            boxListIdxOffset = 0

            for featMapIdx, inputFeatMapSize in enumerate(self.inputFeatMapSizes):

                gx = box[0] * self.inputFeatMapSizes[featMapIdx][0]
                gy = box[1] * self.inputFeatMapSizes[featMapIdx][1]
                gw = box[2] * self.inputFeatMapSizes[featMapIdx][0]
                gh = box[3] * self.inputFeatMapSizes[featMapIdx][1]
                label = int(box[4])


                gi = int(gx)
                gj = int(gy)

                gtBox = torch.tensor(np.array([0, 0, gw, gh]), dtype=torch.float32).unsqueeze(0)
                anchorBoxes = torch.tensor(np.concatenate((np.zeros((self.numAnchors[featMapIdx], 2)),
                                                            np.array(self.scaledAnchors[featMapIdx])), 1), dtype=torch.float32)
                ious = bboxIOU(gtBox, anchorBoxes)
                
                for idx,iou in enumerate(ious):
                    if iou > 0.5:
                        boxListIdx = ((gi + (gj*self.inputFeatMapSizes[featMapIdx][1])) * self.numAnchors[featMapIdx]) + idx + boxListIdxOffset
                        noObjMask[boxListIdx] = 0
                        
                # Find the best matching anchor box
                bestIdx = torch.argmax(ious)

                boxListIdx = (gi + (gj*self.inputFeatMapSizes[featMapIdx][1]) ) * self.numAnchors[featMapIdx] + bestIdx + boxListIdxOffset 

                objMask[boxListIdx] = 1
                noObjMask[boxListIdx] = 0

                target[boxListIdx,0] = gx - gi #x
                target[boxListIdx,1] = gy - gj #y
                
                target[boxListIdx,2] = math.log(gw / self.scaledAnchors[featMapIdx][bestIdx][0] + 1e-16) #h
                target[boxListIdx,3] = math.log(gh / self.scaledAnchors[featMapIdx][bestIdx][1] + 1e-16) #w
                target[boxListIdx,4] = 1.0      #objscore
                target[boxListIdx,5+label] = 1.0    # one-hot for class

                boxListIdxOffset += self.inputFeatMapSizes[featMapIdx][0] * self.inputFeatMapSizes[featMapIdx][1] * self.numAnchors[featMapIdx]
                
        return target, objMask, noObjMask
    
    def decode(self, inputFeatMaps):
        predictions = []
        for featMapIdx, inputFeatMap in enumerate(inputFeatMaps):
            '''
            prediction ==> (batchNo, numBoxes, 5+numClass)
            [ 
              [x,y,w,h,objScore,classScore,....],
              [x,y,w,h,objScore,classScore,....],
            ]
            '''
            prediction = inputFeatMap.permute(0, 2, 3, 1).contiguous().view(inputFeatMap.shape[0], -1, self.numClass+5) 
            predictions.append(prediction)

        predictions = torch.cat(predictions,1)

        predictions[:,:,0] = torch.sigmoid(predictions[:,:,0]) + self.anchorBoxes[:,0]  # x
        predictions[:,:,1] = torch.sigmoid(predictions[:,:,1]) + self.anchorBoxes[:,1]  # y
        predictions[:,:,2] = torch.exp(predictions[:,:,2]) * self.anchorBoxes[:,2] # w
        predictions[:,:,3] = torch.exp(predictions[:,:,3]) * self.anchorBoxes[:,3] #
        predictions[:,:,:4] = predictions[:,:,0:4] * self.widthHeightRatios # scale back coordinate to inputImageSize
        predictions[:,:,4:] = torch.sigmoid(predictions[:,:,4:]) # objscore-classconf
        
        return predictions

def YOLOBoxPostProcess(xyCenterBoxes, numClass, objectThreshold=0.65, nmsThreshold=0.45):
    # xyCenterBoxes ==> [ [[x-center, y-center, width, height, objectnessConf, classScores....]] ]
    # Convert from (x-center, y-center, width, height) to (x1, y1, x2, y2) coordinate
    xyBoxes = torch.empty(xyCenterBoxes.size())
    xyBoxes[:, :, 0] = xyCenterBoxes[:, :, 0] - xyCenterBoxes[:, :, 2] / 2
    xyBoxes[:, :, 1] = xyCenterBoxes[:, :, 1] - xyCenterBoxes[:, :, 3] / 2
    xyBoxes[:, :, 2] = xyCenterBoxes[:, :, 0] + xyCenterBoxes[:, :, 2] / 2
    xyBoxes[:, :, 3] = xyCenterBoxes[:, :, 1] + xyCenterBoxes[:, :, 3] / 2
    xyBoxes[:, :, 4:] = xyCenterBoxes[:, :, 4:]
    boxesOutput = []
    for imageNo in range(xyBoxes.shape[0]):
        imageBoxes = xyBoxes[imageNo, :, :]
        
        # Filter out boxes with low objectness score
        thresholdMask = (imageBoxes[:, 4] >= objectThreshold).squeeze()
        imageBoxes = imageBoxes[thresholdMask]

        # No remaining boxes
        if not imageBoxes.size(0):
            boxesOutput.append(None)
            continue

        # For each box, grab class with highest confidence
        classConf, classPred = torch.max(
            imageBoxes[:, 5:5 + numClass], 1,  keepdim=True)

        # Detections ordered as (x1, y1, x2, y2, classConf, classPred)
        detectionResults = torch.cat(
            (imageBoxes[:, :5], classConf.float(), classPred.float()), 1)

        # Back to CPU
        detectionResultsNp = detectionResults.clone().cpu().detach().numpy()

        # Grab detected class labels
        detectedClasses = np.unique(detectionResultsNp[:, -1])

        imageOutputBoxes = np.empty((0, 6))
        # Iterate over each detected class
        for detectedClass in detectedClasses:
            boxIdx = np.where(detectionResultsNp[:, -1] == detectedClass)[0]
            classBoxes = detectionResultsNp[boxIdx, :7]
            nmsClassBoxes = (classBoxes[nms(classBoxes, nmsThreshold), :])[
                :, [0, 1, 2, 3, 4, 5]]  # Remove class conf column
            imageOutputBoxes = np.vstack((imageOutputBoxes, nmsClassBoxes))
        boxesOutput.append(imageOutputBoxes)
    return boxesOutput
