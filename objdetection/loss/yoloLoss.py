import torch
import torch.nn as nn
import numpy as np

class YOLOLoss(nn.Module):
    def __init__(self, numClass):
        super(YOLOLoss, self).__init__()
        self.numClass = numClass
        self.bceLoss = nn.BCEWithLogitsLoss()
        self.mseLoss = nn.MSELoss()

    def forward(self, inputFeatMaps, target, objMask, noObjMask):
        predictions = []
        for featMapIdx, inputFeatMap in enumerate(inputFeatMaps):
            prediction = inputFeatMap.permute(0, 2, 3, 1).contiguous().view(inputFeatMap.shape[0], -1, self.numClass+5) 
            predictions.append(prediction)
        predictions = torch.cat(predictions,1)

        predObj = predictions[objMask]
        targetObj = target[objMask]
        
        predNoObj = predictions[noObjMask]
        targetNoObj = target[noObjMask]

        #### Loc ####
        lossX = self.bceLoss(predObj[:,0] , targetObj[:,0] )
        lossY = self.bceLoss(predObj[:,1] , targetObj[:,1] )
        lossW = self.mseLoss(predObj[:,2] , targetObj[:,2] )
        lossH = self.mseLoss(predObj[:,3] , targetObj[:,3] )

        #### Objectness ####
        lossObjConf = self.bceLoss(predObj[:,4], targetObj[:,4])
        loossNoObjConf = 0.5*self.bceLoss(predNoObj[:,4], targetNoObj[:,4])

        #### Class ####
        lossClassConf = self.bceLoss(predObj[:,5:], targetObj[:,5:]) 

        totalLoss = (lossX + lossY + lossW + lossH) + lossObjConf+ loossNoObjConf+  lossClassConf

        return totalLoss, lossX , lossY , lossW , lossH , lossObjConf , loossNoObjConf,  lossClassConf
        
