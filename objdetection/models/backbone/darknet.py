import torch
import torch.nn as nn
from collections import OrderedDict

class DarkNetBasicBlock(nn.Module):
    def __init__(self, inplanes, planes):
        super(DarkNetBasicBlock, self).__init__()

        self.conv1 = nn.Conv2d(inplanes, planes[0], kernel_size=1,stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(planes[0])
        self.relu1 = nn.LeakyReLU(0.1)

        self.conv2 = nn.Conv2d(planes[0], planes[1], kernel_size=3,stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes[1])
        self.relu2 = nn.LeakyReLU(0.1)

    '''
            x -+-> conv1 --> bn1 --> relu1 --> conv2 --> bn2 --> relu2 -+-> out
               |________________________________________________________|
            x and out size is equal
    '''
    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        out += residual
        return out

class DarkNet(nn.Module):
    def __init__(self, layers):
        super(DarkNet, self).__init__()

        self.layers_out_filters = [64, 128, 256, 512, 1024]
        self.inplanes = 32
        
        ###############
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu1 = nn.LeakyReLU(0.1)
        ###############
        
        ### each elements in 'layers' represent the number of residule basic block
        ### make_layers ==> [in,out] filter dim
        self.layer1 = self._make_layer([32, 64], layers[0])
        self.layer2 = self._make_layer([64, 128], layers[1])
        self.layer3 = self._make_layer([128, 256], layers[2])
        self.layer4 = self._make_layer([256, 512], layers[3])
        self.layer5 = self._make_layer([512, 1024], layers[4])

    def _make_layer(self, planes, blocks):
        
        layers = []
        #  downsample
        layers.append(("ds_conv", nn.Conv2d(self.inplanes, planes[1], kernel_size=3, stride=2, padding=1, bias=False)))
        layers.append(("ds_bn", nn.BatchNorm2d(planes[1])))
        layers.append(("ds_relu", nn.LeakyReLU(0.1)))

        #  blocks
        self.inplanes = planes[1]
        for i in range(blocks):
            layers.append(("residual_{}".format(i), DarkNetBasicBlock(self.inplanes, planes)))
        return nn.Sequential(OrderedDict(layers))

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.layer1(x)
        x = self.layer2(x)

        out3 = self.layer3(x)
        out4 = self.layer4(out3)
        out5 = self.layer5(out4)

        return out3, out4, out5

def darknet21():
    return DarkNet([1, 1, 2, 2, 1])
    

def darknet53():
    return DarkNet([1, 2, 8, 8, 4])