import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from .backbone.darknet import darknet53, darknet21
from .backbone.nasnetamobile import NASNetAMobile


class YOLONet(nn.Module):
    def __init__(self, numClass, backboneName='', backboneWeightFile=None):
        super(YOLONet, self).__init__()
        self.backboneName = backboneName
        if backboneName == 'darknet53':
            self.backbone = darknet53()
            self.outputFeatMapSizes = [(13, 13), (26, 26), (52, 52)]
        elif backboneName == 'darknet21':
            self.backbone = darknet21()
            self.outputFeatMapSizes = [(13, 13), (26, 26), (52, 52)]
        elif backboneName == 'nasnetamobile':
            self.backbone = NASNetAMobile()
            self.outputFeatMapSizes = [(13, 13), (13, 13), (26, 26)]
        else:
            print('Unspecified backbone name',backboneName)
            sys.exit()
        
        if backboneWeightFile is not None:
            self.backbone.load_state_dict(torch.load(backboneWeightFile))

        _out_filters = self.backbone.layers_out_filters

        #  embedding0
        final_out_filter0 = 3 * (5 + numClass)
        self.embedding0 = self._make_embedding(
            [512, 1024], _out_filters[-1], final_out_filter0)

        #  embedding1
        final_out_filter1 = 3 * (5 + numClass)
        self.embedding1_cbl = self._make_cbl(512, 256, 1)
        self.embedding1 = self._make_embedding(
            [256, 512], _out_filters[-2] + 256, final_out_filter1)

        #  embedding2
        final_out_filter2 = 3 * (5 + numClass)
        self.embedding2_cbl = self._make_cbl(256, 128, 1)
        self.embedding2 = self._make_embedding(
            [128, 256], _out_filters[-3] + 128, final_out_filter2)

    def _make_cbl(self, _in, _out, ks):
        ''' cbl = conv + batch_norm + leaky_relu
        '''
        pad = (ks - 1) // 2 if ks else 0
        return nn.Sequential(OrderedDict([
            ("conv", nn.Conv2d(_in, _out, kernel_size=ks,
                               stride=1, padding=pad, bias=False)),
            ("bn", nn.BatchNorm2d(_out)),
            ("relu", nn.LeakyReLU(0.1)),
        ]))

    def _make_embedding(self, filters_list, in_filters, out_filter):
        m = nn.ModuleList([
            self._make_cbl(in_filters, filters_list[0], 1),
            self._make_cbl(filters_list[0], filters_list[1], 3),
            self._make_cbl(filters_list[1], filters_list[0], 1),
            self._make_cbl(filters_list[0], filters_list[1], 3),
            self._make_cbl(filters_list[1], filters_list[0], 1),
            self._make_cbl(filters_list[0], filters_list[1], 3)])
        m.add_module("conv_out", nn.Conv2d(filters_list[1], out_filter, kernel_size=1,
                                           stride=1, padding=0, bias=True))
        return m

    def forward(self, x):
        def _branch(_embedding, _in):
            for i, e in enumerate(_embedding):
                _in = e(_in)
                if i == 4:
                    out_branch = _in
            return _in, out_branch
        #  backbone
        x2, x1, x0 = self.backbone(x)

        #  yolo branch 0
        out0, out0_branch = _branch(self.embedding0, x0)
        #  yolo branch 1
        x1_in = self.embedding1_cbl(out0_branch)
        if self.backboneName == 'darknet53' or self.backboneName == 'darknet21':
            x1_in = F.interpolate(x1_in, scale_factor=2, mode='nearest')
        x1_in = torch.cat([x1_in, x1], 1)
        out1, out1_branch = _branch(self.embedding1, x1_in)

        #  yolo branch
        x2_in = self.embedding2_cbl(out1_branch)
        x2_in = F.interpolate(x2_in, scale_factor=2, mode='nearest')
        x2_in = torch.cat([x2_in, x2], 1)
        out2, out2_branch = _branch(self.embedding2, x2_in)
        return out0, out1, out2
