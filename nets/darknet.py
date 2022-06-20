import math
from collections import OrderedDict
import warnings
import torch

import torch.nn as nn


#---------------------------------------------------------------------#
#   残差结构
#   利用一个1x1卷积下降通道数，然后利用一个3x3卷积提取特征并且上升通道数
#   最后接上一个残差边
#---------------------------------------------------------------------#
class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes,Activation_F):
        super(BasicBlock, self).__init__()
        self.conv1  = nn.Conv2d(inplanes, planes[0], kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1    = nn.BatchNorm2d(planes[0])
        self.relu1  = eval(Activation_F)
        
        self.conv2  = nn.Conv2d(planes[0], planes[1], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2    = nn.BatchNorm2d(planes[1])
        self.relu2  = eval(Activation_F)

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

#基本卷积神经网络结构conv+bn+relu
def conv2d(filter_in, filter_out, kernel_size,Activation_F):
    pad = (kernel_size - 1) // 2 if kernel_size else 0
    return nn.Sequential(OrderedDict([
        ("conv", nn.Conv2d(filter_in, filter_out, kernel_size=kernel_size, stride=1, padding=pad, bias=False)),
        ("bn", nn.BatchNorm2d(filter_out)),
        ("relu", eval(Activation_F)),
    ]))

class SPP(nn.Module):
    # Spatial Pyramid Pooling (SPP) layer https://arxiv.org/abs/1406.4729
    def __init__(self,k=(5, 9, 13)):
        super(SPP,self).__init__()
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  # suppress torch 1.9.0 max_pool2d() warning
            return torch.cat([x] + [m(x) for m in self.m], 1)


class SPPF(nn.Module):
    # Spatial Pyramid Pooling - Fast (SPPF) layer for  by Glenn Jocher
    def __init__(self,k=5):  # equivalent to SPP(k=(5, 9, 13))
        super(SPPF,self).__init__()
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  # suppress torch 1.9.0 max_pool2d() warning
            y1 = self.m(x)
            y2 = self.m(y1)
            return torch.cat([x, y1, y2, self.m(y2)], 1)

class DarkNet(nn.Module):
    def __init__(self, layers,Activation_F,SPP_SPPF = ''):
        super(DarkNet, self).__init__()
        self.inplanes = 32
        # 416,416,3 -> 416,416,32
        self.conv1  = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1    = nn.BatchNorm2d(self.inplanes)
        self.relu1  = eval(Activation_F)
        self.SPP_SPPF = SPP_SPPF

        # 416,416,32 -> 208,208,64
        self.layer1 = self._make_layer([32, 64], layers[0],Activation_F)
        # 208,208,64 -> 104,104,128
        self.layer2 = self._make_layer([64, 128], layers[1],Activation_F)
        # 104,104,128 -> 52,52,256
        self.layer3 = self._make_layer([128, 256], layers[2],Activation_F)
        # 52,52,256 -> 26,26,512
        self.layer4 = self._make_layer([256, 512], layers[3],Activation_F)
        # 26,26,512 -> 13,13,1024
        self.layer5 = self._make_layer([512, 1024], layers[4],Activation_F)

        if self.SPP_SPPF=='':
            self.layers_out_filters = [64, 128, 256, 512, 1024]
        elif self.SPP_SPPF == 'SPP':
            self.final_conv = self.make_final_layers([512,1024],1024,Activation_F)
            self.SPP = SPP()
            self.layers_out_filters = [64, 128, 256, 512, 2048]
        elif self.SPP_SPPF == 'SPPF':
            self.final_conv = self.make_final_layers([512, 1024], 1024, Activation_F)
            self.SPPF = SPPF()
            self.layers_out_filters = [64, 128, 256, 512, 2048]

        # 进行权值初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    #---------------------------------------------------------------------#
    #   在每一个layer里面，首先利用一个步长为2的3x3卷积进行下采样
    #   然后进行残差结构的堆叠
    #---------------------------------------------------------------------#
    def _make_layer(self, planes, blocks,Activation_F):
        layers = []
        # 下采样，步长为2，卷积核大小为3
        layers.append(("ds_conv", nn.Conv2d(self.inplanes, planes[1], kernel_size=3, stride=2, padding=1, bias=False)))
        layers.append(("ds_bn", nn.BatchNorm2d(planes[1])))
        layers.append(("ds_relu", eval(Activation_F)))
        # 加入残差结构
        self.inplanes = planes[1]
        for i in range(0, blocks):
            layers.append(("residual_{}".format(i), BasicBlock(self.inplanes, planes,Activation_F)))
        return nn.Sequential(OrderedDict(layers))

    def make_final_layers(self,filters_list, in_filters, Activation_F):
        m = nn.Sequential(
            conv2d(in_filters, filters_list[0], 1, Activation_F),
            conv2d(filters_list[0], filters_list[1], 3, Activation_F),
            conv2d(filters_list[1], filters_list[0], 1, Activation_F),
        )
        return m

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        out3 = self.layer3(x)
        out4 = self.layer4(out3)
        out5 = self.layer5(out4)
        if self.SPP_SPPF == '':
            return out3, out4, out5
        elif self.SPP_SPPF == 'SPP':
            x = self.final_conv(out5)
            out = self.SPP(x)
            return out3, out4, out
        elif self.SPP_SPPF == 'SPPF':
            x = self.final_conv(out5)
            out = self.SPPF(x)
            return out3, out4, out

def darknet53(Activation_F):
    model = DarkNet([1, 2, 8, 8, 4],Activation_F)
    return model
def darknet53_SPP(Activation_F,SPP_SPPF):
    model = DarkNet([1, 2, 8, 8, 4],Activation_F,SPP_SPPF)
    return model
