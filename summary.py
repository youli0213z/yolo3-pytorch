#--------------------------------------------#
#   该部分代码用于看网络结构
#--------------------------------------------#
import torch
from thop import clever_format, profile
from torchsummary import summary
from torch import nn

from nets.yolo import YoloBody
from nets.yolo import YoloBodySPP

if __name__ == "__main__":
    input_shape     = [608, 608]
    anchors_mask    = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
    num_classes     = 23
    
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Activation_F = 'nn.SiLU()'
    #Activation_F = 'nn.LeakyReLU(0.1)'
    #m = YoloBody(anchors_mask, num_classes, Activation_F).to(device)
    m       = YoloBodySPP(anchors_mask, num_classes,Activation_F,SPP_SPPF = 'SPP').to(device)
    summary(m, (3, input_shape[0], input_shape[1]))
    
    dummy_input     = torch.randn(1, 3, input_shape[0], input_shape[1]).to(device)
    flops, params   = profile(m.to(device), (dummy_input, ), verbose=False)
    #--------------------------------------------------------#
    #   flops * 2是因为profile没有将卷积作为两个operations
    #   有些论文将卷积算乘法、加法两个operations。此时乘2
    #   有些论文只考虑乘法的运算次数，忽略加法。此时不乘2
    #   本代码选择乘2，参考YOLOX。
    #--------------------------------------------------------#
    flops           = flops
    flops, params   = clever_format([flops, params], "%.3f")
    print('Total GFLOPS: %s' % (flops))
    print('Total params: %s' % (params))
