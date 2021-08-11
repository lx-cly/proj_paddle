"""
要用的卷积块
"""

import paddle.nn as nn
import paddle.nn.functional as F
import paddle



#-------------------------------------------------#
#   MISH激活函数
#-------------------------------------------------#
class Mish(nn.Layer):
    def __init__(self):
        super(Mish, self).__init__()

    def forward(self, x):
        return x * paddle.tanh(F.softplus(x))

#---------------------------------------------------#
#   卷积块 -> 卷积 + 标准化 + 激活函数
#   Conv2d + BatchNormalization + Mish
#---------------------------------------------------#
class CBM(nn.Layer):
    def __init__(self, in_ch, out_ch,kernel_size,stride=1):
        super(CBM, self).__init__()
        self.conv = nn.Conv2D(in_ch, out_ch,kernel_size,stride=stride,padding=kernel_size//2)
        self.bn = nn.BatchNorm2D(out_ch)
        self.relu = Mish()

    def forward(self, x):
        x = self.relu(self.bn(self.conv(x)))
        return x

#---------------------------------------------------#
#   卷积块 -> 卷积 + 标准化 + 激活函数
#   Conv2d + BatchNormalization + LeakyReLU
#---------------------------------------------------#
class CBL(nn.Layer):
    def __init__(self, in_ch, out_ch,kernel_size,stride=1):
        super(CBL, self).__init__()
        pad = (kernel_size - 1) // 2 if kernel_size else 0
        self.conv = nn.Conv2D(in_ch, out_ch,kernel_size,stride=stride, padding=pad)
        self.bn = nn.BatchNorm2D(out_ch)
        self.relu = nn.LeakyReLU(0.1)

    def forward(self, x):
        x = self.relu(self.bn(self.conv(x)))
        return x


#---------------------------------------------------#
#   卷积 + 上采样
#---------------------------------------------------#
class Upsample(nn.Layer):
    def __init__(self, in_channels, out_channels):
        super(Upsample, self).__init__()

        self.upsample = nn.Sequential(
            CBL(in_channels, out_channels, 1),
            nn.Upsample(scale_factor=2, mode='nearest')
        )

    def forward(self, x,):
        x = self.upsample(x)
        return x

if __name__ == "__main__":
    print('test')
    ms = Upsample(32,16)
    x = paddle.ones((32,32,32,32))#NCHW
    print(x.shape)
    y = ms(x)
    print(y.shape)
