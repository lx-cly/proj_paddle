import paddle.nn as nn
import numpy as np
import paddle
from models.block.block import CBM


#---------------------------------------------------#
#   CSPdarknet的结构块的组成部分
#   内部堆叠的残差块
#---------------------------------------------------#
class Resblock(nn.Layer):
    def __init__(self, channels, hidden_channels=None):
        super(Resblock, self).__init__()

        if hidden_channels is None:
            hidden_channels = channels

        self.block = nn.Sequential(
            CBM(channels, hidden_channels, 1),
            CBM(hidden_channels, channels, 3)
        )

    def forward(self, x):
        return x + self.block(x)

#--------------------------------------------------------------------#
#   CSPdarknet的结构块
#   首先利用ZeroPadding2D和一个步长为2x2的卷积块进行高和宽的压缩
#   然后建立一个大的残差边shortconv、这个大残差边绕过了很多的残差结构
#   主干部分会对num_blocks进行循环，循环内部是残差结构。
#   对于整个CSPdarknet的结构块，就是一个大残差块+内部多个小残差块
#--------------------------------------------------------------------#
class Resblock_body(nn.Layer):
    def __init__(self, in_channels, out_channels, num_blocks, first):
        super(Resblock_body, self).__init__()
        #----------------------------------------------------------------#
        #   利用一个步长为2x2的卷积块进行高和宽的压缩
        #----------------------------------------------------------------#
        self.downsample_conv = CBM(in_channels, out_channels, 3, stride=2)

        if first:
            #--------------------------------------------------------------------------#
            #   然后建立一个大的残差边self.split_conv0、这个大残差边绕过了很多的残差结构
            #--------------------------------------------------------------------------#
            self.split_conv0 = CBM(out_channels, out_channels, 1)

            #----------------------------------------------------------------#
            #   主干部分会对num_blocks进行循环，循环内部是残差结构。
            #----------------------------------------------------------------#
            self.split_conv1 = CBM(out_channels, out_channels, 1)
            self.blocks_conv = nn.Sequential(
                Resblock(channels=out_channels, hidden_channels=out_channels//2),
                CBM(out_channels, out_channels, 1)
            )

            self.concat_conv = CBM(out_channels*2, out_channels, 1)
        else:
            #--------------------------------------------------------------------------#
            #   然后建立一个大的残差边self.split_conv0、这个大残差边绕过了很多的残差结构
            #--------------------------------------------------------------------------#
            self.split_conv0 = CBM(out_channels, out_channels//2, 1)

            #----------------------------------------------------------------#
            #   主干部分会对num_blocks进行循环，循环内部是残差结构。
            #----------------------------------------------------------------#
            self.split_conv1 = CBM(out_channels, out_channels//2, 1)
            self.blocks_conv = nn.Sequential(
                *[Resblock(out_channels//2) for _ in range(num_blocks)],
                CBM(out_channels//2, out_channels//2, 1)
            )

            self.concat_conv = CBM(out_channels, out_channels, 1)

    def forward(self, x):
        x = self.downsample_conv(x)

        x0 = self.split_conv0(x)

        x1 = self.split_conv1(x)
        x1 = self.blocks_conv(x1)

        #------------------------------------#
        #   将大残差边再堆叠回来
        #------------------------------------#
        x = paddle.concat([x1, x0],axis=1)#x = torch.cat([x1, x0], dim=1)
        #------------------------------------#
        #   最后对通道数进行整合
        #------------------------------------#
        x = self.concat_conv(x)

        return x


#---------------------------------------------------#
#   CSPdarknet53 的主体部分
#   输入为一张416x416x3的图片
#   输出为三个有效特征层
#---------------------------------------------------#
class CSPDarkNet(nn.Layer):
    def __init__(self, layers):
        super(CSPDarkNet, self).__init__()
        self.inplanes = 32
        # 416,416,3 -> 416,416,32
        self.conv1 = CBM(3, self.inplanes, kernel_size=3, stride=1)
        self.feature_channels = [64, 128, 256, 512, 1024]

        self.stages = nn.LayerList([
            # 416,416,32 -> 208,208,64
            Resblock_body(self.inplanes, self.feature_channels[0], layers[0], first=True),
            # 208,208,64 -> 104,104,128
            Resblock_body(self.feature_channels[0], self.feature_channels[1], layers[1], first=False),
            # 104,104,128 -> 52,52,256
            Resblock_body(self.feature_channels[1], self.feature_channels[2], layers[2], first=False),
            # 52,52,256 -> 26,26,512
            Resblock_body(self.feature_channels[2], self.feature_channels[3], layers[3], first=False),
            # 26,26,512 -> 13,13,1024
            Resblock_body(self.feature_channels[3], self.feature_channels[4], layers[4], first=False)
        ])

        self.num_features = 1

        for m in self.sublayers():#初始化
            if isinstance(m, nn.Conv2D):
                n = m.weight.shape[0] * m.weight.shape[1] * m.weight.shape[2]
                v = np.random.normal(loc=0., scale=np.sqrt(2. / n), size=m.weight.shape).astype('float32')
                m.weight.set_value(v)
            elif isinstance(m, nn.BatchNorm2D):
                m.weight.set_value(np.ones(m.weight.shape).astype('float32'))
                m.bias.set_value(np.zeros(m.bias.shape).astype('float32'))

    def forward(self, x):
        x = self.conv1(x)

        x = self.stages[0](x)
        x = self.stages[1](x)
        out3 = self.stages[2](x)
        out4 = self.stages[3](out3)
        out5 = self.stages[4](out4)

        return out3, out4, out5

def darknet53(pretrained, **kwargs):
    model = CSPDarkNet([1, 2, 8, 8, 4])
    if pretrained:
        if isinstance(pretrained, str):
            model.set_state_dict(paddle.load(pretrained))
        else:
            raise Exception("darknet request a pretrained path. got [{}]".format(pretrained))
    return model

if __name__ == "__main__":
    print('test')
    model = darknet53(pretrained='')
    print(model)
