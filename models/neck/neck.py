"""
spp 以及 实现PAN的块
"""
import paddle.nn as nn
import paddle
from models.block.block import CBL

#---------------------------------------------------#
#   SPP结构，利用不同大小的池化核进行池化
#   池化后堆叠
#---------------------------------------------------#
class SpatialPyramidPooling(nn.Layer):
    def __init__(self, pool_sizes=[5, 9, 13]):
        super(SpatialPyramidPooling, self).__init__()

        self.maxpools = nn.LayerList([nn.MaxPool2D(pool_size, 1, pool_size//2) for pool_size in pool_sizes])

    def forward(self, x):
        features = [maxpool(x) for maxpool in self.maxpools[::-1]]
        features = paddle.concat(features + [x], axis=1)

        return features


#---------------------------------------------------#
#   三次卷积块
#---------------------------------------------------#
class make_three_conv(nn.Layer):
    def __init__(self,filters_list, in_filters):
        super(make_three_conv, self).__init__()
        self.make_three_conv = nn.Sequential(
            CBL(in_filters, filters_list[0], 1),
            CBL(filters_list[0], filters_list[1], 3),
            CBL(filters_list[1], filters_list[0], 1),
        )

    def forward(self,x):
        return self.make_three_conv(x)

#---------------------------------------------------#
#   五次卷积块
#---------------------------------------------------#
class make_five_conv(nn.Layer):
    def __init__(self,filters_list, in_filters):
        super(make_five_conv, self).__init__()
        self.make_five_conv = nn.Sequential(
            CBL(in_filters, filters_list[0], 1),
            CBL(filters_list[0], filters_list[1], 3),
            CBL(filters_list[1], filters_list[0], 1),
            CBL(filters_list[0], filters_list[1], 3),
            CBL(filters_list[1], filters_list[0], 1),
        )
    def forward(self,x):
        return self.make_five_conv(x)




# class Neck(nn.Layer):
#     # 主要实现FPN+PAN结构
#     def __init__(self):
#         super(Neck, self).__init__()
#
#
#
#     def forward(self):
#         return


if __name__ == "__main__":
    print('test')
    ms = make_five_conv((32,16),32)
    # x = paddle.ones((32,32,32,32))#NCHW
    # print(x.shape)
    # y = ms(x)
    print(ms)
