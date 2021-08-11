from models.block.block import CBL,Upsample
from models.neck.neck import SpatialPyramidPooling,make_three_conv,make_five_conv
from models.head.yolo_head import yolo_head
from models.backbone.CSPdarknet import darknet53

import paddle.nn as nn
import paddle
import numpy as np

#---------------------------------------------------#
#   yolo_body
#---------------------------------------------------#
class YoloBody(nn.Layer):
    def __init__(self, num_anchors, num_classes,pretrained=None):
        super(YoloBody, self).__init__()
        #---------------------------------------------------#
        #   生成CSPdarknet53的主干模型 输入416*416*3
        #   获得三个有效特征层，他们的shape分别是：
        #   52,52,256
        #   26,26,512
        #   13,13,1024
        #---------------------------------------------------#
        self.backbone = darknet53(pretrained)

        self.conv1 = make_three_conv([512,1024],1024)
        self.SPP = SpatialPyramidPooling()
        self.conv2 = make_three_conv([512,1024],2048)

        self.upsample1 = Upsample(512,256)
        self.conv_for_P4 = CBL(512,256,1)
        self.make_five_conv1 = make_five_conv([256, 512],512)

        self.upsample2 = Upsample(256,128)
        self.conv_for_P3 = CBL(256,128,1)
        self.make_five_conv2 = make_five_conv([128, 256],256)

        # 3*(5+num_classes) = 3*(5+80) = 3*(4+1+80)=255
        final_out_filter2 = num_anchors * (5 + num_classes)
        self.yolo_head3 = yolo_head([256, final_out_filter2],128)

        self.down_sample1 = CBL(128,256,3,stride=2)
        self.make_five_conv3 = make_five_conv([256, 512],512)

        # 3*(5+num_classes) = 3*(5+80) = 3*(4+1+80)=255
        final_out_filter1 =  num_anchors * (5 + num_classes)
        self.yolo_head2 = yolo_head([512, final_out_filter1],256)

        self.down_sample2 = CBL(256,512,3,stride=2)
        self.make_five_conv4 = make_five_conv([512, 1024],1024)

        # 3*(5+num_classes) = 3*(5+80) = 3*(4+1+80)=255
        final_out_filter0 =  num_anchors * (5 + num_classes)
        self.yolo_head1 = yolo_head([1024, final_out_filter0],512)

        #############初始化类型#############
        # if isinstance(m, nn.Conv2D):
        #     n = m.weight.shape[0] * m.weight.shape[1] * m.weight.shape[2]
        #     v = np.random.normal(loc=0., scale=np.sqrt(2. / n), size=m.weight.shape).astype('float32')
        #     m.weight.set_value(v)
        # elif isinstance(m, nn.BatchNorm):
        #     m.weight.set_value(np.ones(m.weight.shape).astype('float32'))
        #     m.bias.set_value(np.zeros(m.bias.shape).astype('float32'))
        for m in self.sublayers():#初始化
            if isinstance(m, nn.Conv2D):
                n = m.weight.shape[0] * m.weight.shape[1] * m.weight.shape[2]
                v = np.random.normal(loc=0., scale=np.sqrt(2. / n), size=m.weight.shape).astype('float32')
                m.weight.set_value(v)
            elif isinstance(m, nn.BatchNorm2D):
                m.weight.set_value(np.ones(m.weight.shape).astype('float32'))
                m.bias.set_value(np.zeros(m.bias.shape).astype('float32'))



    def forward(self, x):
        #  backbone
        x2, x1, x0 = self.backbone(x)

        # 13,13,1024 -> 13,13,512 -> 13,13,1024 -> 13,13,512 -> 13,13,2048
        P5 = self.conv1(x0)
        P5 = self.SPP(P5)
        # 13,13,2048 -> 13,13,512 -> 13,13,1024 -> 13,13,512
        P5 = self.conv2(P5)

        # 13,13,512 -> 13,13,256 -> 26,26,256
        P5_upsample = self.upsample1(P5)
        # 26,26,512 -> 26,26,256
        P4 = self.conv_for_P4(x1)
        # 26,26,256 + 26,26,256 -> 26,26,512
        P4 = paddle.concat([P4,P5_upsample],axis=1)
        # 26,26,512 -> 26,26,256 -> 26,26,512 -> 26,26,256 -> 26,26,512 -> 26,26,256
        P4 = self.make_five_conv1(P4)

        # 26,26,256 -> 26,26,128 -> 52,52,128
        P4_upsample = self.upsample2(P4)
        # 52,52,256 -> 52,52,128
        P3 = self.conv_for_P3(x2)
        # 52,52,128 + 52,52,128 -> 52,52,256
        P3 = paddle.concat([P3,P4_upsample],axis=1)
        # 52,52,256 -> 52,52,128 -> 52,52,256 -> 52,52,128 -> 52,52,256 -> 52,52,128
        P3 = self.make_five_conv2(P3)

        # 52,52,128 -> 26,26,256
        P3_downsample = self.down_sample1(P3)
        # 26,26,256 + 26,26,256 -> 26,26,512
        P4 = paddle.concat([P3_downsample,P4],axis=1)
        # 26,26,512 -> 26,26,256 -> 26,26,512 -> 26,26,256 -> 26,26,512 -> 26,26,256
        P4 = self.make_five_conv3(P4)

        # 26,26,256 -> 13,13,512
        P4_downsample = self.down_sample2(P4)
        # 13,13,512 + 13,13,512 -> 13,13,1024
        P5 = paddle.concat([P4_downsample,P5],axis=1)
        # 13,13,1024 -> 13,13,512 -> 13,13,1024 -> 13,13,512 -> 13,13,1024 -> 13,13,512
        P5 = self.make_five_conv4(P5)

        #---------------------------------------------------#
        #   第三个特征层
        #   y3=(batch_size,255,52,52)
        #---------------------------------------------------#
        out2 = self.yolo_head3(P3)
        #---------------------------------------------------#
        #   第二个特征层
        #   y2=(batch_size,255,26,26)
        #---------------------------------------------------#
        out1 = self.yolo_head2(P4)
        #---------------------------------------------------#
        #   第一个特征层
        #   y1=(batch_size,255,13,13)
        #---------------------------------------------------#
        out0 = self.yolo_head1(P5)

        return out0, out1, out2


if __name__ == "__main__":
    print('test')
    model = YoloBody(3,80,pretrained=None)
    x = paddle.ones((1,3,416, 416))  # NCHW
    print(x.shape)
    a,b,c = model(x)
    print(a.shape)
    print(b.shape)
    print(c.shape)