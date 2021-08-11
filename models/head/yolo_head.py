from models.block.block import CBL
import paddle.nn as nn


#---------------------------------------------------#
#   最后获得yolov4的输出
#---------------------------------------------------#
class yolo_head(nn.Layer):
    def __init__(self,filters_list, in_filters):
        super(yolo_head, self).__init__()
        self.head = nn.Sequential(
            CBL(in_filters, filters_list[0], 3),
            nn.Conv2D(filters_list[0], filters_list[1], 1),
        )

    def forward(self, x):
        return self.head(x)
