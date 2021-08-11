##测试模块是否正确##
import numpy as np
import paddle
import paddle.optimizer as optim
from paddle.io import DataLoader
from tqdm import tqdm

from models.yolov4 import YoloBody
from utils.loss import LossHistory, YOLOLoss#, weights_init
from utils.dataloader import YoloDataset, yolo_dataset_collate

# ---------------------------------------------------#
#   获得类和先验框
# ---------------------------------------------------#
def get_classes(classes_path):
    '''loads the classes'''
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names

def get_anchors(anchors_path):
    '''loads the anchors from a file'''
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape([-1, 3, 2])[::-1, :, :]

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']



if __name__=="__main__":
    # ----------------------------------------------------#
    #   获得图片路径和标签
    # ----------------------------------------------------#
    train_annotation_path = 'data/2017_train.txt'
    val_annotation_path = 'data/2017_val.txt'
    with open(train_annotation_path) as f:
        train_lines = f.readlines()
    with open(val_annotation_path) as f:
        val_lines = f.readlines()
    np.random.seed(10101)
    np.random.shuffle(train_lines)
    np.random.shuffle(val_lines)
    np.random.seed(None)
    num_train = len(train_lines)
    num_val = len(val_lines)
    #
    Batch_size = 1
    input_shape = (416, 416)
    mosaic = True
    Cosine_lr = True
    smoooth_label = 0
    lr = 2e-3
    #

    train_dataset = YoloDataset(train_lines, (input_shape[0], input_shape[1]), mosaic=mosaic, is_train=True)
    val_dataset = YoloDataset(val_lines, (input_shape[0], input_shape[1]), mosaic=False, is_train=False)
    gen = DataLoader(train_dataset, shuffle=True, batch_size=Batch_size, num_workers=4,
                     drop_last=True, collate_fn=yolo_dataset_collate)
    gen_val = DataLoader(val_dataset, shuffle=True, batch_size=Batch_size, num_workers=4,
                         drop_last=True, collate_fn=yolo_dataset_collate)

    Cuda = True
    normalize = False
    anchors_path = 'data/coco_anchors.txt'
    classes_path = 'data/coco_classes.txt'
    class_names = get_classes(classes_path)
    anchors = get_anchors(anchors_path)
    num_classes = len(class_names)
    print('读入成功')
    print(len(anchors[0]))
    model = YoloBody(len(anchors[0]), num_classes)
    #net = model
    # net = net.cuda()
    if Cosine_lr:
        lr_scheduler = optim.lr.CosineAnnealingDecay(lr, T_max=5, eta_min=1e-5)
        # optimizer = optim.Adam(parameters=net.parameters(), learning_rate=lr_scheduler)
    else:
        lr_scheduler = optim.lr.StepDecay(lr, step_size=1, gamma=0.92)

    optimizer = optim.Adam(parameters=model.parameters(), learning_rate=lr_scheduler)

    print('***')
    # yolo_loss = YOLOLoss(np.reshape(anchors, [-1, 2]), num_classes, (input_shape[1], input_shape[0]), smoooth_label,
    #                      Cuda, normalize)
    # for iteration, batch in enumerate(gen_val):
    #     print(iteration)
    #     images, targets = batch[0], batch[1]
    #     outputs = model(images)
    #     print('*' * 50)
    #     num_pos_all = 0
    #     total_loss = 0
    #     losses = []
    #     for i in range(3):
    #         loss_item, num_pos = yolo_loss(outputs[i], targets)
    #         losses.append(loss_item)
    #         num_pos_all += num_pos
    #     loss = sum(losses) / num_pos_all
    #     total_loss += loss.item()
    #     print(total_loss)
    #     print(num_pos_all)
    #     break

    #########数据集加载成功#########



