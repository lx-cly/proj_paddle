# -------------------------------------#
#       对数据集进行训练
# -------------------------------------#
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

def get_lr(lr_scheduler):
    #for param_group in lr_scheduler._learning_rate_:
    return lr_scheduler.last_lr


def fit_one_epoch(net,lr_scheduler, optimizer,yolo_loss, epoch, epoch_size, epoch_size_val, gen, genval, Epoch, cuda):

    total_loss = 0
    val_loss = 0
    best_loss = 1e5
    net.train()
    print('Start Train')
    with tqdm(total=epoch_size, desc='Epoch{}/{}'.format(Epoch+1,Epoch), postfix=dict, mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen):
            if iteration >= epoch_size:
                break
            images, targets = batch[0], batch[1]#
            # #(tensor)images.shape = B C H W  归一化了
            # targets 是list len为 B 每一个是tensor.shape = number 5
            #with torch.no_grad():
            if cuda:
                images = paddle.to_tensor(images,dtype='float32').cuda()#torch.from_numpy(images).type(torch.FloatTensor).cuda()
                targets = [paddle.to_tensor(ann,dtype='float32').cuda() for ann in targets]#[torch.from_numpy(ann).type(torch.FloatTensor) for ann in targets]
                 # else:
                #     images = torch.from_numpy(images).type(torch.FloatTensor)
                #     targets = [torch.from_numpy(ann).type(torch.FloatTensor) for ann in targets]
            # ----------------------#
            #   清零梯度
            # ----------------------#
            optimizer.clear_grad()
            # ----------------------#
            #   前向传播
            # ----------------------#
            outputs = net(images)
            losses = []
            num_pos_all = 0
            # ----------------------#
            #   计算损失
            # ----------------------#
            for i in range(3):
                loss_item, num_pos = yolo_loss(outputs[i], targets)
                losses.append(loss_item)
                num_pos_all += num_pos

            loss = sum(losses) / num_pos_all
            total_loss += loss.item()

            # ----------------------#
            #   反向传播
            # ----------------------#
            loss.backward()
            optimizer.step()

            pbar.set_postfix(**{'total_loss': total_loss / (iteration + 1),
                                'lr': get_lr(lr_scheduler)})
            pbar.update(1)

    # 将loss写入tensorboard，下面注释的是每个世代保存一次
    # if Tensorboard:
    #     writer.add_scalar('Train_loss', total_loss/(iteration+1), epoch)
    net.eval()
    print('Start Validation')
    with tqdm(total=epoch_size_val, desc='Epoch{}/{}'.format(Epoch+1,Epoch), postfix=dict, mininterval=0.3) as pbar:
        for iteration, batch in enumerate(genval):
            if iteration >= epoch_size_val:
                break
            images_val, targets_val = batch[0], batch[1]

            #with torch.no_grad():
            if cuda:
                images_val =  paddle.to_tensor(images_val,dtype='float32').cuda()#torch.from_numpy(images_val).type(torch.FloatTensor).cuda()
                targets_val = [paddle.to_tensor(ann,dtype='float32').cuda() for ann in targets_val]#[torch.from_numpy(ann).type(torch.FloatTensor) for ann in targets_val]
            # else:
            #     images_val = torch.from_numpy(images_val).type(torch.FloatTensor)
            #     targets_val = [torch.from_numpy(ann).type(torch.FloatTensor) for ann in targets_val]

            optimizer.clear_grad()

            outputs = net(images_val)
            losses = []
            num_pos_all = 0
            for i in range(3):
                loss_item, num_pos = yolo_loss(outputs[i], targets_val)
                losses.append(loss_item)
                num_pos_all += num_pos
            loss = sum(losses) / num_pos_all
            val_loss += loss.item()

            # 将loss写入tensorboard, 下面注释的是每一步都写
            # if Tensorboard:
            #     writer.add_scalar('Val_loss', loss, val_tensorboard_step)
            #     val_tensorboard_step += 1
            pbar.set_postfix(**{'total_loss': val_loss / (iteration + 1)})
            pbar.update(1)

    # # 将loss写入tensorboard，每个世代保存一次
    # if Tensorboard:
    #     writer.add_scalar('Val_loss', val_loss / (epoch_size_val + 1), epoch)
    loss_history.append_loss(total_loss / (epoch_size + 1), val_loss / (epoch_size_val + 1))
    print('Finish Validation')
    print('Epoch:' + str(epoch + 1) + '/' + str(Epoch))
    print('Total Loss: %.4f || Val Loss: %.4f ' % (total_loss / (epoch_size + 1), val_loss / (epoch_size_val + 1)))
    print('Saving state, iter:', str(epoch + 1))
    if (val_loss / (epoch_size_val + 1)) < best_loss:
        best_loss = (val_loss / (epoch_size_val + 1))
        paddle.save(model.state_dict(), 'logs/Epoch%d-Total_Loss%.4f-Val_Loss%.4f.pth' % (
        (epoch + 1), total_loss / (epoch_size + 1), best_loss))

    else:
        print('val_loss no downing,so dont save model!')


# ----------------------------------------------------#
#   检测精度mAP和pr曲线计算参考视频
#   https://www.bilibili.com/video/BV1zE411u7Vw
# ----------------------------------------------------#
if __name__ == "__main__":
    # -------------------------------#
    #   是否使用Tensorboard
    # -------------------------------#
    Tensorboard = False
    # -------------------------------#
    #   是否使用Cuda
    #   没有GPU可以设置成False
    # -------------------------------#
    Cuda = True
    # ------------------------------------------------------#
    #   是否对损失进行归一化，用于改变loss的大小
    #   用于决定计算最终loss是除上batch_size还是除上正样本数量
    # ------------------------------------------------------#
    normalize = False
    # -------------------------------#
    #   输入的shape大小
    #   显存比较小可以使用416x416
    #   显存比较大可以使用608x608
    # -------------------------------#
    input_shape = (416, 416)
    # ----------------------------------------------------#
    #   classes和anchor的路径，非常重要
    #   训练前一定要修改classes_path，使其对应自己的数据集
    # ----------------------------------------------------#
    anchors_path = 'data/coco_anchors.txt'
    classes_path = 'data/coco_classes.txt'
    # ------------------------------------------------------#
    #   Yolov4的tricks应用
    #   mosaic 马赛克数据增强 True or False
    #   实际测试时mosaic数据增强并不稳定，所以默认为False
    #   Cosine_scheduler 余弦退火学习率 True or False
    #   label_smoothing 标签平滑 0.01以下一般 如0.01、0.005
    # ------------------------------------------------------#
    mosaic = True
    Cosine_lr = True
    smoooth_label = 0

    # ----------------------------------------------------#
    #   获取classes和anchor
    # ----------------------------------------------------#
    class_names = get_classes(classes_path)#class list ['person',...]
    anchors = get_anchors(anchors_path)# shape:3,3,2
    num_classes = len(class_names) #80

    # ------------------------------------------------------#
    #   创建yolo模型
    #   训练前一定要修改classes_path和对应的txt文件
    # ------------------------------------------------------#
    model = YoloBody(len(anchors[0]), num_classes)
    # weights_init(model)

    # # ------------------------------------------------------#
    # #   权值文件请看README，百度网盘下载
    # # ------------------------------------------------------#
    # model_path = "model_data/yolo4_weights.pth"
    # print('Loading weights into state dict...')
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model_dict = model.state_dict()
    # pretrained_dict = torch.load(model_path, map_location=device)
    # pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) == np.shape(v)}
    # model_dict.update(pretrained_dict)
    # model.load_state_dict(model_dict)
    # print('Finished!')

    #net = model#.train()

    # if Cuda:
    #     #net = paddle.DataParallel(model)
    #     model = model.cuda()

    yolo_loss = YOLOLoss(np.reshape(anchors, [-1, 2]), num_classes, (input_shape[1], input_shape[0]), smoooth_label,
                         Cuda, normalize)
    loss_history = LossHistory("logs/")

    # ----------------------------------------------------#
    #   获得图片路径和标签
    # ----------------------------------------------------#
    train_annotation_path = 'data/2017_train.txt'
    val_annotation_path = 'data/2017_val.txt'
    # ----------------------------------------------------------------------#
    #   验证集的划分在train.py代码里面进行
    #   2007_test.txt和2007_val.txt里面没有内容是正常的。训练不会使用到。
    #   当前划分方式下，验证集和训练集的比例为1:9
    # ----------------------------------------------------------------------#

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

    # if Tensorboard:
    #     from tensorboardX import SummaryWriter
    #
    #     writer = SummaryWriter(log_dir='logs', flush_secs=60)
    #     if Cuda:
    #         graph_inputs = torch.randn(1, 3, input_shape[0], input_shape[1]).type(torch.FloatTensor).cuda()
    #     else:
    #         graph_inputs = torch.randn(1, 3, input_shape[0], input_shape[1]).type(torch.FloatTensor)
    #     writer.add_graph(model, graph_inputs)
    #     train_tensorboard_step = 1
    #     val_tensorboard_step = 1

    # ------------------------------------------------------#
    #   主干特征提取网络特征通用，冻结训练可以加快训练速度
    #   也可以在训练初期防止权值被破坏。
    #   Init_Epoch为起始世代
    #   Freeze_Epoch为冻结训练的世代
    #   Epoch总训练世代
    #   提示OOM或者显存不足请调小Batch_size
    # ------------------------------------------------------#
    if True:
        lr = 2e-3
        Batch_size = 8
        Init_Epoch = 0
        Freeze_Epoch = 50

        # ----------------------------------------------------------------------------#
        #   我在实际测试时，发现optimizer的weight_decay起到了反作用，
        #   所以去除掉了weight_decay，大家也可以开起来试试，一般是weight_decay=5e-4
        # ----------------------------------------------------------------------------#

        if Cosine_lr:
            lr_scheduler = optim.lr.CosineAnnealingDecay(lr, T_max=5, eta_min=1e-5)
           # optimizer = optim.Adam(parameters=net.parameters(), learning_rate=lr_scheduler)
        else:
            lr_scheduler = optim.lr.StepDecay(lr, step_size=1, gamma=0.92)

        optimizer = optim.Adam(parameters=model.parameters(), learning_rate=lr_scheduler)

        train_dataset = YoloDataset(train_lines, (input_shape[0], input_shape[1]), mosaic=mosaic, is_train=True)
        val_dataset = YoloDataset(val_lines, (input_shape[0], input_shape[1]), mosaic=False, is_train=False)
        gen = DataLoader(train_dataset, shuffle=True, batch_size=Batch_size, num_workers=0,
                         drop_last=True, collate_fn=yolo_dataset_collate)
        gen_val = DataLoader(val_dataset, shuffle=True, batch_size=Batch_size, num_workers=0,
                             drop_last=True, collate_fn=yolo_dataset_collate)

        epoch_size = num_train // Batch_size
        epoch_size_val = num_val // Batch_size

        if epoch_size == 0 or epoch_size_val == 0:
            raise ValueError("数据集过小，无法进行训练，请扩充数据集。")
        # ------------------------------------#
        #   冻结一定部分训练
        # ------------------------------------#
        for param in model.backbone.parameters():
            param.requires_grad = False

        for epoch in range(Init_Epoch, Freeze_Epoch):
            fit_one_epoch(model,lr_scheduler,optimizer, yolo_loss, epoch, epoch_size, epoch_size_val, gen, gen_val, Freeze_Epoch, Cuda)#TODO:重改
            lr_scheduler.step()

    if True:
        lr = 2e-4
        Batch_size = 4
        Freeze_Epoch = 50
        Unfreeze_Epoch = 100

        # ----------------------------------------------------------------------------#
        #   我在实际测试时，发现optimizer的weight_decay起到了反作用，
        #   所以去除掉了weight_decay，大家也可以开起来试试，一般是weight_decay=5e-4
        # ----------------------------------------------------------------------------#

        if Cosine_lr:
            lr_scheduler = optim.lr.CosineAnnealingDecay(lr, T_max=5, eta_min=1e-5)
        else:
            lr_scheduler = optim.lr.StepDecay(lr, step_size=1, gamma=0.92)
        optimizer = optim.Adam(parameters=model.parameters(), learning_rate=lr_scheduler)

        train_dataset = YoloDataset(train_lines, (input_shape[0], input_shape[1]), mosaic=mosaic, is_train=True)
        val_dataset = YoloDataset(val_lines, (input_shape[0], input_shape[1]), mosaic=False, is_train=False)
        gen = DataLoader(train_dataset, shuffle=True, batch_size=Batch_size, num_workers=0, pin_memory=True,
                         drop_last=True, collate_fn=yolo_dataset_collate)
        gen_val = DataLoader(val_dataset, shuffle=True, batch_size=Batch_size, num_workers=0, pin_memory=True,
                             drop_last=True, collate_fn=yolo_dataset_collate)

        epoch_size = num_train // Batch_size
        epoch_size_val = num_val // Batch_size

        if epoch_size == 0 or epoch_size_val == 0:
            raise ValueError("数据集过小，无法进行训练，请扩充数据集。")
        # ------------------------------------#
        #   解冻后训练
        # ------------------------------------#
        for param in model.backbone.parameters():
            param.requires_grad = True

        for epoch in range(Freeze_Epoch, Unfreeze_Epoch):
            fit_one_epoch(model,lr_scheduler,optimizer, yolo_loss, epoch, epoch_size, epoch_size_val, gen, gen_val, Unfreeze_Epoch, Cuda)
            lr_scheduler.step()
