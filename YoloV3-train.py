#------------------------------------------------
# Project: paddle-Traffic
# Author:Peilin Wu - Najing Normal University
# File name :TrafficLight.py.py
# Created time :2020/05
#------------------------------------------------
import os
from utils import *
import time
import os
import paddle
import paddle.fluid as fluid
from Darknet53 import *
from paddle.fluid.dygraph.base import to_variable

ANCHORS = [10, 13, 16, 30, 33, 23, 30, 61, 62, 45, 59, 119, 116, 90, 156, 198, 373, 326]
ANCHOR_MASKS = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
IGNORE_THRESH = .8
NUM_CLASSES = 2

catagories = ['green','red']#分类
TRAINDIR = "./dataset/train"
VALIDDIR = "./dataset/dev"
TESTDIR = "./dataset/test"
MAX_EPOCH = 160

def get_lr(base_lr = 0.0001, lr_decay = 0.1):
    bd = [10000, 20000]
    lr = [base_lr, base_lr * lr_decay, base_lr * lr_decay * lr_decay]
    learning_rate = fluid.layers.piecewise_decay(boundaries=bd, values=lr)
    return learning_rate

if __name__ == '__main__':

    catagories_id = get_catagories_id(catagories)
    annotation_info = get_annotations(catagories_id,TRAINDIR,"xml","img")
    with fluid.dygraph.guard():
        model = YOLOv3('yolov3',num_classes=NUM_CLASSES,is_train=True) #创建模型
        learning_rate = get_lr()
        opt = fluid.optimizer.Momentum(
            learning_rate = learning_rate,
            momentum=0.9,
            regularization=fluid.regularizer.L2Decay(0.0005),
            parameter_list=model.parameters()
        )#创建优化器

        # 读取数据
        train_loader = multithread_loader(catagories_id,'./dataset/train','xml','img', batch_size=10, mode='train')
        valid_loader = multithread_loader(catagories_id,'./dataset/train','xml','img', batch_size=10, mode='valid')

        for epoch in range(MAX_EPOCH):
            for i,data in enumerate(train_loader()):
                img, gt_boxes, gt_labels, img_scale = data
                gt_scores = np.ones(gt_labels.shape).astype('float32')
                gt_scores = to_variable(gt_scores)
                img = to_variable(img)
                gt_boxes = to_variable(gt_boxes)
                gt_labels = to_variable(gt_labels)
                outputs = model(img)  # 前向传播，输出[P0, P1, P2]
                loss = model.get_loss(outputs, gt_boxes, gt_labels, gtscore=gt_scores,
                                      anchors=ANCHORS,
                                      anchor_masks=ANCHOR_MASKS,
                                      ignore_thresh=IGNORE_THRESH,
                                      use_label_smooth=False)  # 计算损失函数
                loss.backward()  # 反向传播计算梯度
                opt.minimize(loss)  # 更新参数
                model.clear_gradients()
                if i % 1 == 0:
                    timestring = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
                    print('{}[TRAIN]epoch {}, iter {}, output loss: {}'.format(timestring, epoch, i, loss.numpy()))
            # save params of model
            if (epoch % 5 == 0) or (epoch == MAX_EPOCH -1):
                fluid.save_dygraph(model.state_dict(), '521-GR-yolo_epoch{}'.format(epoch))
            # 每个epoch结束之后在验证集上进行测试
            model.eval()
            for i, data in enumerate(valid_loader()):
                img, gt_boxes, gt_labels, img_scale = data
                gt_scores = np.ones(gt_labels.shape).astype('float32')
                gt_scores = to_variable(gt_scores)
                img = to_variable(img)
                gt_boxes = to_variable(gt_boxes)
                gt_labels = to_variable(gt_labels)
                outputs = model(img)
                loss = model.get_loss(outputs, gt_boxes, gt_labels, gtscore=gt_scores,
                                      anchors=ANCHORS,
                                      anchor_masks=ANCHOR_MASKS,
                                      ignore_thresh=IGNORE_THRESH,
                                      use_label_smooth=False)
                if i % 1 == 0:
                    timestring = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
                    print('{}[VALID]epoch {}, iter {}, output loss: {}'.format(timestring, epoch, i, loss.numpy()))
            model.train()