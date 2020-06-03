import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import paddlex as pdx
import paddle.fluid as fluid
from paddlex.det import transforms
train_transforms = transforms.Compose([
        transforms.MixupImage(alpha=1.5, beta=1.5, mixup_epoch=-1),
        transforms.RandomExpand(),
        transforms.RandomCrop(),
        transforms.Resize(target_size=480),
        transforms.RandomHorizontalFlip(prob=0.5),
        transforms.Normalize()
])

eval_transforms = transforms.Compose([
    #transforms.Resize(target_size=480, interp='RANDOM'),
    transforms.Resize(target_size=480),
    transforms.Normalize()
])
#读取数据
train_dataset = pdx.datasets.VOCDetection(data_dir = './dataset', 
                                    file_list = './dataset/train_list.txt', 
                                    label_list = './dataset/label_list.txt', 
                                    transforms=train_transforms)
eval_dataset = pdx.datasets.VOCDetection(data_dir = './dataset', 
                                    file_list = './dataset/dev_list.txt', 
                                    label_list = './dataset/label_list.txt', 
                                    transforms=eval_transforms)

num_classes = len(train_dataset.labels)
model = pdx.det.YOLOv3(num_classes=num_classes, 
    backbone='DarkNet53', 
    anchors= [[6, 5],[10, 13], [16, 30], [33, 23], [30, 61], [62, 45], [59, 119], [116, 90], [156, 198]], 
    anchor_masks=None, 
    ignore_threshold=0.7,
    nms_score_threshold=0.01,
    nms_topk=1000, nms_keep_topk=100,
    nms_iou_threshold=0.45,
    label_smooth=False, 
    train_random_shapes=[320, 352, 384, 416, 448, 480, 512, 544, 576, 608])

model.train(num_epochs = 100, train_dataset = train_dataset, 
            train_batch_size=8, eval_dataset = eval_dataset, 
            save_interval_epochs=1, log_interval_steps=5,save_dir='output', 
            pretrain_weights=None, 
            optimizer=fluid.optimizer.AdamOptimizer(learning_rate=0.001, beta1=0.9, beta2=0.999,
            epsilon=1e-08, parameter_list=None, regularization=None, grad_clip=None, name=None, lazy_mode=False),
            resume_checkpoint=None
            )







