# 设置使用0号GPU卡（如无GPU，执行此代码后仍然会使用CPU训练模型）
import matplotlib
matplotlib.use('Agg')
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import paddlex as pdx
from paddlex.det import transforms

train_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.Padding(coarsest_stride=32),
    transforms.RandomDistort(contrast_prob = 0.01,hue_prob = 0,saturation_prob = 0.01),
    transforms.Normalize()
])

eval_transforms = transforms.Compose([
    transforms.RandomDistort(),
    transforms.Padding(coarsest_stride=32),
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

num_classes = len(train_dataset.labels) + 1
model = pdx.det.FasterRCNN(num_classes=num_classes, 
                            backbone='ResNet101_vd', with_fpn=True,
                            aspect_ratios=[0.5, 1.0, 2.0],
                            anchor_sizes=[8,16, 32, 64, 128, 256])

model.train(num_epochs = 20, train_dataset = train_dataset, 
            train_batch_size=8, eval_dataset = eval_dataset, 
            save_interval_epochs=1, log_interval_steps=2,save_dir='output', 
            pretrain_weights='IMAGENET', optimizer=None, 
            learning_rate=0.0025, warmup_steps=500, 
            warmup_start_lr=1.0/1200, 
            lr_decay_epochs=[8, 11], lr_decay_gamma=0.1,  use_vdl=True,
            resume_checkpoint=None)







