#------------------------------------------------
# Project: paddle-Traffic
# Author:Peilin Wu - Najing Normal University
# File name :test.py.py
# Created time :2020/05
#------------------------------------------------
import json
import paddle.fluid as fluid
from Darknet53 import YOLOv3
import numpy as np
from paddle.fluid.dygraph.base import to_variable
from utils import *
import argparse

def arg_parse():#传参函数
    parser = argparse.ArgumentParser(description="YoloV3红绿灯测试结果")
    parser.add_argument("--multi",dest = 'multi',help = "是否允许单张图中存在多个结果",
                        default='true',type = str)
    parser.add_argument("--valid",dest = 'valid_thresh',default=0.2,help = "判定存在目标的阈值")
    parser.add_argument("--nms",dest = 'nms_thresh',default = 0.05, help = "非极大值抑制的重合度阈值")

if __name__ == '__main__':
    args = arg_parse()
    ANCHORS = [10, 13, 16, 30, 33, 23, 30, 61, 62, 45, 59, 119, 116, 90, 156, 198, 373, 326]
    ANCHOR_MASKS = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
    VALID_THRESH = float(args.valid_thresh)
    ALLOW_MULTI = str(args.multi)
    NMS_THRESH = float(args.nms_thresh)
    NMS_TOPK = 400
    NMS_POSK = 100
    NUM_CLASSES = 2

    TESTDIR = './dataset/test'
    MODELDIR = './520yolo_epoch55.pdparams'
    JSONFILE = './55epoch-valid{}-nms{}-multi{}.json'.format(VALID_THRESH,NMS_THRESH,ALLOW_MULTI)

    with fluid.dygraph.guard():
        model = YOLOv3('yolov3', num_classes=NUM_CLASSES, is_train=False)
        params_file_path = MODELDIR
        model_state_dict, _ = fluid.load_dygraph(params_file_path)
        model.load_dict(model_state_dict)
        model.eval()

        total_results = []
        test_loader = test_data_loader(TESTDIR, batch_size= 1, mode='test')
        for i, data in enumerate(test_loader()):
            img_name, img_data, img_scale_data = data
            img = to_variable(img_data)
            img_scale = to_variable(img_scale_data)

            outputs = model.forward(img)
            bboxes, scores = model.get_pred(outputs,
                                     im_shape=img_scale,
                                     anchors=ANCHORS,
                                     anchor_masks=ANCHOR_MASKS,
                                     valid_thresh = VALID_THRESH)

            bboxes_data = bboxes.numpy()
            scores_data = scores.numpy()
            result = multiclass_nms(bboxes_data, scores_data,
                          score_thresh=VALID_THRESH,
                          nms_thresh=NMS_THRESH,
                          pre_nms_topk=NMS_TOPK,
                          pos_nms_topk=NMS_POSK)
            for j in range(len(result)):
                result_j = result[j]
                img_name_j =int(img_name[j])
                if len(result_j) == 0:
                    total_results.append([img_name_j,[]])
                else:
                    np.rint(result_j[:,])
                    total_results.append([img_name_j, result_j.tolist()])
            print('processed {} pictures'.format(len(total_results)))

        print('')
        json.dump(total_results, open(JSONFILE, 'w'))

