import paddlex as pdx
import os
from tqdm import tqdm
import json
JSONFILE = './RCNN101.json'
results = []
#读取训练文件并且排序
test_file = './dataset/test'
img_list = os.listdir(test_file)
img_list.sort(key = lambda x : x.split('.')[0])
#加载模型
model = pdx.load_model('./output/best_model')
#训练
for i ,data in enumerate(tqdm(img_list)):
    buffer_result = []
    targets = []
    buffer_result.append(int(data.split('.')[0])) #图片序号
    img_path = os.path.join(test_file,data)
    result = model.predict(img_path)
    if len(result)  != 0:#如果有结果
        for j,data in enumerate(result):
            target = []
            if data['category'] == 'green':
                target.append(0)
            else:
                target.append(1)
            target.append(data['score'])
            target.append(data['bbox'][0]),target.append(data['bbox'][1])
            target.append(data['bbox'][2]),target.append(data['bbox'][3])
            targets.append(target)
        buffer_result.append(targets)
    else:#没有检测到
        buffer_result.append(targets)
    results.append(buffer_result)#保存结果
#生成json格式
json.dump(results,open(JSONFILE,'w'))