import os
import yaml

epoch_list = []
dir_list = os.listdir(os.getcwd())
model_xml = 'model.yml'

for i,data in enumerate(dir_list):
    if data.split('_')[0] == 'epoch':
        epoch_list.append(data)
    else:
        continue
epoch_list.sort(key = lambda x : int(x.split('_')[1]))
del(epoch_list[-1])
max_socre = 0.0
score_list = []
for i,data in enumerate(epoch_list):
    buffer_data = []
    xml_path = os.path.join(data,model_xml)
    fd = open(xml_path,'r')
    fd = yaml.load(fd)
    score = float(fd['_Attributes']['eval_metrics']['bbox_map'])
    buffer_data.append(data)
    buffer_data.append(score)
    score_list.append(buffer_data)
    if score>max_socre:
        max_socre = score
        epoch = data
    else:
        continue
score_list.sort(key = lambda x:float(x[1]),reverse = True)
print('-------------------------------------------------------------')
print("验证集最优结果为{0},出自{1}".format(score_list[0][1],score_list[0][0]))
print('-------------------------------------------------------------')
print('模型序号                 分数')
for i,data in enumerate(score_list):
    print("{0}                 {1}".format(data[0],data[1]))