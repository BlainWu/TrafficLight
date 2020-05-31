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
for i,data in enumerate(epoch_list):
    xml_path = os.path.join(data,model_xml)
    fd = open(xml_path,'r')
    fd = yaml.load(fd)
    score = float(fd['_Attributes']['eval_metrics']['bbox_map'])
    if score>max_socre:
        max_socre = score
        epoch = data
    else:
        continue
print("验证集最优结果为{0},出自{1}".format(max_socre,epoch))

#fd = open("epoch_1/model.yml",'r')
#fd = yaml.load(fd)
#print(fd['_Attributes']['eval_metrics']['bbox_map'])