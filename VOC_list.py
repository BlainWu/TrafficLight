import os
#常量转变量
root_path = './dataset'
train = 'train'
dev = 'dev'
xml = 'xml'
img = 'img'
#路径设置
train_xml_path = os.path.join(root_path,train,xml)
train_img_path = os.path.join(root_path,train,img)
dev_xml_path = os.path.join(root_path,dev,xml)
dev_img_path = os.path.join(root_path,dev,img)
#读取文件名
train_xml_list = os.listdir(train_xml_path)
train_img_list = os.listdir(train_img_path)
dev_xml_list = os.listdir(dev_xml_path)
dev_img_list = os.listdir(dev_img_path)
#排序
train_xml_list.sort(key = lambda x:int(x.split('.')[0]))
train_img_list.sort(key = lambda x:int(x.split('.')[0]))
dev_xml_list.sort(key = lambda x:int(x.split('.')[0]))
dev_img_list.sort(key = lambda x:int(x.split('.')[0]))
#写train.list
f_train = open("./dataset/train_list.txt","w")
for i in range(len(train_img_list)):
    if not train_img_list[i].split('.')[0] == train_xml_list[i].split('.')[0]:
        print("标签不对应！")
    line = os.path.join(train,img,train_img_list[i]) + ' ' + os.path.join(train,xml,train_xml_list[i]) + '\n'
    f_train.write(line)
f_train.close()

#写dev.list
f_dev = open("./dataset/dev_list.txt","w")
for i in range(len(dev_img_list)):
    if not dev_img_list[i].split('.')[0] == dev_xml_list[i].split('.')[0]:
        print("标签不对应！")
    line = os.path.join(dev,img,dev_img_list[i]) + ' ' + os.path.join(dev,xml,dev_xml_list[i]) + '\n'
    f_dev.write(line)
f_dev.close()