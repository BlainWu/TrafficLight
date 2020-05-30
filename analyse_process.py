#------------------------------------------------
# Project: paddle-Traffic
# Author:Peilin Wu - Najing Normal University
# File name :analyse_process.py.py
# Created time :2020/05
#------------------------------------------------
import json
import os
import numpy as np
#统计单个结果文件情况
def file_result(file_path):
    count_empty = 0
    count_multy = 0
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
        # <class 'dict'>,JSON文件读入到内存以后，就是一个Python中的字典。
        # 字典是支持嵌套的，
        for i, data in enumerate(data):
            if len(data[1]) == 0:
                count_empty += 1
            if len(data[1]) > 1:
                count_multy += 1
    print(file_path, "中包含{0}个空结果，{1}个多结果".format(count_empty, count_multy))

#统计整个文件夹中的所有数据结果
def fold_results(fold_path):
    file_lists = os.listdir(fold_path)
    for file_list in file_lists:
        filename = os.path.join(fold_path,file_list)
        file_result(filename)

#转换标签，仅针对一开始训练出错的情况，一次性函数别用
def invert_result_label(origin_path,save_path):
    file_lists = os.listdir(origin_path)
    new_data = []
    for file_list in file_lists:
        filename = os.path.join(origin_path, file_list)
        with open(filename, 'r', encoding='utf-8') as file:
            datas = json.load(file)
            for i, data in enumerate(datas):
                data_buffer = data
                for j in range(len(data_buffer[1])):
                    if data_buffer[1][j][0] == 0.0:
                        data_buffer[1][j][0] = 1.0
                    else:
                        data_buffer[1][j][0] = 0.0
                new_data.append(data_buffer)
        print("处理完成文件：{}".format(filename))
        json.dump(new_data, open(os.path.join(save_path, file_list), 'w'))

#将bbox的浮点数改成整数，实验性函数，对分数没影响
def bbox_float_to_int(origin_path,save_path):
    file_lists = os.listdir(origin_path)
    new_data = []
    for file_list in file_lists:
        filename = os.path.join("./results", file_list)
        with open(filename, 'r', encoding='utf-8') as file:
            datas = json.load(file)
            for i, data in enumerate(datas):
                data_buffer = data
                for j in range(len(data_buffer[1])):
                    data_buffer[1][j][2] = int(data[1][j][2])
                    data_buffer[1][j][3] = int(data[1][j][3])
                    data_buffer[1][j][4] = int(data[1][j][4])
                    data_buffer[1][j][5] = int(data[1][j][5])
                new_data.append(data_buffer)
        print("处理完成文件：{}".format(filename))
        json.dump(new_data, open(os.path.join(save_path, file_list), 'w'))

#去除单张图片中的多个结果，仅留一个
def de_multi(origin_file_path):
    new_datas = []
    file_name = origin_file_path.split('/')[-1] #取文件名
    save_name = "[de_multi]_" + file_name #保存的文件名
    save_path = origin_file_path.replace(file_name, save_name) #保存的文件地址
    print("原结果统计：")
    file_result(origin_file_path)

    with open(origin_file_path,'r',encoding='utf-8') as file:
        datas = json.load(file)
        for i,data in enumerate(datas):
            if len(data[1]) > 1:#有不止一个结果
                scores = []
                inds = []
                buffer_data = []
                saved_data =[]
                for i,multi_resul in enumerate(data[1]):
                    scores.append(multi_resul[1])#按顺序保存所有结果分数
                inds = np.argsort(scores)  # 升排列索引
                buffer_data.append(data[0])
                saved_data.append(data[1][inds[-1]])
                buffer_data.append(saved_data)
                new_datas.append(buffer_data)#多嵌套一层，要不然格式不一致
            else:
                new_datas.append(data)
    with open(save_path,'w',encoding='utf-8') as f:
        json.dump(new_datas,f)
    file_result(save_path)

#de_multi('./uploaded/90.8809765epoch-valid22-nms05.json')
#fold_results('./result_histories')
fold_results("./uploaded")
#fold_results('./results')
#file_result('./results/65epoch-valid2-nms05.json')
#invert_result_label('./result_buffer','./correct_result')