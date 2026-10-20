import random
from torch.utils.data import DataLoader
# from torch.utils.data import Dataset
# import numpy as np
# from .data_utils import get_all_files, filter_image
import os
# import SimpleITK as sitk
# from tools.data_utils import save_sitk
from .datagen import *

# 主模型训练 第一阶段 预训练特征提取
def dataset(data, DataGen=DataGen1, TestGen=None, percent=0.7):
    def is_set(name):
        value = data.get(name) if hasattr(data, "get") else data.get(name, None)
        return value is not None and value != "None"

    if TestGen is None: TestGen = DataGen
    train_list, test_list = None, None
    if is_set('train_list') and is_set('test_list'):
        pass
    else:
        if is_set('fold'):
            data.same_as = f"{data.fold_root}/fold{data.fold}"
        if is_set('same_as'):
            # 三种方式：train_list/test_list > fold > same_as 如果均未指定，则随机划分
            # assert is_set('same_as'), 'fold or same_as should be given'
            if '/' in data.same_as.__str__():
                data.train_list = f"{data.same_as}/train.txt"
                data.test_list = f"{data.same_as}/test.txt"
            else:
                data.train_list = f"{data.name}/{data.same_as}/train.txt"
                data.test_list = f"{data.name}/{data.same_as}/test.txt"

    if data.tag == 'train':
        if is_set('train_list') and is_set('test_list'):
            with open(data.train_list, "r") as f:
                train_list = f.read()
            # train_list = train_list.splitlines()
            train_list = [line.strip() for line in train_list.splitlines() if line.strip()]
            # 说明:
            # splitlines() 之后得到的是列表，里面可能包含 ""（空字符串）或 " "（只有空格）。
            # if line.strip() 会过滤掉这些"假空行"。
            # line.strip() 顺便去掉每行首尾的空白字符（换行符、空格、Tab），防止文件名前后带空格导致后续读取失败。

            with open(data.test_list, "r") as f:
                test_list = f.read()
            test_list = test_list.splitlines()

        else: 
            data_list = os.listdir(data.root)
            random.shuffle(data_list)
            percent_7 = int(len(data_list)*percent)
            train_list, test_list = data_list[:percent_7], data_list[percent_7:]
        
        path = f"data/{data.data_name}/{data.train_name}"
        os.makedirs(path, exist_ok=True)
        separator = '\n'
        with open(f"{path}/train.txt","w") as f:
            f.write(separator.join(train_list))
        with open(f"{path}/test.txt","w") as f:
            f.write(separator.join(test_list))

    else:
        assert is_set('test_list')
        with open(data.test_list, "r") as f:
            test_list = f.read()
        test_list = test_list.splitlines()


    test_set = TestGen(data.root, test_list, data.clinical_root)
    test_loader = DataLoader(test_set, shuffle=False, drop_last=False, batch_size=data.test_batch_size, num_workers=data.num_workers)

    if data.tag == 'train':
        train_set = DataGen(data.root, train_list, data.clinical_root,train=True)
        train_loader = DataLoader(train_set, shuffle=True, drop_last=True, batch_size=data.train_batch_size, num_workers=data.num_workers)
        return train_loader, test_loader
    else: return test_loader


# 第二阶段 分类微调
def dataset2(data):
    return dataset(data, DataGen=DataGen2)

