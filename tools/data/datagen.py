import json
import os
import pickle
import random
from torch.utils.data import Dataset
import numpy as np
import SimpleITK as sitk


# DataGen1_20260324 exception: MAMA_MIA没有临床信息嵌入
class DataGen1(Dataset):  # 加上临床信息嵌入
    def __init__(self, root, name_list, clinical_root, train=False):
        self.root = root
        self.name_list = name_list
        self.pcr0_list = []
        self.pcr1_list = []
        self.get_data = False
        # self.train = train
        
        if clinical_root is not None:
            with open(clinical_root, 'rb') as f:
                self.clinical_emb = pickle.load(f)

        try:
            for name in name_list:
                image = sitk.GetArrayFromImage(sitk.ReadImage(f'{root}/{name}')).astype('float32')  # d, w, h
                Min, Max = np.min(image), np.max(image)
                image = (image - Min) / (Max - Min) if (Max - Min > 0) else np.zeros_like(image)
                if len(image.shape) == 3: image = image[np.newaxis, :]
                assert len(image.shape) == 4  # c, d, h, w
                label = int(name[-5])
                if label == 1:
                    # self.pcr1_list.append([name[:-4], image[np.newaxis, :]])
                    self.pcr1_list.append([name[:-4], image])
                else: self.pcr0_list.append([name[:-4], image])
            self.get_data = True
        except Exception as e:
            print(e)
            exit()
            for name in name_list:
                label = int(name[-5])
                if label == 1: self.pcr1_list.append(name[:-4])
                else: self.pcr0_list.append(name[:-4])
            print('fail to preload')


    def __getitem__(self, index):
        if self.get_data:
            name1, image1 = self.pcr0_list[index]
            i = random.randint(0, len(self.pcr0_list)-2)  # 包括len(self.pcr0_list)-2
            if i>=index: i+=1
            name2, image2 = self.pcr0_list[i]

            name3, image3 = self.pcr1_list[index]
            j = random.randint(0, len(self.pcr1_list)-2)
            if j>=index: j+=1
            name4, image4 = self.pcr1_list[j]

            # print(name2[6:10])
            # print(int(name2[6:10]))
            # exit()
            if 'ISPY' in self.root:
                c1 = self.clinical_emb[int(name1[6:10])]
                c2 = self.clinical_emb[int(name2[6:10])]
                c3 = self.clinical_emb[int(name3[6:10])]
                c4 = self.clinical_emb[int(name4[6:10])]
            elif 'pilot' in self.root:
                c1 = self.clinical_emb[name1[:10]]
                c2 = self.clinical_emb[name2[:10]]
                c3 = self.clinical_emb[name3[:10]]
                c4 = self.clinical_emb[name4[:10]]
            else: raise ValueError()
            c2 = np.concatenate([c1, c2], axis=1)
            c4 = np.concatenate([c3, c4], axis=1)
            # print(c1.shape, c2.shape, c3.shape, c4.shape)
            # # (1, 7) (1, 14) (1, 7) (1, 14)
            # exit()
            return [name1,name2,name3,name4], image1, image2, image3, image4, c2, c4
            
        else: 
            raise KeyError('!')
    
    def __len__(self):
        return min(len(self.pcr0_list), len(self.pcr1_list))


class DataGen2(Dataset):  # 第二阶段 分类微调 加临床信息
    def __init__(self, root, name_list, clinical_root, train=False):
        self.root = root
        self.name_list = name_list
        self.clinical_root = clinical_root
        self.data_list = [] 
        self.get_data = False
        
        if clinical_root is not None:
            with open(clinical_root, 'rb') as f:
                self.clinical_emb = pickle.load(f)
        
        try:
            for name in name_list:
                image = sitk.GetArrayFromImage(sitk.ReadImage(f'{root}/{name}')).astype('float32')  # d, w, h
                Min, Max = np.min(image), np.max(image)
                image = (image - Min) / (Max - Min) if (Max - Min > 0) else np.zeros_like(image)
                label = int(name[-5])
                self.data_list.append([name[:-4], image[np.newaxis, :], label])
            self.get_data = True
        except:
            print('fail to preload')

    def __getitem__(self, index):
        if self.get_data:
            name, image, label = self.data_list[index]
            if 'ISPY' in self.root:
                c = self.clinical_emb[int(name[6:10])]
            elif 'pilot' in self.root:
                c = self.clinical_emb[name[:10]]
            else: raise ValueError()
            return name, image, label, np.concatenate([c, c], axis=1)
        else:
            name = self.name_list[index]
            image = sitk.GetArrayFromImage(sitk.ReadImage(f'{self.root}/{name}')).astype('float32')  # d, w, h
            Min, Max = np.min(image), np.max(image)
            image = (image - Min) / (Max - Min) if (Max - Min > 0) else np.zeros_like(image)
            label = int(name[-5])

            if 'ISPY' in self.root:
                c = self.clinical_emb[int(name[6:10])]
            elif 'pilot' in self.root:
                c = self.clinical_emb[name[:10]]
            else: raise ValueError()

            return name[:-4], image[np.newaxis, :], label, np.concatenate([c, c], axis=1)  # name: ISPY1_0000_PCR1, PCR1要留着，方便观察真实值
    
    def __len__(self):
        return len(self.name_list)

