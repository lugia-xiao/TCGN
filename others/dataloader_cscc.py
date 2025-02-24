from torch.utils.data import Dataset, DataLoader # 载入数据专用
from torchvision import transforms # 图像变换专用
import torch # pytorch
import os
import numpy as np
import pandas as pd
from PIL import Image # used for read graph
from dataset import pk_load

use_gpu = torch.cuda.is_available() # gpu加速

class ST_cscc_Dataset(Dataset):
    def __init__(self,test_sample_number,transforms,train_or_test='train'):
        self.train_or_test=train_or_test
        cscc_set = pk_load(fold=test_sample_number,mode=train_or_test,dataset_name='cscc')
        #print(type(cscc_set),type(pk_load(fold=test_sample_number,mode=train_or_test,dataset_name='cscc')))
        cscc_loader = DataLoader(cscc_set, batch_size=1, num_workers=0, shuffle=True)
        self.imgs=[]
        self.exps=[]
        self.transforms=transforms
        for stepi, input in enumerate(cscc_loader, 1):
            '''
            img <class 'torch.Tensor'> torch.Size([1, 325, 3, 112, 112])
            position <class 'torch.Tensor'> torch.Size([1, 325, 2])
            exp <class 'torch.Tensor'> torch.Size([1, 325, 785])
            <class 'torch.Tensor'> torch.Size([1, 325, 325])
            <class 'torch.Tensor'> torch.Size([1, 325, 785])
            <class 'torch.Tensor'> torch.Size([1, 325])
            <class 'torch.Tensor'> torch.Size([1, 325, 2])
            '''
            self.imgs.append(self.transforms[0](input[0].squeeze(dim=0)/255))
            self.exps.append(input[2].squeeze(dim=0))
        self.imgs=torch.concat(self.imgs,dim=0)
        self.exps=torch.concat(self.exps,dim=0)

    def __getitem__(self, index):
        if self.train_or_test=='train':
            return self.transforms[1](self.imgs[index]),self.exps[index]
        elif self.train_or_test=='test':
            return self.imgs[index],self.exps[index]

    def __len__(self):
        return self.exps.shape[0]

def make_csc_dataset(test_sample_number,transform,batch_size=32):
    patients_all=["P2_ST_rep1","P2_ST_rep2","P2_ST_rep3","P5_ST_rep1","P5_ST_rep2","P5_ST_rep3","P9_ST_rep1",
                  "P9_ST_rep2","P9_ST rep3","P10_ST_rep1","P10_ST_rep2","P10_ST_rep3"]
    test_sample=patients_all[test_sample_number]
    print("test sample:",test_sample)
    train_dataset=ST_cscc_Dataset(test_sample_number,transform,train_or_test='train')
    test_dataset = ST_cscc_Dataset(test_sample_number,transform,train_or_test='test')
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0,drop_last = True)
    test_loader=DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=0,drop_last = True)
    return train_loader,test_loader,test_sample

if __name__=="__main__":
    import time

    time_begin = time.time()
    from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

    train_transform = transforms.Compose(
        [transforms.RandomRotation(180),
         transforms.RandomHorizontalFlip(0.5),  # randomly 水平旋转
         transforms.RandomVerticalFlip(0.5),  # 随机竖直翻转
         ]
    )
    basic_transform = transforms.Compose(
        [transforms.Resize((224, 224), antialias=True),  # resize to 256x256 square
         transforms.ConvertImageDtype(torch.float),
         transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)  # 归一化
         ]
    )
    my_transforms = [basic_transform, train_transform]
    # /export/home/bs2021
    train_loader, test_loader, test_sample = make_csc_dataset(0,my_transforms)
    print(len(test_loader), len(train_loader))
    print("finish loading")
    for stepi, (img, gene) in enumerate(train_loader, 1):
        imgs = img.cuda()
        if stepi % 20 == 0:
            print(stepi)
            print(imgs.shape,gene.shape)
        pass
    time_end = time.time()
    print("improved:", time_end - time_begin)
