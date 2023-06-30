import torch
from torchvision import datasets, transforms, models
from torch.utils.data import Dataset, DataLoader # 载入数据专用

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

import os
import pandas as pd
import datetime
import numpy as np
import json
import cv2

from dataloader import ST_HER2_Dataset
from model import TCGN
import torch.nn as nn

use_gpu = torch.cuda.is_available() # gpu加速
#use_gpu=False
os.environ['CUDA_VISIBLE_DEVICES'] = "0" # 确定显卡
torch.cuda.empty_cache() # 清除显卡缓存

'''
['SCD' 'GNAS' 'FASN' 'MYL12B']
[227, 134, 366, 746]
C4
'''

def make_HER2_validation_dataset(transform,test_sample_number,batch_size=1):
    sample_all = ['A2', 'A3', 'A4', 'A5', 'A6']
    for patient in ['B', 'C', 'D']:
        for sectioni in range(1, 7):
            sample_all.append(patient + str(sectioni))
    for patient in ['E', 'F', 'G']:
        for sectioni in range(1, 4):
            sample_all.append(patient + str(sectioni))
    test_sample = sample_all[test_sample_number]
    print("test sample:", test_sample)
    test_dataset = ST_HER2_Dataset(test_sample_number, transform, train_or_test='test')
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False)
    return test_loader, test_sample

def get_tensor_input(sample_number=14,img_number=1):
    '''model_name = "TCGN"
    sample_name = "C4"
    model_path = "record-TCGN/A2-ST_Net-TCGN-best.pth"'''

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
    test_loader, test_sample = make_HER2_validation_dataset(my_transforms, sample_number)
    print("finish loading")

    '''my_model = TCGN()
    if use_gpu:
        my_model.load_state_dict(torch.load(model_path), strict=True)
        my_model = my_model.cuda()'''

    img_output = None
    for stepi, (imgs, genes) in enumerate(test_loader, 1):
        #print(stepi)
        if stepi==img_number:
            img_output = imgs
            break
    return img_output

def get_original_img(img_input):
    img=img_input.clone().squeeze(dim=0).cpu().numpy().transpose(1,2,0)
    img=(img*IMAGENET_DEFAULT_STD+IMAGENET_DEFAULT_MEAN)*255
    img=img[:,:,[2,1,0]]
    #print(img.shape)
    #img=cv2.imread("D:/project/Graduate/data/piece_images/C4-10x15.jpg")
    #img=cv2.resize(img,(224,224))
    return img

if __name__=="__main__":
    img_input=get_tensor_input()
    print(img_input)
    print(torch.mean(img_input))


    img=get_original_img(img_input)
    print(np.mean(img))
    cv2.imshow("xx",img)
    cv2.waitKey(10000)