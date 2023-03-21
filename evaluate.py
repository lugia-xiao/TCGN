import torch
from torchvision import datasets, transforms, models
from torch.utils.data import Dataset, DataLoader # 载入数据专用

import os
import pandas as pd
import datetime
import numpy as np

from dataloader import ST_HER2_Dataset
from real_metric import compare_prediction_label_list
from model import TCGN
import torch.nn as nn

use_gpu = torch.cuda.is_available() # gpu加速
#use_gpu=False
os.environ['CUDA_VISIBLE_DEVICES'] = "0" # 确定显卡
torch.cuda.empty_cache() # 清除显卡缓存

def get_size():
    model=TCGN()
    #model=models.densenet121()
    total = sum([param.nelement() for param in model.parameters()])
    print("Number of parameter: %.2fM" % (total / 1e6))

    param_size = 0
    param_sum = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
        param_sum += param.nelement()
    buffer_size = 0
    buffer_sum = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
        buffer_sum += buffer.nelement()
    all_size = (param_size + buffer_size) / 1024 / 1024
    print('模型总大小为：{:.3f}MB'.format(all_size))
    print(param_size, param_sum, buffer_size, buffer_sum, all_size)


def make_HER2_validation_dataset(transform,test_sample_number,batch_size=32):
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
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=True)
    return test_loader, test_sample

def calculate_pcc(x,y):
    n=len(x)
    sum_xy = np.sum(np.sum(x * y))
    sum_x = np.sum(np.sum(x))
    sum_y = np.sum(np.sum(y))
    sum_x2 = np.sum(np.sum(x * x))
    sum_y2 = np.sum(np.sum(y * y))
    pc = (n * sum_xy - sum_x * sum_y) / np.sqrt((n * sum_x2 - sum_x * sum_x) * (n * sum_y2 - sum_y * sum_y))
    return pc

def get_pccs(prediction_list,real_list,meta_path="../data/meta.csv"):
    header = list(pd.read_csv(meta_path,index_col=None).columns)[4:]
    prediction_df = pd.DataFrame(columns=header, data=prediction_list)
    real_df = pd.DataFrame(columns=header, data=real_list)
    header = list(prediction_df.columns)
    pccs = []
    for genei in header:
        predictioni = prediction_df.loc[:, genei]
        reali = real_df.loc[:, genei]
        pcci = calculate_pcc(predictioni, reali)
        pccs.append(pcci)
    pccs = np.array(pccs)
    return pccs

def find_sample_number(sample_name):
    sample_list = ["A2", "A3", "A4", "A5", "A6", "B1", "B2", "B3", "B4", "B5", "B6", "C1", "C2", "C3", "C4", "C5", "C6",
                   "D1", "D2", "D3", "D4", "D5", "D6", "E1", "E2", "E3", "F1", "F2", "F3", "G1", "G2", "G3"]
    for i in range(len(sample_list)):
        if sample_list[i]==sample_name:
            return i

def evaluate_model(test_sample_number,model_path):
    sample_list = ["A2", "A3", "A4", "A5", "A6", "B1", "B2", "B3", "B4", "B5", "B6", "C1", "C2", "C3", "C4", "C5", "C6",
                   "D1", "D2", "D3", "D4", "D5", "D6", "E1", "E2", "E3", "F1", "F2", "F3", "G1", "G2", "G3"]
    model_name = "TCGN"
    sample_name = sample_list[test_sample_number]
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
    test_loader, test_sample = make_HER2_validation_dataset(my_transforms, test_sample_number)
    print("finish loading")

    my_model = TCGN()
    if use_gpu:
        my_model.load_state_dict(torch.load(model_path), strict=True)
        my_model = my_model.cuda()

    epoch_real_record_val = []
    epoch_predict_record_val = []
    my_model.eval()

    loss_train_sum=0
    step_val=0
    loss_func = nn.MSELoss()

    for stepi, (imgs, genes) in enumerate(test_loader, 1):
        step_val = stepi
        with torch.no_grad():
            if use_gpu:
                imgs = imgs.cuda()
                genes = genes.cuda()
            predictions = my_model(imgs)
            loss = loss_func(predictions, genes)
            loss_train_sum += loss.cpu().item()

        if use_gpu:
            predictions = predictions.cpu().detach().numpy()
        else:
            predictions = predictions.detach().numpy()
        epoch_real_record_val += list(genes.cpu().numpy())
        epoch_predict_record_val += list(predictions)
    pccs = get_pccs(epoch_predict_record_val, epoch_real_record_val)
    print(sample_name,"median pccs:",np.nanmedian(pccs))
    sample_names = [sample_name for i in range(len(pccs))]
    model_names = [model_name for i in range(len(pccs))]
    df = pd.DataFrame({"Model": model_names, "Sample": sample_names, "PCC": pccs})

    if sample_name in ["B1","C1","D1","E1","F1","G2"]:
        gene_predicted=np.array(epoch_predict_record_val)
        np.save(sample_name+".npy", gene_predicted)
    return df,loss_train_sum/step_val

def evaluate_all_TCGNv6():
    sample_list = ["A2", "A3", "A4", "A5", "A6", "B1", "B2", "B3", "B4", "B5", "B6", "C1", "C2", "C3", "C4", "C5", "C6",
                   "D1", "D2", "D3", "D4", "D5", "D6", "E1", "E2", "E3", "F1", "F2", "F3", "G1", "G2", "G3"]
    sample_list=["A2"]
    model_name = "TCGN"
    df_all=None
    loss_all=[]
    for samplei in sample_list:
        test_sample_numberi = find_sample_number(samplei)
        model_pathi="./record-TCGN/"+samplei+"-ST_Net-TCGN-best.pth"
        dfi,lossi = evaluate_model(test_sample_numberi, model_pathi)
        loss_all.append(lossi)
        if samplei=="A2":
            df_all=dfi
        else:
            df_all=pd.concat([df_all,dfi])
    df_all.to_csv("./TCGN_evaluate.csv")
    df_loss = pd.DataFrame({"Model":["TCGN" for i in range(len(loss_all))],
                            "Sample": sample_list,"Loss": loss_all})
    df_loss.to_csv("./loss_TCGN.csv")


if __name__=="__main__":
    get_size()
    #evaluate_all_TCGNv6()

