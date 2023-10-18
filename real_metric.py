import pandas as pd
import os
import torch
from torch import nn
from torchvision import datasets, transforms, models
from torch.utils.data import Dataset, DataLoader # 载入数据专用
import numpy as np

def get_prediciton(model,loader):
    model.eval()
    prediction_record=[]
    real_record=[]
    for stepi, (imgs, genes, samples) in enumerate(loader, 1):
        with torch.no_grad():
            predictions = model(imgs)
        predictionsi = list(predictions.detach().numpy().item())
        genesi=list(genes.numpy().item())
        prediction_record+=predictionsi
        real_record+=genesi
    header=list(pd.read_csv("../data/meta.csv",
                       index_col=None).columns)[4:]
    prediction_df=pd.DataFrame(columns=header, data=prediction_record)
    real_df=pd.DataFrame(columns=header, data=real_record)
    return prediction_df,real_df

def calculate_pcc(x,y):
    n=len(x)
    sum_xy = np.sum(np.sum(x * y))
    sum_x = np.sum(np.sum(x))
    sum_y = np.sum(np.sum(y))
    sum_x2 = np.sum(np.sum(x * x))
    sum_y2 = np.sum(np.sum(y * y))
    pc = (n * sum_xy - sum_x * sum_y) / np.sqrt((n * sum_x2 - sum_x * sum_x) * (n * sum_y2 - sum_y * sum_y))
    return pc

def compare_prediction_label_list(prediction_list,real_list):
    header = list(range(785))
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
    median_pcc = np.nanmedian(pccs)
    return median_pcc

def compare_prediction_label(prediction_df, real_df):
    header=list(prediction_df.columns)
    pccs=[]
    for genei in header:
        predictioni=prediction_df.loc[:,genei]
        reali=real_df.loc[:,genei]
        pcci=calculate_pcc(predictioni,reali)
        pccs.append(pcci)
    pccs_df = pd.DataFrame(columns=header, data=[pccs])
    pccs=np.array(pccs)
    mean_pcc=np.nanmean(pccs)
    median_pcc=np.nanmedian(pccs)
    print("mean:",mean_pcc)
    print("median:",median_pcc)
    return pccs_df,mean_pcc,median_pcc


def find_min_loss_epoch(root,sample_name):
    model_name = root.split("/")[-1].split("-")[1]
    final_epoch = 350
    if model_name == "DenseNet121":
        final_epoch = 200
    loss_record_path = root + "/loss-ST_Net-" + model_name + "-" + sample_name + "-" + str(final_epoch) + ".csv"
    loss_record = pd.read_csv(loss_record_path)
    i = 1
    min_val_loss = loss_record.loc[i * 50 - 1, "val_loss"]
    min_loss_epoch = i * 50
    while i * 50 <= final_epoch:
        val_lossi = loss_record.loc[i * 50 - 1, "val_loss"]
        if val_lossi < min_val_loss:
            min_val_loss = val_lossi
            min_loss_epoch = i * 50
        i += 1
    min_loss_model_path = root + "/ST_Net-" + model_name + "-" + sample_name + "-" + str(min_loss_epoch) + ".pth"
    print("epoch with the least validation loss:", min_loss_epoch)
    return min_loss_model_path


if __name__=="__main__":
    #file_path="/dssg/home/acct-clslh/clslh/xiaoxiao/ST/HER2_positive-breast-tumor-ST/CNN" \
    #         "/other-CNNs/record-CMT/ST_Net-CMT-A2-350.pth"
    '''file_path="/dssg/home/acct-clslh/clslh/xiaoxiao/ST/HER2_positive-breast-tumor-ST/CNN/other-CNNs/record-CMT/"+\
              "ST_Net-CMT-A5-50.pth"'''
    '''file_path = "/dssg/home/acct-clslh/clslh/xiaoxiao/ST/HER2_positive-breast-tumor-ST/CNN/ST-Net/record-DenseNet121/" + \
                "ST_Net-DenseNet121-A3-150.pth"
    evaluate(file_path,use_gpu=False,write_pccs=False)'''
    root="../../CNN/other-CNNs/record-CMT"
    sample_name="A6"
    '''min_loss_model_path=find_min_loss_epoch(root,sample_name)
    mean_pcc,median_pcc=evaluate(min_loss_model_path,use_gpu=False,write_pccs=False)'''
    #best_model_path, max_median_pcc=find_model_with_max_pcc(root,sample_name,use_gpu=True,write_predictions=False,write_pccs=False)







