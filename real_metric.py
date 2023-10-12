import pandas as pd
import os
import torch
from torch import nn
from torchvision import datasets, transforms, models
from torch.utils.data import Dataset, DataLoader
import numpy as np


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



if __name__=="__main__":
    pass
