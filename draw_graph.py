import os
import pandas as pd
import datetime
import numpy as np
import json
import cv2
import torch

from dataloader import ST_HER2_Dataset
from model import TCGN
from heatmap_func import get_original_img,get_tensor_input

def get_connection():
    net = TCGN().cuda()
    net.load_state_dict(torch.load("record-TCGN/A2-ST_Net-TCGN-best.pth"))
    # print(net)

    net.eval()  # 8
    img_input = get_tensor_input(sample_number=0, img_number=2).cuda()
    img = get_original_img(img_input)
    output = net(img_input)
    np.save("img.npy",img)

def get_position(position):
    x=position%14
    y=position//14
    x=x*16+8
    y=y*16+8
    return (int(x),int(y))

def draw_connection(img,i,j):
    position1=get_position(i)
    position2=get_position(j)
    img=cv2.line(img,position1,position2,(0,255,0))
    return img

if __name__=="__main__":
    #original_img=get_connection()
    img1=np.load("img.npy")
    img2 = np.load("img.npy")
    connection1=np.load("connection1.npy").squeeze(0)
    connection2=np.load("connection2.npy").squeeze(0)
    count=0
    for i in range(connection1.shape[0]):
        for j in range(connection1.shape[1]):
            if count>10:
                break
            if connection1[i][j]!=0:
                count+=1
                draw_connection(img1,i,j)
    cv2.imwrite("img1.png",img1)

    count = 0
    for i in range(connection2.shape[0]):
        for j in range(connection2.shape[1]):
            if count > 10:
                break
            if connection2[i][j] != 0:
                count += 1
                draw_connection(img2, i, j)
    cv2.imwrite("img2.png", img2)

