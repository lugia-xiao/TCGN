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

# Grad-Cam图
## 定义获取梯度的函数
grad_block=[]
fmap_block=[]
def backward_hook(module, grad_in, grad_out):
    grad_block.append(grad_out[0].detach())

## 定义获取特征图的函数
def farward_hook(module, input, output):
    fmap_block.append(output)

class Hook(object):
    def __init__(self):
        self.module_name = []
        self.features_in_hook = []
        self.features_out_hook = []

    def __call__(self,module, fea_in, fea_out):
        print("hooker working", self)
        self.module_name.append(module.__class__)
        self.features_in_hook.append(fea_in)
        self.features_out_hook.append(fea_out)
        return None

def cam_show_img(img, feature_map, grads, out_dir="./"):
    H, W, _ = img.shape
    cam = np.zeros(feature_map.shape[1:], dtype=np.float32)		# 4
    grads = grads.reshape([grads.shape[0],-1])					# 5
    weights = np.mean(grads, axis=1)							# 6
    for i, w in enumerate(weights):
        cam += w * feature_map[i, :, :]							# 7
    cam = np.maximum(cam, 0)
    cam = cam / cam.max()
    cam = cv2.resize(cam, (W, H))

    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    cam_img = 0.3 * heatmap + 0.7 * img

    path_cam_img = os.path.join(out_dir, "cam.jpg")
    cv2.imwrite(path_cam_img, cam_img)

if __name__=="__main__":
    net = TCGN().cuda()
    net.load_state_dict(torch.load("record-TCGN/A2-ST_Net-TCGN-best.pth"))
    #print(net)

    net.eval()  # 8
    net._fc.register_forward_hook(farward_hook)
    net._fc.register_backward_hook(backward_hook)

    #get input
    img_input = get_tensor_input(sample_number=0,img_number=2).cuda()
    img=get_original_img(img_input)

    # forward
    output = net(img_input)

    '''
    ['GNAS' 'RHOB' 'MYL12B' 'CLDN4' 'SCD' 'FASN']
    [134, 368, 746, 431, 227, 366]
    '''
    idx =227

    # backward
    net.zero_grad()
    class_loss = output[0, idx]
    class_loss.backward()

    # 生成cam
    grads_val = grad_block[0].cpu().data.numpy().squeeze()
    fmap = fmap_block[0].cpu().data.numpy().squeeze()

    # 保存cam图片
    cam_show_img(img, fmap, grads_val)