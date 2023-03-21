import torch
from torchvision import datasets, transforms, models
import torch.nn as nn

import pandas as pd
import datetime

from dataloader import make_HER2_dataset
from model import TCGN

import psutil
import os
import gc

use_gpu = torch.cuda.is_available() # gpu加速
#use_gpu=False
os.environ['CUDA_VISIBLE_DEVICES'] = "0" # 确定显卡
torch.cuda.empty_cache() # 清除显卡缓存

xs_kwargs = dict(
        qkv_bias=True, embed_dims=[52,104,208,416], stem_channel=16, num_heads=[1,2,4,8],
        depths=[3,3,12,3], mlp_ratios=[3.77,3.77,3.77,3.77], qk_ratio=1, sr_ratios=[8,4,2,1])
small_kwargs=dict(
        qkv_bias=True, embed_dims=[64,128,256,512], stem_channel=32, num_heads=[1,2,4,8],
        depths=[3,3,16,3], mlp_ratios=[4,4,4,4], qk_ratio=1, sr_ratios=[8,4,2,1])
base_kwargs = dict(
        qkv_bias=True, embed_dims=[76,152,304,608], stem_channel=38, num_heads=[1,2,4,8],
        depths=[4,4,20,4], mlp_ratios=[4,4,4,4], qk_ratio=1, sr_ratios=[8,4,2,1], dp=0.3)

def ST_TCGN(test_sample_number,model_size="tiny"):
    batch_size=32
    epoch=26 if test_sample_number<23 else 36
    print("GPU available:", use_gpu)
    # load data
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
    #/export/home/bs2021
    train_loader, test_loader, test_sample = make_HER2_dataset(test_sample_number,my_transforms,batch_size)
    print("finish loading")

    # initialize model
    model_name = "TCGN"
    dirs = "./record-" + model_name
    if not os.path.exists(dirs):
        os.makedirs(dirs)
    my_model=None
    if model_size=="tiny":
        my_model = TCGN()
        if use_gpu:
            my_model = my_model.cuda()
            my_model.load_state_dict(torch.load('./pretrained/cmt_tiny.pth'), strict=False)
    elif model_size=="xs":
        my_model = TCGN(**xs_kwargs)
        if use_gpu:
            my_model = my_model.cuda()
            my_model.load_state_dict(torch.load('./pretrained/cmt_xs.pth'), strict=False)
    elif model_size=="small":
        my_model = TCGN(**small_kwargs)
        if use_gpu:
            my_model = my_model.cuda()
            my_model.load_state_dict(torch.load('./pretrained/cmt_small.pth'), strict=False)
    elif model_size=="base":
        my_model = TCGN(**base_kwargs)
        if use_gpu:
            my_model = my_model.cuda()
            my_model.load_state_dict(torch.load('./pretrained/cmt_base.pth'), strict=False)

    # train the model
    # v5: remove the first GNN
    optimizer = torch.optim.Adam(my_model.parameters(),lr=1e-5, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

    loss_func = nn.MSELoss()
    dfhistory = pd.DataFrame(columns=["epoch", "train_loss", "val_loss", "train_median_pcc", "val_median_pcc"])
    print("Start Training...")
    nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_step_freq = 20

    print("==========" * 8 + "%s" % nowtime)
    from real_metric import compare_prediction_label_list
    best_val_median_pcc = 0
    record_file = open('./record-' + model_name + '/'+test_sample+'-'+'best_epoch.csv', mode='w')
    record_file.write("epoch,best_val_median_pcc\n")
    for epoch in range(1, epoch):
        my_model.train()
        loss_train_sum = 0.0
        epoch_median_pcc_val = None
        epoch_median_pcc_train = None

        epoch_real_record_train = []
        epoch_predict_record_train = []
        step_train = 0
        for stepi, (imgs, genes) in enumerate(train_loader, 1):
            #print(stepi,end="")
            step_train = stepi
            optimizer.zero_grad()
            if use_gpu:
                imgs = imgs.cuda()
                genes = genes.cuda()

            predictions = my_model(imgs)
            loss = loss_func(predictions, genes)
            ## 反向传播求梯度
            loss.backward()  # 反向传播求各参数梯度
            optimizer.step()  # 用optimizer更新各参数

            if use_gpu:
                predictions = predictions.cpu().detach().numpy()
            else:
                predictions = predictions.detach().numpy()
            epoch_real_record_train += list(genes.cpu().numpy())
            epoch_predict_record_train += list(predictions)
            epoch_median_pcc_train = compare_prediction_label_list(epoch_predict_record_train, epoch_real_record_train)

            if use_gpu:
                loss_train_sum += loss.cpu().item()  # 返回数值要加.item
            else:
                loss_train_sum += loss.item()

            gc.collect()
            if stepi % log_step_freq == 0:  # 当多少个batch后打印结果
                print(("training: [epoch = %d, step = %d, images = %d] loss: %.3f, " + "median pearson coefficient" + ": %.3f") %
                      (epoch, stepi, stepi*batch_size,loss_train_sum / stepi, epoch_median_pcc_train))


        my_model.eval()
        loss_val_sum = 0.0
        epoch_real_record_val = []
        epoch_predict_record_val = []
        step_val = 0
        for stepi, (imgs, genes) in enumerate(test_loader, 1):
            #print(stepi, end="")
            step_val = stepi
            with torch.no_grad():
                if use_gpu:
                    imgs = imgs.cuda()
                    genes = genes.cuda()
                predictions = my_model(imgs)
                loss = loss_func(predictions, genes)

                if use_gpu:
                    loss_val_sum += loss.cpu().item()  # 返回数值要加.item
                else:
                    loss_val_sum += loss.item()

                if use_gpu:
                    predictions = predictions.cpu().detach().numpy()
                else:
                    predictions = predictions.detach().numpy()

            epoch_real_record_val += list(genes.cpu().numpy())
            epoch_predict_record_val += list(predictions)
            epoch_median_pcc_val = compare_prediction_label_list(epoch_predict_record_val, epoch_real_record_val)

            if stepi * 2 % log_step_freq == 0:  # 当多少个batch后打印结果
                print("validation sample", test_sample)
                print(("validation: [step = %d] loss: %.3f, " + "median pearson coefficient" + ": %.3f") %
                      (stepi, loss_val_sum / stepi, epoch_median_pcc_val))

        historyi = (
            epoch, loss_train_sum / step_train, loss_val_sum / step_val, epoch_median_pcc_train, epoch_median_pcc_val)
        dfhistory.loc[epoch - 1] = historyi

        print(model_name)
        print((
                  "\nEPOCH = %d, loss_train_avg = %.3f, loss_val_avg = %.3f, epoch_median_pcc_train = %.3f, epoch_median_pcc_val = %.3f")
              % historyi)

        nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print("\n" + "==========" * 8 + "%s" % nowtime)
        if epoch >= 1:
            if epoch_median_pcc_val > best_val_median_pcc:
                best_val_median_pcc = epoch_median_pcc_val
                print("Sample:",test_sample,"best epoch now:", epoch)
                record_file.write(str(epoch) + "," + str(epoch_median_pcc_val) + "\n")
                record_file.flush()
                torch.save(my_model.state_dict(),
                           "./record-" + model_name + "/"  + test_sample+"-"+ "ST_Net-" + model_name + "-best.pth")
    record_file.close()

if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--fold', type=int, default=0, help='test section')
    parser.add_argument('--type', type=str, default="tiny", help='test section')
    args = parser.parse_args()
    ST_TCGN(args.fold,args.type)
