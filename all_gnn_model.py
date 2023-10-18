from CMT_block import PatchEmbed,partial,Block,Attention,trunc_normal_
from gnn_block import Graph_Encoding_Block
from transformer_block import Channel_Attention
from convnext import ConvNexStage
import torch.nn as nn
import torch
from TCG import TCG

class MemoryEfficientSwish(nn.Module):
    def forward(self, x):
        return SwishImplementation.apply(x)

class SwishImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_tensors[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))

class All_GNN(nn.Module):
    def __init__(self,num_classes=250,img_size=112,depths=[ 3, 6, 3, 3], dims=[46,92,192,384],dp=0.1):
        super(All_GNN, self).__init__()
        self.stem=ConvNexStage(in_features=3,out_features=dims[0],depth=depths[0])
        self.down_sample1=TCG(img_size=img_size//2, patch_size=img_size//(2*14), num_feature_in=dims[0],
                                 embed_dim=dims[1], num_feature_graph_hidden=dims[1]
                                 , num_feature_out=dims[1], flatten=True, num_heads=4,topk=2,
                              scale=True,img_out_shape=img_size//4)
        self.stage1=nn.ModuleList([
            TCG(img_size=img_size // 4, patch_size=img_size // (4 * 14), num_feature_in=dims[1],
                embed_dim=dims[1], num_feature_graph_hidden=dims[1]
                , num_feature_out=dims[1], flatten=True, num_heads=4, topk=2,
                scale=False, img_out_shape=img_size//4)
        for i in range(depths[1]-1)])

        self.down_sample2 = TCG(img_size=img_size // 4, patch_size=img_size // (4 * 14), num_feature_in=dims[1],
                                embed_dim=dims[2], num_feature_graph_hidden=dims[2]
                                , num_feature_out=dims[2], flatten=True, num_heads=4, topk=2,
                                scale=True, img_out_shape=img_size//8)
        self.stage2_0 = nn.ModuleList([
            TCG(img_size=img_size // 8, patch_size=img_size // (8 * 14), num_feature_in=dims[2],
                embed_dim=dims[2], num_feature_graph_hidden=dims[2]
                , num_feature_out=dims[2], flatten=True, num_heads=4, topk=2,
                scale=False, img_out_shape=img_size//8)
            for i in range((depths[2] - 1)//2+1)])

        self.stage2_1 = nn.ModuleList([
            TCG(img_size=img_size // 8, patch_size=img_size // (4 * 14), num_feature_in=dims[2],
                embed_dim=dims[2], num_feature_graph_hidden=dims[2]
                , num_feature_out=dims[2], flatten=True, num_heads=4, topk=2,
                scale=False, img_out_shape=img_size//8)
            for i in range((depths[2] - 1) // 2)])

        self.down_sample3 = TCG(img_size=img_size // 8, patch_size=img_size // (8*7), num_feature_in=dims[2],
                                embed_dim=dims[3], num_feature_graph_hidden=dims[3]
                                , num_feature_out=dims[3], flatten=True, num_heads=4, topk=9,
                                scale=True, img_out_shape=img_size//16)
        self.stage3 = nn.ModuleList([
            TCG(img_size=img_size // 16, patch_size=img_size // (16*7), num_feature_in=dims[3],
                embed_dim=dims[3], num_feature_graph_hidden=dims[3]
                , num_feature_out=dims[3], flatten=True, num_heads=4, topk=2,
                scale=False, img_out_shape=img_size // 16)
            for i in range(depths[3]-1)])

        self._fc = nn.Conv2d(dims[-1], dims[-1], kernel_size=1)
        self._bn = nn.BatchNorm2d(dims[-1], eps=1e-5)
        self._swish = MemoryEfficientSwish()
        self._avg_pooling = nn.AdaptiveAvgPool2d(1)
        self._drop = nn.Dropout(dp)
        self.head = nn.Sequential(nn.Linear(dims[-1], dims[-1] * 4, bias=True), nn.GELU(),
                                  nn.Linear(dims[-1] * 4, num_classes, bias=True))

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
            if isinstance(m, nn.Conv2d) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


    def forward_feature(self,x):
        x=self.stem(x)
        x1,x2=self.down_sample1(x)
        for layeri in self.stage1:
            x1,x2=layeri(x1,x2)
        x=x1
        x1, x2 = self.down_sample2(x)
        for layeri in self.stage2_0:
            x1, x2 = layeri(x1, x2)
        for layeri in self.stage2_1:
            x1, x2 = layeri(x1, x2)
        x=x1
        x1, x2 = self.down_sample3(x)
        for layeri in self.stage3:
            x1, x2 = layeri(x1, x2)
        x=x1
        x = self._fc(x)
        x = self._bn(x)
        x = self._swish(x)
        x = self._avg_pooling(x).flatten(start_dim=1)
        x = self._drop(x)
        return x

    def forward(self,x):
        x=self.forward_feature(x)
        x=self.head(x)
        return x

def get_size():
    model=All_GNN()
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


if __name__=="__main__":
    get_size()
    img=torch.randn((2,3,112,112)).cuda()
    my_model=All_GNN().cuda()
    output=my_model(img)
    print(output.shape)