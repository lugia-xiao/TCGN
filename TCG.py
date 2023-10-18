from CMT_block import PatchEmbed,partial,Block,Attention,trunc_normal_
from gnn_block import Graph_Encoding_Block
from transformer_block import Cross_Attention_Transformer
from convnext import BottleNeckBlock,ConvNexStage
import torch.nn as nn
import torch
import torch.nn.functional as F

class Upsample(nn.Module):
    def __init__(self, channels,scale_factor=2, use_conv=True):
        super().__init__()
        self.scale_factor=scale_factor
        self.use_conv = use_conv
        if use_conv:
            self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        #print(x.shape,self.scale_factor)
        x = F.interpolate(x, scale_factor=self.scale_factor, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x

class TCG(nn.Module):
    def __init__(self,img_size,patch_size,num_feature_in,embed_dim,num_heads,num_feature_graph_hidden,
                 num_feature_out,sr_ratio=1.,flatten=True,processed=False,num_patches=None
                 ,topk=4,scale=True,img_out_shape=28):
        super(TCG, self).__init__()
        self.scale=scale
        self.GNN=Graph_Encoding_Block(img_size,patch_size,num_feature_in,embed_dim,num_heads,num_feature_graph_hidden,
                 num_feature_out,sr_ratio,flatten,processed,num_patches,topk)
        self.CNN=None
        if self.scale==True:
            self.CNN=ConvNexStage(in_features=num_feature_in,out_features=num_feature_out,depth=1)
        else:
            self.CNN=BottleNeckBlock(in_features=num_feature_in,out_features=num_feature_out)

        scale_factor=round(img_out_shape/(img_size/patch_size))
        print(scale_factor)
        self.upsample=Upsample(channels=num_feature_out,scale_factor=scale_factor)

        num_dim_cross_attention=img_out_shape*img_out_shape
        self.cross_attention_conv=Cross_Attention_Transformer(num_feature_out,num_feature_out)
        self.cross_attention_graph = Cross_Attention_Transformer(num_feature_out,num_feature_out)

    def forward(self,x,y=None):
        if self.scale:
            conv_feature = self.CNN(x)
            B, C, H, W = conv_feature.shape
            conv_feature=conv_feature.view(B,C,-1).permute(0,2,1) # B, C, H, W -> B, H*W, C
            graph_feature = self.GNN(x)
            #print(graph_feature.shape)
            graph_feature = self.upsample(graph_feature).view(B,C,-1).permute(0,2,1)
            conv_feature1=self.cross_attention_conv(conv_feature
                                                   ,graph_feature).permute(0,2,1).reshape(B, C, H, W)
            graph_feature1=self.cross_attention_graph(graph_feature
                                                     ,conv_feature).permute(0,2,1).reshape(B, C, H, W)
            return conv_feature1,graph_feature1
        else:
            conv_feature = self.CNN(x)
            B, C, H, W = conv_feature.shape
            conv_feature = conv_feature.view(B, C, -1).permute(0, 2, 1)  # B, C, H, W -> B, H*W, C
            graph_feature = self.GNN(y)
            graph_feature = self.upsample(graph_feature).view(B, C, -1).permute(0, 2, 1)
            conv_feature1 = self.cross_attention_conv(conv_feature
                                                     , graph_feature).permute(0, 2, 1).reshape(B, C, H, W)
            graph_feature1 = self.cross_attention_graph(graph_feature
                                                       , conv_feature).permute(0, 2, 1).reshape(B, C, H, W)
            return conv_feature1, graph_feature1

if __name__=="__main__":
    model=ConvNexStage(46,92,1).cuda()
    x=torch.randn((2,46,56,56)).cuda()
    out=model(x)
    print(out.shape)






