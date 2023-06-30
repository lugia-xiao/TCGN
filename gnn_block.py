from censnet_block import GCN,GCN_big
from transformer_block import Multi_head_cross_attention_with_graph_output,Edge_Node_Transformer_fusion
from transformer_block import Edge_Node_Transformer_fusion_big
from CMT_block import Block
from convnext import ConvNexStage

import torch
import torch.nn as nn
from timm.models.layers import to_2tuple, trunc_normal_

import time


class Graph_Encoding_Block(nn.Module):
    def __init__(self,img_size,patch_size,num_feature_in,embed_dim,num_heads,num_feature_graph_hidden,
                 num_feature_out,sr_ratio=1.,flatten=True,processed=False,num_patches=None):
        super(Graph_Encoding_Block, self).__init__()
        self.flatten=flatten
        self.processed=processed
        self.relative_pos_flag=False
        self.patch_size=patch_size

        if processed:
            self.patch_embedding=nn.Identity()
        else:
            self.patch_embedding = PatchEmbed(img_size=img_size,patch_size=patch_size
                                          , in_chans=num_feature_in, embed_dim=embed_dim)
            self.relative_pos = nn.Parameter(torch.randn(
                num_heads, self.patch_embedding.num_patches,
                self.patch_embedding.num_patches))
            self.relative_pos_flag = True

        if num_patches is not None:
            self.relative_pos_flag = True
            self.relative_pos = nn.Parameter(torch.randn(
                num_heads, num_patches,num_patches))

        self.CMT_encoder1=Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0.,
                                attn_drop=0.,drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                                qk_ratio=1, sr_ratio=sr_ratio)

        if patch_size==4:
            self.convnext1 = ConvNexStage(in_features=num_feature_in, out_features=num_feature_in, depth=2)
            self.convnext2 = ConvNexStage(in_features=num_feature_in, out_features=embed_dim, depth=1)
        elif patch_size==2:
            self.convnext1 = ConvNexStage(in_features=num_feature_in, out_features=embed_dim, depth=2)

        self.MHSA=Multi_head_cross_attention_with_graph_output(emb_size = embed_dim,
                                                              num_heads=num_heads,add_diag=True)
        self.censnet=GCN(nfeat_v=embed_dim, nfeat_e=3+num_heads+1, nhid=num_feature_graph_hidden,
                         nfeat_v_out=num_feature_out, dropout=0.1)
        self.transformer_fusion_encoder=Edge_Node_Transformer_fusion(emb_dim_node=num_feature_out,
                                                               emb_dim_edge=2*(3+num_heads+1),
                                                               num_heads=num_heads)
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

    def forward(self,x,relative_pos=None):
        '''

        :param x: (batch,num_nodes,num_feature)
        :return: (batch,num_nodes,num_feature)
        '''
        if self.patch_size==4:
            conv_feature=self.convnext2(self.convnext1(x)).flatten(2).transpose(1, 2)
        else:
            conv_feature = self.convnext1(x).flatten(2).transpose(1, 2)
        #print(conv_feature.shape)

        B = x.shape[0]
        if self.processed == False:
            x, (H, W) = self.patch_embedding(x)
        else:
            H = round((x.shape[1]) ** 0.5)
            W = H
        if self.relative_pos_flag==True:
            relative_pos=self.relative_pos

        x=self.CMT_encoder1(x, H, W, relative_pos)

        x,graph=self.MHSA(conv_feature,x)

        x,edges,topk,T=self.censnet(x,graph)

        x=self.transformer_fusion_encoder(x,edges,H, W, relative_pos,topk,T)

        if self.flatten:
            x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])

        assert img_size[0] % patch_size[0] == 0 and img_size[1] % patch_size[1] == 0, \
            f"img_size {img_size} should be divided by patch_size {patch_size}."

        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)

        H, W = H // self.patch_size[0], W // self.patch_size[1]
        return x, (H, W)

class Graph_Encoding_Block_big(nn.Module):
    def __init__(self,img_size,patch_size,num_feature_in,embed_dim,num_heads,num_feature_graph_hidden,
                 num_feature_out,flatten,sr_ratio=1.,processed=False,num_patches=None):
        super(Graph_Encoding_Block_big, self).__init__()
        self.flatten=flatten
        self.processed=processed
        self.patch_embedding=None
        self.relative_pos_flag = False
        if processed:
            self.patch_embedding=nn.Identity()
        else:
            self.patch_embedding = PatchEmbed(img_size=img_size,patch_size=patch_size
                                          , in_chans=num_feature_in, embed_dim=embed_dim)
            self.relative_pos = nn.Parameter(torch.randn(
                num_heads, self.patch_embedding.num_patches,
                self.patch_embedding.num_patches))
            self.relative_pos_flag = True

        if num_patches is not None:
            self.relative_pos_flag = True
            self.relative_pos = nn.Parameter(torch.randn(
                num_heads, num_patches, num_patches))

        self.CMT_encoder1 = Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                                  drop=0.,
                                  attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                                  qk_ratio=1, sr_ratio=sr_ratio)
        self.CMT_encoder2 = Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                                  drop=0.,
                                  attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                                  qk_ratio=1, sr_ratio=sr_ratio)

        self.convnext1 = ConvNexStage(in_features=num_feature_in, out_features=num_feature_in, depth=2)
        self.convnext2 = ConvNexStage(in_features=num_feature_in, out_features=embed_dim, depth=1)

        self.MHSA = Multi_head_cross_attention_with_graph_output(emb_size=embed_dim,
                                                                num_heads=num_heads, add_diag=True)
        self.censnet = GCN_big(nfeat_v=embed_dim, nfeat_e=3 + num_heads + 1, nhid=num_feature_graph_hidden,
                           nfeat_v_out=num_feature_out, dropout=0.1)
        self.transformer_fusion_encoder = Edge_Node_Transformer_fusion_big(emb_dim_node=num_feature_out,
                                                                 emb_dim_edge=2 * (3 + num_heads + 1),
                                                                 num_heads=num_heads)
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

    def forward(self, x, relative_pos=None):
        '''
        :param x: (batch,num_nodes,num_feature)
        :return: (batch,num_nodes,num_feature)
        '''
        conv_feature = self.convnext2(self.convnext1(x)).flatten(2).transpose(1, 2)

        B = x.shape[0]
        if self.processed == False:
            x, (H, W) = self.patch_embedding(x)
        else:
            H=round((x.shape[1])**0.5)
            W=H
        if self.relative_pos_flag == True:
            relative_pos = self.relative_pos

        x = self.CMT_encoder1(x, H, W, relative_pos)
        x = self.CMT_encoder2(x, H, W, relative_pos)

        x, graph = self.MHSA(conv_feature,x)

        x, edges,topk,T = self.censnet(x, graph)

        x = self.transformer_fusion_encoder(x, edges, H, W, relative_pos,topk,T)

        if self.flatten:
            x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        return x

if __name__=="__main__":
    x=torch.randn([1, 46, 56, 56]).cuda()
    model=Graph_Encoding_Block_big(img_size=56,patch_size=4,num_feature_in=46,num_feature_out=92,
                               embed_dim=92,num_heads=4,num_feature_graph_hidden=92,flatten=True).cuda()
    x=model(x)
    print(x.shape)
    '''model=PatchEmbed(img_size=224, patch_size=16, in_chans=3, embed_dim=768).cuda()
    x=torch.randn(2,3,224,224).cuda()
    for i in range(10):
        start = time.time()
        y = model(x)
        print(time.time() - start)'''
