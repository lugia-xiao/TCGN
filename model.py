from CMT_block import *
from gnn_block import Graph_Encoding_Block_big,Graph_Encoding_Block
from transformer_block import Channel_Attention

import torch.nn as nn
import torch

from collections import OrderedDict
from memory_profiler import profile

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

class TCGN(nn.Module):
    def __init__(self, img_size=224, in_chans=3, num_classes=785, embed_dims=[46, 92, 184, 368], stem_channel=16,
                 fc_dim=1280,
                 num_heads=[1, 2, 4, 8], mlp_ratios=[3.6, 3.6, 3.6, 3.6], qkv_bias=True, qk_scale=None,
                 representation_size=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.0, hybrid_backbone=None, norm_layer=None,
                 depths=[2, 2, 10, 2], qk_ratio=1, sr_ratios=[8, 4, 2, 1], dp=0.1):
        super().__init__()
        self.fc_dim = fc_dim
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dims[-1]
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)

        self.stem_conv1 = nn.Conv2d(3, stem_channel, kernel_size=3, stride=2, padding=1, bias=True)
        self.stem_relu1 = nn.GELU()
        self.stem_norm1 = nn.BatchNorm2d(stem_channel, eps=1e-5)

        self.stem_conv2 = nn.Conv2d(stem_channel, stem_channel, kernel_size=3, stride=1, padding=1, bias=True)
        self.stem_relu2 = nn.GELU()
        self.stem_norm2 = nn.BatchNorm2d(stem_channel, eps=1e-5)

        self.stem_conv3 = nn.Conv2d(stem_channel, stem_channel, kernel_size=3, stride=1, padding=1, bias=True)
        self.stem_relu3 = nn.GELU()
        self.stem_norm3 = nn.BatchNorm2d(stem_channel, eps=1e-5)

        self.patch_embed_a = PatchEmbed(
            img_size=img_size // 2, patch_size=2, in_chans=stem_channel, embed_dim=embed_dims[0])
        self.patch_embed_b = PatchEmbed(
            img_size=img_size // 4, patch_size=2, in_chans=embed_dims[0], embed_dim=embed_dims[1])
        self.patch_embed_c = PatchEmbed(
            img_size=img_size // 8, patch_size=2, in_chans=embed_dims[1], embed_dim=embed_dims[2])
        self.patch_embed_d = PatchEmbed(
            img_size=img_size // 16, patch_size=2, in_chans=embed_dims[2], embed_dim=embed_dims[3])

        self.relative_pos_a = nn.Parameter(torch.randn(
            num_heads[0], self.patch_embed_a.num_patches,
            self.patch_embed_a.num_patches // sr_ratios[0] // sr_ratios[0]))
        self.relative_pos_b = nn.Parameter(torch.randn(
            num_heads[1], self.patch_embed_b.num_patches,
            self.patch_embed_b.num_patches // sr_ratios[1] // sr_ratios[1]))
        self.relative_pos_c = nn.Parameter(torch.randn(
            num_heads[2], self.patch_embed_c.num_patches,
            self.patch_embed_c.num_patches // sr_ratios[2] // sr_ratios[2]))
        self.relative_pos_d = nn.Parameter(torch.randn(
            num_heads[3], self.patch_embed_d.num_patches,
            self.patch_embed_d.num_patches // sr_ratios[3] // sr_ratios[3]))

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0
        self.blocks_a = nn.ModuleList([
            Block(
                dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias,
                qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i],
                norm_layer=norm_layer, qk_ratio=qk_ratio, sr_ratio=sr_ratios[0])
            for i in range(depths[0])])
        cur += depths[0]
        self.blocks_b = nn.ModuleList([
            Block(
                dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias,
                qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i],
                norm_layer=norm_layer, qk_ratio=qk_ratio, sr_ratio=sr_ratios[1])
            for i in range(depths[1])])
        cur += depths[1]
        self.blocks_c = nn.ModuleList([
            Block(
                dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias,
                qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i],
                norm_layer=norm_layer, qk_ratio=qk_ratio, sr_ratio=sr_ratios[2])
            for i in range(depths[2])])
        cur += depths[2]
        self.blocks_d = nn.ModuleList([
            Block(
                dim=embed_dims[3], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3], qkv_bias=qkv_bias,
                qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i],
                norm_layer=norm_layer, qk_ratio=qk_ratio, sr_ratio=sr_ratios[3])
            for i in range(depths[3])])

        # Representation layer
        if representation_size:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(self.embed_dim, representation_size)),
                ('act', nn.Tanh())#('act', nn.GELU())
            ]))
        else:
            self.pre_logits = nn.Identity()

        # Graph learning
        self.gnn_block0 = Graph_Encoding_Block(img_size=56, patch_size=4, num_feature_in=embed_dims[0],
                                               embed_dim=embed_dims[0]*2, num_feature_graph_hidden=embed_dims[0]*2
                                               , num_feature_out=48, flatten=False, num_heads=2)
        self.channel_attention0 = Channel_Attention(num_nodes=196)

        self.gnn_block1 = Graph_Encoding_Block(img_size=28, patch_size=2, num_feature_in=embed_dims[1],
                                               embed_dim=embed_dims[1]*2, num_feature_graph_hidden=embed_dims[1]*2
                                               , num_feature_out=48, flatten=False, num_heads=2)
        self.channel_attention1 = Channel_Attention(num_nodes=196)

        # Classifier head
        self._fc = nn.Conv2d(embed_dims[-1], fc_dim, kernel_size=1)
        self._bn = nn.BatchNorm2d(fc_dim, eps=1e-5)
        self._swish = MemoryEfficientSwish()
        self._avg_pooling = nn.AdaptiveAvgPool2d(1)
        self._drop = nn.Dropout(dp)
        self.head = nn.Sequential(nn.Linear(fc_dim, fc_dim*4, bias=True),nn.GELU(),nn.Linear(fc_dim*4,num_classes, bias=True))
        # nn.Linear(fc_dim, num_classes, bias=True) if num_classes > 0 else nn.Identity()
        # nn.Sequential(nn.Linear(fc_dim, fc_dim*4, bias=True),nn.Linear(fc_dim*4,num_classes))
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

    def update_temperature(self):
        for m in self.modules():
            if isinstance(m, Attention):
                m.update_temperature()

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.fc_dim, num_classes, bias=True) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        B = x.shape[0]
        x = self.stem_conv1(x)
        x = self.stem_relu1(x)
        x = self.stem_norm1(x)

        x = self.stem_conv2(x)
        x = self.stem_relu2(x)
        x = self.stem_norm2(x)

        x = self.stem_conv3(x)
        x = self.stem_relu3(x)
        x = self.stem_norm3(x)

        x, (H, W) = self.patch_embed_a(x)
        for i, blk in enumerate(self.blocks_a):
            x = blk(x, H, W, self.relative_pos_a)

        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        graph_feature0 = self.gnn_block0(x,self.relative_pos_a)

        x, (H, W) = self.patch_embed_b(x)
        for i, blk in enumerate(self.blocks_b):
            x = blk(x, H, W, self.relative_pos_b)

        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        graph_feature1 = self.gnn_block1(x, self.relative_pos_c)

        x, (H, W) = self.patch_embed_c(x)

        x = self.channel_attention0(x, graph_feature0)


        for i, blk in enumerate(self.blocks_c):
            x = blk(x, H, W, self.relative_pos_c)

        x=self.channel_attention1(x,graph_feature1)

        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        x, (H, W) = self.patch_embed_d(x)

        for i, blk in enumerate(self.blocks_d):
            x = blk(x, H, W, self.relative_pos_d)

        #print("+", end="")

        B, N, C = x.shape
        x = self._fc(x.permute(0, 2, 1).reshape(B, C, H, W))
        x = self._bn(x)
        x = self._swish(x)
        x = self._avg_pooling(x).flatten(start_dim=1)
        x = self._drop(x)
        x = self.pre_logits(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x

if __name__=="__main__":
    import psutil
    import os
    torch.manual_seed(6)
    x=torch.randn([2, 3, 224, 224]).cuda()
    x_test = torch.randn([4, 3, 224, 224]).cuda()
    y = torch.randn([2, 785]).cuda()

    xs_kwargs = dict(
        qkv_bias=True, embed_dims=[52, 104, 208, 416], stem_channel=16, num_heads=[1, 2, 4, 8],
        depths=[3, 3, 12, 3], mlp_ratios=[3.77, 3.77, 3.77, 3.77], qk_ratio=1, sr_ratios=[8, 4, 2, 1])
    model_size = "xs"
    if model_size=="xs":
        my_model = TCGN(**xs_kwargs)
        my_model = my_model.cuda()
        my_model.load_state_dict(torch.load('./pretrained/cmt_xs.pth'), strict=False)

    print(my_model(x_test).shape)
    gnn_params = list(map(id, my_model.gnn_block0.parameters()))
    base_params = filter(lambda p: id(p) not in gnn_params, my_model.parameters())
    optimizer = torch.optim.Adam([{'params': base_params},
                                  {'params': my_model.gnn_block0.parameters(), 'lr': 2.5e-4}],
                                 lr=1e-5, betas=(0.9, 0.999), eps=1e-08, weight_decay=0,amsgrad=False)
    loss_function = torch.nn.MSELoss().cuda()

    import time
    with torch.autograd.set_detect_anomaly(True):
        for i in range(100):
            print(u'memory:%.4f GB' % (psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024))
            start = time.time()
            out = my_model(x)
            end = time.time()
            print("farward", end - start)
            start = time.time()
            loss = loss_function(out, y)
            loss.backward()
            optimizer.step()
            end = time.time()
            print("backward", end - start)
