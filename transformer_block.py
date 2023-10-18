import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

from CMT_block import Block

class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size: int = 92, num_heads: int = 4, dropout: float = 0,add_diag:bool=False):
        super().__init__()
        self.add_diag=add_diag
        self.emb_size = emb_size
        self.num_heads = num_heads
        # 将查询、键和值融合到一个矩阵中
        self.qkv = nn.Linear(emb_size, emb_size * 3)
        self.att_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor,mask: torch.Tensor = None):
        # 分割num_heads中的键、查询和值
        qkv = rearrange(self.qkv(x), "b n (h d qkv) -> (qkv) b h n d", h=self.num_heads, qkv=3)
        queries, keys, values = qkv[0], qkv[1], qkv[2]
        # 最后一个轴上求和
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)  # batch, num_heads, query_len, key_len
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)

        scaling = self.emb_size ** (1 / 2)
        if self.add_diag==True:
            added=torch.diag_embed(torch.ones(
                [energy.shape[0], energy.shape[1], energy.shape[2]])).cuda()
            # energy BxHxNxN
            energy = F.softmax((energy + added) / scaling, dim=-1)
        else:
            energy=F.softmax(energy / scaling, dim=-1)
        att = self.att_drop(energy)
        # 在第三个轴上求和
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        return out


class MultiHeadCrossAttention(nn.Module):
    def __init__(self,embed_k,embed_q,num_heads,dropout=0.1):
        super().__init__()
        self.emb_size = embed_k
        self.num_heads = num_heads
        # 将查询、键和值融合到一个矩阵中
        self.k = nn.Linear(embed_k, embed_k)
        self.q=nn.Linear(embed_q, embed_k)
        self.att_drop = nn.Dropout(dropout)
        self.v = nn.Linear(embed_k, embed_k)

    def forward(self, key: torch.Tensor, query: torch.Tensor):
        # 分割num_heads中的键、查询和值
        queries = rearrange(self.q(query), "b n (h d) -> b h n d", h=self.num_heads)
        keys = rearrange(self.k(key), "b n (h d) -> b h n d", h=self.num_heads)
        values=rearrange(self.v(key), "b n (h d) -> b h n d", h=self.num_heads)
        # 最后一个轴上求和
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)  # batch, num_heads, query_len, key_len

        scaling = self.emb_size ** (1 / 2)
        energy=F.softmax(energy / scaling, dim=-1)
        att = self.att_drop(energy)
        # 在第三个轴上求和
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        return out

class Multi_head_self_attention_with_graph_output(nn.Module):
    def __init__(self,emb_size=768,num_heads=8,add_diag=False):
        super(Multi_head_self_attention_with_graph_output, self).__init__()
        self.add_diag=add_diag
        self.emb_size = emb_size
        self.num_heads = num_heads
        # 将查询、键和值融合到一个矩阵中
        self.qkv = nn.Linear(emb_size, emb_size * 2)
        self.value_transform=nn.Linear(emb_size,emb_size)

    def forward(self, x: torch.Tensor, relative_pos: torch.Tensor, mask: torch.Tensor = None):
        # 分割num_heads中的键、查询和值
        qkv = rearrange(self.qkv(x), "b n (h d qk) -> (qk) b h n d", h=self.num_heads, qk=2)
        queries, keys = qkv[0], qkv[1]
        # 最后一个轴上求和
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)  # batch, num_heads, query_len, key_len
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)

        scaling = self.emb_size ** (1 / 2)
        energy=energy / scaling
        if self.add_diag:
            energy=energy+torch.diag_embed(torch.ones([energy.shape[0],energy.shape[1],energy.shape[2]])).cuda()
        energy = F.softmax(energy + relative_pos, dim=-1)
        values=self.value_transform(x)
        return values,energy

class Multi_head_cross_attention_with_graph_output(nn.Module):
    def __init__(self,emb_size=768,num_heads=8,add_diag=False):
        super(Multi_head_cross_attention_with_graph_output, self).__init__()
        self.add_diag=add_diag
        self.emb_size = emb_size
        self.num_heads = num_heads
        # 将查询、键和值融合到一个矩阵中
        self.k=nn.Linear(emb_size, emb_size)
        self.q = nn.Linear(emb_size, emb_size)
        self.value_transform=nn.Linear(emb_size,emb_size)

    def forward(self, key: torch.Tensor, query: torch.Tensor):
        # 分割num_heads中的键、查询和值
        queries = rearrange(self.q(query), "b n (h d) -> b h n d", h=self.num_heads)

        keys = rearrange(self.k(key), "b n (h d) -> b h n d", h=self.num_heads)
        # 最后一个轴上求和
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)  # batch, num_heads, query_len, key_len

        scaling = self.emb_size ** (1 / 2)
        energy=energy / scaling
        if self.add_diag:
            energy=energy+torch.diag_embed(torch.ones([energy.shape[0],energy.shape[1],energy.shape[2]])).cuda()
        energy = F.softmax(energy, dim=-1)
        values=self.value_transform(key)
        return values,energy

class Cross_Attention_Transformer(nn.Module):
    def __init__(self,emb_sizek,emb_sizeq, drop_p=0.1, forward_expansion=4, num_heads=4):
        super(Cross_Attention_Transformer, self).__init__()
        emb_size=emb_sizek
        self.layernormk = nn.LayerNorm(emb_sizek)
        self.layernormq = nn.LayerNorm(emb_sizeq)
        self.Attention = MultiHeadCrossAttention(emb_sizek,emb_sizeq, num_heads=num_heads, dropout=drop_p)
        self.dropout1 = nn.Dropout(drop_p)
        self.layernorm2 = nn.LayerNorm(emb_size)
        self.FFN = nn.Sequential(
            nn.Linear(emb_size, forward_expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(forward_expansion * emb_size, emb_size),
        )
        self.dropout2 = nn.Dropout(drop_p)

    def forward(self,key,query):
        #print(key.shape,query.shape)
        key = key + self.dropout1(self.Attention(self.layernormk(key),self.layernormq(query)))
        key = key + self.dropout2(self.FFN(self.layernorm2(key)))
        return key


class Edge_Node_Transformer_fusion(nn.Module):
    def __init__(self,emb_dim_node,emb_dim_edge,num_heads=4):
        super(Edge_Node_Transformer_fusion, self).__init__()
        self.edge_transformer=TransformerEncoderBlock(emb_size=emb_dim_edge,num_heads=num_heads)
        self.node_transformer=Block(dim=emb_dim_node, num_heads=num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0.,
                                attn_drop=0.,drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                                qk_ratio=1, sr_ratio=1)
        self.channel_attention_196 = TransformerEncoderBlock(emb_size=196, num_heads=4)
        self.linear_fusion = nn.Linear(emb_dim_edge + emb_dim_node, emb_dim_node)

    def forward(self,node_feature,edge_feature,H, W, relative_pos,topk,T):
        edge_feature=(edge_feature.transpose(-1,-2)@T.to_dense().transpose(-2,-1)).transpose(-2,-1)/topk

        node_feature=self.node_transformer(node_feature,H, W, relative_pos)
        edge_feature=self.edge_transformer(edge_feature)

        features = torch.concat((node_feature, edge_feature), dim=-1)  # B,N,C

        features = self.channel_attention_196(features.transpose(-1, -2)).transpose(-1, -2).contiguous()
        features = self.linear_fusion(features)
        return features

class Edge_Node_Transformer_fusion_big(nn.Module):
    def __init__(self,emb_dim_node,emb_dim_edge,num_heads=4):
        super().__init__()
        self.edge_transformer1=TransformerEncoderBlock(emb_size=emb_dim_edge,num_heads=num_heads)
        self.node_transformer1=Block(dim=emb_dim_node, num_heads=num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0.,
                                attn_drop=0.,drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                                qk_ratio=1, sr_ratio=1)
        self.edge_transformer2 = TransformerEncoderBlock(emb_size=emb_dim_edge, num_heads=num_heads)
        self.node_transformer2 = Block(dim=emb_dim_node, num_heads=num_heads, mlp_ratio=4., qkv_bias=False,
                                       qk_scale=None, drop=0.,
                                       attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                                       qk_ratio=1, sr_ratio=1)
        self.channel_attention_196 = nn.Sequential(TransformerEncoderBlock(emb_size=196, num_heads=4),
                                                   TransformerEncoderBlock(emb_size=196, num_heads=4))
        self.linear_fusion = nn.Linear(emb_dim_edge + emb_dim_node, emb_dim_node)

    def forward(self,node_feature,edge_feature,H, W, relative_pos,topk,T):
        edge_feature=(edge_feature.transpose(-1,-2)@T.to_dense().transpose(-2,-1)).transpose(-2,-1)/topk

        node_feature=self.node_transformer1(node_feature,H, W, relative_pos)
        node_feature = self.node_transformer2(node_feature, H, W, relative_pos)

        edge_feature=self.edge_transformer1(edge_feature)
        edge_feature = self.edge_transformer2(edge_feature)

        features = torch.concat((node_feature, edge_feature), dim=-1)  # B,N,C

        features = self.channel_attention_196(features.transpose(-1, -2)).transpose(-1, -2).contiguous()
        features = self.linear_fusion(features)
        return features

class TransformerEncoderBlock(nn.Module):
    def __init__(self, emb_size=768, drop_p=0.1, forward_expansion=4, num_heads=8
                 ,attention_add_diag=False):
        super(TransformerEncoderBlock, self).__init__()
        self.layernorm1 = nn.LayerNorm(emb_size)
        self.Attention = MultiHeadAttention(emb_size=emb_size, num_heads=num_heads, dropout=drop_p
                                            ,add_diag=attention_add_diag)
        self.dropout1 = nn.Dropout(drop_p)
        self.layernorm2 = nn.LayerNorm(emb_size)
        self.FFN = nn.Sequential(
            nn.Linear(emb_size, forward_expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(forward_expansion * emb_size, emb_size),
        )
        self.dropout2 = nn.Dropout(drop_p)
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

    def forward(self, x):
        x = x + self.dropout1(self.Attention(self.layernorm1(x)))
        x = x + self.dropout2(self.FFN(self.layernorm2(x)))
        return x

'''class Cross_Attention(nn.Module):
    def __init__(self,embed_k,embed_q,num_heads,drop_p=0.1,forward_expansion=4):
        super(Cross_Attention, self).__init__()
        self.layernorm_k = nn.LayerNorm(embed_k)
        self.layernorm_q = nn.LayerNorm(embed_q)
        self.MHCA=MultiHeadCrossAttention(embed_k,embed_q,num_heads)
        self.dropout1 = nn.Dropout(drop_p)
        self.layernorm2 = nn.LayerNorm(embed_k)
        self.FFN = nn.Sequential(
            nn.Linear(embed_k, forward_expansion * embed_k),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(forward_expansion * embed_k, embed_k),
        )
        self.dropout2 = nn.Dropout(drop_p)

    def forward(self,x,q):
        x = x + self.dropout1(self.MHCA(self.layernorm_k(x),self.layernorm_q(q)))
        x = x + self.dropout2(self.FFN(self.layernorm2(x)))
        return x'''

class Channel_Attention(nn.Module):
    def __init__(self,num_nodes):
        super(Channel_Attention, self).__init__()
        self.num_heads=1 if num_nodes==49 else 4
        self.transformer=TransformerEncoderBlock(emb_size=num_nodes,num_heads=self.num_heads,attention_add_diag=True)

    def forward(self,x1,x2):
        #print(">",end="")
        num_channel_out=x1.shape[2]
        x=torch.concat((x1,x2),dim=-1) # B,N,C1+C2 -> B,C1+C2,N
        x=self.transformer(x.transpose(-1,-2))
        x=x.transpose(-1,-2).contiguous() # B,C1+C2,N -> B,N,C1+C2
        x=x[:,:,:num_channel_out]
        #print("> ", end="")
        return x

if __name__=="__main__":
    pass
