import torch
from torch.nn import Parameter
import torch.nn as nn
import torch.nn.functional as F

import math
import numpy as np
import scipy.sparse as sp

from build_graph_tensor import build_edge_feature_tensor,normalize_tensor
use_gpu = torch.cuda.is_available() # gpu加速


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features_v, out_features_v, in_features_e, out_features_e, bias=True, node_layer=True):
        super(GraphConvolution, self).__init__()
        self.in_features_e = in_features_e
        self.out_features_e = out_features_e
        self.in_features_v = in_features_v
        self.out_features_v = out_features_v
        if node_layer:
            self.node_layer = True
            self.weight = Parameter(torch.FloatTensor(in_features_v, out_features_v))
            self.p = Parameter(torch.from_numpy(np.random.normal(size=(1, in_features_e))).float())
            if bias:
                self.bias = Parameter(torch.FloatTensor(out_features_v))
            else:
                self.register_parameter('bias', None)
        else:
            self.node_layer = False
            self.weight = Parameter(torch.FloatTensor(in_features_e, out_features_e))
            self.p = Parameter(torch.from_numpy(np.random.normal(size=(1, in_features_v))).float())
            if bias:
                self.bias = Parameter(torch.FloatTensor(out_features_e))
            else:
                self.register_parameter('bias', None)
        self.activation=nn.GELU()
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward_single_batch(self, H_v, H_e, adj_e, adj_v, T):
        if self.node_layer:
            multiplier1 = torch.mm(T, torch.diag((H_e @ self.p.t()).t()[0])) @ T.to_dense().t() # to dense
            mask1 = torch.eye(multiplier1.shape[0]).cuda()
            M1 = mask1 * torch.ones(multiplier1.shape[0]).cuda() + (1. - mask1)*multiplier1
            adjusted_A = torch.mul(M1, adj_v)
            output = torch.mm(adjusted_A, torch.mm(H_v, self.weight))
            ret=self.activation(output)
            if self.bias is not None:
                ret = output + self.bias
            return ret, H_e

        else:
            multiplier2 = torch.spmm(T.t(), torch.diag((H_v @ self.p.t()).t()[0])) @ T.to_dense()
            mask2 = torch.eye(multiplier2.shape[0]).cuda()
            M3 = mask2 * torch.ones(multiplier2.shape[0]).cuda() + (1. - mask2) * multiplier2
            adjusted_A = torch.mul(M3, adj_e)
            normalized_adjusted_A = adjusted_A / adjusted_A.max(0, keepdim=True)[0]
            output = torch.mm(normalized_adjusted_A, torch.mm(H_e, self.weight))
            ret=self.activation(output)
            if self.bias is not None:
                ret = output + self.bias
            return H_v, ret

    def forward(self, H_v, H_e, adj_e, adj_v, T):
        if self.node_layer:
            ret=torch.stack([self.forward_single_batch(H_v[i], H_e[i], adj_e[i], adj_v[i], T[i])[0] for i in range(H_v.shape[0])])
            return ret,H_e
        else:
            ret=torch.stack([self.forward_single_batch(H_v[i], H_e[i], adj_e[i], adj_v[i], T[i])[1] for i in range(H_v.shape[0])])
            return H_v,ret


class GCN(nn.Module):
    def __init__(self,nfeat_v, nfeat_e, nhid, nfeat_v_out,dropout,topk=4):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(nfeat_v, nhid, nfeat_e, nfeat_e, node_layer=True)
        self.gc2 = GraphConvolution(nhid, nhid, nfeat_e, nfeat_e*2, node_layer=False)
        self.gc3 = GraphConvolution(nhid, nfeat_v_out, nfeat_e*2, nfeat_e*2, node_layer=True)
        self.dropout = dropout
        self.topk=topk

    def forward(self, X,graph):
        Z, adj_e, adj_v, T,topk=process_feature(X,graph,self.topk)
        if use_gpu:
            Z=Z.cuda()
            adj_e=adj_e.cuda()
            adj_v=adj_v.cuda()
            T=T.cuda()
        # print x
        gc1 = self.gc1(X, Z, adj_e, adj_v, T)
        X, Z = F.relu(gc1[0]), F.relu(gc1[1])

        X = F.dropout(X, self.dropout, training=self.training)
        Z = F.dropout(Z, self.dropout, training=self.training)

        gc2 = self.gc2(X, Z, adj_e, adj_v, T)
        X, Z = F.relu(gc2[0]), F.relu(gc2[1])

        X = F.dropout(X, self.dropout, training=self.training)
        Z = F.dropout(Z, self.dropout, training=self.training)

        X, Z = self.gc3(X, Z, adj_e, adj_v, T)
        return X,Z,topk,T

class GCN_big(nn.Module):
    def __init__(self,nfeat_v, nfeat_e, nhid, nfeat_v_out, dropout):
        super(GCN_big, self).__init__()
        self.gc1 = GraphConvolution(nfeat_v, nhid, nfeat_e, nfeat_e, node_layer=True)
        self.gc2 = GraphConvolution(nhid, nhid, nfeat_e, nfeat_e*2, node_layer=False)
        self.gc3 = GraphConvolution(nhid, nhid, nfeat_e*2, nfeat_e*2, node_layer=True)
        self.gc4=GraphConvolution(nhid, nhid, nfeat_e*2, nfeat_e*2, node_layer=False)
        self.gc5 = GraphConvolution(nhid, nfeat_v_out, nfeat_e * 2, nfeat_e*2 , node_layer=True)
        self.dropout = dropout

    def forward(self, X,graph):
        Z, adj_e, adj_v, T,topk=process_feature(X,graph)
        if use_gpu:
            Z=Z.cuda()
            adj_e=adj_e.cuda()
            adj_v=adj_v.cuda()
            T=T.cuda()
        # print x
        gc1 = self.gc1(X, Z, adj_e, adj_v, T)
        X, Z = F.relu(gc1[0]), F.relu(gc1[1])

        X = F.dropout(X, self.dropout, training=self.training)
        Z = F.dropout(Z, self.dropout, training=self.training)

        gc2 = self.gc2(X, Z, adj_e, adj_v, T)
        X, Z = F.relu(gc2[0]), F.relu(gc2[1])

        X = F.dropout(X, self.dropout, training=self.training)
        Z = F.dropout(Z, self.dropout, training=self.training)

        X, Z = self.gc3(X, Z, adj_e, adj_v, T)

        X = F.dropout(X, self.dropout, training=self.training)
        Z = F.dropout(Z, self.dropout, training=self.training)

        X, Z = self.gc4(X, Z, adj_e, adj_v, T)

        X = F.dropout(X, self.dropout, training=self.training)
        Z = F.dropout(Z, self.dropout, training=self.training)

        X, Z = self.gc5(X, Z, adj_e, adj_v, T)
        return X,Z,topk,T

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    with torch.no_grad():
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse.FloatTensor(indices, values, shape)

def create_transition_matrix_new(vertex_adj_all_input):
    '''create N_v * N_e transition matrix'''
    with torch.no_grad():
        batch_size = vertex_adj_all_input.shape[0]
        vertex_adj_all = vertex_adj_all_input.cpu().numpy()
        transition_matrix = None
        for batch in range(batch_size):
            vertex_adj = vertex_adj_all[batch]
            # np.fill_diagonal(vertex_adj, 0)
            edge_index = np.nonzero(vertex_adj)
            num_edge = int(len(edge_index[0]))
            edge_name = [x for x in zip(edge_index[0], edge_index[1])]

            row_index = [i for sub in edge_name for i in sub]
            col_index = np.repeat([i for i in range(num_edge)], 2)

            data = np.ones(num_edge * 2)
            T = sp.csr_matrix((data, (row_index, col_index)),
                              shape=(vertex_adj.shape[0], num_edge))
            T = sparse_mx_to_torch_sparse_tensor(T).unsqueeze(dim=0)
            if transition_matrix == None:
                transition_matrix = T
            else:
                transition_matrix = torch.concat((T, transition_matrix), dim=0)

        return transition_matrix

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1)).astype("float")
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def create_edge_adj_tensor(vertex_adj_all_input,flag_normalize=True):
    '''
    :param vertex_adj_all_input: adjacent matrix (Batch,Num_nodes,Channel)
    :param flag_normalize: Whether to normalize the edge adj matrix
    :return: edge adjacent matrix
    '''
    vertex_adj_all=vertex_adj_all_input.cpu()
    with torch.no_grad():
        batch_size = vertex_adj_all.shape[0]
        edge_index = torch.nonzero(vertex_adj_all).view(batch_size, -1, 3)
        num_edge_per_batch = edge_index.shape[1]
        edge_adj = torch.zeros((batch_size, num_edge_per_batch, num_edge_per_batch))

        edge_index_row = edge_index[:, :, 1].unsqueeze(dim=-1) + 1
        edge_index_col = edge_index[:, :, 2].unsqueeze(dim=-1) + 1

        connection_row_row = edge_index_row @ edge_index_row.transpose(-2, -1)
        edge_adj[connection_row_row == edge_index_row ** 2] = 1

        connection_row_col = edge_index_row @ edge_index_col.transpose(-2, -1)
        edge_adj[connection_row_col.transpose(-2, -1) == edge_index_col ** 2] = 1

        connection_col_row = edge_index_col @ edge_index_row.transpose(-2, -1)
        edge_adj[connection_col_row.transpose(-2, -1) == edge_index_row ** 2] = 1

        connection_col_col = edge_index_col @ edge_index_col.transpose(-2, -1)
        edge_adj[connection_col_col == edge_index_col ** 2] = 1
        if flag_normalize:
            edge_adj = normalize_tensor(edge_adj, plus_one=True)
        return edge_adj

'''number=0'''
def process_feature(x,graph,topk=2):
    distance_adj, edge_features,topk = build_edge_feature_tensor(x,graph,topk)
    '''global number
    number+=1
    np.save("connection"+str(number)+".npy",distance_adj.cpu().numpy())'''
    #print(">", end="")
    transition_matrix = create_transition_matrix_new(distance_adj)
    #print(">", end="")
    edge_adj = create_edge_adj_tensor(distance_adj)
    #print("> ", end="")
    return edge_features,edge_adj,normalize_tensor(distance_adj.to(torch.float32)),transition_matrix,topk

if __name__ == '__main__':
    x = torch.randn((12, 196, 256)).cuda()
    graph=torch.randn(12,4,196,196).cuda()
    GNN=GCN(nfeat_v=256,nfeat_v_out=32, nfeat_e=8, nhid=64, dropout=0.1).cuda()
    import time
    start=time.time()
    x=GNN(x,graph)
    print(time.time()-start)
    print(x.shape)
    print(x[0,0,:])
