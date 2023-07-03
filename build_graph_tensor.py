import torch
import numpy as np


class Global_Adj():
    def __init__(self):
        self.adj_v_196 = None
        self.adj_v_49 = None
        self.batch_size=0

    def initialize(self,batch_size):
        self.init_global_adj(batch_size)
        self.batch_size=batch_size

    @torch.no_grad()
    def init_global_adj(self,batch_size):
        self.adj_v_196 = np.zeros((196, 196))
        num_nodes = int(self.adj_v_196.shape[0])
        height = int(num_nodes ** 0.5)
        for i in range(num_nodes):
            position1 = [i // height, i % height]
            for j in range(num_nodes):
                position2 = [j // height, j % height]
                distancei = ((position1[0] - position2[0]) ** 2 + (position1[1] - position2[1]) ** 2) ** 0.5
                distancei = 1 / (distancei / 3 + 1)
                self.adj_v_196[i][j] = distancei
                self.adj_v_196[j][i] = distancei
        self.adj_v_196 = torch.from_numpy(self.adj_v_196)
        self.adj_v_196 = self.adj_v_196.unsqueeze(dim=0).repeat(batch_size, 1, 1).to(torch.float32)

        self.adj_v_49 = np.zeros((49, 49))
        num_nodes = int(self.adj_v_49.shape[0])
        height = int(num_nodes ** 0.5)
        for i in range(num_nodes):
            position1 = [i // height, i % height]
            for j in range(num_nodes):
                position2 = [j // height, j % height]
                distancei = ((position1[0] - position2[0]) ** 2 + (position1[1] - position2[1]) ** 2) ** 0.5
                distancei = 1 / (distancei / 3 + 1)
                self.adj_v_49[i][j] = distancei
                self.adj_v_49[j][i] = distancei
        self.adj_v_49 = torch.from_numpy(self.adj_v_49)
        self.adj_v_49 = self.adj_v_49.unsqueeze(dim=0).repeat(batch_size, 1, 1).to(torch.float32)

global_adj=Global_Adj()


def build_edge_feature_tensor(x_input,graph_input,topk):
    '''
    :param x_input: (batch_size,num_nodes,embed_dim)
    :param graph_input: (batch_size,num_heads,num_nodes,num_nodes)
    :param topk: int, each node has k neighbors
    :return: (batch_size,num_nodes,num_nodes),(batch_size,num_edges,num_heads+1+3)
    '''
    x = x_input.cpu()
    graph=graph_input.cpu()
    batch_size = x.shape[0]
    num_heads=graph.shape[1]
    num_nodes=x.shape[1]

    if global_adj.batch_size!=batch_size:
        global_adj.initialize(batch_size)

    adj_v=None
    if num_nodes==196:
        adj_v=global_adj.adj_v_196.clone()
    elif num_nodes==49:
        adj_v=global_adj.adj_v_49.clone()
    else:
        print("wrong input")
    scale = x.shape[2]//2

    graph0 = torch.sum(graph, dim=1)/num_heads #(batch_size,num_nodes,num_nodes)


    # to avoid if the top k_th number occurs twice
    # maybe this is wrong, try adding: with torch.no_grad():
    graph0=graph0.view(batch_size,-1) # (batch_size,num_nodes,num_nodes) -> (batch_size,num_nodes*num_nodes)

    #print(torch.sum(graph0 == torch.topk(graph0, dim=-1, k=topk * num_nodes)[0][:, topk * num_nodes - 1].unsqueeze(-1)))
    #print(torch.topk(graph0, dim=-1, k=topk*num_nodes)[0][:, topk*num_nodes - 1].unsqueeze(-1).shape)
    num_selection=topk*num_nodes
    print_flag=False
    with torch.no_grad():
        while torch.sum(graph0 == torch.topk(graph0, dim=-1, k=num_selection)[0][:, num_selection - 1].unsqueeze(-1)) != \
                batch_size:
            print_flag=True
            num_selection+=1
            topk=num_selection/num_nodes
            if topk==graph.shape[2]:
                break

    # build adjacent matrix
    graph0[graph0 < torch.topk(graph0, dim=-1, k=num_selection)[0][:, num_selection - 1].unsqueeze(-1)] = 0

    adjacent=torch.ones_like(graph0)
    adjacent[graph0 < torch.topk(graph0, dim=-1, k=num_selection)[0][:, num_selection - 1].unsqueeze(-1)] = 0

    graph0=graph0.view(batch_size,num_nodes,num_nodes)
    adjacent=adjacent.view(batch_size,num_nodes,num_nodes)
    '''if x.shape[1]==49:
        print(adjacent[:, 0, :])'''

    # build L2 distance between different nodes
    x_inner = -2 * torch.matmul(x, x.transpose(2, 1))
    x_square = torch.sum(torch.mul(x, x), dim=-1, keepdim=True)
    L2_distance = torch.sqrt(((x_square + x_inner + x_square.transpose(2, 1))+scale)/scale)
    L2_distance = 1/L2_distance

    L2_distance[graph0==0] = 0

    # build cos distance between different nodes
    x_length = torch.sqrt(torch.sum(x * x, dim=-1)).unsqueeze(dim=-1).repeat(1, 1, x.shape[-1])
    x_normalized = x / x_length
    cos_distance = x_normalized @ x_normalized.transpose(-1, -2)
    cos_distance = (cos_distance + 1)/2
    cos_distance[graph0==0] = 0

    adj_v[graph0==0] = 0

    att_feature = []
    for i in range(num_heads):
        graphi = graph[:, i, :, :]
        graphi[graph0==0] = 0
        graphi = graphi[torch.nonzero(L2_distance, as_tuple=True)].view(batch_size, -1)
        graphi = graphi.unsqueeze(dim=-1)
        att_feature.append(graphi)

    #print(adjacent.shape)

    graph0 = graph0[torch.nonzero(L2_distance, as_tuple=True)].view(batch_size, -1)
    graph0 = graph0.unsqueeze(dim=-1)
    #print(graph0.shape)
    att_feature.append(graph0)
    #print(graph0.shape)

    adj_v = adj_v[torch.nonzero(adj_v, as_tuple=True)].view(batch_size, -1)
    cos_distance = cos_distance[torch.nonzero(cos_distance, as_tuple=True)].view(batch_size, -1)
    L2_distance = L2_distance[torch.nonzero(L2_distance, as_tuple=True)].view(batch_size, -1)

    adj_v = adj_v.unsqueeze(dim=-1)
    cos_distance = cos_distance.unsqueeze(dim=-1)
    L2_distance = L2_distance.unsqueeze(dim=-1)
    '''adj_v=adj_v.cuda()
    cos_distance=cos_distance.cuda()
    L2_distance=L2_distance.cuda()'''

    edge_features = att_feature + [adj_v, L2_distance, cos_distance]
    #print(adjacent.shape,[xi.shape for xi in edge_features])
    return adjacent,torch.concat(edge_features,dim=-1),topk#normalize_tensor(adj_v)

def normalize_tensor(x,plus_one=False):
    '''
    :param x: (batch_size,num_nodes,num_nodes)
    :param plus_one: whether to add self-loop
    :return:
    '''
    with torch.no_grad():
        row_sum = None
        if plus_one:
            rowsum = torch.sum(x + torch.diag_embed(torch.ones(x.shape[0], x.shape[1])), dim=-1)
        else:
            rowsum = torch.sum(x, dim=-1)
        r_inv_sqrt = rowsum ** (-0.5)
        r_inv_sqrt = torch.where(torch.isnan(r_inv_sqrt), torch.full_like(r_inv_sqrt, 0), r_inv_sqrt)
        r_inv_sqrt = torch.diag_embed(r_inv_sqrt)
    return r_inv_sqrt@x@r_inv_sqrt.transpose(-2,-1)

if __name__=="__main__":
    np.random.seed(43)
    a=np.random.randn(16,196,256)
    x=torch.Tensor(a)
    graph=torch.Tensor(np.random.randn(16,4,196,196))

    import time

    start=time.time()
    #print(out1[0,:,:])
    print("numpy",time.time()-start)
    start = time.time()
    distance_adj1,adj1 = build_edge_feature_tensor(x,graph,9)
    print(distance_adj1.shape,adj1.shape)
    print("tensor", time.time() - start)
