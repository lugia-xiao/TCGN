import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import ImageFile, Image
import torch
import os
import numpy as np
from collections import defaultdict as dfd
import scprep as scp
import pandas as pd
from torch.utils.data import DataLoader
from graph_construction import calcADJ

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None


class ViT_HER2ST(torch.utils.data.Dataset):
    """Some Information about HER2ST"""

    def __init__(self, train=True, fold=0, r=4, flatten=True, ori=False, adj=False, prune='Grid', neighs=4):
        super(ViT_HER2ST, self).__init__()

        self.cnt_dir = '../her2st/data/ST-cnts'
        self.img_dir = '../her2st/data/ST-imgs'
        self.pos_dir = '../her2st/data/ST-spotfiles'
        self.lbl_dir = '../her2st/data/ST-pat/lbl'
        self.r = 224 // r

        # gene_list = list(np.load('data/her_hvg.npy',allow_pickle=True))
        gene_list = list(np.load('data/her_hvg_cut_1000.npy', allow_pickle=True))
        self.gene_list = gene_list
        names = os.listdir(self.cnt_dir)
        names.sort()
        names = [i[:2] for i in names]
        self.train = train
        self.ori = ori
        self.adj = adj
        # samples = ['A1','B1','C1','D1','E1','F1','G2','H1']
        samples = names[1:33]

        te_names = [samples[fold]]
        print(te_names)
        tr_names = list(set(samples) - set(te_names))

        if train:
            self.names = tr_names
        else:
            self.names = te_names

        print('Loading imgs...')
        self.img_dict = {i: torch.Tensor(np.array(self.get_img(i))) for i in self.names}
        print('Loading metadata...')
        self.meta_dict = {i: self.get_meta(i) for i in self.names}
        self.label = {i: None for i in self.names}
        self.lbl2id = {
            'invasive cancer': 0, 'breast glands': 1, 'immune infiltrate': 2,
            'cancer in situ': 3, 'connective tissue': 4, 'adipose tissue': 5, 'undetermined': -1
        }
        if not train and self.names[0] in ['A1', 'B1', 'C1', 'D1', 'E1', 'F1', 'G2', 'H1', 'J1']:
            self.lbl_dict = {i: self.get_lbl(i) for i in self.names}
            # self.label={i:m['label'].values for i,m in self.lbl_dict.items()}
            idx = self.meta_dict[self.names[0]].index
            lbl = self.lbl_dict[self.names[0]]
            lbl = lbl.loc[idx, :]['label'].values
            # lbl=torch.Tensor(list(map(lambda i:self.lbl2id[i],lbl)))
            self.label[self.names[0]] = lbl
        elif train:
            for i in self.names:
                idx = self.meta_dict[i].index
                if i in ['A1', 'B1', 'C1', 'D1', 'E1', 'F1', 'G2', 'H1', 'J1']:
                    lbl = self.get_lbl(i)
                    lbl = lbl.loc[idx, :]['label'].values
                    lbl = torch.Tensor(list(map(lambda i: self.lbl2id[i], lbl)))
                    self.label[i] = lbl
                else:
                    self.label[i] = torch.full((len(idx),), -1)
        self.gene_set = list(gene_list)
        self.exp_dict = {
            i: scp.transform.log(scp.normalize.library_size_normalize(m[self.gene_set].values))
            for i, m in self.meta_dict.items()
        }
        if self.ori:
            self.ori_dict = {i: m[self.gene_set].values for i, m in self.meta_dict.items()}
            self.counts_dict = {}
            for i, m in self.ori_dict.items():
                n_counts = m.sum(1)
                sf = n_counts / np.median(n_counts)
                self.counts_dict[i] = sf
        self.center_dict = {
            i: np.floor(m[['pixel_x', 'pixel_y']].values).astype(int)
            for i, m in self.meta_dict.items()
        }
        self.loc_dict = {i: m[['x', 'y']].values for i, m in self.meta_dict.items()}
        self.adj_dict = {
            i: calcADJ(m, neighs, pruneTag=prune)
            for i, m in self.loc_dict.items()
        }
        self.patch_dict = dfd(lambda: None)
        self.lengths = [len(i) for i in self.meta_dict.values()]
        self.cumlen = np.cumsum(self.lengths)
        self.id2name = dict(enumerate(self.names))
        self.flatten = flatten

    def __getitem__(self, index):
        ID = self.id2name[index]
        im = self.img_dict[ID]
        im = im.permute(1, 0, 2)
        # im = torch.Tensor(np.array(self.im))
        exps = self.exp_dict[ID]
        if self.ori:
            oris = self.ori_dict[ID]
            sfs = self.counts_dict[ID]
        centers = self.center_dict[ID]
        loc = self.loc_dict[ID]
        adj = self.adj_dict[ID]
        patches = self.patch_dict[ID]
        positions = torch.LongTensor(loc)
        patch_dim = 3 * self.r * self.r * 4
        label = self.label[ID]
        exps = torch.Tensor(exps)
        if patches is None:
            n_patches = len(centers)
            if self.flatten:
                patches = torch.zeros((n_patches, patch_dim))
            else:
                patches = torch.zeros((n_patches, 3, 2 * self.r, 2 * self.r))
            for i in range(n_patches):
                center = centers[i]
                x, y = center
                patch = im[(x - self.r):(x + self.r), (y - self.r):(y + self.r), :]
                if self.flatten:
                    patches[i] = patch.flatten()
                else:
                    patches[i] = patch.permute(2, 0, 1)
            self.patch_dict[ID] = patches
        data = [patches, positions, exps]
        if self.adj:
            data.append(adj)
        if self.ori:
            data += [torch.Tensor(oris), torch.Tensor(sfs)]
        data.append(torch.Tensor(centers))
        return data

    def __len__(self):
        return len(self.exp_dict)

    def get_img(self, name):
        pre = self.img_dir + '/' + name[0] + '/' + name
        fig_name = os.listdir(pre)[0]
        path = pre + '/' + fig_name
        im = Image.open(path)
        return im

    def get_cnt(self, name):
        path = self.cnt_dir + '/' + name + '.tsv'
        df = pd.read_csv(path, sep='\t', index_col=0)

        return df

    def get_pos(self, name):
        path = self.pos_dir + '/' + name + '_selection.tsv'
        # path = self.pos_dir+'/'+name+'_labeled_coordinates.tsv'
        df = pd.read_csv(path, sep='\t')

        x = df['x'].values
        y = df['y'].values
        x = np.around(x).astype(int)
        y = np.around(y).astype(int)
        id = []
        for i in range(len(x)):
            id.append(str(x[i]) + 'x' + str(y[i]))
        df['id'] = id

        return df

    def get_meta(self, name, gene_list=None):
        cnt = self.get_cnt(name)
        pos = self.get_pos(name)
        meta = cnt.join((pos.set_index('id')))

        return meta

    def get_lbl(self, name):
        # path = self.pos_dir+'/'+name+'_selection.tsv'
        path = self.lbl_dir + '/' + name + '_labeled_coordinates.tsv'
        df = pd.read_csv(path, sep='\t')

        x = df['x'].values
        y = df['y'].values
        x = np.around(x).astype(int)
        y = np.around(y).astype(int)
        id = []
        for i in range(len(x)):
            id.append(str(x[i]) + 'x' + str(y[i]))
        df['id'] = id
        df.drop('pixel_x', inplace=True, axis=1)
        df.drop('pixel_y', inplace=True, axis=1)
        df.drop('x', inplace=True, axis=1)
        df.drop('y', inplace=True, axis=1)
        df.set_index('id', inplace=True)
        return df

def pk_load(fold,mode='train',flatten=False,dataset='her2st',r=4,ori=True,adj=True,prune='Grid',neighs=4):
    assert dataset in ['her2st','cscc']
    if dataset=='her2st':
        dataset = ViT_HER2ST(
            train=(mode=='train'),fold=fold,flatten=flatten,
            ori=ori,neighs=neighs,adj=adj,prune=prune,r=r
        )
    '''elif dataset=='cscc':
        dataset = ViT_SKIN(
            train=(mode=='train'),fold=fold,flatten=flatten,
            ori=ori,neighs=neighs,adj=adj,prune=prune,r=r
        )'''
    return dataset

if __name__=="__main__":
    trainset = pk_load(0, 'train', False, 'her2st', neighs=4, prune='Grid')
    train_loader = DataLoader(trainset, batch_size=1, num_workers=0, shuffle=True)
    for stepi, input in enumerate(train_loader, 1):
        for i in range(len(input)):
            '''
            img <class 'torch.Tensor'> torch.Size([1, 325, 3, 112, 112])
            position <class 'torch.Tensor'> torch.Size([1, 325, 2])
            exp <class 'torch.Tensor'> torch.Size([1, 325, 785])
            <class 'torch.Tensor'> torch.Size([1, 325, 325])
            <class 'torch.Tensor'> torch.Size([1, 325, 785])
            <class 'torch.Tensor'> torch.Size([1, 325])
            <class 'torch.Tensor'> torch.Size([1, 325, 2])
            '''
            print(type(input[i]),input[i].shape)


