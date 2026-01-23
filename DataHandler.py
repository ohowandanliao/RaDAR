import pickle
import numpy as np
from scipy.sparse import csr_matrix, coo_matrix, dok_matrix
from Params import args
import scipy.sparse as sp
from Utils.TimeLogger import log
import torch as t
import torch.utils.data as data
import torch.utils.data as dataloader
import networkx as nx
import dgl
import os


class DataHandler:
    def __init__(self):
        if args.debug:
            predir = './Datasets/lastFM/'
        elif args.data == 'yelp':
            predir = './Datasets/sparse_yelp/'
        elif args.data == 'lastfm':
            predir = './Datasets/lastFM/'
        elif args.data == 'beer':
            predir = './Datasets/beerAdvocate/'
        elif args.data == 'tmall':
            # Weighted-edge dataset: choose weighted or binary based on use_weighted_edges flag
            subdir = 'weighted' if getattr(args, 'use_weighted_edges', 0) == 1 else 'binary'
            predir = f'./Datasets/tmall/{subdir}/'
        elif args.data == 'retail_rocket':
            subdir = 'weighted' if getattr(args, 'use_weighted_edges', 0) == 1 else 'binary'
            predir = f'./Datasets/retail_rocket/{subdir}/'
        elif args.data == 'ijcai_15':
            subdir = 'weighted' if getattr(args, 'use_weighted_edges', 0) == 1 else 'binary'
            predir = f'./Datasets/ijcai_15/{subdir}/'
        else:
            # Default fallback: try to use data name directly
            predir = f'./Datasets/{args.data}/'
        self.predir = predir
        self.trnfile = predir + 'trnMat.pkl'  # train file
        self.tstfile = predir + 'tstMat.pkl'  # test file

    def loadOneFile(self, filename):
        with open(filename, 'rb') as fs:
            ret = (pickle.load(fs) != 0).astype(np.float32)
        if type(ret) != coo_matrix:
            ret = sp.coo_matrix(ret)
        return ret

    def normalizeAdj(self, mat):
        degree = np.array(mat.sum(axis=-1))
        dInvSqrt = np.reshape(np.power(degree, -0.5), [-1])
        dInvSqrt[np.isinf(dInvSqrt)] = 0.0
        dInvSqrtMat = sp.diags(dInvSqrt)
        return mat.dot(dInvSqrtMat).transpose().dot(dInvSqrtMat).tocoo()

    def makeTorchAdj(self, mat):
        # make ui adj
        a = sp.csr_matrix((args.user, args.user))
        b = sp.csr_matrix((args.item, args.item))
        mat = sp.vstack([sp.hstack([a, mat]),
                         sp.hstack([mat.transpose(), b])])  # [(args.user + args.item) , (args.user + args.item)]
        mat = (mat != 0) * 1.0
        mat = (mat + sp.eye(mat.shape[0])) * 1.0  # A + I = ~A
        mat = self.normalizeAdj(mat)

        # make cuda tensor
        idxs = t.from_numpy(np.vstack([mat.row, mat.col]).astype(np.int64))
        vals = t.from_numpy(mat.data.astype(np.float32))
        shape = t.Size(mat.shape)
        return t.sparse.FloatTensor(idxs, vals, shape).cuda()

    def LoadData(self):
        trnMat = self.loadOneFile(self.trnfile)
        tstMat = self.loadOneFile(self.tstfile)
        self.trnMat = trnMat
        args.user, args.item = trnMat.shape
        self.torchBiAdj = self.makeTorchAdj(trnMat)  # ( D^{-1/2} A D^{-1/2} )
        trnData = TrnData(trnMat)
        self.trnLoader = dataloader.DataLoader(trnData, batch_size=args.batch, shuffle=True, num_workers=0)
        tstData = TstData(tstMat, trnMat)
        self.tstLoader = dataloader.DataLoader(tstData, batch_size=args.tstBat, shuffle=False, num_workers=0)

    def graph_generator(self):
        G = nx.MultiDiGraph()
        if args.user is None or args.item is None:
            args.user, args.item = self.trnMat.shape

        entities_list = []
        for i, j in zip(self.trnMat.row, self.trnMat.col):
            G.add_edge(str(i), str(args.user + j))
            G.add_edge(str(args.user + j), str(i))
        for user_idx in range(args.user):
            entities_list.append(str(user_idx))
        for item_idx in range(args.item):
            entities_list.append(str(item_idx + args.user))
        return G, entities_list

    def dgl_graph_generator(self):
        if self.trnMat is None:
            self.trnMat = self.loadOneFile(self.trnfile)
        user_indices = self.trnMat.row
        item_indices = self.trnMat.col
        num_users = args.user
        num_items = args.item
        num_nodes = num_users + num_items
        item_indices += num_users

        device = t.device('cuda' if t.cuda.is_available() else 'cpu')
        dgl_graph = dgl.graph((user_indices, item_indices), num_nodes=num_nodes)
        dgl_graph = dgl.to_bidirected(dgl_graph, copy_ndata=True).to(device)
        all_nodes = set(range(num_nodes))
        connected_nodes = set(user_indices).union(item_indices)
        isolated_nodes = all_nodes - connected_nodes
        return dgl_graph


class TrnData(data.Dataset):
    def __init__(self, coomat):
        self.rows = coomat.row  # 非零元素 行索引
        self.cols = coomat.col  # 非零元素 列索引
        self.dokmat = coomat.todok()  # {(row,col), value}
        self.negs = np.zeros(len(self.rows)).astype(np.int32)  # 非零元素 行索引同等长度的list

    def negSampling(self):
        for i in range(len(self.rows)):
            u = self.rows[i]
            while True:
                iNeg = np.random.randint(args.item)
                if (u, iNeg) not in self.dokmat:
                    break
            self.negs[i] = iNeg

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        return self.rows[idx], self.cols[idx], self.negs[idx]


class TstData(data.Dataset):
    def __init__(self, coomat, trnMat):
        self.csrmat = (trnMat.tocsr() != 0) * 1.0
        self.csrmat_transposed = self.csrmat.T

        tstLocs = [None] * coomat.shape[0]
        tstUsrs = set()
        for i in range(len(coomat.data)):
            row = coomat.row[i]
            col = coomat.col[i]
            if tstLocs[row] is None:
                tstLocs[row] = list()
            tstLocs[row].append(col)
            tstUsrs.add(row)

        tstUsrs = np.array(list(tstUsrs))
        self.tstUsrs = tstUsrs
        self.tstLocs = tstLocs
        print("init TstData done!")

    def __len__(self):
        return len(self.tstUsrs)

    def __getitem__(self, idx):
        return self.tstUsrs[idx], np.reshape(self.csrmat[self.tstUsrs[idx]].toarray(), [-1])


if __name__ == '__main__':
    log('Start')
    handler = DataHandler()
    handler.LoadData()
    log('Load Data')
