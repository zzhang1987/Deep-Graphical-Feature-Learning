import torch
import numpy as np
from torch.utils.data import Dataset

from .utils import gen_random_graph_2d, pc_normalize, knn


class RandomGraphDataset(Dataset):
    def __init__(self,
                 nMinIn,
                 nMaxIn,
                 nMinOu,
                 nMaxOu,
                 k=8,
                 min_scale=0.9,
                 max_scale=1.2,
                 max_noise_level=0.05):
        self.nMinIn = nMinIn
        self.nMaxIn = nMaxIn
        self.nMinOu = nMinOu
        self.nMaxOu = nMaxOu

        self.min_scale = min_scale
        self.max_scale = max_scale
        self.max_noise_level = max_noise_level
        self.nMaxNodes = self.nMaxIn + self.nMaxOu
        self.k = k

    def __len__(self):
        return 1024 * 1024

    def __getitem__(self, idx):

        nIns = np.random.randint(self.nMinIn, self.nMaxIn)
        if self.nMaxOu == self.nMinOu:
            nOus = self.nMinOu
        else:
            nOus = np.random.randint(self.nMinOu, self.nMaxOu)

        theta = np.random.uniform(0, 2 * np.pi)
        scale = np.random.uniform(self.min_scale, self.max_scale)
        noise_level = np.random.uniform(0, self.max_noise_level)

        pt1, pt2 = gen_random_graph_2d(nIns, nOus, scale, noise_level, theta)

        n_of_nodes = nIns + nOus

        gTruth = np.random.permutation(n_of_nodes)
        pt1 = pt1[gTruth, :]

        pt1 = pc_normalize(pt1).astype(np.float32)
        knn_pt1 = knn(pt1, self.k)

        pt2 = pc_normalize(pt2).astype(np.float32)
        knn_pt2 = knn(pt2, self.k)

        pad_offset = self.nMaxNodes - n_of_nodes

        if pad_offset > 0:
            # print(pad_offset)
            pt1 = np.pad(
                pt1, [[0, pad_offset], [0, 0]], 'constant', constant_values=0)
            knn_pt1 = np.pad(
                knn_pt1, [[0, pad_offset], [0, 0]],
                'constant',
                constant_values=n_of_nodes)
            pt2 = np.pad(
                pt2, [[0, pad_offset], [0, 0]], 'constant', constant_values=0)
            knn_pt2 = np.pad(
                knn_pt2, [[0, pad_offset], [0, 0]],
                'constant',
                constant_values=n_of_nodes)
            gTruth = np.pad(
                gTruth, [[0, pad_offset]], 'constant', constant_values=-100)

        gTruth[nIns:] = -100

        mask = [1.0] * n_of_nodes + [0.0] * pad_offset
        mask = np.asarray(mask)

        data_item = [
            pt1.astype(np.float32),
            knn_pt1.astype(np.int64),
            pt2.astype(np.float32),
            knn_pt2.astype(np.int64),
            mask.astype(np.float32),
            gTruth.astype(np.int64)
        ]
        data_item = [torch.from_numpy(d) for d in data_item]

        return data_item
