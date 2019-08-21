import torch
import numpy as np
import scipy.io as sio
import os
import skimage.io
from .utils import rotate_pt, pc_normalize, knn

class PascalPFCategory:
    def __init__(self, data_path, cpair, label, random_rotate=False):
        self.data_path = data_path
        self.cpair = cpair
        self.label = label
        self.random_rotate = random_rotate
        
    def __len__(self):
        return self.cpair.shape[0]

    def __getitem__(self, idx):
        pidx = idx
        p1 = self.cpair[pidx, 0][0]
        p2 = self.cpair[pidx, 1][0]

        pt1 = sio.loadmat(
            os.path.join(self.data_path, 'Annotations', self.label,
                         p1 + '.mat'))['kps']
        pt2 = sio.loadmat(
            os.path.join(self.data_path, 'Annotations', self.label,
                         p2 + '.mat'))['kps']

        I1 = skimage.io.imread(
            os.path.join(self.data_path, 'JPEGImages/', p1 + '.jpg'))
        I2 = skimage.io.imread(
            os.path.join(self.data_path, 'JPEGImages/', p2 + '.jpg'))

        pt1 = pt1[~np.isnan(pt1).any(axis=1)]
        pt2 = pt2[~np.isnan(pt2).any(axis=1)]

        gTruth = np.random.permutation(pt1.shape[0])
        orig_pt1 = pt1[gTruth, :]
        if self.random_rotate:
            pt1 = rotate_pt(orig_pt1)
        else:
            pt1 = orig_pt1
        npt1 = pc_normalize(pt1)
        npt2 = pc_normalize(pt2)
        nn_idx1 = knn(npt1, 5)
        nn_idx2 = knn(npt2, 5)
        mask = np.asarray([1.0] * npt1.shape[0]).astype(np.float32)

        return gTruth, npt1.astype(np.float32), npt2.astype(np.float32), nn_idx1, nn_idx2, mask, orig_pt1, pt2, I1, I2


class PascalPF:
    def __init__(self, data_path, random_rotate=False):
        self.data_path = data_path
        self.pairs = sio.loadmat(
            os.path.join(self.data_path, 'parsePascalVOC.mat'))
        self.random_rotate = random_rotate
    def __len__(self):
        return 20

    def __getitem__(self, idx):
        cpair = self.pairs['PascalVOC']['pair'][0, 0][0, idx]
        label = self.pairs['PascalVOC']['class'][0, 0][0, idx][0]

        return label, PascalPFCategory(self.data_path, cpair, label, self.random_rotate)
