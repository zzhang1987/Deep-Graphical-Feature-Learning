import torch
import numpy
import scipy.io as sio
import os



class PascalPFCategory:
    def __init__(self, data_path, cpair, label):
        self.data_path = data_path
        self.cpair = cpair
        self.label = label

    def __len__(self):
        return cpair.shape[0]

    def __getitem__(self, idx):
        p1 = cpair[pidx, 0][0]
        p2 = cpair[pidx, 1][0]

        pt1 = sio.loadmat(os.path.join(self.path, 'Annotations', self.label, p1 + '.mat'))['kps']
        pt2 = sio.loadmat(os.path.join(self.path, 'Annotations', self.label, p2 + '.mat'))['kps']

        I1 = skimage.io.imread(os.path.join(self.path, 'JPEGImages/', p1 + '.jpg'))
        I2 = skimage.io.imread(os.path.join(self.path, 'JPEGImages/', p2 + '.jpg'))

        pt1 = pt1[~np.isnan(pt1).any(axis=1)]
        pt2 = pt2[~np.isnan(pt2).any(axis=1)]

        gTruth = np.random.permutation(pt1.shape[0])
        orig_pt1 = pt1[gTruth, :]
        pt1 = rotate_pt(orig_pt1)
        npt1 = pc_normalize(pt1)
        npt2 = pc_normalize(pt2)
        nn_idx1 = knn(npt1, 5)
        nn_idx2 = knn(npt2, 5)
        mask = np.asarray([1.0] * npt1.shape[0] ).astype(np.float32)

        return gTruth, npt1, npt2, nn_idx1, nn_idx2, mask, orig_pt1, orig_pt2, I1, I2

class PascalPF:
    def __init__(self, data_path):
        self.data_path = data_path
        self.pairs = sio.loadmat(os.path.join(self.data_path, 'parsePascalVOC.mat'))

    def __len__(self):
        return 20

    def __getitem__(self, idx):
        cpair = self.pairs['PascalVOC']['pair'][0, 0][0, idx]
        label = self.pairs['PascalVOC']['class'][0, 0][0, idx][0]

        return PascalPFCategory(self.data_path, cpair, label)
