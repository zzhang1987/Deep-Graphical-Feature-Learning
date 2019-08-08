import torch
from .utils import get_edge_feature, get_nn_node_feature
from .mp_conv import etype_net, mp_conv_v2, gconv_residual

SyncBatchNorm = torch.nn.BatchNorm2d


class feature_network(torch.nn.Module):
    def __init__(self, with_residual=True, with_global=False, output_dim=512):
        super(feature_network, self).__init__()
        self.with_global = with_global
        self.etype_net = etype_net(16, 64, 2)

        self.mp_conv1 = mp_conv_v2(2, 64, 16)
        self.conv1 = torch.nn.Sequential(torch.nn.Conv2d(64, 128, 1),
                                         SyncBatchNorm(128),
                                         torch.nn.ReLU(inplace=True))

        self.mp_residual1 = gconv_residual(128,
                                           64,
                                           16,
                                           with_residual=with_residual)

        self.conv2 = torch.nn.Sequential(torch.nn.Conv2d(128, 128, 1),
                                         SyncBatchNorm(128),
                                         torch.nn.ReLU(inplace=True))
        self.mp_conv2 = mp_conv_v2(128, 128, 16)
        self.conv3 = torch.nn.Sequential(torch.nn.Conv2d(128, 256, 1),
                                         SyncBatchNorm(256),
                                         torch.nn.ReLU(inplace=True))

        self.mp_residual2 = gconv_residual(256,
                                           128,
                                           16,
                                           with_residual=with_residual)
        if self.with_global:
            self.conv4 = torch.nn.Sequential(torch.nn.Conv2d(512, 256, 1),
                                             SyncBatchNorm(256),
                                             torch.nn.ReLU(inplace=True))
        else:
            self.conv4 = torch.nn.Sequential(torch.nn.Conv2d(256, 256, 1),
                                             SyncBatchNorm(256),
                                             torch.nn.ReLU(inplace=True))
        self.mp_conv3 = mp_conv_v2(256, 256, 16)
        self.conv5 = torch.nn.Sequential(torch.nn.Conv2d(256, 512, 1),
                                         SyncBatchNorm(512),
                                         torch.nn.ReLU(inplace=True))

        self.mp_residual3 = gconv_residual(512,
                                           256,
                                           16,
                                           with_residual=with_residual)

        self.affine = torch.nn.Sequential(
            torch.nn.Conv2d(512, output_dim, 1, bias=False))

    def forward(self, pts, nn_idx, mask):
        pts_knn = get_nn_node_feature(pts, nn_idx)
        efeature = get_edge_feature(pts_knn, pts)

        etype = self.etype_net(efeature)

        nfeature = self.mp_conv1(
            pts.view(pts.shape[0], pts.shape[1], pts.shape[2], 1), nn_idx,
            etype)
        nfeature = self.conv1(nfeature)
        # nfeature, _ = nfeature.max(dim=3, keepdim=True)

        nfeature = self.mp_residual1(nfeature, etype, nn_idx)

        nfeature = self.conv2(nfeature)
        nfeature = self.mp_conv2(nfeature, nn_idx, etype)
        nfeature = self.conv3(nfeature)

        nfeature = self.mp_residual2(nfeature, etype, nn_idx)

        if self.with_global:
            global_feature, _ = nfeature.max(dim=2, keepdim=True)
            nfeature = torch.cat(
                [nfeature,
                 global_feature.repeat(1, 1, nfeature.shape[2], 1)],
                dim=1)

        nfeature = self.conv4(nfeature)
        nfeature = self.mp_conv3(nfeature, nn_idx, etype)
        nfeature = self.conv5(nfeature)

        nfeature = self.mp_residual3(nfeature, etype, nn_idx)
        nfeature = self.affine(nfeature)

        nfeature = torch.nn.functional.normalize(nfeature, 2)
        nfeature = nfeature.squeeze() * mask.unsqueeze(1)

        return nfeature
