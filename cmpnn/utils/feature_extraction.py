import torch


def get_edge_feature(pts_knn, pts):
    return pts.view(pts.shape[0], pts.shape[1], pts.shape[2], 1) - pts_knn


def get_nn_node_feature(node_feature, nn_idx):
    batch_size = nn_idx.shape[0]
    node_feature = node_feature.squeeze()
    if batch_size == 1:
        node_feature = node_feature.unsqueeze(0)
    # print(node_feature.shape)
    # print(nn_idx.shape)
    assert (batch_size == node_feature.shape[0])
    npts = nn_idx.shape[1]
    # assert (npts == nodel_feature.shape[2])
    k = nn_idx.shape[2]

    nidx = nn_idx.view(batch_size,
                       -1).unsqueeze(1).repeat(1, node_feature.shape[1], 1)

    pts_knn = node_feature.gather(2, nidx).view(batch_size, -1, npts, k)

    return pts_knn
