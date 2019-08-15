import torch


def pairwise_distance(node_feature):
    """
    Compute the l2 distance between any two features in a group of features
    :param node_feature: a group of feature vectors organized as batch_size x channels x number_of_nodes
    
    """
    batch_size = node_feature.shape[0]
    node_feature = node_feature.squeeze()
    if batch_size == 1:
        node_feature = node_feature.unsqueeze(0)
    # print(node_feature.shape)
    assert (len(node_feature.shape) == 3)

    node_feature_t = node_feature.permute(0, 2, 1)
    node_feature_inner = -2 * torch.bmm(node_feature_t, node_feature)
    node_feature_square = node_feature**2
    node_feature_square = node_feature_square.sum(dim=1, keepdim=True)
    node_feature_square_t = node_feature_square.permute(0, 2, 1)

    res = node_feature_square + node_feature_square_t + node_feature_inner
    return res
