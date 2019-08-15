import numpy as np


def gen_random_graph_2d(nIns, nOus, scale, noise_level, theta, OusScale=1.5):
    inliers = np.random.uniform(-1, 1, [nIns, 2])

    rot_mat = np.zeros([2, 2])
    rot_mat[0][0] = np.cos(theta)
    rot_mat[0][1] = np.sin(theta)
    rot_mat[1][0] = -np.sin(theta)
    rot_mat[1][1] = np.cos(theta)

    trans_mat = scale * rot_mat

    ninliers = inliers.dot(trans_mat) + np.random.randn(nIns, 2) * noise_level

    pt1 = [inliers, np.random.uniform(-OusScale, OusScale, [nOus, 2])]
    pt2 = [ninliers, np.random.uniform(-OusScale, OusScale, [nOus, 2])]

    return np.concatenate(
        pt1, axis=0).astype(np.float32), np.concatenate(
            pt2, axis=0).astype(np.float32)


def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc


def adj_mat(pc):
    pc_square = np.sum(pc**2, axis=1, keepdims=True)
    ptp = -2 * np.matmul(pc, np.transpose(pc, [1, 0]))
    pt_square = np.transpose(pc_square, [1, 0])
    res = pc_square + pt_square + ptp
    return res


def knn(pc, k):
    adj = adj_mat(pc)
    distance = np.argsort(adj, axis=1)
    return distance[:, :k]


def rotate_pt(pt):
    theta = np.random.uniform(0, 1) * 2 * np.pi
    rot_mat = np.zeros([2, 2])
    rot_mat[0][0] = np.cos(theta)
    rot_mat[0][1] = np.sin(theta)
    rot_mat[1][0] = -np.sin(theta)
    rot_mat[1][1] = np.cos(theta)

    return pt.dot(rot_mat).astype(np.float32)
