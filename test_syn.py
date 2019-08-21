import cmpnn
from utils import str2bool, ComputeAccuracyPas, to_tensor, to_cuda
from data import gen_random_graph_2d, pc_normalize, knn
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

import torch
import os
import argparse
import numpy as np
import time


def parse_arguments():

    parser = argparse.ArgumentParser(description='Training opts')
    parser.add_argument("--with_residual",
                        type=str2bool,
                        nargs='?',
                        const=True,
                        default=True,
                        help='Activate residual link')
    parser.add_argument("--with_global_pool",
                        type=str2bool,
                        nargs='?',
                        const=True,
                        default=True,
                        help='Activate global pooling')
    parser.add_argument("--param_path",
                        type=str,
                        required=True,
                        nargs='?',
                        const=True,
                        help="path to trained parameters")
    parser.add_argument("--use_cuda",
                        type=str2bool,
                        default=True,
                        help="Use cuda or not")

    args = parser.parse_args()
    if not torch.cuda.is_available():
        args.use_cuda = False
    return args


def run_test_instance(model,
                      nIns,
                      nOus,
                      scale,
                      noise_level,
                      ntheta,
                      use_cuda=True):
    PT1, PT2 = gen_random_graph_2d(nIns,
                                   nOus,
                                   scale,
                                   noise_level,
                                   ntheta,
                                   OusScale=1.0)
    gTruth = np.random.permutation(nIns + nOus)
    PT1 = PT1[gTruth, :]

    pt1 = pc_normalize(PT1)
    pt2 = pc_normalize(PT2)
    nn_idx1 = knn(pt1, 8)
    nn_idx2 = knn(pt2, 8)
    mask = np.asarray([1.0] * pt1.shape[0]).astype(np.float32)

    with torch.no_grad():
        pt1, pt2, nn_idx1, nn_idx2, mask = to_tensor(
            [pt1, pt2, nn_idx1, nn_idx2, mask], use_cuda)

    with torch.no_grad():
        # print(pt1.shape, nn_idx1.shape, mask.shape)
        feature1 = model(pt1.permute(0, 2, 1), nn_idx1.contiguous(), mask)
        feature2 = model(pt2.permute(0, 2, 1), nn_idx2.contiguous(), mask)
        sim = torch.bmm(feature1.permute(0, 2, 1), feature2)

    cost = -sim[0].cpu().numpy()
    row_ind, col_ind = linear_sum_assignment(cost)
    acc = ComputeAccuracyPas(col_ind, gTruth, nIns)

    return acc


def test_noise(model, nIns, nOus, scale, noise_start, noise_end, noise_step,
               seed, use_cuda):
    np.random.seed(seed)

    avg_accs = []
    for noise_level in tqdm(np.arange(noise_start, noise_end, noise_step)):
        avg_acc = []
        for idx in tqdm(range(0, 100)):
            ntheta = np.random.uniform(0, np.pi)
            avg_acc.append(
                run_test_instance(model, nIns, nOus, scale, noise_level,
                                  ntheta, use_cuda))
        avg_accs.append(np.mean(avg_acc))
    return avg_accs


def test_outlier(model, nIns, noise_level, scale, nOus_start, nOus_end,
                 nOus_step, seed, use_cuda):
    np.random.seed(seed)

    avg_accs = []

    for nOus in tqdm(range(nOus_start, nOus_end, nOus_step)):
        avg_acc = []
        for idx in tqdm(range(0, 100)):
            ntheta = np.random.uniform(0, np.pi)
            avg_acc.append(
                run_test_instance(model, nIns, nOus, scale, noise_level,
                                  ntheta, use_cuda))
        avg_accs.append(np.mean(avg_acc))
    return avg_accs



def main():
    args = parse_arguments()

    print(args)

    model = cmpnn.graph_matching.feature_network(
        with_residual=args.with_residual, with_global=args.with_global_pool)
    if args.use_cuda:
        model.cuda()

    if args.use_cuda:
        params = torch.load(args.param_path)
    else:
        params = torch.load(args.param_path, map_location=torch.device('cpu'))

    model.load_state_dict(params['model_state_dict'])
    model.eval()

    avg_accs = test_noise(model, 20, 0, 1.0, 0, 0.101, 0.01, 123456,
                          args.use_cuda)
    print("avg accs: ", avg_accs)

    avg_accs = test_outlier(model, 20, 0, 1.0, 0, 11, 1, 123456, args.use_cuda)
    print("avg accs: ", avg_accs)

    avg_accs = test_outlier(model, 20, 0.025, 1.0, 0, 11, 1, 123456,
                            args.use_cuda)
    print("avg accs: ", avg_accs)


if __name__ == '__main__':
    main()
