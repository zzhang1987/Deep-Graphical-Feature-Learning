import cmpnn
from utils import str2bool, ComputeAccuracyPas, to_tensor
from data import gen_random_graph_2d, pc_normalize, knn
from scipy.optimize import linear_sum_assignment

import torch
import os
import argparse
import numpy as np
import time


def parse_arguments():

    parser = argparse.ArgumentParser(description='Training opts')
    parser.add_argument(
        "--with_residual",
        type=str2bool,
        nargs='?',
        const=True,
        default=True,
        help='Activate residual link')
    parser.add_argument(
        "--with_global_pool",
        type=str2bool,
        nargs='?',
        const=True,
        default=True,
        help='Activate global pooling')
    parser.add_argument(
        "--param_path",
        type=str,
        required=True,
        nargs='?',
        const=True,
        help="path to trained parameters")

    args = parser.parse_args()
    return args


def run_test_instance(model, nIns, nOus, scale, noise_level, ntheta):
    PT1, PT2 = gen_random_graph_2d(nIns, nOus, scale, noise_level, ntheta)
    gTruth = np.random.permutation(nIns + nOus)
    PT1 = PT1[gTruth, :]

    pt1 = pc_normalize(PT1)
    pt2 = pc_normalize(PT2)
    nn_idx1 = knn(pt1, 8)
    nn_idx2 = knn(pt2, 8)
    mask = np.asarray([1.0] * pt1.shape[0]).astype(np.float32)

    with torch.no_grad():
        pt1, pt2, nn_idx1, nn_idx2, mask = to_tensor(
            [pt1, pt2, nn_idx1, nn_idx2, mask])

    cuda_start = torch.cuda.Event(enable_timing=True)
    cuda_end = torch.cuda.Event(enable_timing=True)
    with torch.no_grad():
        cuda_start.record()
        feature1 = model(pt1.permute(0, 2, 1), nn_idx1, mask)
        feature2 = model(pt2.permute(0, 2, 1), nn_idx2, mask)
        sim = torch.bmm(feature1.permute(0, 2, 1), feature2)
        cuda_end.record()
    torch.cuda.synchronize()

    nninfer_time = cuda_start.elapsed_time(cuda_end)
    cost = -sim[0].cpu().numpy()
    start = time.time()
    row_ind, col_ind = linear_sum_assignment(cost)
    lapinfer_time = time.time() - start
    acc = ComputeAccuracyPas(col_ind, gTruth, nIns)

    return acc, nninfer_time, lapinfer_time


def test_noise(model, nIns, nOus, scale, noise_start, noise_end, noise_step,
               seed):
    np.random(seed)

    avg_accs = []
    avg_time_nninfer = []
    avg_time_lapinfer = []

    for noise_level in np.arange(noise_start, noise_end, noise_step):
        cavg_acc = []
        cavg_time_nninfer = []
        cavg_time_lapinfer = []


def test_theta(model, nIns, nOus, scale, noise_level, seed=123456):
    np.random.seed(seed)

    avg_accs = []
    avg_time_nninfer = []
    avg_time_lapinfer = []

    for theta in np.arange(0, 1.01, 0.05):
        cavg_acc = []
        cavg_time_nninfer = []
        cavg_time_lapinfer = []

        for idx in range(0, 100):
            ntheta = theta * np.pi  #np.random.uniform(0, 2 * np.pi)
            PT1, PT2 = gen_random_graph_2d(nIns, nOus, scale, noise_level,
                                           ntheta)
            gTruth = np.random.permutation(nIns + nOus)
            PT1 = PT1[gTruth, :]

            pt1 = pc_normalize(PT1)
            pt2 = pc_normalize(PT2)
            nn_idx1 = knn(pt1, 8)
            nn_idx2 = knn(pt2, 8)
            mask = np.asarray([1.0] * pt1.shape[0]).astype(np.float32)

            with torch.no_grad():
                pt1, pt2, nn_idx1, nn_idx2, mask = to_tensor(
                    [pt1, pt2, nn_idx1, nn_idx2, mask])

            cuda_start = torch.cuda.Event(enable_timing=True)
            cuda_end = torch.cuda.Event(enable_timing=True)
            with torch.no_grad():
                cuda_start.record()
                feature1 = model(pt1.permute(0, 2, 1), nn_idx1, mask)
                feature2 = model(pt2.permute(0, 2, 1), nn_idx2, mask)
                sim = torch.bmm(feature1.permute(0, 2, 1), feature2)
                cuda_end.record()
            torch.cuda.synchronize()
            cavg_time_nninfer.append(cuda_start.elapsed_time(cuda_end))

            cost = -sim[0].cpu().numpy()
            start = time.time()
            row_ind, col_ind = linear_sum_assignment(cost)
            cavg_time_lapinfer.append(time.time() - start)
            acc = ComputeAccuracyPas(col_ind, gTruth, nIns)
            cavg_acc.append(acc)

        cacc = np.mean(cavg_acc)
        cinfer_time_nn = np.mean(cavg_time_nninfer)
        cinfer_time_lap = np.mean(cavg_time_lapinfer)

        avg_accs.append(cacc)
        avg_time_nninfer.append(cinfer_time_nn)
        avg_time_lapinfer.append(cinfer_time_lap)
        print("theta = {} acc = {} nntime = {} laptime ={}".format(
            ntheta, cacc, cinfer_time_nn, cinfer_time_lap))
    return avg_accs, avg_time_nninfer, avg_time_lapinfer


def main():
    args = parse_arguments()

    print(args)

    model = cmpnn.graph_matching.feature_network(
        with_residual=args.with_residual, with_global=args.with_global_pool)
    model.cuda()

    params = torch.load(args.param_path)
    model.load_state_dict(params['model_state_dict'])
    model.eval()

    avg_accs, avg_time_nninfer, avg_time_lapinfer = test_theta(
        model, 20, 2, 1.0, 0)

    print("avg accs: ", avg_accs)
    print("avg nn infer time: ", avg_time_nninfer)
    print("avg lap infer time: ", avg_time_lapinfer)

    avg_accs, avg_time_nninfer, avg_time_lapinfer = test_theta(
        model, 20, 0, 1.0, 0.025)

    print("avg accs: ", avg_accs)
    print("avg nn infer time: ", avg_time_nninfer)
    print("avg lap infer time: ", avg_time_lapinfer)


if __name__ == '__main__':
    main()
