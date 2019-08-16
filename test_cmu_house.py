import cmpnn
from utils import str2bool, ComputeAccuracyPas, to_tensor, to_cuda
from data import load_cmu_house, pc_normalize, knn, rotate_pt
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
    parser.add_argument("--data_path",
                        type=str, required=True, nargs='?', const=True, help="path to CMU house datasets") 
    parser.add_argument("--use_cuda",
                        type=str2bool,
                        default=True,
                        help="Use cuda or not")

    args = parser.parse_args()
    if not torch.cuda.is_available():
        args.use_cuda = False
    return args

def run_cmu_test_instance(model, HouseImg, HouseData, idx, sep, use_cuda=True):
    NofNodes = 30
    PT1 = np.copy(HouseData[idx]).astype(np.float32)
    PT2 = np.copy(HouseData[idx+sep]).astype(np.float32)
    gTruth = np.random.permutation(NofNodes)
    PT1 = PT1[gTruth, :]
    
    pt1 = pc_normalize(PT1)
    pt2 = pc_normalize(PT2)
    pt1 = rotate_pt(pt1)

    nn_idx1 = knn(pt1, 8)
    nn_idx2 = knn(pt2, 8)
    mask = np.asarray([1.0] * pt1.shape[0] ).astype(np.float32)

    with torch.no_grad():
        pt1, pt2, nn_idx1, nn_idx2, mask = to_tensor([pt1, pt2, nn_idx1, nn_idx2, mask], use_cuda)
        feature1 = model(pt1.permute(0, 2, 1), nn_idx1.contiguous(), mask)
        feature2 = model(pt2.permute(0, 2, 1), nn_idx2.contiguous(), mask)
        sim = torch.bmm(feature1.permute(0, 2, 1), feature2)

    cost = - sim[0].cpu().numpy()
    row_ind, col_ind = linear_sum_assignment(cost)
    acc = ComputeAccuracyPas(col_ind, gTruth, NofNodes)
    return acc 





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

    HouseData, HouseImg = load_cmu_house(args.data_path)
    
    for sep in range(10, 101, 10):
        accs = []
        for image_idx in range(0, 111 - sep):
            accs.append(run_cmu_test_instance(model, HouseImg, HouseData, image_idx, sep, args.use_cuda))
        print('sep = {} avg acc = {}'.format(sep, np.mean(accs)))


if __name__ == '__main__':
    main()
