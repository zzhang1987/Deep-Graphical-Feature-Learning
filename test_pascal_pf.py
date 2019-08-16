import cmpnn
from utils import str2bool, ComputeAccurancy, to_tensor, to_cuda
from data import PascalPF
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
    parser.add_argument("--random_rotate", type=str2bool, required=True, help="Random rotate dataset or not")
    parser.add_argument("--use_cuda",
                        type=str2bool,
                        default=True,
                        help="Use cuda or not")

    args = parser.parse_args()
    if not torch.cuda.is_available():
        args.use_cuda = False
    return args

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

    pascal_pf = PascalPF(args.data_path, args.random_rotate)

    for category, data in pascal_pf:
        accs = []
        for data_item in tqdm(data):
            gTruth, pt1, pt2, nn_idx1, nn_idx2, mask, PT1, PT2, I1, I2 = data_item
            NofNodes = pt1.shape[0]
            # print(NofNodes)
            with torch.no_grad():
                pt1, pt2, nn_idx1, nn_idx2, mask = to_tensor([pt1, pt2, nn_idx1, nn_idx2, mask], args.use_cuda)
                feature1 = model(pt1.permute(0, 2, 1), nn_idx1.contiguous(), mask)
                feature2 = model(pt2.permute(0, 2, 1), nn_idx2.contiguous(), mask)
                sim = torch.bmm(feature1.permute(0, 2, 1), feature2)

            cost = - sim[0].cpu().numpy()
            row_ind, col_ind = linear_sum_assignment(cost)
            # print(col_ind)
            # print(gTruth)
            acc, _ = ComputeAccurancy(col_ind, gTruth, NofNodes)
            # print(acc)
            accs.append(acc)
        print('class = {} avg acc = {}'.format(category, np.mean(accs)))
    

if __name__ == '__main__':
    main()
