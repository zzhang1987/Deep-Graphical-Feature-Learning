import torch
import numpy as np


def to_tensor(feature_list):
    return [
        torch.unsqueeze(torch.from_numpy(t), 0).cuda() for t in feature_list
    ]


def ComputeAccuracyPas(decode, gTruth, NofInliers):
    Ccnt = 0
    for i in range(len(gTruth)):
        if ((decode[i] == gTruth[i]) and (gTruth[i] < NofInliers)):
            Ccnt += 1
    return 1.0 * Ccnt / NofInliers


def ComputeAccurancy(Decode, gTruth, NofNodes):
    NIns = np.sum(gTruth[0:NofNodes] != -1)
    #print(NIns)
    Ccnt = 0
    for i in range(NofNodes):
        if (Decode[i] == gTruth[i]):
            Ccnt += 1
    acc = 1.0 * Ccnt / NIns
    return acc, Ccnt


def ComputeAccuracyPas(decode, gTruth, NofInliers):
    Ccnt = 0
    for i in range(len(gTruth)):
        if ((decode[i] == gTruth[i]) and (gTruth[i] < NofInliers)):
            Ccnt += 1
    return 1.0 * Ccnt / NofInliers
