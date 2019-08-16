import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from skimage.transform import rescale, resize, downscale_local_mean


def draw_matches_with_outlier(I1,
                              I2,
                              Pt1,
                              Pt2,
                              matches,
                              gTruth,
                              NofInliers,
                              StoreFileName,
                              l1_scale=1.0,
                              l2_scale=1.0,
                              rotate_xy=True):

    I1_rescale = np.asarray(255 * rescale(I1, l1_scale), dtype=np.uint8)
    I2_rescale = np.asarray(255 * rescale(I2, l2_scale), dtype=np.uint8)
    I1 = I1_rescale
    I2 = I2_rescale

    if rotate_xy:
        nPt1 = np.zeros_like(Pt1)
        nPt2 = np.zeros_like(Pt2)
        nPt1[:, 0] = Pt1[:, 1]
        nPt1[:, 1] = Pt1[:, 0]
        nPt2[:, 0] = Pt2[:, 1]
        nPt2[:, 1] = Pt2[:, 0]
        Pt1 = nPt1
        Pt2 = nPt2

    Pt1_rescale = l1_scale * Pt1
    Pt2_rescale = l2_scale * Pt2
    Pt1 = Pt1_rescale
    Pt2 = Pt2_rescale

    h1 = I1.shape[0]
    w1 = I1.shape[1]

    h2 = I2.shape[0]
    w2 = I2.shape[1]
    # print('h1: %d  w1: %d, h2:%d w2:%d' % (h1, w1, h2, w2))

    fig = plt.figure(1)
    fig.clf()
    h = max(h1, h2)

    diff1 = h - h1
    diff2 = h - h2

    d11 = int(diff1 * 0.5)
    d12 = diff1 - d11

    d21 = int(diff2 * 0.5)
    d22 = diff2 - d21
    ax1 = plt.axes(frameon=False)

    ax1.axes.get_yaxis().set_visible(False)
    ax1.axes.get_xaxis().set_visible(False)
    I = np.zeros([h, w1 + w2, 3], dtype=np.uint8)
    if len(I1.shape) == 2 or I1.shape[2] == 1:
        for j in range(3):
            #print('Here')
            I[d11:(h1 + d11), 0:w1, j] = I1
    else:
        I[d11:(h1 + d11), 0:w1] = I1[0:h1, 0:w1]

    if len(I2.shape) == 2 or I2.shape[2] == 1:
        for j in range(3):
            #print('Here')
            I[d21:(h2 + d21), (w1):(w1 + w2), j] = I2
    else:
        I[d21:(h2 + d21), (w1):(w1 + w2)] = I2[0:h2, 0:w2]

    plt.imshow(I)

    plt.plot([w1, w1], [0, h], color='#eeefff', linewidth=1.0)

    plt.plot(Pt1[:, 1], Pt1[:, 0] + d11, 'ro', markersize=3)
    plt.plot(Pt2[:, 1] + w1, Pt2[:, 0] + d21, 'ro', markersize=3)

    for i in range(matches.shape[0]):
        if (matches[i] == gTruth[i] and gTruth[i] < NofInliers):
            plt.plot([Pt1[i, 1], Pt2[matches[i], 1] + w1],
                     [Pt1[i, 0] + d11, Pt2[matches[i], 0] + d21], 'g')
        elif ((gTruth[i] == -1 and matches[i] not in gTruth)
              or (gTruth[i] >= NofInliers and matches[i] >= NofInliers)):
            plt.plot([Pt1[i, 1], Pt2[matches[i], 1] + w1],
                     [Pt1[i, 0] + d11, Pt2[matches[i], 0] + d21], 'y')
        else:
            plt.plot([Pt1[i, 1], Pt2[matches[i], 1] + w1],
                     [Pt1[i, 0] + d11, Pt2[matches[i], 0] + d21], 'm')

    plt.xlim([0, w1 + w2])
    plt.ylim([h, 0])
    plt.draw()
    plt.savefig(StoreFileName)
    # plt.show()
    # plt.close()


def drawMatches(I1, I2, Pt1, Pt2, G1, G2, matches, IsCorrect):
    matplotlib.use('AGG')
    h1 = I1.shape[0]
    w1 = I1.shape[1]

    h2 = I2.shape[0]
    w2 = I2.shape[1]

    fig = plt.figure(1)

    h = max(h1, h2)

    diff1 = h - h1
    diff2 = h - h2

    d11 = int(diff1 * 0.5)
    d12 = diff1 - d11

    d21 = int(diff2 * 0.5)
    d22 = diff2 - d21

    I = np.zeros([h, w1 + w2, 3])
    if len(I1.shape) == 2 or I1.shape[2] == 1:
        for j in range(3):
            I[d11:(h1 + d11), 0:w1, j] = I1
    else:
        I[d11:(h1 + d12), 0:w1, :] = I1

    if len(I1.shape) == 2 or I2.shape[2] == 1:
        for j in range(3):
            I[d21:(h2 + d21), (w1):(w1 + w2), j] = I2
    else:
        I[d21:(h2 + d22), (w1):(w1 + w2), :] = I2

    plt.imshow(I)

    plt.plot([w1, w1], [0, h], color='#eeefff', linewidth=2.0)

    plt.plot(Pt1[:, 1], Pt1[:, 0] + d11, 'ro')
    plt.plot(Pt2[:, 1] + w1, Pt2[:, 0] + d21, 'ro')

    for i in range(len(IsCorrect)):
        if (IsCorrect[i] == 1):
            plt.plot([Pt1[i, 1], Pt2[matches[i], 1] + w1],
                     [Pt1[i, 0] + d11, Pt2[matches[i], 0] + d21], 'g')
        else:
            plt.plot([Pt1[i, 1], Pt2[matches[i], 1] + w1],
                     [Pt1[i, 0] + d11, Pt2[matches[i], 0] + d21], 'y')

    plt.xlim([0, w1 + w2])
    plt.ylim([h, 0])
    plt.draw()
