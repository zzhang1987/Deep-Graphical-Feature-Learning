import numpy as np
import scipy.misc as misc
import os

def load_cmu_house(data_path):
    res = np.zeros([111, 30, 2])
    label_path = os.path.join(data_path, 'label')
    img_path = os.path.join(data_path, 'images')
    Imgs = {}
    for i in range(0, 111):
        res[i] = np.loadtxt('{}/house{}'.format(label_path, i + 1) )
        Imgs[i] = misc.imread('{}/house.seq{}.png'.format(img_path, i))
    return res, Imgs
