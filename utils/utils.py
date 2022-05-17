import os
from numpy import dtype
import paddle
from  paddle.callbacks import Callback
from paddle.distributed import ParallelEnv
import matplotlib.pyplot as plt
import paddle.nn as nn
import numpy as np

def get_classes(path):
    """
    path: 数据集路径
    返回种类数量以及种类
    """
    return len(os.listdir(path=path)),os.listdir(path=path)

def show_img(img, predict, label=None):
    plt.figure()
    if label:
        plt.title('predict: {}, real_label: {}'.format(predict, label))
    else:
        plt.title('predict: {}'.format(predict))
    img = np.array(img)
    if img.shape[0] == 3:
        img = np.transpose(img, [1,2,0])
    plt.imshow(img)
    plt.show()
    


def get_total_images(path):
    """
    path: 数据集路径
    返回数据集数量
    """
    with open(path, mode='r', encoding='utf-8') as f:
        lines = f.readlines()
    return len(lines)