'''
Author: li yifeng
Date: 2021-10-17 23:19:31
LastEditors: li yifeng
LastEditTime: 2021-11-24 16:05:05
Description:
'''
from types import FunctionType
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
from PIL import Image


def pca(x, k):
    '''
    description: PCA主成分分析
    param  x：高维度样本
           k:降维后的维度
    return x_cen:中心化的样本
           mu：样本均值
           vectors：最大的k个特征值对应的特征向量
    '''
    n = x.shape[0]
    mu = np.sum(x, axis=0) / n
    x_cen = x - mu  # 中心化
    cov = (x_cen.T @ x_cen) / n  # 求协方差矩阵
    values, vectors = np.linalg.eig(cov)
    index = np.argsort(values)  # 从小到大排序后的下标序列
    vectors = vectors[:, index[:-(k+1):-1]].T  # 把序列逆向排列然后取前k个，转为行向量
    return x_cen, mu, vectors

# 三维压缩投影


def scatter_pca_3D(x):
    """
    三维降到二维
    """
    x_cen, mu, vectors = pca(x, 2)
    x_pca = x_cen @ vectors.T @ vectors + mu  # 压缩后的投影点
    plt.style.use('seaborn')
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(x[:, 0], x[:, 1], x[:, 2], c="b", label='Origin Data')
    ax.scatter(x_pca[:, 0], x_pca[:, 1], x_pca[:, 2], c='r', label='PCA Data')
    ax.plot_trisurf(x_pca[:, 0], x_pca[:, 1],
                    x_pca[:, 2], color='k', alpha=0.3)
    ax.legend()
    plt.show()
    plt.style.use('default')


def faces(path):
    """
    读取指定目录下的所有文件
    """
    file_list = os.listdir(path)
    data = []
    i = 0
    for file in file_list:
        file_path = os.path.join(path, file)
        pic = Image.open(file_path).convert('L')  # 读入图片转换为灰度图
        pic = np.array(pic)
        h, w = pic.shape
        pic = pic.reshape(h * w)
        data.append(pic)
    return np.array(data).T, h, w


def origin_faces(path):
    data, h, w = faces(path)
    plt.figure(dpi=50, figsize=(20, 5*data.shape[1]/4))
    plt.suptitle('origin faces picture,dimision='+str(data.shape[1]))
    plt.suptitle()
    for i in range(data.shape[1]):
        plt.subplot((data.shape[1]/4)+1, 4, i+1)
        plt.axis('off')
        plt.imshow(data[:, i].reshape(h, w))
    plt.show()


def faces_pca(path, k):
    """
    path: 文件路径
    """
    data, h, w = faces(path)
    x_cen, mu, vectors = pca(data, k)  # PCA降维
    x_pca = x_cen @ vectors.T @ vectors + mu  # 重建数据
    plt.figure(dpi=50, figsize=(20, 5*data.shape[1]/4))
    plt.suptitle('reduce dimision,dimision='+str(k))
    for i in range(x_pca.shape[1]):
        plt.subplot((data.shape[1]/4)+1, 4, i + 1)
        plt.axis('off')
        plt.title('psnrvalue:' +
                  '{:.2f}'.format(psnr(data[:, i], x_pca[:, i])))
        plt.imshow(x_pca[:, i].reshape(h, w))
    plt.show()


def psnr(source, target):
    """
    计算峰值信噪比
    """
    rmse = np.sqrt(np.mean((source - target) ** 2))
    return 20 * np.log10(255.0 / rmse)
