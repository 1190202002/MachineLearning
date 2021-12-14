'''
Author: li yifeng
Date: 2021-10-17 23:19:31
LastEditors: li yifeng
LastEditTime: 2021-11-23 23:07:15
Description: 
'''
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


def faces_pca(path, k_list):
    """
    path: 文件路径
    k_list: 降到的不同维度
    """
    x_list = faces(path)
    for x in x_list:
        x_pca_list = []
        x_psnr_list = []
        for k in k_list:
            x_cen, mu, vectors = pca(x, k)  # PCA降维
            x_pca = x_cen @ vectors.T @ vectors + mu  # 重建数据
            x_pca_list.append(x_pca)
            x_psnr_list.append(psnr(x, x_pca))
        show_faces(x, x_pca_list, k_list, x_psnr_list)


def faces(path):
    """
    读取指定目录下的所有文件
    """
    file_list = os.listdir(path)
    x_list = []
    for file in file_list:
        file_path = os.path.join(path, file)
        pic = Image.open(file_path).convert('RGBA')  # 读入图片，并将三通道转换为灰度图
        x_list.append(np.asarray(pic))
    return x_list


def psnr(source, target):
    """
    计算峰值信噪比
    """
    rmse = np.sqrt(np.mean((source - target) ** 2))
    return 20 * np.log10(255.0 / rmse)


def show_faces(x, x_pca_list, k_list, x_psnr_list):
    """
    在一行展示降维后的结果
    """
    plt.figure(figsize=(12, 8),  frameon=False)
    size = np.ceil((len(k_list) + 1) / 2)
    plt.subplot(2, size, 1)
    plt.title('Real Image')
    plt.imshow(x)
    plt.axis('off')  # 去掉坐标轴
    for i in range(len(k_list)):
        plt.subplot(2, size, i+2)
        plt.title('k = ' + str(k_list[i]) +
                  ', PSNR = ' + '{:.2f}'.format(x_psnr_list[i]))
        plt.imshow(x_pca_list[i])
        plt.axis('off')
    plt.show()
