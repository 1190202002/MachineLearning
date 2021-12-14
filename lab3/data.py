import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.shape_base import column_stack
from numpy.ma.core import count
import pandas as pd
from pandas.core.reshape.melt import wide_to_long
from scipy.stats import multivariate_normal
from itertools import permutations


def generate_data(k, n, dim, mu_list, sigma_list):
    '''
    description: 生成一个包含k类高斯分布的集合，每类高斯分布包含n个点
    param {
       k：高斯分布的类数
       n:每个高斯分布包含的样本数
       dimsion:维度
       mu_list:每个高斯分布的均值 k*dimsion
       sigma_list：每个高斯分布的方差 k*dimsion*dimsion
    }
    return {
        X:一个包含k类高斯分布的集合，最后一列为正确的分类标签
    }
    '''    
    X = np.zeros((n * k, dim + 1))
    for i in range(k):
        X[i * n : (i + 1) * n, :dim] = np.random.multivariate_normal(
            mu_list[i], sigma_list[i], size=n
        )
        X[i * n : (i + 1) * n, dim : dim + 1] = i
    return X


def kmeans(X, k, epsilon):
    '''
    description: 
    param {
        X:一个包含k类高斯分布的集合，最后一列为正确的分类标签
        k：高斯分布的类数
        epsilon：精度
    }
    return {
        X:高斯分布的集合分类结果
        center：每个类中心点
        iterations：循环次数
    }
    '''    
    center = np.zeros((k, X.shape[1] - 1))
    for i in range(k):
        center[i, :] = X[np.random.randint(0, high=X.shape[0]), :-1]  # 初始化随机选取k个中心点
    iterations = 0
    while True:
        iterations += 1
        distance = np.zeros(k)
        # 根据中心重新给每个点贴分类标签
        for i in range(X.shape[0]):
            for j in range(k):
                distance[j] = np.linalg.norm(X[i, :-1] - center[j, :])  # 点到每个中心的距离
            X[i, -1] = np.argmin(distance)
        # 根据每个点新的标签计算它的中心
        new_center = np.zeros((k, X.shape[1] - 1))
        sum = np.zeros(k)
        for i in range(X.shape[0]):
            new_center[int(X[i, -1]), :] += X[i, :-1]  # 对每个类的所有点坐标求和
            sum[int(X[i, -1])] += 1
        for i in range(k):
            new_center[i, :] = new_center[i, :] / sum[i]  # 对每个类的所有点坐标求平均值
        if np.linalg.norm(new_center - center) < epsilon:  # 如果两个中心的距离
            break
        else:
            center = new_center
    return X, center, iterations


def e_step(x, mu_list, sigma_list, pi_list):
    """
    description:
    param {
       x：一个包含k类高斯分布的集合
       mu_list:每个高斯分布的均值 k*dimsion
       sigma_list：每个高斯分布的方差 k*dimsion*dimsion 
       pi_list：每一个类中样本数所占比例
    }
    return {
        gamma_z：各个样本由各个高斯分布生成的后验概率
    }
    """
    k = mu_list.shape[0]
    gamma_z = np.zeros((x.shape[0], k))
    for i in range(x.shape[0]):
        pipdf_sum = 0
        pipdf = np.zeros(k)
        for j in range(k):
            pipdf[j] = pi_list[j] * multivariate_normal.pdf(
                x[i], mean=mu_list[j], cov=sigma_list[j]
            )
            pipdf_sum += pipdf[j]
        for j in range(k):
            gamma_z[i, j] = pipdf[j] / pipdf_sum
    return gamma_z


def m_step(x, mu_list, gamma_z):
    """
    description:根据样本点在各个模型下的期望，更新模型，使模型逐渐拟合
    param {*}
    return {*}
    """
    k = mu_list.shape[0]
    mu_list_new = np.zeros(mu_list.shape)
    sigma_list_new = np.zeros((k, x.shape[1], x.shape[1]))
    pi_list_new = np.zeros(k)
    for j in range(k):
        n_j = np.sum(gamma_z[:, j])
        pi_list_new[j] = n_j / x.shape[0] # 计算新的pi

        gamma = gamma_z[:, j]
        gamma = gamma.reshape(x.shape[0], 1)
        mu_list_new[j, :] = (gamma.T @ x) / n_j  # 计算新的mu
        sigma_list_new[j] = (
            (x - mu_list[j]).T @ np.multiply((x - mu_list[j]), gamma)
        ) / n_j  # 计算新的sigma
    return mu_list_new, sigma_list_new, pi_list_new


def gmm(X, k, epsilon=1e-5):
    '''
    description: GMM算法,随机初始化，e求期望，m更新模型，直至两次中心点相差在精度内
    param {*}
    return {*}
    '''
    x = X[:, :-1]
    pi_list = np.ones(k) * (1.0 / k)
    sigma_list = np.array([0.1 * np.eye(x.shape[1])] * k)
    # 随机选第1个初始点，依次选择与当前mu中样本点距离最大的点作为初始簇中心点
    mu_list0=np.zeros((k,x.shape[1]))
    for i in range(k):
        mu_list0[i,:]=x[np.random.randint(0,x.shape[0]),:]
    iterations = 0
    while True:
        gamma_z = e_step(x, mu_list0, sigma_list, pi_list)
        mu_list1, sigma_list, pi_list = m_step(x, mu_list0, gamma_z)
        if np.linalg.norm(mu_list1-mu_list0) < epsilon:
            break
        mu_list0=mu_list1
        iterations += 1
    # 计算标签
    for i in range(X.shape[0]):
        X[i, -1] = np.argmax(gamma_z[i, :])
    return X, iterations,mu_list0


def accuracy(real_lable, train_lable, k):

    '''
    description: 计算准确率，真实聚类标签与生成的比对，算出正确率
    param {*}
    return {*}
    '''    
    classes = list(permutations(range(k), k))
    sums = np.zeros(len(classes))
    for i in range(len(classes)):
        for j in range(real_lable.shape[0]):
            if int(real_lable[j]) == classes[i][int(train_lable[j])]:
                sums[i] += 1
    return np.max(sums) / real_lable.shape[0]


def show(X, center=None, title=None):
    '''
    description: 显示图像，各点只显示前两维，中心点特殊标注
    param {*}
    return {*}
    '''    
    plt.style.use("seaborn")
    plt.scatter(X[:, 0], X[:, 1], c=X[:, -1], marker=".", cmap="Set1")
    if not center is None:
        plt.scatter(center[:, 0], center[:, 1], c="b", marker="*",s=100)
    if not title is None:
        plt.title(title)
    plt.show()


def uci_iris():
    """
    对uci数据集聚类
    """
    se=[]
    with open("ucidata.csv",'r') as f:
        lines=f.readlines()
        for line in lines:
            se.append(line.strip("\n").split(","))
    X=np.zeros((len(se),len(se[0])))
    for i in range(X.shape[0]):
        for j in range(X.shape[1]-1):
            X[i,j]=float(se[i][j])
    for i in range(X.shape[0]):
        X[i,-1]=int(se[i][-1])
    return X