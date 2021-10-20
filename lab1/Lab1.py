import numpy as np
from numpy import matlib
import matplotlib.pyplot as plt
import math
from numpy import random
from numpy.ma.core import power
from numpy.matrixlib.defmatrix import matrix
train_size = 30  # 训练数据集大小
order = 3  # 阶数
test_size = 40  # 测试数据集大小
punish_lam = 0.0001  # 惩罚项参数
train_method = ["analysis", "punish_analysis", "gradient", "conjugate"]  # 训练方法
color = ["r", "y", "g", "p"]  # 每个方法画图时对应颜色
learn_rate = 0.01  # 学习率

# 生成训练数据集，x,y


def generate_f():
    x = np.random.rand(train_size, 1)*2*np.math.pi
    y = np.sin(x)
    error = np.random.randn(train_size, 1)*0.1
    y = y+error
    return np.matrix(x), np.matrix(y)

# 生成x的范德蒙矩阵


def Vander(x, data_size, order):
    X = np.ones([data_size, 1])
    for i in range(1, order+1):
        X = np.concatenate((X, np.power(x, i)), 1)
    return X

# 解析法，根据样本点目标值y与预测函数之间的误差公式，计算导数为0时对应的参数w


def analysis(X, y):
    w = (X.T*X).I*X.T*y
    return w

# 带惩罚项的解析法，解析法中加入惩罚参数，防止阶数过大发生过拟合


def punish_analysis(X, y):
    w = ((X.T*X+punish_lam*matlib.eye(order+1)).I*X.T*y)
    return w

# 计算误差公式的梯度
def gradient(X, y, w):
    return (X.T*X*w-X.T*y+punish_lam*w)

# 梯度下降法，速度非常慢
def loss(x, y, w):
     diff = x * w  - y
     loss = 1.0/(2*train_size)*(diff.T * diff+punish_lam*w.T*w)
     return loss[0,0]

def gradient_descent(X, y):
    global learn_rate
    w1 = matlib.zeros([order+1, 1])
    loss1=loss(X,y,w1)
    for i in range(300000):  # 下降梯度接近0时停止，说明达到极值点
        loss0 = loss1
        grad = gradient(X,y,w1)
        w0 = w1
        w1 = w1-learn_rate*grad   # 根据下降梯度找到沿此梯度方向的下个值
        loss1 = loss(X, y, w1)
        if abs(loss0) < abs(loss1):  # 如果损失函数增大了，降低学习率重新找
            learn_rate /= 2
            w1=w0-learn_rate*grad
            loss1 = loss(X, y, w1)
        if abs(loss0-loss1) < 1e-8:  # 如果几乎不再下降，则停止
            break
    print(i)
    return w1

# 共轭梯度法


def conjugate_gradient(X, y):
    w0 = matlib.zeros([order+1, 1])  # 初始化w0
    k = 0
    r0 = -gradient(X, y, w0)  # r0为w0的下降梯度
    p0 = r0
    A = (X.T * X + punish_lam*matlib.eye(order+1))  # 根据最小二乘误差公式得出的特征矩阵
    # while (abs(r0)>1e-1).any():
    for i in range(order*3):
        k += 1
        a = ((r0.T*r0)/(p0.T*A*p0))[0, 0]
        w1 = w0+a*p0
        r1 = r0-a*A*p0
        p1 = r1+((r1.T*r1)/(r0.T*r0))[0, 0]*p0
        w0 = w1
        r0 = r1
        p0 = p1
    print(k)
    return w1

# 画出训练数据集和要拟合的函数


def draw(x, y):
    plt.plot(x, y, 'bo', label="train data")  # 所有训练集的点
    lam_x = np.linspace(0, 2*np.math.pi, 1000)
    plt.plot(lam_x, np.sin(lam_x), 'm', label="sin(x)")  # 要拟合的目标函数
    plt.legend()
    plt.title("train data size:"+str(train_size)+"   "+"order:"+str(order))
    plt.show()

# 根据参数计算预测值


def test_count(w):
    pre_x = np.linspace(0, 2*np.math.pi, test_size).reshape([test_size, 1])
    pre_y = Vander(pre_x, test_size, order)*w  # 计算预测值
    return pre_x, pre_y


def main():
    x, y = generate_f()
    X = Vander(x, train_size, order)
    print("选择一个训练方法\n1：解析式\n2：带惩罚项解析式\n3：梯度下降法\n4：共轭梯度法（选择多种方法以，分隔）")
    methods = input().split(",")
    for method in methods:  # 四种方法可选择，如（1,2）选择1,2两种方法
        if method == "1":
            w = analysis(X, y)
            pre_x, pre_y = test_count(w)
            plt.plot(pre_x, pre_y, 'r', label="analysis")
        elif method == "2":
            w = punish_analysis(X, y)
            pre_x, pre_y = test_count(w)
            plt.plot(pre_x, pre_y, 'y', label="punish_analysis")
        elif method == "3":
            w = gradient_descent(X, y)
            pre_x, pre_y = test_count(w)
            plt.plot(pre_x, pre_y, 'g', label="gradient_descent")
        elif method == "4":
            w = conjugate_gradient(X, y)
            pre_x, pre_y = test_count(w)
            plt.plot(pre_x, pre_y, 'c', label="conjugate_gradient")

    draw(x, y)


main()
