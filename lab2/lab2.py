import numpy as np
from numpy import matlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
pos_num = 50    #正例个数
pos_mean = [1, 2]    #正例均值，二维
pos_cov = neg_cov = np.mat([[0.3, 0], [0, 0.4]])      #满足朴素贝叶斯正反例协方差
fpos_cov = fneg_cov = np.mat([[0.3, 0.3], [0.3, 0.4]])    #不满足朴素贝叶斯正反例协方差
neg_num = 50         #反例个数
neg_mean = [-1, -2]    #反例均值，二维
lam = np.exp(-10)      #惩罚项
cur = 1e-5            #精度
learn_rate = 0.01       #学习率
# 生成训练集x，根据均值与方差随机生成正例反例的坐标，在y中进行标记
def generate_f(pos_num, pos_mean, pos_cov, neg_num, neg_mean, neg_cov):
    x = np.zeros((pos_num+neg_num, 2))
    y = np.zeros(pos_num+neg_num)
    x[:pos_num, :] = np.random.multivariate_normal(
        pos_mean, pos_cov, size=pos_num)          #前面放由均值和协方差生成的正例
    x[pos_num:, :] = np.random.multivariate_normal(
        neg_mean, neg_cov, size=neg_num)           #后面放由均值和协方差生成的反例
    y[:pos_num] = 1          #前面正例为1，后面反例为0
    return x, y


def sig(x):
    return 1/(1+np.exp(-x))        

# 损失函数
def loss(X, y, w, lam):
    YWX = 0
    ln = 0
    for i in range(X.shape[0]):
        YWX += y[i]*w@X[i].T
        ln += np.log(1 + np.exp(w@X[i].T))
    loss = -YWX+ln+(lam*w@w.T)/2
    return loss/(X.shape[1])

# 梯度下降法求参数
def diff(x, y, learn_rate, lam):
    X = np.concatenate((np.ones((x.shape[0], 1)), x), 1)  # 扩增x一个维度为X，第一维均为1
    w1 = np.ones((1, X.shape[1]))  # 扩增w与X维度相同
    loss1 = loss(X, y, w1, lam)
    for i in range(100000):
        loss0 = loss1
        t = np.zeros((X.shape[0], 1))
        t = X@w1.T
        grad = - (y - sig(t.T)) @ X / X.shape[0]  # 计算下降梯度
        w0 = w1
        w1 = w1 - learn_rate * lam * w1 - learn_rate * grad    # 根据下降梯度找到沿此梯度方向的下个值
        loss1 = loss(X, y, w1, lam)
        if abs(loss0) < abs(loss1):  # 如果损失函数增大了，降低学习率重新找
            w1 = w0
            learn_rate /= 2
            loss1 = loss(X, y, w1, lam)
        if abs(loss0-loss1) < cur:  # 如果几乎不再下降，则停止
            break
    print(i)
    w1 = w1.reshape(X.shape[1])
    coef = -(w1 / w1[X.shape[1]-1])[0:X.shape[1]-1]  # 对w做归一化，得到方程系数
    print(w1)
    print(coef)
    return coef, w1
# def hess(x, w):
    hess = matlib.zeros((x.shape[1], x.shape[1]))
    for i in range(pos_num+neg_num):
        hess += x[i].T * x[i] * sig(w.T*x[i].T) * (1 - sig(w.T*x[i].T))
    hess += lam*matlib.eye((x.shape[1]))
    return hess.I
# def newton(x, y):
    w = matlib.zeros((x.shape[1], 1))
    grad = diff(x, y, w)
    w = w-hess(x, w)*grad
    while np.linalg.norm(grad) > cur:
        grad = diff(x, y, w)
        w = w-hess(x, w)*grad
    return w

#画出二维散点图和分类函数线
def plt2d(x, y, discriminant, title):
    print('Discriminant function: y = ', discriminant)
    plt.scatter(x[:, 0], x[:, 1], c=y, marker='o', cmap=plt.cm.Spectral)
    res_x = np.linspace(min(x[:, 0]), max(x[:, 0]), x.shape[0])
    res_y = discriminant(res_x)
    plt.plot(res_x, res_y, 'r', label='discriminant')
    plt.title(title)
    plt.show()
#画出三维散点图和分类函数面
def plt3d(x, y, coef, title):
    d3 = Axes3D(plt.figure())
    d3.scatter(x[:, 0], x[:, 1], x[:, 2], c=y, cmap=plt.cm.Spectral)
    res_x = np.linspace(np.min(x[:, 0])-20, np.max(x[:, 0])+20, 200)
    res_y = np.linspace(np.min(x[:, 1])-20, np.max(x[:, 1])+20, 200)
    res_X, res_Y = np.meshgrid(res_x, res_y)
    res_z = coef[0] + coef[1] * res_X + coef[2] * res_Y
    d3.plot_surface(res_x, res_y, res_z)
    d3.set_title(title)
    plt.show()

#进行二维测试
def test(lam, pos_cov, neg_cov):
    x, y = generate_f(pos_num, pos_mean, pos_cov, neg_num,neg_mean, neg_cov)  # 生成训练数据集
    coef, w = diff(x, y, learn_rate, lam)           #计算出方程系数
    discriminant = np.poly1d(coef[::-1])
    print('Train data accuracy:', fit(x, y, w))
    plt2d(x, y, discriminant, 'Train data')      #画出训练集二维散点图和分类函数线
    test_x, test_y = generate_f(pos_num*4, pos_mean, pos_cov, neg_num*4, neg_mean, neg_cov)  # 生成测试数据集
    print('Test data accuracy:', fit(test_x, test_y, w))
    plt2d(test_x, test_y, discriminant, 'Test data')     #画出测试集二维散点图和分类函数线

# 从文件读取uci数据，分离训练数据集和测试数据集
def uciread(path):
    data = np.loadtxt(path, dtype=np.int32)
    np.random.shuffle(data)  # 随机打乱数据，便于选取数据
    dim = data.shape[1]
    if data.shape[0] < 5000:
        train_size = int(0.6 * data.shape[0])  # 按照6：4的比例分配训练集和测试集
        test_size = int(0.4 * data.shape[0])
    else:
        train_size = 3000
        test_size = 2000
    x = data[:train_size, 0:dim-1]
    y = data[:train_size, dim-1] - 1
    test_x = data[train_size:train_size+test_size, 0:dim-1]
    test_y = data[train_size:train_size+test_size, dim-1] - 1
    return x, y, test_x, test_y

#进行三维uci数据集测试
def uci_test(path):
    x, y, test_x, test_y = uciread(path)          # 生成训练数据集和测试数据集
    coef, w = diff(x, y, learn_rate, lam)          #计算出方程系数
    print('Train data accuracy:', fit(x, y, w))
    plt3d(x, y, coef, 'Train data set')            #画出训练集三维散点图和分类函数面
    print('Test data accuracy:', fit(test_x, test_y, w))
    plt3d(test_x, test_y, coef, 'Test data set')       #画出测试集三维散点图和分类函数面

#计算训练集和数据集进行相应划分后的正确率
def fit(x, y, w):
    count = 0
    X = np.concatenate((np.ones((x.shape[0], 1)), x), 1)   # 扩增x一个维度为X，第一维均为1
    for i in range(X.shape[0]):
        if w@X[i].T >= 0 and y[i] == 1:
            count += 1
        if w@X[i].T < 0 and y[i] == 0:
            count += 1
    return count / X.shape[0]


def main():
    test(0, pos_cov, neg_cov)  # 满足朴素贝叶斯，无惩罚项
    test(lam,pos_cov,neg_cov)    #满足朴素贝叶斯，有惩罚项
    test(0,fpos_cov,fneg_cov)      #不满足朴素贝叶斯，无惩罚项
    test(lam,fpos_cov,fneg_cov)        #不满足朴素贝叶斯，有惩罚项
    uci_test("lab2\Skin_NonSkin.data")


main()
