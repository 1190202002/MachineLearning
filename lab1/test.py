import numpy as np
import math
import matplotlib.pyplot as plt

# 全局变量
m = 9     # 拟合阶数
data_size = 30  # 训练集规模
data_range = math.pi * 2  # x的范围[0,data_range]
lambd = 0.01  # 超参数,用于惩罚项中
learning_rate = 0.0001   # 学习率, 用于梯度下降
method_name = ''  # 选择方法

   # 主函数

def main():

        # 生成数据
        x0, y = generate_data()
        print(x0)
        print("\n")
        print(y)
        x = Vandermonde_x(x0)
        print(x)

    # 以下是四种方法
        global method_name
        method_name_set = ['analytical_solution', 'analytical_solution_reg',
                           'gradient_descent', 'conjugate_gradient']
        method_name = method_name_set[2]  # 从列表中选择方法, 选2时较慢(梯度下降)

        if method_name == 'analytical_solution':
            w = analytical_solution(x, y)
        elif method_name == 'analytical_solution_reg':
            w = analytical_solution_reg(x, y)
        elif method_name == 'gradient_descent':
            w = gradient_descent(x, y, learning_rate)
        else:
            w = conjugate_gradient(x, y)

        test(x0, y, w)  # 测试w的拟合情况

    # 以sin(x)生成数据,附带噪声

def generate_data():
    x0 = np.random.random((data_size, 1)) * data_range
    x0 = np.sort(x0, 0)  # 按列排序, 便于画图
    loss = np.random.randn(data_size, 1) * 0.1  # 以标准的正态分布*0.1产生噪音
    y = np.sin(x0) + loss
    return np.matrix(x0), np.matrix(y)

    # 生成范德蒙德行列式
def Vandermonde_x(x0):
    x = np.ones((data_size, 1))
    for i in range(1, m):
      x = np.hstack((x, np.power(x0, i)))
    return np.matrix(x)

    # 解析解（不带惩罚项）
def analytical_solution(x, y):
    w = (x.T * x).I * x.T * y
    return w

    # 解析解（带惩罚项）
def analytical_solution_reg(x, y):
    w = (x.T * x + lambd * np.eye(m)).I * x.T * y
    return w

    # 梯度函数

def gradient_function(x, y, w):
    return (1.0/m) * (x.T * x * w - x.T * y + 0.001 * w)

    # 损失函数 loss
def loss_function(x, y, w):
    diff = x * w - y
    loss = 1.0/(2*data_size)*(diff.T * diff)
    return loss[0, 0]

    # 梯度下降法
def gradient_descent(x, y, learning_rate):
    w = np.zeros((m, 1))
    gradient = gradient_function(x, y, w)
    loss0 = 0
    loss1 = loss_function(x, y, w)
    count = 0
    xw = np.linspace(0, 2*math.pi, 40)
    while abs(loss1 - loss0) > 1e-8:  # 当两次loss相差不大时, 表示接近极值点了.
      count += 1
      w = w - learning_rate * gradient  # 注意此处
      loss0 = loss1
      loss1 = loss_function(x, y, w)
    # 若发现迭代过程中loss变大， 则减小学习率
      if np.all(loss1-loss0 > 0):
         learning_rate *= 0.5
         gradient = gradient_function(x, y, w)

    # 每1000次迭代绘一次图
      if count > 1000:
        print(loss1-loss0)
        count -= 1000
        yw = np.polyval(w[::-1], xw)
        plt.scatter(x[:, 1].tolist(), y.tolist(), edgecolor="b")
        plt.plot(xw, yw, 'r')
        plt.pause(0.001)
        plt.clf()
    return w

    # 共轭梯度法

def conjugate_gradient(x, y):
    lambd = 0.001
    Q = (1 / m) * (x.T * x + lambd * np.mat(np.eye(x.shape[1])))
    w = np.mat(np.zeros(x.shape[1])).T
    r = -gradient_function(x, y, w)
    p = r
    count = 0
    for i in range(1, x.shape[1]):
        count += 1
        a = float((r.T * r) / (p.T * Q * p))
        r_prev = r
        w = w + a * p
        r = r - a * Q * p
        p = r + float((r.T * r) / (r_prev.T * r_prev)) * p
    return w

    # 测试效果
def test(x0, y, w):
        # 画出给定点的散点图
    plt.scatter(x0.tolist(), y.tolist(), edgecolor="b", label="training data")
    # 画出结果的模拟曲线
    xw = np.linspace(0, 2*math.pi, 40)
    yw = np.polyval(w[::-1], xw)
    plt.plot(xw, np.sin(xw), 'y', label='sin(x)')
    info = 'm = ' + str(w.shape[0])+'   training_data_size = '+str(data_size)
    plt.plot(xw, yw, 'r', label=method_name)
    plt.legend()
    plt.title(info)
    plt.show()

if __name__ == '__main__':
    main()
