import numpy as np
import matplotlib.pyplot as plt


# ---------------1. 准备数据----------
data = np.array([[1, 14],
                 [3, 24],
                 [2, 18],
                 [1, 17],
                 [3, 27]])

# 提取data中的两列数据，分别作为x，y
x = data[:, 0]
y = data[:, 1]


# 用plt画出散点图
# plt.scatter(x, y)
# plt.show()

# -----------2. 定义损失函数 均方误差 ------------------
# 损失函数是系数的函数，另外还要传入数据的x，y
def compute_cost(w, b, points):
    total_cost = 0
    M = len(points)

    # 逐点计算平方损失误差，然后求平均数
    for i in range(M):
        x = points[i, 0]
        y = points[i, 1]
        total_cost += (y - w * x - b) ** 2

    return total_cost / M


# ------------3.定义算法拟合函数-----------------
# 先定义一个求均值的函数
def average(data):
    sum = 0
    num = len(data)
    for i in range(num):
        sum += data[i]
    return sum / num


# 定义核心拟合函数
def fit(points):
    M = len(points)
    x_bar = average(points[:, 0])

    sum_yx = 0
    sum_x2 = 0
    sum_delta = 0

    for i in range(M):
        x = points[i, 0]
        y = points[i, 1]
        sum_yx += y * (x - x_bar)
        sum_x2 += x ** 2
    # 根据公式计算w
    w = sum_yx / (sum_x2 - M * (x_bar ** 2))

    for i in range(M):
        x = points[i, 0]
        y = points[i, 1]
        sum_delta += (y - w * x)
    b = sum_delta / M

    return w, b


# ------------4. 测试------------------
w, b = fit(data)

print("w is: ", w)
print("b is: ", b)

cost = compute_cost(w, b, data)

print("cost is: ", cost)

# ---------5. 画出拟合曲线------------
plt.scatter(x, y)
# 针对每一个x，计算出预测的y值
pred_y = w * x + b

plt.plot(x, pred_y, c='r')
plt.show()