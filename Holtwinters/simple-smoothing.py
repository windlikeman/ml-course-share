import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import SimpleExpSmoothing

# 解决中文乱码
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['font.size'] = 10
plt.rcParams['axes.unicode_minus']=False


# 简单的指数平滑法
def ses():
    number = 30
    x1 = np.round(np.linspace(0, 1, number), 4)
    y1 = pd.Series(np.multiply(x1, (x1 - 0.5)) + np.random.randn(number))

    print(x1)
    # fitted部分是直线或者是曲线，受到原始数据影响。
    # 多次测试显示，直线的概率高。
    # ets1 = SimpleExpSmoothing(endog=y1, initialization_method='estimated')
    ets1 = SimpleExpSmoothing(endog=y1, initialization_method='heuristic')
    r1 = ets1.fit()
    pred1 = r1.predict(start=len(y1), end=len(y1) + len(y1)//2)

    pd.DataFrame({
        'origin': y1,
        'fitted': r1.fittedvalues,
        '预测值': pred1
    }).plot()
    plt.show()

ses()
