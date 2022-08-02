import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import Holt

# 解决中文乱码
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['font.size'] = 10
plt.rcParams['axes.unicode_minus']=False


# Holt 扩展了简单指数平滑，使其可以用来预测带有趋势的时间序列。
def holt():
    # 某种产品最近15个月的销售量
    ydata = [10,
          15,
          8,
          20,
          10,
          18,
          20,
          22,
          24,
          20,
          26,
          27,
          29,
          29]
    y2 = pd.Series(ydata)
    # fitted部分是直线或者是曲线，受到原始数据影响。
    # 多次测试显示，直线的概率高。
    ets2 = Holt(endog=y2, initialization_method='estimated')
    # ets2 = Holt(endog=y2, initialization_method='heuristic')
    # ets2 = Holt(endog=y2, initialization_method='estimated', damped_trend=True)
    r2 = ets2.fit()
    pred2 = r2.predict(start=len(y2), end=len(y2) + len(y2) // 2)

    pd.DataFrame({
        'origin': y2,
        'fitted': r2.fittedvalues,
        '预测值': pred2
    }).plot(legend=True)
    plt.show()


holt()
