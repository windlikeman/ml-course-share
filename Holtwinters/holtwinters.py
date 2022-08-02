import pandas as pd

import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# 解决中文乱码
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['font.size'] = 10
plt.rcParams['axes.unicode_minus']=False


# 某医院科室3年内每个月的耗电量
def holtwinters():

    ydata = [4114.00,
          3861.10,
          3487.60,
          2995.90,
          2875.60,
          2898.90,
          3509.70,
          3375.50,
          2941.80,
          3298.00,
          3328.70,
          3281.80,
          4269.26,
          4034.27,
          3630.43,
          3192.08,
          2891.73,
          3006.27,
          3519.84,
          3553.32,
          3036.50,
          3364.69,
          3438.81,
          3366.85,
          4398.81,
          4083.59,
          3714.07,
          3381.54,
          2941.99,
          3185.68,
          3708.26,
          3673.45,
          3160.81,
          3518.36,
          3594.87,
          3528.12];
    y3 = pd.Series(ydata)
    ets3 = ExponentialSmoothing(y3, trend='add', seasonal='add', seasonal_periods=12)
    # ets3 = ExponentialSmoothing(y3, trend='mul', seasonal='mul', seasonal_periods=12)
    # ets3 = ExponentialSmoothing(y3, trend='mul', seasonal='mul', damped_trend=True, seasonal_periods=12)
    r3 = ets3.fit()
    pred3 = r3.predict(start=len(y3), end=len(y3) + len(y3) // 2)

    print(pred3)
    pd.DataFrame({
        'origin': y3,
        'fitted': r3.fittedvalues,
        '预测值': pred3
    }).plot(legend=True)
    plt.show()


holtwinters()
