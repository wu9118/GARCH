import sys
import numpy as np
from numpy import cumsum, log, polyfit, sqrt, std, subtract
from numpy.random import randn
import pandas as pd
from pandas_datareader import data as web
import seaborn as sns
from pylab import rcParams
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from arch import arch_model
from numpy.linalg import LinAlgError
from scipy import stats
import statsmodels.api as sm
import statsmodels.tsa.api as tsa
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import acf, q_stat, adfuller
from sklearn.metrics import mean_squared_error
from scipy.stats import probplot, moment
from arch import arch_model
from arch.univariate import ConstantMean, GARCH, Normal
from sklearn.model_selection import TimeSeriesSplit
import warnings

#matplotlib inline
pd.set_option('display.max_columns', None)
warnings.filterwarnings('ignore')
sns.set(style="darkgrid", color_codes=True)
rcParams['figure.figsize'] = 8,4

# 读取数据,观察原始数据，做简单处理

data0 = pd.read_csv('BTCUSDT-trades-2021-12-03.csv', header=None)

# 添加列名
data0.columns = ['trade Id','price','qty','quoteQty','timestamp','isBuyerMaker']


# 将时间戳（1637971200004）转换为2021-12-03-00-00-000这种格式，单位为ms
# 并且新增一列 ，列名'time',
data0['time']=pd.to_datetime(data0['timestamp'],unit='ms')
# ‘time'列设为索引列
data0.set_index('time', inplace=True)

#print(data0)

start = pd.Timestamp('2021-12-03 00:00:00.010')
end = pd.Timestamp('2021-12-03 00:00:05.400 ')
data=data0[start:end]
#print(data0[start:end])

data_price=data.groupby('time')['price'].last()
#print(data_price)

time_space='1ms'
time_jump='100ms'
jump_num=100
vol2_retn_fill=[]
price_fill=data_price.resample(time_space).ffill().dropna()     # resample 默认从0点开始
#print(price_fill)  # 0.010、0.011、。。。。5.400

price_jump=price_fill[::jump_num]
#print(price_jump)  # 0.010 、0.110  、。。。5.210 ，5.310
#计算收益率
retn_price_fill=price_fill.pct_change().fillna(0)*1e+05
#print(retn_price_fill)   # 0.010、0.011。。。5.400
retn_price_jump=price_jump.pct_change().fillna(0)*1e+05
#print(retn_price_jump)  # 0.010 、0.110  、。。。5.210 ，5.310
# rv
for j in range(len(retn_price_jump) - 1):  # 100
    j = jump_num * j
    vol2_retn_fill = np.append(vol2_retn_fill, (retn_price_fill ** 2)[j:j + jump_num + 1].sum())
vol_retn_fill = np.sqrt(vol2_retn_fill)     # [010-110]、[110-210]...[5.210-5.310]

# Specify GARCH model assumptions
basic_gm = arch_model(retn_price_jump, p = 1, q = 1,
                      mean = 'zero', vol = 'GARCH',  dist = 't')
gjr_gm = arch_model(retn_price_jump, p = 1, q = 1, o = 1, vol = 'GARCH', dist = 't')
egarch_gm = arch_model(retn_price_jump, p = 1, q = 1, o = 1, vol = 'EGARCH', dist = 't')
# Fit the model
gm_result = basic_gm.fit(update_freq = 0)
gjrgm_result = gjr_gm.fit(disp = 'off')
egarch_result = egarch_gm.fit(disp = 'off')

# Display model fitting summary
print(gm_result.summary())
print(gjrgm_result.summary())
print(egarch_result.summary())
# Get model estimated volatility
gm_vol = gm_result.conditional_volatility


def evaluate(observation, forecast):
    # Call sklearn function to calculate MAE
    mae = mean_absolute_error(observation, forecast)
    print(f'Mean Absolute Error (MAE): {round(mae,3)}')
    # Call sklearn function to calculate MSE
    mse = mean_squared_error(observation, forecast)
    print(f'Mean Squared Error (MSE): {round(mse,3)}')
    return mae, mse