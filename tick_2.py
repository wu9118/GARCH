import sys
import math
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

start = pd.Timestamp('2021-12-03 09:40:33.866')
end = pd.Timestamp('2021-12-03 13:36:16.610 ')  #  当前时刻
data=data0[start:end]
#print(data0[start:end])

data_price=data.groupby('time')['price'].last()
#print(data_price)

time_space='1000ms'
time_jump='10ms'
jump_num=10
window_size=460
pre_num=100

gm_pre=[]
vol2_retn_fill=[]
vol_gm_pre=[]
gm_pre_2=[]


price_fill=data_price.resample(time_space).ffill().dropna()     # resample 默认从0点开始
#print(price_fill)

price_jump=price_fill[::jump_num]
#print(price_jump)
#计算收益率
retn_price_fill=price_fill.pct_change().fillna(0)*1e+05
#print(retn_price_fill)
retn_price_jump=price_jump.pct_change().fillna(0)*1e+05
#print(retn_price_jump)
# rv
for j in range(len(retn_price_jump) - 1):  # 100
    j = jump_num * j
    vol2_retn_fill = np.append(vol2_retn_fill, (retn_price_fill ** 2)[j:j + jump_num + 1].sum())
    #print((retn_price_fill ** 2)[j:j + jump_num + 1])
vol_retn_fill = np.sqrt(vol2_retn_fill)


for i in range(pre_num):

    retn_fit=retn_price_jump[i:window_size+i] # 1s的滑动
    print(retn_fit)
    basic_gm = arch_model( retn_fit, p=1, q=1,o = 1,
                         mean='zero', vol='EGARCH', dist='t')
    gm_result = basic_gm.fit(update_freq = 0)
    gm_vol2 = gm_result.conditional_volatility
    gm_vol =  np.sqrt(gm_vol2)
    gm_for2 = gm_result.forecast()
    gm_forecast2 = gm_for2.variance[-1:]
    gm_pre_2=np.append( gm_pre_2, gm_forecast2['h.1'])
    #gm_pre=np.append(gm_pre,np.sqrt(gm_forecast2['h.1']))

   # # 调整预测值
   #  if gm_pre[-1] > np.max(gm_vol):
   #      gm_pre[-1]  = np.max(gm_vol)
   #  if gm_pre[-1] .item()< np.min(gm_vol):
   #      gm_pre[-1]  = np.min(gm_vol)
   #  if gm_pre[-1] .item()> gm_vol[-1] * 2:
   #      gm_pre[-1]  = gm_vol[-1] * 2
    #vol_gm_pre = np.append( vol_gm_pre , gm_pre[-1])

vol_retn_fill_pre=vol_retn_fill[window_size:window_size+pre_num]
gm_pre=np.sqrt(gm_pre_2)


def evaluate(observation, forecast):
    # Call sklearn function to calculate MAE
    mae = mean_absolute_error(observation, forecast)
    print(f'Mean Absolute Error (MAE): {round(mae,3)}')
    # Call sklearn function to calculate MSE
    mse = mean_squared_error(observation, forecast)
    print(f'Mean Squared Error (MSE): {round(mse,3)}')
    return mae, mse

print('vol_gm_pre',  gm_pre)
print('vol_retn_fill_pre',vol_retn_fill_pre)

evaluate(vol_retn_fill_pre, gm_pre)

x=range(pre_num)
plt.plot(x,gm_pre,label='con_vol')
plt.plot(x,vol_retn_fill_pre,label='his_rv')
# plt.plot(res.conditional_volatility,label='conditional_volatility')
plt.legend(loc=0)
plt.show()
