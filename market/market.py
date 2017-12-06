# -*- coding: utf-8 -*-
# -*- author: nik -*-
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tushare as ts

import sklearn

#获得历史数据 江南嘉捷
data = pd.read_csv("hist_data.csv")

#data = data.sort_values(by='date',ascending=True)
print(data.info())
print(data.head())
print(data.head(-5))
# print(data.shape)

ax1 = plt.subplot(311)
t1 = np.arange(630)

#reverse the data
plt.plot(t1, data.open[::-1], label="open")
plt.plot(t1, data.high[::-1], label="max")
plt.plot(t1, data.close[::-1], label="close")
plt.plot(t1, data.low[::-1], label="min")
#plt.bar(t1,data.volume,label="volume")

#plt.xlim(1,10)
leg = plt.legend(loc='upper left', ncol=4, mode="expand", shadow=True, fancybox=True)
leg.get_frame().set_alpha(0.5)
ax2 = plt.subplot(312)
plt.bar(t1,data.volume[::-1],label="volume")

ax3 = plt.subplot(313)
plt.scatter(data.open[::-1],data.close[::-1])
plt.show()



