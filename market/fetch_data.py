# -*- coding: utf-8 -*-
# -*- author: nik -*-
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tushare as ts

import sklearn
#获取今天分笔数据
today_ticks = ts.get_today_ticks('601313')
print(today_ticks.info())
print(today_ticks)

#获取实时分笔数据
realtime = ts.get_realtime_quotes('601313')
print(realtime.info())
print(realtime)

#获得实时行情数据
#all = ts.get_today_all()
#print(all.head())

#获得历史数据 江南嘉捷
data = ts.get_hist_data('601313')
#data = data.sort_values(by='date',ascending=True)
print(data.info())
print(data.head())
print(data.head(-5))
data.to_csv("hist_data.csv")
exit(0)
# print(data.shape)

ax1 = plt.subplot(211)
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
ax2 = plt.subplot(212)
plt.bar(t1,data.volume[::-1],label="volume")

plt.show()
