# -*- coding: utf-8 -*-
# -*- author: nik -*-
import matplotlib.pyplot as plt
import numpy as np
import tushare as ts

class FetchData:

    def __init__(self, code):
        self.code = code

    def desc_data(self, data_frame):
        print("\n---------------desc data start ------------------------\n")
        print(data_frame.info())
        print("\n-----------------------------------------------\n")
        print(data_frame.head())
        print("\n-----------------------------------------------\n")
        print(data_frame.tail())

        print("\n----------------desc data end-------------------------\n")
    '''
    #获取今天分笔数据
    '''
    def get_today_ticks(self):
        #获取今天分笔数据
        today_ticks = ts.get_today_ticks(self.code)
        self.desc_data(today_ticks)
        today_ticks.to_csv("today_data.csv")
    '''
    #获取实时分笔数据
    '''
    def get_realtime_quotes(self):
        #获取实时分笔数据
        realtime = ts.get_realtime_quotes(self.code)
        self.desc_data(realtime)
        realtime.to_csv("realtime_data.csv")
    '''
    #获得所有发行股票实时行情数据
    '''
    def get_today_all(self):
        #获得所有发行股票实时行情数据
        all = ts.get_today_all()
        self.desc_data(all)
        all.to_csv("today_all.csv")
    '''
    获得历史交易数据
    '''
    def get_hist_data(self):
        #获得历史数据
        data = ts.get_hist_data(self.code)
        #data = data.sort_values(by='date',ascending=True)
        self.desc_data(data)
        data.to_csv("hist_data.csv")


if __name__ == "__main__":
    fd = FetchData("601313")
    fd.get_hist_data()
    fd.get_today_all()
    fd.get_realtime_quotes()
    fd.get_today_ticks()


