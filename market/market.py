# -*- coding: utf-8 -*-
# -*- author: nik -*-
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np



#获得历史数据 江南嘉捷
data = pd.read_csv("hist_data.csv")

ax1 = plt.subplot(311)
t1 = np.arange(630)

#reverse the data
plt.plot(t1, data.open[::-1], label="open")
plt.plot(t1, data.high[::-1], label="max")
plt.plot(t1, data.close[::-1], label="close")
plt.plot(t1, data.low[::-1], label="min")


#plt.xlim(1,10)
leg = plt.legend(loc='upper left', ncol=4, mode="expand", shadow=True, fancybox=True)
leg.get_frame().set_alpha(0.5)
ax2 = plt.subplot(312)
plt.bar(t1,data.volume[::-1],label="volume")

ax3 = plt.subplot(313)
plt.scatter(data.volume[::-1],data.close[::-1])
#plt.show()

##############
#数据准备
y = data.close.values.reshape(-1,1)
#drop column close
X = data.drop('close',axis=1)
X = data.drop('date',axis=1)
print(y.shape)
print(X.shape)

from sklearn.model_selection import train_test_split
# 依然如故，我们对数据进行分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state=33)

from sklearn.preprocessing import StandardScaler

# 正规化的目的在于避免原始特征值差异过大，导致训练得到的参数权重不一
scalerX = StandardScaler().fit(X_train)
X_train = scalerX.transform(X_train)
X_test = scalerX.transform(X_test)

scalery = StandardScaler().fit(y_train)
y_train = scalery.transform(y_train)
y_test = scalery.transform(y_test)

# 先把评价模块写好，依然是默认5折交叉验证，只是这里的评价指标不再是精度，而是另一个函数R2，大体上，这个得分多少代表有多大百分比的回归结果可以被训练器覆盖和解释
from sklearn.model_selection import KFold,cross_val_score

def train_and_evaluate(clf, X_train, y_train):
    cv = KFold( n_splits = 5, shuffle=True, random_state=33)
    scores = cross_val_score(clf, X_train, y_train, cv=cv)
    print ('Average coefficient of determination using 5-fold cross validation:', np.mean(scores))

# 线性模型尝试， SGD_Regressor
from sklearn import linear_model
# 这里有一个正则化的选项penalty，目前14维特征也许不会有太大影响
clf_sgd = linear_model.SGDRegressor(loss='squared_loss', penalty=None, random_state=42)
train_and_evaluate(clf_sgd, X_train, y_train)

# 再换一个SGD_Regressor的penalty参数为l2,结果貌似影响不大，因为特征太少，正则化意义不大
clf_sgd_l2 = linear_model.SGDRegressor(loss='squared_loss', penalty='l2', random_state=42)
train_and_evaluate(clf_sgd_l2, X_train, y_train)
# 再看看SVM的regressor怎么样（都是默认参数）,
from sklearn.svm import SVR
# 使用线性核没有啥子提升，但是因为特征少，所以可以考虑升高维度
clf_svr = SVR(kernel='linear')
train_and_evaluate(clf_svr, X_train, y_train)

clf_svr_poly = SVR(kernel='poly')
# 升高维度，效果明显，但是此招慎用@@，特征高的话, CPU还是受不了，内存倒是小事。其实到了现在，连我们自己都没办法直接解释这些特征的具体含义了。
train_and_evaluate(clf_svr_poly, X_train, y_train)

clf_svr_rbf = SVR(kernel='rbf')
# RBF (径向基核更是牛逼！)
train_and_evaluate(clf_svr_rbf, X_train, y_train)
# 再来个更猛的! 极限回归森林，放大招了！！！
from sklearn import ensemble
clf_et = ensemble.ExtraTreesRegressor()
train_and_evaluate(clf_et, X_train, y_train)
# 最后看看在测试集上的表现
clf_et.fit(X_train, y_train)
clf_et.score(X_test, y_test)