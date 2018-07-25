# !/usr/bin/python
# -*- coding: utf-8 -*-

from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from linear_regression_model import linearRegressionModel as lrm
from sklearn.linear_model import LinearRegression as LRM
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

if __name__ == '__main__':
    # 生成线性回归数据集
    x, y = make_regression(7000)
    # 数据集分割成训练集和测试集
    x_train,x_test,y_train, y_test = train_test_split(x, y, test_size=0.5)
    y_train = y_train.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)

    # 自己实现的线性回归模型
    LRM1 = lrm(x.shape[1])
    LRM1.train(x_train, y_train, x_test, y_test)
    y_predict = LRM1.predict(x_test)
    print("Tensorflow R2: ", r2_score(y_test.ravel(), y_predict.ravel()))
    print("Tensorflow MSE: ", mean_squared_error(y_test.ravel(), y_predict.ravel()))

    # sklearn里面的线性回归模型
    LRM2 = LRM()
    y_predict = LRM2.fit(x_train, y_train).predict(x_test)
    print("Sklearn R2: ", r2_score(y_test.ravel(), y_predict.ravel()))
    print("Sklearn MSE: ", mean_squared_error(y_test.ravel(), y_predict.ravel()))
