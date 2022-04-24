from asyncio.base_tasks import _task_print_stack
from cgi import print_arguments
import imp
from operator import mod
from statistics import mode
import time
from json import load
from pyexpat import model
from tkinter import OUTSIDE
import pandas as pd
import numpy as np
from pyparsing import col
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.keras.models import Sequential, load_model
from tensorflow.python.keras.layers import Dense, Dropout
from tensorflow.python.keras.layers import LSTM
import matplotlib.pyplot as plt
import math


scaler = MinMaxScaler(feature_range=(0, 1))


def load_data(fileName):
    file = pd.read_csv(fileName, header=None)
    return file


def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    # 偷網路上的code==.
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # 輸入序列(t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))

        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # 預測序列 (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # 將他們整合在一起
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # print(names)
    # 刪除那些包含空值(NaN)的行
    if dropnan:
        agg.dropna(inplace=True)
    return agg


def modelTraining(training_data):
    train_x, train_y = training_data[:-30, :-10], training_data[:-30, -10:]
    train_x = train_x.reshape((train_x.shape[0], 1, train_x.shape[1]))
    test_x, test_y = training_data[-30:, :-10], training_data[-30:, -10:]
    test_x = test_x.reshape((test_x.shape[0], 1, test_x.shape[1]))
    print(train_x.shape)
    print(train_y.shape)
    model = Sequential()
    model.add(LSTM(50, input_shape=(
        train_x.shape[1], train_x.shape[2]), return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(50, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(50))
    model.add(Dense(10))
    model.compile(loss='mae', optimizer='adam')
    print(model.summary())

    tmp_str = time.strftime("%Y-%m-%d-%H-%M", time.localtime())
    tmp_str += '-model.h5'
    history = model.fit(train_x, train_y, epochs=100, batch_size=72, validation_data=(
        test_x, test_y), verbose=2, shuffle=False)
    model.save(tmp_str)
    # Visualize loss.
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    plt.show()


def modelPredict(testing_data):
    val_x, val_y = testing_data[:, :-10], testing_data[:, -10:]
    val_x = val_x.reshape((val_x.shape[0], 1, val_x.shape[1]))

    model = load_model('2022-04-22-17-29-model.h5')
    for i in range(len(val_x)):
        print('model predict')
        print(val_x[:i+1, :, :])
        y = model.predict(val_x[:i+1, :, :])

    y = y.reshape(len(y), 10)
    val_x = val_x.reshape(val_x.shape[0], val_x.shape[2])
    y = np.concatenate((y, val_x), axis=1)
    inv_y = scaler.inverse_transform(y)

    # 實際值逆標準化.
    val_y = val_y.reshape(len(val_y), 10)
    val_y = np.concatenate((val_y, val_x), axis=1)
    val_y = scaler.inverse_transform(val_y)
    print('real value: ', inv_y[:, 0])
    print('predict value: ', val_y[:, 0])

    # 方均跟差、視覺化.
    plt.plot(val_y[:, 0])
    plt.plot(inv_y[:, 0])
    plt.show()

    rmse = np.sqrt(mean_squared_error(val_y[:, 0:9], inv_y[:, 0:9]))
    print(rmse)

# 我自己寫的


def model_predict(testing_data):
    model = load_model('2022-04-22-17-29-model.h5')
    testing_data = testing_data.reshape(
        (testing_data.shape[0], 1, testing_data.shape[1]))
    # print(testing_data)
    y = model.predict(testing_data)
    y = y.reshape(len(y), 10)
    y = np.delete(y, [4, 5, 6, 7, 8, 9], 1)
    return y


if __name__ == '__main__':
    # You should not modify this part.
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--training',
                        default='training.csv',
                        help='input training data file name')
    parser.add_argument('--testing',
                        default='testing.csv',
                        help='input testing data file name')
    parser.add_argument('--output',
                        default='output.csv',
                        help='output file name')
    args = parser.parse_args()

    # The following part is an example.
    # You can modify it at will.
    training_data = load_data(args.training)
    testing_data = load_data(args.testing)

    # Normalization.

    # Transform data to supervised series.
    training_data = series_to_supervised(training_data, 1, 10, True)
    training_data.drop(training_data.columns[[5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 6, 10, 14,
                       18, 22, 26, 30, 34, 38, 42, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43]], axis=1, inplace=True)

    testing_data = series_to_supervised(testing_data, 1, 10, True)
    testing_data.drop(testing_data.columns[[5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 6, 10, 14,
                      18, 22, 26, 30, 34, 38, 42, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43]], axis=1, inplace=True)
    training_data = scaler.fit_transform(training_data)
    testing_data = scaler.fit_transform(testing_data)

    # Train or Test the Model.
    # modelTraining(training_data)

    # modelPredict(testing_data)

    # train和test串在一起
    testing_data = load_data(args.testing)
    training_data = load_data(args.training)
    new_test = np.concatenate([training_data, testing_data])

    flag = -1
    signal_buy = []
    signal_sell = []
    for i in range(len(testing_data)-1):
        print(i)

        # 第i天的預測過去9天和當天作為標準化的依據(共10筆)
        neww_test = new_test[(-(len(testing_data))-9+i)                             :(-(len(testing_data))+i+1)]
        neww_test = scaler.fit_transform(neww_test)

        # _predict是預測出來10天的開盤價
        _predict = model_predict(neww_test)
        _predict = scaler.inverse_transform(_predict)
        _predict = np.delete(_predict, [1, 2, 3], 1)
        print(_predict)

        # 算短平均(3日)
        tmp = new_test[(-(len(testing_data))+i-1):(-(len(testing_data))+i+1)]
        tmp = np.delete(tmp, [1, 2, 3], 1)
        tmp = np.concatenate((tmp, np.reshape(_predict[0], (1, 1))))
        short = np.mean(tmp)
        print(short)

        # 算長平均(11日)
        tmp = new_test[(-(len(testing_data))+i-5):(-(len(testing_data))+i+1)]
        tmp = np.delete(tmp, [1, 2, 3], 1)
        ttmp = _predict[0:5]
        tmp = np.concatenate((tmp, ttmp))
        long = np.mean(tmp)
        print(long)

        # 當天
        now = new_test[-(len(testing_data))+i]
        now = now[0]
        print(now)
        # 判斷買賣
        if short > long:
            if flag != 1:  # 之前的短期未超過長期，即黃金交叉
                signal_buy.append(now)
                signal_sell.append(np.nan)
                flag = 1
            else:
                signal_buy.append(np.nan)
                signal_sell.append(np.nan)
        elif long > short:
            if flag != 0:  # 之前的長期未超過短期，即死亡交叉
                signal_buy.append(np.nan)
                signal_sell.append(now)
                flag = 0
            else:
                signal_buy.append(np.nan)
                signal_sell.append(np.nan)
        else:
            signal_buy.append(np.nan)
            signal_sell.append(np.nan)
    print('buy and sell')
    print(signal_buy)
    print(signal_sell)
    # 大賠特賠
