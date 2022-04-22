from asyncio.base_tasks import _task_print_stack
from cgi import print_arguments
import imp
from operator import mod
from statistics import mode
import time
from json import load
from pyexpat import model
import pandas as pd
import numpy as np
from pyparsing import col
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.keras.models import Sequential, load_model
from tensorflow.python.keras.layers import Dense, Dropout
from tensorflow.python.keras.layers import LSTM
import matplotlib.pyplot as plt


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
    model.add(LSTM(50, input_shape=(train_x.shape[1], train_x.shape[2]), return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(50, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(50))
    model.add(Dense(10))
    model.compile(loss='mae', optimizer='adam')
    print(model.summary())
    
    tmp_str = time.strftime("%Y-%m-%d-%H-%M", time.localtime())
    tmp_str += '-model.h5'
    history = model.fit(train_x, train_y, epochs=100, batch_size=72, validation_data=(test_x, test_y), verbose=2, shuffle=False)
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
        y = model.predict(val_x[:i+1, :, :])

    y = y.reshape(len(y),10)
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
    training_data.drop(training_data.columns[[5,9,13,17,21,25,29,33,37,41,6,10,14,18,22,26,30,34,38,42,7,11,15,19,23,27,31,35,39,43]], axis=1, inplace=True)
    

    testing_data = series_to_supervised(testing_data, 1, 10, True)
    testing_data.drop(testing_data.columns[[5,9,13,17,21,25,29,33,37,41,6,10,14,18,22,26,30,34,38,42,7,11,15,19,23,27,31,35,39,43]], axis=1, inplace=True)
    training_data = scaler.fit_transform(training_data)
    testing_data = scaler.fit_transform(testing_data)

    # Train or Test the Model.
    # modelTraining(training_data)
    modelPredict(testing_data)

    # 剩下交給你ㄌ...