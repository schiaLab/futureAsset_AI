from keras import models
from keras import layers
from keras.models import load_model
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import maxabs_scale


def xy(trainData):

    train_x = trainData.loc[:, ['매도고객수', '평균매수수량', '평균매도수량', '매수가격_중앙값', '매도가격_중앙값']]

    train_y = trainData.loc[:, '매수고객수']

    train_x = train_x.to_numpy()

    train_y = train_y.to_numpy()

    train_x = maxabs_scale(train_x)

    return (train_x, train_y)

trainData = pd.read_csv("train/trade_train.csv")

print(trainData.columns)

trainData = trainData.loc[:, ['매수고객수', '매도고객수', '평균매수수량', '평균매도수량', '매수가격_중앙값', '매도가격_중앙값']]


trainData, testData = train_test_split(trainData, train_size=0.75)



train_x, train_y = xy(trainData)

test_x, test_y = xy(testData)



model = models.Sequential()
model.add(layers.Dense(10, activation='relu',
                           input_shape=(train_x.shape[1],)))
model.add(layers.Dense(10, activation='relu'))

model.add(layers.Dense(1))
model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])


model.fit(train_x, train_y,validation_data=(test_x, test_y) , epochs=15, batch_size=20)

