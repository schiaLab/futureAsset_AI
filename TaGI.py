from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_validate
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import sys
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from keras import models
from keras import layers
from keras.models import load_model
from sklearn.model_selection import train_test_split

def dateChanger(date): #scaling the date if the data between 0 and 1



    date1 = date // 100

    date2 = date % 100

    date1 = date1.where(date1 == 2019, 12)

    date1 = date1.where(date1 == 12, 0)

    date2 = date1 + date2

    date2  = date2 - 7

    date2 = date2 / 12

    return date2

def dateAddition(data): #seperating time data and changing with dataChanger()

    timeData = data.loc[:, "기준년월"]
    timeData = dateChanger(timeData)
    trainX = data.drop("기준년월", axis=1)


    return (timeData, trainX)






def split_XY(data, testColumn): #getting ready for data separating with input and output.
    x = data.drop(testColumn, axis=1)
    y = data.loc[:, testColumn]

    return (x, y)


def reset(data):

    return data.reset_index()

def groupbyMean(data, column):

    data = data.groupby(column).mean()

    data = data.reset_index()

    return data


def queryCleaner(data):

    data.iloc[:, 0] = data.iloc[:, 0].str.replace("-", "").astype(int)

    data.iloc[:, 0] = data.iloc[:, 0] // 100


    print("쿼리데이터 정제 완료")

    print(data)

    return data

def pairing(data1, data2, columnName, element = None):

    if element == None:

        result = pd.merge(data1, data2, on = columnName)


        print(result)

        return result

    else:

        result1 = data1[(data1.loc[:, columnName] == element)]

        result2 = data2[(data2.loc[:, columnName] == element)]

        return (result1, result2)

def Gi(x, testColumn=None, y=None, ALLE=None, time=False, additional= np.array([]), batch=1):


    timeData = None

    if testColumn != None and y==None:

        trainX, trainY = split_XY(x, testColumn)



        if time:

            timeData = trainX.loc[:, "기준년월"]
            timeData = dateChanger(timeData)
            timeData = timeData.to_numpy().reshape(-1, 1)
            trainX = trainX.drop("기준년월", axis=1)

        if ALLE == None:

            ALLE = OneHotEncoder()
            trainX = ALLE.fit_transform(trainX).toarray()

            if time:
                trainX = np.append(trainX, timeData, axis=1)




        else:

            ALLE = ALLE

            trainX = ALLE.transform(trainX).toarray()

            if time:
                trainX = np.append(trainX, timeData, axis=1)





    else:

        trainX = x

        if time:
            timeData = trainX.loc[:, "기준년월"]
            timeData = dateChanger(timeData)
            timeData = timeData.to_numpy().reshape(-1, 1)
            trainX = trainX.drop("기준년월", axis=1)

        if ALLE == None:

            ALLE = OneHotEncoder()

            trainX = ALLE.fit_transform(trainX).toarray()

            if time:

                trainX = np.append(trainX, timeData, axis=1)



        else:

            ALLE = ALLE

            trainX = ALLE.transform(trainX).toarray()



            if time:
                trainX = np.append(trainX, timeData, axis=1)


        trainY = y.reshape(-1, 1)

    if additional.size != 0:

        trainX = np.append(trainX, additional, axis=1)

    print(trainX)

    Gi = models.Sequential()
    Gi.add(layers.Dense(128, activation='relu',
                        input_shape=(trainX.shape[1],)))
    Gi.add(layers.Dense(128, activation='relu'))
    Gi.add(layers.Dense(128, activation='relu'))

    Gi.add(layers.Dense(1))
    Gi.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])





    Gi.fit(trainX, trainY, batch_size=batch, epochs=30)

    return (ALLE, Gi)


def GDE(x, testColumn):




    trainX, trainY = split_XY(x, testColumn)



    mmsc = MinMaxScaler()

    trainX = mmsc.fit_transform(trainX)
    GDE = models.Sequential()
    GDE.add(layers.Dense(10, activation='relu',
                        input_shape=(trainX.shape[1],)))
    GDE.add(layers.Dense(10, activation='relu'))

    GDE.add(layers.Dense(1))
    GDE.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])


    GDE.fit(trainX, trainY, batch_size=1, epochs=30)

    return (mmsc, GDE)

name = input("File Name: ")

stock = pd.read_csv("train/stocks.csv", index_col=0)


train = pd.read_csv("train/trade_train.csv", index_col=0)

stock = stock.where(stock != 0)

stock = stock.dropna()

scData = stock.loc[:, ['시장구분', '표준산업구분코드_대분류', '표준산업구분코드_중분류']].drop_duplicates()

submission = stock.loc[(stock.loc[:, "20년7월TOP3대상여부"] == "Y")]

submission = submission[submission.loc[:, "기준일자"] > 20200700]

submission = submission.rename({"기준일자": "기준년월"}, axis=1)

print(submission)

poli = pd.read_csv("train/politics.csv")

poli = poli.iloc[:, -2:]



kofi = pd.read_csv("train/koreaFinance.csv")


poli = queryCleaner(poli)

kofi = queryCleaner(kofi)

poli = groupbyMean(poli, "날짜.2")

kofi = groupbyMean(kofi, "날짜")

kofiPre = kofi.loc[(kofi.loc[:, "날짜"] == 202007)].iloc[:, 1]

kofiPre = float(kofiPre)

poliPre = poli.loc[(poli.loc[:, "날짜.2"] == 202007)].iloc[:, 1]

poliPre = float(poliPre)



stock.loc[:, "기준일자"] = stock.loc[:, "기준일자"] // 100

stock = stock.rename({"기준일자": "기준년월"}, axis=1)

stock2 = stock.drop("종목명", axis=1)

stock2 = stock2.groupby(by=["기준년월", '종목번호', '20년7월TOP3대상여부', '시장구분', '표준산업구분코드_대분류',
       '표준산업구분코드_중분류', '표준산업구분코드_소분류']).mean()

stock2 = reset(stock2)


print("stock2:", stock2)

kofi2 = kofi.rename({"날짜":"기준년월"}, axis=1)
poli2 = poli.rename({"날짜.2":"기준년월"}, axis=1)


train2 = pd.merge(train, kofi2, on="기준년월")

train2 = pd.merge(train2, poli2, on="기준년월")

print(train2)

paired = pairing(train2, stock2, ["종목번호", "기준년월"])

paired = paired.drop_duplicates()

print("Section B")
print(paired.columns)
print(paired)

stock = stock.groupby("기준년월").mean()

stock = reset(stock)

train = train.groupby("기준년월").mean()

train = reset(train)







jun = ((stock.loc[:, "종목고가"] - stock.loc[:, "종목저가"]) /
       stock.loc[:, "종목시가"] * (stock.loc[:, "거래량"] / stock.loc[:, "거래량"].sum()))

stock["jun"] = jun

stockPre = stock.loc[(stock.loc[:, "기준년월"] == 202007)]


junPre = stockPre.loc[:, "jun"]

tradeMoneyPre = float(stockPre.loc[:, "거래금액_만원단위"])

tradeMoney = stock.loc[:, "거래금액_만원단위"]


#공시변수 설정

train["jun"] = jun.iloc[:-1]
train["tradeMoney"] = tradeMoney


kofi = kofi.rename({"날짜":"기준년월"}, axis=1)
poli = poli.rename({"날짜.2":"기준년월"}, axis=1)


train = pd.merge(train, kofi, on="기준년월")

train = pd.merge(train, poli, on="기준년월")

sagi = train.loc[:, ["경제위기", "부동산정책", "jun"]] #tradeMoney는 제


y = train.loc[:, "매수고객수"]

y2 = train.loc[:, ["기준년월", "매수고객수"]]

sc = StandardScaler()

saGi = LinearRegression()

saGiPip = Pipeline([("scal", sc), ("model", saGi)])


saGiPip.fit(sagi, y)

septem = np.array([[kofiPre, poliPre, junPre]])

saGiPre = saGiPip.predict(septem)

gi2 = saGiPip.predict(sagi)


y2["gi2"] = gi2

y2 = y2.loc[:, ["기준년월", "gi2"]]

paired = pd.merge(paired, y2, on= "기준년월")

paired["giy"] = paired.loc[:, "매수고객수"] - paired.loc[:, "gi2"]


giy = paired.loc[:, ["종목번호", "giy"]]


print(giy)

print("2020.07 saGi Prediction:", saGiPre)

LargeE = OneHotEncoder()
SmallE = OneHotEncoder()
MarketE = OneHotEncoder()

print(paired.columns)

jun = ((paired.loc[:, "종목고가"] - paired.loc[:, "종목저가"]) /
       paired.loc[:, "종목시가"] * (paired.loc[:, "거래량"] / paired.loc[:, "거래량"].sum()))

paired["jun"] = jun


data = paired.loc[:, ['기준년월', '시장구분', '표준산업구분코드_대분류', '표준산업구분코드_중분류',"경제위기", "부동산정책", "jun" ,  "giy"]]


X = reset(data)

print("Error Expected Less than: ", data.loc[:, "giy"].abs().mean())

sc = OneHotEncoder()

sc.fit(scData)

Add = X.loc[:, ["경제위기", "부동산정책", "jun"]]

giMMS = MinMaxScaler()

Add = giMMS.fit_transform(Add)




X = X.loc[:, ['기준년월', '시장구분', '표준산업구분코드_대분류', '표준산업구분코드_중분류',  "giy"]]



ALLE2, Gi2 = Gi(X, testColumn="giy", ALLE=sc, additional=Add, time=True, batch=30)

x, y = split_XY(X, "giy")

x = x.loc[:, ['시장구분', '표준산업구분코드_대분류', '표준산업구분코드_중분류']]

x["a"] = 1

alpha = x.loc[:, "a"]

x = x.drop("a", axis=1)

print(x)

X = ALLE2.transform(x).toarray()


X = np.append(X, alpha.to_numpy().reshape(-1, 1), axis=1)

X = np.append(X, Add, axis=1)

print(X)


pre = Gi2.predict(X)

#Test
gde = y.to_numpy().reshape(-1, 1) - pre


data["gdey"] = gde

data = data.drop("giy", axis =1)

paired = pd.merge(paired, data, on=['기준년월', '시장구분', '표준산업구분코드_대분류', '표준산업구분코드_중분류',"경제위기", "부동산정책", "jun" ])


print(paired.columns)

print("Distribution0:", paired.loc[:, "매수고객수"].std())
print("Distribution1:", paired.loc[:, "giy"].std())
print("Distribution2:", np.std(gde))

#ProtoType Prediction

indis = paired.loc[:, "그룹번호"].unique()

GDElist = dict()
GiList = dict()

for indi in indis:

    data = paired.loc[(paired.loc[:, "그룹번호"] == indi)].drop_duplicates()

    print("Expected Less than: ", data.loc[:, "giy"].abs().mean())

    X = data.loc[:, ['그룹내고객수', "경제위기", "부동산정책", "jun", "gdey"]]

    mmsc, model = GDE(X, "gdey")

    GDElist[indi] = (mmsc, model)

    x = X.loc[:, ['그룹내고객수', "경제위기", "부동산정책", "jun"]]

    y= X.loc[:, "gdey"].to_numpy().reshape(-1, 1)

    X = mmsc.transform(x)

    y2 = model.predict(X)

    y3 = y - y2

    X = data.loc[:, ["기준년월", '시장구분', '표준산업구분코드_대분류', '표준산업구분코드_중분류']]


    print(X)
    print("="*30)

    ALLE, model = Gi(X, y=y3, ALLE=sc, time=True)

    GiList[indi] = (ALLE, model)




stocks = submission.loc[:, "종목번호"].unique()

submission["jun"] = ((submission.loc[:, "종목고가"] - submission.loc[:, "종목저가"]) /
       submission.loc[:, "종목시가"] * (submission.loc[:, "거래량"] / submission.loc[:, "거래량"].sum()))


data = submission.loc[:, ['시장구분', '표준산업구분코드_대분류', '표준산업구분코드_중분류']]


X = ALLE2.transform(data).toarray()

jun2 = float(submission[(submission.loc[:, "종목번호"] == stock)].loc[:, "jun"].mean())

Add2 = giMMS.transform(np.array([[kofiPre, poliPre, jun2]]))

X = np.append(X, Add2, axis=1)

y = Gi2.predict(X)

submission["gi"] = pd.Series(y.flatten(), index=submission.index)


sub = dict()


for indi in indis:

    mmsc1, model1 = GDElist[indi]

    ALLE2, model2 = GiList[indi]

    cus = float(paired[(paired.loc[:, "그룹번호"] == indi)].loc[:, '그룹내고객수'].mean())

    max = []


    print("   ", indi)


    for stock in stocks:

        data = submission[(submission.loc[:, "종목번호"] == stock)]


        jun = float(data.loc[:, 'jun'].mean())

        giData = data.loc[:, ['시장구분', '표준산업구분코드_대분류', '표준산업구분코드_중분류']]
        giData = giData.drop_duplicates()


        x = np.array([[cus, kofiPre, poliPre, jun]])


        X = mmsc1.transform(x)

        y = float(model1.predict(X))


        y2 = float((data.loc[:, "gi"]).mean())

        y2 = y + y2

        X = ALLE2.transform(giData).toarray()

        X = np.append(X, np.array([[1]]), axis=1)

        y3 = float(model2.predict(X))

        y3 = y3 + y2

        max.append([stock, y3])

        print(stock)


    first = 0

    second = 0

    thrid = 0

    first1 = ""

    second1 = ""

    thrid1 = ""

    for n in range(len(max)):

        num = max[n][1]

        if num == np.nan :

            print(n ,max[n])
            sys.exit()

        print(num)

        if num > first:

            first = max[n][1]

            first1 = max[n][0]



        elif num > second:

            second = max[n][1]

            second1 = max[n][0]


        elif num > thrid:

            thrid = max[n][1]

            thrid1 = max[n][0]

        else:

            continue

        print(n)



    me = [first1, second1, thrid1]

    me.sort()

    sub[indi] = me


submission = pd.DataFrame.from_dict(sub, orient="index")

submission = reset(submission)

submission = submission.sort_values(by=["index"], axis=0)


submission.to_csv("%s.csv"%name, index=False)












