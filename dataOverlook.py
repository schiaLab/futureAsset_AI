import pandas as pd
import matplotlib.pyplot as plt


def reset(data):

    return data.reset_index()


def pairing(data1, data2, columnName, element = None):

    if element == None:

        result = pd.merge(data1, data2, on = columnName)


        print(result)

        return result

    else:

        result1 = data1[(data1.loc[:, columnName] == element)]

        result2 = data2[(data2.loc[:, columnName] == element)]

        return (result1, result2)


stock = pd.read_csv("train/stocks.csv", index_col=0)

print(stock.columns)

print(stock.종목번호.unique)

