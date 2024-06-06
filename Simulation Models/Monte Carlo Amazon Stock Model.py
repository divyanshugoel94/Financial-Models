import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()

Amzndata = pd.read_csv("C:/Users/goeld/OneDrive/Desktop/Financial Analytics/AMZN.csv", header = 0, usecols= ['Date', 'Close'], parse_dates= True, index_col= 'Date')

print(Amzndata.info())

print(Amzndata.describe())

plt.figure(figsize = (20, 10))

plt.plot(Amzndata)

plt.show()

Amzndatapctchange = Amzndata.pct_change()

Amznlogreturns = np.log(1 + Amzndatapctchange)
print(Amznlogreturns.tail(10))

plt.figure(figsize= (20, 10))

plt.plot(Amznlogreturns)

plt.show()

Meanlogreturns = np.array(Amznlogreturns.mean())

Varlogreturns = np.array(Amznlogreturns.var())

Stddevlogreturns = np.array(Amznlogreturns.std())

drift = Meanlogreturns - (0.5* Varlogreturns)

print("Drift =", drift)

numintervals = 2514

iterations = 20

np.random.seed(7)

SBmotion = norm.ppf(np.random.rand(numintervals, iterations))

dailyreturns = np.exp((drift  + Stddevlogreturns*SBmotion))

startstockprices = Amzndata.iloc[0]

stockprice = np.zeros_like(dailyreturns)

stockprice[0] = startstockprices

for t in range(1, numintervals):
    stockprice[t] = stockprice[t-1]*dailyreturns[t]

plt.figure(figsize= (20,10))

plt.plot(stockprice)

plt.legend()

amzntrend = np.array(Amzndata.iloc[:, 0:1])

plt.plot(amzntrend, 'k*')

plt.show()




