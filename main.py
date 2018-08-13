import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy import stats

plt.style.use('ggplot')

mpl.rcParams['axes.unicode_minus'] = False
train = pd.read_csv('train.csv', parse_dates=["datetime"])
test = pd.read_csv('test.csv')

print(train.head(20))
print(train.temp.describe())
print(train.isnull().sum())

import missingno as msno
# msno.matrix(train, figsize=(12, 5))
train["year"] = train["datetime"].dt.year
train["month"] = train["datetime"].dt.month
train["day"] = train["datetime"].dt.day
train["hour"] = train["datetime"].dt.hour
train["minute"] = train["datetime"].dt.minute
train["second"] = train["datetime"].dt.second

fig1, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(nrows=2, ncols=3)
fig1.set_size_inches(18, 8)

sns.barplot(data=train, x="year", y="count", ax=ax1)
sns.barplot(data=train, x="month", y="count", ax=ax2)
sns.barplot(data=train, x="day", y="count", ax=ax3)
sns.barplot(data=train, x="hour", y="count", ax=ax4)
sns.barplot(data=train, x="minute", y="count", ax=ax5)
sns.barplot(data=train, x="second", y="count", ax=ax6)
# plt.show()

fig2, axes = plt.subplots(nrows=2, ncols=2)
fig2.set_size_inches(12, 10)
sns.boxenplot(data=train, y="count", orient="v", ax=axes[0][0])
sns.boxenplot(data=train, y="count", x="season", orient="v", ax=axes[0][1])
sns.boxenplot(data=train, y="count", x="hour", orient="v", ax=axes[1][0])
sns.boxenplot(data=train, y="count", x="workingday", orient="v", ax=axes[1][1])
# plt.show()

train["dayofweek"] = train["datetime"].dt.dayofweek

train["dayofweek"].value_counts()
fig3, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(nrows=5)
fig3.set_size_inches(18, 25)
sns.pointplot(data=train, x="hour", y="count", ax=ax1)
sns.pointplot(data=train, x="hour", y="count", hue="workingday", ax=ax2)
sns.pointplot(data=train, x="hour", y="count", hue="dayofweek", ax=ax3)
sns.pointplot(data=train, x="hour", y="count", hue="weather", ax=ax4)
sns.pointplot(data=train, x="hour", y="count", hue="season", ax=ax5)
# plt.show()

corrMatt = train[["temp", "atemp", "casual", "registered", "humidity", "windspeed", "count"]]
corrMatt = corrMatt.corr()
print(corrMatt)

mask = np.array(corrMatt)
mask[np.tril_indices_from(mask)] = False

fig4, ax = plt.subplots()
fig4.set_size_inches(20, 10)
sns.heatmap(corrMatt, mask=mask, vmax=.8, square=True, annot=True)
# plt.show()

fig5, (ax1, ax2, ax3) = plt.subplots(ncols=3)
fig5.set_size_inches(12, 5)
sns.regplot(x="temp", y="count", data=train, ax=ax1)
sns.regplot(x="windspeed", y="count", data=train, ax=ax2)
sns.regplot(x="humidity", y="count", data=train, ax=ax3)
# plt.show()

trainWithoutOutliers = train[np.abs(train["count"] - train["count"].mean()) <= (3*train["count"].std())]

print(train.shape)
print(trainWithoutOutliers.shape)

fig6, axes2 = plt.subplots(ncols=2, nrows=2)
fig6.set_size_inches(12, 10)

sns.distplot(train["count"], ax=axes2[0][0])
stats.probplot(train["count"], dist='norm', fit=True, plot=axes2[0][1])
sns.distplot(np.log(trainWithoutOutliers["count"]), ax=axes2[1][0])
stats.probplot(np.log1p(trainWithoutOutliers["count"]), dist='norm', fit=True, plot=axes2[1][1])
plt.show()