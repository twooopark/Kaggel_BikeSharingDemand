import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy import stats

plt.style.use('ggplot')

mpl.rcParams['axes.unicode_minus'] = False
train = pd.read_csv('train.csv', parse_dates=["datetime"])
test = pd.read_csv('test.csv', parse_dates=["datetime"])

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
train["dayofweek"] = train["datetime"].dt.dayofweek
print(train.shape)
train["dayofweek"].value_counts()

trainWithoutOutliers = train[np.abs(train["count"] - train["count"].mean()) <= (3 * train["count"].std())]


test["year"] = test["datetime"].dt.year
test["month"] = test["datetime"].dt.month
test["day"] = test["datetime"].dt.day
test["hour"] = test["datetime"].dt.hour
test["minute"] = test["datetime"].dt.minute
test["second"] = test["datetime"].dt.second
test["dayofweek"] = test["datetime"].dt.dayofweek
print(train.shape)


def DateTime():
    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(nrows=2, ncols=3)
    fig.set_size_inches(18, 8)

    sns.barplot(data=train, x="year", y="count", ax=ax1)
    sns.barplot(data=train, x="month", y="count", ax=ax2)
    sns.barplot(data=train, x="day", y="count", ax=ax3)
    sns.barplot(data=train, x="hour", y="count", ax=ax4)
    sns.barplot(data=train, x="minute", y="count", ax=ax5)
    sns.barplot(data=train, x="second", y="count", ax=ax6)
    plt.show()


def SeasonHourWorkingday():
    fig, axes = plt.subplots(nrows=2, ncols=2)
    fig.set_size_inches(12, 10)

    sns.boxenplot(data=train, y="count", orient="v", ax=axes[0][0])
    sns.boxenplot(data=train, y="count", x="season", orient="v", ax=axes[0][1])
    sns.boxenplot(data=train, y="count", x="hour", orient="v", ax=axes[1][0])
    sns.boxenplot(data=train, y="count", x="workingday", orient="v", ax=axes[1][1])
    plt.show()


def DayOfWeekf():
    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(nrows=5)
    fig.set_size_inches(18, 25)
    sns.pointplot(data=train, x="hour", y="count", ax=ax1)
    sns.pointplot(data=train, x="hour", y="count", hue="workingday", ax=ax2)
    sns.pointplot(data=train, x="hour", y="count", hue="dayofweek", ax=ax3)
    sns.pointplot(data=train, x="hour", y="count", hue="weather", ax=ax4)
    sns.pointplot(data=train, x="hour", y="count", hue="season", ax=ax5)
    plt.show()


def corrHeatmap():
    corrMatt = train[["temp", "atemp", "casual", "registered", "humidity", "windspeed", "count"]]
    corrMatt = corrMatt.corr()
    print(corrMatt)

    mask = np.array(corrMatt)
    mask[np.tril_indices_from(mask)] = False

    fig, ax = plt.subplots()
    fig.set_size_inches(20, 10)
    sns.heatmap(corrMatt, mask=mask, vmax=.8, square=True, annot=True)
    plt.show()


def scatterClimate():
    fig, (ax1, ax2, ax3) = plt.subplots(ncols=3)
    fig.set_size_inches(12, 5)
    sns.regplot(x="temp", y="count", data=train, ax=ax1)
    sns.regplot(x="windspeed", y="count", data=train, ax=ax2)
    sns.regplot(x="humidity", y="count", data=train, ax=ax3)
    plt.show()


def trainWithoutOutliers():
    # trainWithoutOutliers = train[np.abs(train["count"] - train["count"].mean()) <= (3*train["count"].std())]

    print(train.shape)
    print(trainWithoutOutliers.shape)

    fig, axes = plt.subplots(ncols=2, nrows=2)
    fig.set_size_inches(12, 10)

    sns.distplot(train["count"], ax=axes[0][0])
    stats.probplot(train["count"], dist='norm', fit=True, plot=axes[0][1])
    sns.distplot(np.log(trainWithoutOutliers["count"]), ax=axes[1][0])
    stats.probplot(np.log1p(trainWithoutOutliers["count"]), dist='norm', fit=True, plot=axes[1][1])
    plt.show()


def windspeedNa():
    # widspeed 풍속에 0 값이 가장 많다. => 잘못 기록된 데이터를 고쳐 줄 필요가 있음
    fig, axes = plt.subplots(nrows=2)
    fig.set_size_inches(18, 10)

    plt.sca(axes[0])
    plt.xticks(rotations=30, ha='right')
    axes[0].set(ylabel='Count', title="train windspeed")
    sns.countplot(data=train, x="windspeed", ax=axes[0])

    plt.sca(axes[1])
    plt.xticks(rotation=30, ha='right')
    axes[1].set(ylabel='Count', title="test windspeed")
    sns.countplot(data=test, x="windspeed", ax=axes[1])
    plt.show()
    # 풍속의 0 값에 특정 값을 넣어준다.
    # 평균을 구해 일괄적으로 넣어줄 수도 있지만, 예측의 정확도를 높이는 데 도움이 될 것 같진 않다.
    # train.loc[train["windspeed"] == 0, "windspeed"] = train["windspeed"].mean()
    # test.loc[train["windspeed"] == 0, "windspeed"] = train["windspeed"].mean()


# 풍속이 0인 것과 아닌 것의 세트를 나누어 준다.
# def trainWind0():
trainWind0 = train.loc[train['windspeed'] == 0]
trainWindNot0 = train.loc[train['windspeed'] != 0]
print(trainWind0.shape)
print(trainWindNot0.shape)


from sklearn.ensemble import RandomForestClassifier

def predict_windspeed(data):
    # 풍속이 0인 것과 아닌 것을 나누어 준다.
    dataWind0 = data.loc[data['windspeed'] == 0]
    dataWindNot0 = data.loc[data['windspeed'] != 0]

    # 풍속을 예측할 피처를 선택한다.
    wCol = ["season", "weather", "humidity", "month", "temp", "year", "atemp"]

    # 풍속이 0이 아닌 데이터들의 타입을 스트링으로 바꿔준다.
    dataWindNot0["windspeed"] = dataWindNot0["windspeed"].astype("str")

    # 랜덤포레스트 분류기를 사용한다,
    rfModel_wind = RandomForestClassifier()

    # wCol에 있는 피처의 값을 바탕으로 풍속을 학습시킨다.
    rfModel_wind.fit(dataWindNot0[wCol], dataWindNot0["windspeed"])

    # 학습한 값을 바탕으로 풍속이 0으로 기록 된 데이터의 풍속을 예측한다.
    wind0Values = rfModel_wind.predict(X=dataWind0[wCol])

    # 값을 다 예측 후 비교해 보기 위해
    # 예측한 값을 넣어 줄 데이터 프레임을 새로 만든다.
    predictWind0 = dataWind0
    predictWindNot0 = dataWindNot0

    # 값이 0으로 기록 된 풍속에 대해 예측한 값을 넣어준다.
    predictWind0["windspeed"] = wind0Values

    # dataWindNot0 0이 아닌 풍속이 있는 데이터프레임에 예측한 값이 있는 데이터프레임을 합쳐준다.
    data = predictWindNot0.append(predictWind0)

    # 풍속의 데이터 타입을 float으로 지정해 준다.
    data["windspeed"] = data["windspeed"].astype("float")

    data.reset_index(inplace=True)
    data.drop('index', inplace=True, axis=1)

    return data

# 0값을 조정한다.
train = predict_windspeed(train)
# test = predict_windspeed(test)

def windspeed0():
    # windspeed 의 0 값을 조정한 데이터를 시각화
    fig, ax1 = plt.subplots()
    fig.set_size_inches(18, 6)

    plt.sca(ax1)
    plt.xticks(rotation=30, ha='right')
    ax1.set(ylabel='Count',title="train windspeed")
    sns.countplot(data=train, x="windspeed", ax=ax1)
    plt.show()



# Feature Selection
categorical_feature_names = ["season", "holiday", "workingday", "weather",
                             "dayofweek", "month", "year", "hour"]

for var in categorical_feature_names:
    train[var] = train[var].astype("category")
    test[var] = test[var].astype("category")

feature_names = ["season", "weather", "temp", "atemp", "humidity", "windspeed",
                 "year", "hour", "dayofweek", "holiday", "workingday"]

print(feature_names)
X_train = train[feature_names]

print(X_train.shape)
X_train.head()
X_test = test[feature_names]

print(X_test.shape)
X_test.head()

label_name = "count"

y_train = train[label_name]

print(y_train.shape)
y_train.head()

from sklearn.metrics import make_scorer

def rmsle(predicted_values, actual_values):
    # 넘파이로 배열 형태로 바꿔준다.
    predicted_values = np.array(predicted_values)
    actual_values = np.array(actual_values)

    # 예측값과 실제 값에 1을 더하고 로그를 씌워준다.
    log_predict = np.log(predicted_values + 1)
    log_actual = np.log(actual_values + 1)

    # 위에서 계산한 예측값에서 실제값을 빼주고 제곱을 해준다.
    difference = log_predict - log_actual
    # difference = (log_predict - log_actual) ** 2
    difference = np.square(difference)

    # 평균을 낸다.
    mean_difference = difference.mean()

    # 다시 루트를 씌운다.
    score = np.sqrt(mean_difference)

    return score

rmsle_scorer = make_scorer(rmsle)

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

k_fold = KFold(n_splits=10, shuffle=True, random_state=0)


from sklearn.ensemble import RandomForestRegressor

max_depth_list = []

model = RandomForestRegressor(n_estimators=100,
                              n_jobs=-1,
                              random_state=0)
print(model)

score = cross_val_score(model, X_train, y_train, cv=k_fold, scoring=rmsle_scorer)
score = score.mean()
# 0에 근접할수록 좋은 데이터
print("Score= {0:.5f}".format(score))


# 학습시킴, 피팅(옷을 맞출 때 사용하는 피팅을 생각함) - 피처와 레이블을 넣어주면 알아서 학습함
model.fit(X_train, y_train)

# 예측
predictions = model.predict(X_test)

print(predictions.shape)
print(predictions[0:10])

def predictData():
    # 예측한 데이터를 시각화 해본다.
    fig, (ax1, ax2) = plt.subplots(ncols=2)
    fig.set_size_inches(12, 5)
    sns.distplot(y_train, ax=ax1, bins=50)
    ax1.set(title="train")
    sns.distplot(predictions, ax=ax2, bins=50)
    ax2.set(title="test")
    plt.show()


submission = pd.read_csv("sampleSubmission.csv")
submission

submission["count"] = predictions

print(submission.shape)
submission.head()

submission.to_csv("Score_{0:.5f}_submission.csv".format(score), index=False)