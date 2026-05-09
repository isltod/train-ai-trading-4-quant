import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


def bt_knn(filename, split, time_period=None, show_info=False, show_plot=False):
    filename = "data/" + filename
    df = pd.read_csv(filename, index_col=0, parse_dates=True)
    df = df.dropna()
    if time_period is not None:
        df = df[time_period[0] : time_period[1]]
    if show_info:
        print(df.head())

    tmp_df = df[["Open", "High", "Low", "Close"]].copy()
    print(tmp_df.head())

    tmp_df["Open-Close"] = tmp_df["Open"] - tmp_df["Close"]
    tmp_df["High-Low"] = tmp_df["High"] - tmp_df["Low"]
    tmp_df = tmp_df.dropna()
    X = tmp_df[["Open-Close", "High-Low"]]
    Y = np.where(tmp_df["Close"].shift(-1) > tmp_df["Open"].shift(-1), 1, -1)
    print(X.head())
    print(Y)

    x_min, x_max = X["Open-Close"].min() - 0.5, X["Open-Close"].max() + 0.5
    y_min, y_max = X["High-Low"].min() - 0.5, X["High-Low"].max() + 0.5

    if show_plot:
        plt.scatter(
            X["Open-Close"], X["High-Low"], c=Y, cmap=plt.cm.Set1, edgecolor="k"
        )
        plt.xlabel("Open-Close")
        plt.ylabel("High-Low")
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.show()

    split_ratio = split
    split = int(split_ratio * len(tmp_df))
    X_train = X[:split]
    Y_train = Y[:split]

    X_test = X[split:]
    Y_test = Y[split:]

    train_acc = []
    test_acc = []

    # for 반복문을 통해 1~15까지의 k값을 가진 모델을 만들어 본다
    best_acc = 0
    best_n = 0
    for n in range(1, 15):
        clf = KNeighborsClassifier(n_jobs=-1, n_neighbors=n)
        clf.fit(X_train, Y_train)
        prediction = clf.predict(X_test)
        train_acc.append(clf.score(X_train, Y_train))
        test_acc.append((prediction == Y_test).mean())
        if test_acc[-1] > best_acc:
            best_acc = test_acc[-1]
            best_n = n

    if show_plot:
        # 위에서 만든 결과를 그래프로 만들어 준다
        plt.figure(figsize=(12, 9))
        plt.plot(range(1, 15), train_acc, label="TRAIN set")
        plt.plot(range(1, 15), test_acc, label="TEST set")
        plt.xlabel("n_neighbors")
        plt.ylabel("accuracy")
        plt.xticks(np.arange(0, 16, step=1))
        plt.legend()
        plt.show()

    knn = KNeighborsClassifier(n_neighbors=best_n)
    knn.fit(X_train, Y_train)

    accuracy_train = accuracy_score(Y_train, knn.predict(X_train))
    accuracy_test = accuracy_score(Y_test, knn.predict(X_test))

    tmp_df["Predicted_Signal"] = knn.predict(X)

    tmp_df["buy&hold_ret"] = np.log(tmp_df["Close"] / tmp_df["Open"])
    cum_spy_ret = tmp_df[split:]["buy&hold_ret"].cumsum() * 100

    tmp_df["strategy_ret"] = tmp_df["buy&hold_ret"] * tmp_df["Predicted_Signal"].shift(
        1
    )
    cum_st_ret = tmp_df[split:]["strategy_ret"].cumsum() * 100

    print("훈련 정확도 : %.2f" % accuracy_train)
    print("테스트 정확도 : %.2f" % accuracy_test)
    print("TOTAL : ", tmp_df["Predicted_Signal"].count())
    print(
        "UP predict : ",
        tmp_df[tmp_df["Predicted_Signal"] == 1].count().iloc[0],
    )
    print(
        "DOWN predict : ",
        tmp_df[tmp_df["Predicted_Signal"] == -1].count().iloc[0],
    )
    std = cum_st_ret.std()
    sharpe = (cum_st_ret - cum_spy_ret) / std
    sharpe = sharpe.mean()
    print("Sharpe ratio : %.2f" % sharpe)

    plt.figure(figsize=(12, 6))
    plt.plot(cum_spy_ret, color="b", label="spy ret")
    plt.plot(cum_st_ret, color="r", label="strategy ret")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    # 책 예제
    filename = "SPY.csv"
    time_period = ("2012-01-01", "2017-01-01")
    split = 0.7
    # BTC
    time_period = None
    # BTC 1m
    filename = "btc_usdt_1m.csv"
    # BTC 1h
    # filename = "btc_usdt_1h.csv"
    # 테스트
    # bt_knn(filename, split, time_period, show_info=True, show_plot=True)
    # 실행
    bt_knn(filename, split, time_period)
