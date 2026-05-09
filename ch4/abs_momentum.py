import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/SPY.csv")
# print(df.head())

price_df = df.loc[:, ["Date", "Adj Close"]].copy()
# print(price_df.head())

# 몇년 몇월인지 뽑기
price_df["STD_YM"] = price_df["Date"].map(
    lambda x: datetime.datetime.strptime(x, "%Y-%m-%d").strftime("%Y-%m")
)
# print(price_df.head())

# 해당 월의 종가
month_list = price_df["STD_YM"].unique()
month_last_df = pd.DataFrame()
for m in month_list:
    # append가 deprecated 됐는데도 그걸 써놔서...concat으로 바꾸는데 이 난리네...
    new_row = price_df.loc[price_df[price_df["STD_YM"] == m].index[-1], :]
    month_last_df = pd.concat([month_last_df, new_row], axis=1)
month_last_df = month_last_df.T
month_last_df.set_index("Date", inplace=True)
# print(month_last_df.head())

month_last_df["BF_1M_Adj Close"] = month_last_df.shift(1)["Adj Close"]
month_last_df["BF_12M_Adj Close"] = month_last_df.shift(12)["Adj Close"]
month_last_df.dropna(inplace=True)
# print(month_last_df.head(15))

book = price_df.copy()
book.set_index(["Date"], inplace=True)
book["trade"] = ""
# print(book.head())

# trading 부분.
ticker = "SPY"
for x in month_last_df.index:
    signal = ""
    # 절대 모멘텀을 계산한다.
    momentum_index = (
        month_last_df.loc[x, "BF_1M_Adj Close"]
        / month_last_df.loc[x, "BF_12M_Adj Close"]
        - 1
    )
    # 절대 모멘텀 지표 True / False를 판단한다.
    flag = (
        True
        if (
            (momentum_index > 0.0)
            and (momentum_index != np.inf)
            and (momentum_index != -np.inf)
        )
        else False and True
    )
    if flag:
        signal = "buy " + ticker  # 절대 모멘텀 지표가 Positive이면 매수 후 보유.
    print(
        "날짜 : ",
        x,
        " 모멘텀 인덱스 : ",
        momentum_index,
        "flag : ",
        flag,
        "signal : ",
        signal,
    )
    book.loc[x:, "trade"] = signal

print(book.tail(25))


def returns(book, ticker):
    # 손익 계산
    rtn = 1.0
    book["return"] = 1
    buy = 0.0
    sell = 0.0
    for i in book.index:
        if (
            book.loc[i, "trade"] == "buy " + ticker
            and book.shift(1).loc[i, "trade"] == ""
        ):  # long 진입
            buy = book.loc[i, "Adj Close"]
            print("진입일 : ", i, "long 진입가격 : ", buy)
        elif (
            book.loc[i, "trade"] == "buy " + ticker
            and book.shift(1).loc[i, "trade"] == "buy " + ticker
        ):
            # 보유중
            current = book.loc[i, "Adj Close"]
            rtn = (current - buy) / buy + 1
            book.loc[i, "return"] = rtn

        elif (
            book.loc[i, "trade"] == ""
            and book.shift(1).loc[i, "trade"] == "buy " + ticker
        ):  # long 청산
            sell = book.loc[i, "Adj Close"]
            rtn = (sell - buy) / buy + 1  # 손익 계산
            book.loc[i, "return"] = rtn
            print(
                "청산일 : ",
                i,
                "long 진입가격 : ",
                buy,
                " |  long 청산가격 : ",
                sell,
                " | return:",
                round(rtn, 4),
            )

        if book.loc[i, "trade"] == "":  # zero position
            buy = 0.0
            sell = 0.0
            current = 0.0

    acc_rtn = 1.0
    for i in book.index:
        if (
            book.loc[i, "trade"] == ""
            and book.shift(1).loc[i, "trade"] == "buy " + ticker
        ):  # long 청산시
            rtn = book.loc[i, "return"]
            acc_rtn = acc_rtn * rtn  # 누적수익률 계산
            book.loc[i:, "acc return"] = acc_rtn

    print("Accunulated return :", round(acc_rtn, 4))
    return round(acc_rtn, 4)


print(returns(book, ticker))
book["acc return"].plot()
plt.show()
# 이건 시간 절대값에 너무 의존해서 비트코인 1분 1시간으로 바꾸려면 너무 복잡해서 포기...
