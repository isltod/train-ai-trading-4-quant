import pandas as pd
import matplotlib.pyplot as plt


def bollinger_band(price_df, colname, n, sigma):
    bb = price_df.copy()
    bb["center"] = price_df[colname].rolling(n).mean()  # 중앙 이동평균선
    bb["ub"] = bb["center"] + sigma * price_df[colname].rolling(n).std()  # 상단 밴드
    bb["lb"] = bb["center"] - sigma * price_df[colname].rolling(n).std()  # 하단 밴드
    return bb


def create_trade_book(sample, colname):
    book = sample[[colname]].copy()
    book["trade"] = ""
    return book


def tradings(sample, book, colname):
    for i in sample.index:
        if sample.loc[i, colname] > sample.loc[i, "ub"]:  # 상단밴드 이탈시 동작 안함
            book.loc[i, "trade"] = ""
        elif sample.loc[i, "lb"] > sample.loc[i, colname]:  # 하반밴드 이탈시 매수
            if book.shift(1).loc[i, "trade"] == "buy":  # 이미 매수상태라면
                book.loc[i, "trade"] = "buy"  # 매수상태 유지
            else:
                book.loc[i, "trade"] = "buy"
        elif (
            sample.loc[i, "ub"] >= sample.loc[i, colname]
            and sample.loc[i, colname] >= sample.loc[i, "lb"]
        ):  # 볼린저 밴드 안에 있을 시
            if book.shift(1).loc[i, "trade"] == "buy":
                book.loc[i, "trade"] = "buy"  # 매수상태 유지
            else:
                book.loc[i, "trade"] = ""  # 동작 안함
    return book


def returns(book, colname):
    # 손익 계산
    rtn = 1.0
    book["return"] = 1
    buy = 0.0
    sell = 0.0
    for i in book.index:
        if (
            book.loc[i, "trade"] == "buy" and book.shift(1).loc[i, "trade"] == ""
        ):  # long 진입
            buy = book.loc[i, colname]
            print("진입일 : ", i, "long 진입가격 : ", buy)
        elif (
            book.loc[i, "trade"] == "" and book.shift(1).loc[i, "trade"] == "buy"
        ):  # long 청산
            sell = book.loc[i, colname]
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

    acc_rtn = 1.0
    for i in book.index:
        rtn = book.loc[i, "return"]
        acc_rtn = acc_rtn * rtn  # 누적 수익률 계산
        book.loc[i, "acc return"] = acc_rtn

    print("Accunulated return :", round(acc_rtn, 4))
    print(round(acc_rtn, 4))
    return book


def prepare_data(filename, code, colname):
    filename = "data/" + filename
    df = pd.read_csv(filename)
    df["CODE"] = code
    price_df = df.loc[:, ["Date", colname]].copy()
    price_df.set_index(["Date"], inplace=True)
    return price_df


def bt_mrevert_bollinger(filename, code, colname, n, sigma, base_date=None):
    price_df = prepare_data(filename, code, colname)
    bollinger = bollinger_band(price_df, colname, n, sigma)
    if base_date is None:
        base_date = bollinger.index[n - 1]
    sample = bollinger.loc[base_date:]
    book = create_trade_book(sample, colname)
    book = tradings(sample, book, colname)
    book = returns(book, colname)
    book["acc return"].plot()
    plt.show()


if __name__ == "__main__":
    n = 20
    sigma = 2

    # filename = "SPY.csv"
    # code = "SPY"
    # colname = "Adj Close"
    # base_date = "2009-01-02"
    # bt_mrevert_bollinger(filename, code, colname, n, sigma, base_date)

    scale = "m"
    # scale = "h"
    filename = f"btc_usdt_1{scale}.csv"
    code = f"btc_usdt_1{scale}"
    colname = "close"
    bt_mrevert_bollinger(filename, code, colname, n, sigma)
