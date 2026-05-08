import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/AMZN.csv", index_col="Date", parse_dates=["Date"])
print(df.head())

# 레코드 기준으로(axis=1), 결측값이 있으면 출력
print(df[df.isin([np.nan, np.inf, -np.inf]).any(axis=1)])

# 수정 종가를 기준
price_df = df.loc[:, ["Adj Close"]].copy()
# price_df.plot(figsize=(10, 6), title="Adj Close Price")

# 특정 기간
from_date = "1997-01-03"
to_date = "2003-01-03"
# price_df[from_date:to_date].plot(figsize=(10, 6), title="Adj Close Price")

# 전체 기간의 수익률
price_df["daily_rtn"] = price_df["Adj Close"].pct_change()
price_df["st_rtn"] = (1 + price_df["daily_rtn"]).cumprod()
print(
    price_df.loc[price_df.index[0], "Adj Close"],
    price_df.loc[price_df.index[-1], "Adj Close"],
)
# price_df["st_rtn"].plot(figsize=(10, 6))

# 특정 기간의 수익률
base_date = "2011-01-03"
# 기준일의 수익으로 나눠버리면 그 때부터의 수익률로 바뀐다...
# 그냥 st_rtn 컬럼명으로 만들면 serise가 나오고
# tmp_df = price_df.loc[base_date:, "st_rtn"] / price_df.loc[base_date, "st_rtn"]
# [st_rtn]으로 슬라이싱 하면 dataframe이 나온다...
tmp_df = price_df.loc[base_date:, ["st_rtn"]] / price_df.loc[base_date, ["st_rtn"]]
last_date = tmp_df.index[-1]
print(type(tmp_df))
# tmp_df가 series라면 "st_rtn"은 필요 없고, dataframe이면 필요하고
# print("누적 수익:", tmp_df.loc[last_date])
print("누적 수익:", tmp_df.loc[last_date, "st_rtn"])
# tmp_df.plot(figsize=(10, 6))

# 연평균 복리 수익률
CAGR = price_df.loc[last_date, "st_rtn"] ** (252 / len(price_df)) - 1
print("연평균 복리 수익률(CAGR):", round(CAGR * 100, 2), "%")

# 최대 낙폭 MDD
# 레코드 넘어가며 그 때까지 최대값
historical_max = price_df["Adj Close"].cummax()
# 111쪽 최대 낙폭 식 참고...daily로
daily_drawdown = price_df["Adj Close"] / historical_max - 1.0
historical_dd = daily_drawdown.cummin()
MDD = historical_dd.min()
print("최대 낙폭(MDD):", round(-MDD * 100, 2), "%")
# historical_dd.plot()

# 수익률 변동성 Vol
VOL = np.std(price_df["daily_rtn"].dropna()) * np.sqrt(252)
print("수익률 변동성(Vol):", round(VOL * 100, 2), "%")

# 샤프 지수(사후적)
Sharpe = (
    np.mean(price_df["daily_rtn"].dropna())
    / np.std(price_df["daily_rtn"].dropna())
    * np.sqrt(252)
)
print("샤프 지수(Sharpe):", round(Sharpe, 2))

plt.show()
