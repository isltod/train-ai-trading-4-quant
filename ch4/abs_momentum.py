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
