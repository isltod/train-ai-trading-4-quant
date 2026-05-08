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


# plt.show()
