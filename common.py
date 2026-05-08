import pandas as pd


def change_btc_dt(filename):
    df = pd.read_csv(filename)
    df["Date"] = pd.to_datetime(df["timestamp"], unit="ms") + pd.Timedelta(hours=9)
    newname = filename.replace("_cache", "")
    df.drop(["timestamp", "ls_label", "capital"], axis=1, inplace=True)
    cols = ["Date"] + [col for col in df.columns if col != "Date"]
    df = df[cols]
    df.to_csv(newname, index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    change_btc_dt("data/btc_usdt_1h_cache.csv")
    change_btc_dt("data/btc_usdt_1m_cache.csv")
