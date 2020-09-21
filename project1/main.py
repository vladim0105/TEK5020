import pandas as pd


if __name__ == "__main__":
    data = pd.read_csv('data/ds-1.txt', sep="  ", header=None)
    data.columns = ["class", "feature_1", "feature_2", "feature_3", "feature_4"]

    print(data.head(10))