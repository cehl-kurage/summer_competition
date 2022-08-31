import numpy as np
import pandas as pd
from data_loader import data_loader

dataset = data_loader("imputed")  # logするのに負の値使えない

dataset["train"].set_index("id", inplace=True)
dataset["test"].set_index("id", inplace=True)

train_index = dataset["train"].index
test_index = dataset["test"].index

train_test = pd.concat(dataset.values())
train_test.loading = np.log(train_test.loading)

train = train_test.iloc[train_index]
test = train_test.iloc[test_index]

train.to_csv(
    "/home/yusaku/projects/summer_competition/data/log/train.csv",
)
test.to_csv(
    "/home/yusaku/projects/summer_competition/data/log/test.csv",
)
