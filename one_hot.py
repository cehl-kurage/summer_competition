from data_loader import data_loader
import pandas as pd

dataset = data_loader()
train, test = dataset["train"], dataset["test"]

column_onehot = ["attribute_0", "attribute_1"]
train_dummied = pd.get_dummies(train, columns=column_onehot)
test_dummied = pd.get_dummies(test, columns=column_onehot)

# 頻繁に実行するファイルじゃないからパス書き込んじゃった
train_dummied.to_csv("data/onehot/train.csv", index=False)
test_dummied.to_csv("data/onehot/test.csv", index=False)
