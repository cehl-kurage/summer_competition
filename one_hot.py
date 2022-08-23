from data_loader import data_loader
import pandas as pd
import numpy as np


dataset = data_loader()
train, test = dataset["train"], dataset["test"]

column_onehot = ["attribute_0", "attribute_1"]
train_dummied = pd.get_dummies(train, columns=column_onehot)
test_dummied = pd.get_dummies(test, columns=column_onehot)

# attribute_1 == "material_7"の列がtrainにはないので、追加しておく
train_dummied["attribute_1_material_7"] = 0

# attribute_1 == "material_8"の列がtestにはないので、追加しておく
test_dummied["attribute_1_material_8"] = 0

# 頻繁に実行するファイルじゃないからパス書き込んじゃった
train_dummied.to_csv("data/onehot/train.csv", index=False)
test_dummied.to_csv("data/onehot/test.csv", index=False)
