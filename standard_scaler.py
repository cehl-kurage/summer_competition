from sklearn.preprocessing import StandardScaler
from data_loader import data_loader
import subprocess
import pandas as pd
import pathlib

dataset = data_loader("imputed")
train, test = dataset["train"], dataset["test"]
columns_need_scaling = train.drop(
    ["id", "product_code", "failure"], axis="columns"
).columns
scaler = StandardScaler()
_train = pd.DataFrame(
    scaler.fit_transform(train[columns_need_scaling]),
    columns=columns_need_scaling,
)
_test = pd.DataFrame(
    scaler.transform(test[columns_need_scaling]), columns=columns_need_scaling
)
train.update(_train)  # 一致する列や行の値を引数のDataFrameで更新してくれる
test.update(_test)

this_file = pathlib.Path(__file__).resolve()
save_dir = this_file.parent.joinpath("data/scaled")

train.to_csv(save_dir.joinpath("train.csv"), index=False)
test.to_csv(save_dir.joinpath("test.csv"), index=False)
