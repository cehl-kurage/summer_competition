import pandas as pd
import sweetviz as sv
import pathlib

data_directory = pathlib.Path("/home/yusaku/projects/summer_competition/data")
train_path = data_directory.joinpath("train.csv")
print(train_path, data_directory)
train = pd.read_csv()
test = pd.read_csv("../data/test.csv")

report = sv.compare([train, "train"], [test, "test"], "failure")
report.show_html("report.html")
