import pandas as pd
import sweetviz as sv

data_dir = "/home/yusaku/projects/summer_competition/data/"
train = pd.read_csv(data_dir + "train.csv")
test = pd.read_csv(data_dir + "test.csv")

# targetはfailure
report = sv.compare([train, "train"], [test, "test"], "failure")
report.show_html("report.html")
