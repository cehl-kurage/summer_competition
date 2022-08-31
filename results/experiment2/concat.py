import pandas as pd

myresult = pd.read_csv("results/experiment2/submission_proba.csv", index_col="id")
takumi = pd.read_csv("/mnt/c/users/yusak/downloads/logi_v1.csv", index_col="id")

id = takumi.index

concat_result = (myresult + takumi) / 2
concat_result.to_csv("results/submission.csv")
