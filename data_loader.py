from pathlib import Path
import pandas as pd

data_dir = Path(__file__).resolve().parent.joinpath("data")


def data_loader():
    train_csv = data_dir.joinpath("train.csv")
    test_csv = data_dir.joinpath("test.csv")
    train = pd.read_csv(train_csv)
    test = pd.read_csv(test_csv)
    return {"train": train, "test": test}
