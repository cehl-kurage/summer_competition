from pathlib import Path
import pandas as pd

data_dir = Path(__file__).resolve().parent.joinpath("data")


def data_loader(option: str):
    """
    option: str data以下でさらにディレクトリを指定するとき

    return: dict trainデータとtestデータの辞書
    """
    if option is None:
        train = pd.read_csv(data_dir.joinpath("train.csv"))
        test = pd.read_csv(data_dir.joinpath("test.csv"))
    else:
        train = pd.read_csv(data_dir.joinpath(f"{option}/train.csv"))
        test = pd.read_csv(data_dir.joinpath(f"{option}/test.csv"))

    return {"train": train, "test": test}
