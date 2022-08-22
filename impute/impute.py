import numpy as np
import os
import pandas as pd
import pathlib
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import subprocess

# resolveで絶対パスになる
data_dir = pathlib.Path(__file__).resolve().parent.parent.joinpath("data")

train = pd.read_csv(data_dir.joinpath("train.csv"))
test = pd.read_csv(data_dir.joinpath("test.csv"))
