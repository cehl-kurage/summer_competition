import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer
from data_loader import data_loader

# resolveで絶対パスになる
dataset = data_loader("onehot")
train, test = dataset["train"], dataset["test"]

imputer = IterativeImputer(random_state=42)

train_need_imputing = train.drop(["product_code", "failure"], axis="columns")
test_need_imputing = test.drop(["product_code"], axis="columns")
failure = train.failure

print(f"imputing...")
imputed_train = imputer.fit_transform(train_need_imputing)
imputed_test = imputer.transform(test_need_imputing)
print("finished imputing")

_train = pd.DataFrame(imputed_train, columns=train_need_imputing.columns)
_test = pd.DataFrame(imputed_test, columns=test_need_imputing.columns)
train.update(_train)
test.update(_test)

train.to_csv("/home/yusaku/projects/summer_competition/data/imputed/train.csv")
test.to_csv("/home/yusaku/projects/summer_competition/data/imputed/test.csv")
