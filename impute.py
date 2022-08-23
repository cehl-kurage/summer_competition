import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer
from data_loader import data_loader

# resolveで絶対パスになる
dataset = data_loader("onehot")
train, test = dataset["train"], dataset["test"]

imputer = IterativeImputer(random_state=42)

common_columns = train.drop(["product_code", "failure"], axis="columns").columns
failure = train.failure
product_code_train, product_code_test = train.product_code, test.product_code

print(f"imputing...")
imputed_train = imputer.fit_transform(train[common_columns])
imputed_test = imputer.transform(test[common_columns])
print("finished imputing")

train = pd.DataFrame(imputed_train, columns=common_columns)
print(train)
