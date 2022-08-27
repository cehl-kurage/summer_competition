from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from data_loader import data_loader


auc_list = []
test_pred_list = []
importance_list = []
kf = GroupKFold(n_splits=5)

dataset = data_loader("scaled")
train, test = dataset["train"], dataset["test"]
test.drop("product_code", axis="columns", inplace=True)  # product_codeは必要ない

print(train.isna().sum())

for fold, (train_indice, val_indice) in enumerate(
    kf.split(train, train.failure, train.product_code)
):
    # training用
    _train = train.drop("product_code", axis="columns")  # product_codeは必要ない
    X_train, y_train = (
        _train.iloc[train_indice][test.columns],
        _train.failure.iloc[train_indice],
    )
    X_val, y_val = (
        _train.iloc[val_indice][test.columns],
        _train.failure.iloc[val_indice],
    )
    X_test = test.copy()
    logi_regressor = LogisticRegression(
        penalty="l1", random_state=42, solver="liblinear"
    )  # https://qiita.com/hannnari0918/items/a0e2184fb4ff8af9981c
    # 学習
    logi_regressor.fit(X_train, y_train)
    importance_list.appen(
        logi_regressor.named_steps["logisticregression"].coef_.rabel()
    )
    # 検証
    y_val_pred = logi_regressor.predict(y_train)
    auc = roc_auc_score(y_val, y_val_pred)
    print(f"Fold: {fold} => auc = {auc}")
