from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from data_loader import data_loader
import numpy as np
import pandas as pd


auc_list = []
test_pred_list = []
importance_list = []
kf = GroupKFold(n_splits=5)

dataset = data_loader("scaled")
train, test = dataset["train"], dataset["test"]
test_id = test.id
test.drop(["id", "product_code"], axis="columns", inplace=True)  # product_codeは必要ない

train_failure_0 = train.query("failure == 0")
train_failure_1 = train.query("failure == 1")
train_failure_0_sampled = train_failure_0.sample(n=len(train_failure_1.index))
print(f"failure=0: {len(train_failure_0_sampled.index)}")
print(f"failure=1: {len(train_failure_1.index)}")

# failure=1と同じ数しかfailure=0を学習しない
train = pd.concat([train_failure_0_sampled, train_failure_1], axis=0)
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
        C=0.01, penalty="l1", random_state=42, solver="liblinear"
    )  # https://qiita.com/hannnari0918/items/a0e2184fb4ff8af9981c
    # 学習
    logi_regressor.fit(X_train, y_train)
    importance_list.append(logi_regressor.coef_.ravel())
    # 検証
    y_val_pred = logi_regressor.predict_proba(X_val)
    auc = roc_auc_score(y_val, y_val_pred[:, 1])
    print(
        f"Fold: {fold} => auc = {auc}, label_1: {np.argmax(y_val_pred, axis=1).sum()}(total={len(y_val_pred)})"
    )

logi_regressor.fit(train[test.columns], train.failure)
y_test_pred = np.array(logi_regressor.predict_proba(X_test[X_train.columns]))
print(y_test_pred.shape)
pd.Series(y_test_pred[:, 1], index=test_id.astype(int), name="failure").to_csv(
    "/home/yusaku/projects/summer_competition/results/experiment2/submission_proba.csv"
)

features = logi_regressor.feature_names_in_
importance = np.mean(importance_list, axis=0)
feature_importance = pd.DataFrame([importance], columns=features)
feature_importance.to_csv(
    "/home/yusaku/projects/summer_competition/results/experiment2/feature_importance.csv",
    index=False,
)

print("-" * 30)
test_pred_labels = pd.Series(np.argmax(y_test_pred, axis=1), name="failure")
submission = pd.concat([test_id.astype(int), test_pred_labels], axis="columns")
submission.to_csv(
    "/home/yusaku/projects/summer_competition/results/experiment2/submission.csv",
    index=False,
)
print(f"Test => label=1: {submission.failure.sum()}")
