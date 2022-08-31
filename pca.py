# loadingを対数変換したものに対して予測してみる

from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import TruncatedSVD
from data_loader import data_loader
import numpy as np
import pandas as pd

n_components = 8
auc_list = []
test_pred_list = []
importance_list = []
kf = GroupKFold(n_splits=5)

dataset = data_loader("scaled")
train, test = dataset["train"], dataset["test"]
train.set_index("id", inplace=True)
test.set_index("id", inplace=True)

train_failure_0 = train.query("failure == 0")
train_failure_1 = train.query("failure == 1")
train_failure_0_sampled = train_failure_0.sample(n=len(train_failure_1.index))
print(f"failure=0: {len(train_failure_0_sampled.index)}")
print(f"failure=1: {len(train_failure_1.index)}")


# failure=1と同じ数しかfailure=0を学習しない
train = pd.concat([train_failure_0_sampled, train_failure_1], axis=0)

tsvd = TruncatedSVD(n_components=n_components)
tsvd_columns = train.drop(["product_code", "failure"], axis=1).columns
reduced_train = pd.DataFrame(tsvd.fit_transform(train[tsvd_columns])).set_index(
    train.index
)
reduced_test = tsvd.fit_transform(test[tsvd_columns])

for fold, (train_indice, val_indice) in enumerate(
    kf.split(reduced_train, train.failure, train.product_code)
):
    # training用
    X_train, y_train = (
        reduced_train.iloc[train_indice],
        train.failure.iloc[train_indice],  # こっちがy
    )

    X_val = reduced_train.iloc[val_indice]
    y_val = train.failure.iloc[val_indice]

    X_test = reduced_test.copy()

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

    y_test_pred = logi_regressor.predict_proba(X_test)


importance = np.mean(importance_list, axis=0)
feature_importance = pd.DataFrame([importance], columns=range(n_components))
feature_importance.to_csv(
    f"/home/yusaku/projects/summer_competition/results/pca/n_coponents{n_components}submission.csv",
    index=False,
)

print("-" * 30)
submission = pd.Series(
    np.argmax(y_test_pred, axis=1), name="failure", index=test.index.astype(int)
)

submission.to_csv(
    f"/home/yusaku/projects/summer_competition/results/pca/n_components{n_components}/submission.csv",
    index=True,
)
print(f"Test => label=1: {submission.sum()}(total: {len(submission.index)})")
