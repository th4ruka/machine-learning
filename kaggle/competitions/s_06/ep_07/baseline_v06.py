"""s_06_ep_07 v06: blend LightGBM + XGBoost probabilities.

Concepts in this version
------------------------
- A single model family hit its ceiling (v03 ~= v05 ~= 0.9495). The next gain
  comes from ensembling: two decent models that make *different* mistakes can
  average to a better prediction than either alone. What matters is error
  decorrelation, not just individual strength.
- XGBoost builds trees differently from LightGBM (depth-wise growth, different
  handling of categoricals and missing values), so its errors are only partly
  correlated - a good blend partner.
- XGBoost has no class_weight parameter; we pass per-row sample_weight
  instead, which is the same thing stated differently (each row weighted
  inversely to its class frequency).
- XGBoost's early stopping *minimizes* a callable metric, so we hand it
  NEGATIVE balanced accuracy (same trick as minimizing -f to maximize f).
- The blend weight alpha (lgb_prob * alpha + xgb_prob * (1 - alpha)) is a
  1-parameter decision tuned on out-of-fold predictions, like v04 - one knob
  on 690k held-out rows, so overfitting risk is negligible.
"""

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

N_SPLITS = 5
CAT_COLS = [
    "diet_type",
    "stress_level",
    "sleep_quality",
    "physical_activity_level",
    "smoking_alcohol",
    "gender",
]
LABEL_MAP = {"at-risk": 0, "unhealthy": 1, "fit": 2}
INV_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}

train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

for df in (train, test):
    for col in CAT_COLS:
        df[col] = df[col].astype("category")

X = train.drop(columns=["id", "health_condition"])
y = train["health_condition"].map(LABEL_MAP)
X_test = test.drop(columns=["id"])
y_arr = y.to_numpy()


def balanced_accuracy_eval(y_true, y_prob):
    return "balanced_accuracy", balanced_accuracy_score(y_true, y_prob.argmax(axis=1)), True


def neg_bal_acc(y_true, y_prob):  # xgboost minimizes callables
    return -balanced_accuracy_score(y_true, y_prob.argmax(axis=1))


skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
oof = {"lgb": np.zeros((len(train), 3)), "xgb": np.zeros((len(train), 3))}
test_probs = {"lgb": np.zeros((len(test), 3)), "xgb": np.zeros((len(test), 3))}

for fold, (idx_tr, idx_va) in enumerate(skf.split(X, y)):
    X_tr, X_va = X.iloc[idx_tr], X.iloc[idx_va]
    y_tr, y_va = y.iloc[idx_tr], y.iloc[idx_va]

    lgbm = lgb.LGBMClassifier(
        n_estimators=8000,
        learning_rate=0.05,
        num_leaves=63,
        class_weight="balanced",
        metric="None",
        random_state=42,
        verbose=-1,
    )
    lgbm.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        eval_metric=balanced_accuracy_eval,
        callbacks=[lgb.early_stopping(200, verbose=False)],
    )
    oof["lgb"][idx_va] = lgbm.predict_proba(X_va)
    test_probs["lgb"] += lgbm.predict_proba(X_test) / N_SPLITS

    xgbm = XGBClassifier(
        n_estimators=8000,
        learning_rate=0.05,
        max_depth=8,
        tree_method="hist",
        enable_categorical=True,
        eval_metric=neg_bal_acc,
        early_stopping_rounds=200,
        random_state=42,
        verbosity=0,
    )
    xgbm.fit(
        X_tr,
        y_tr,
        sample_weight=compute_sample_weight("balanced", y_tr),
        eval_set=[(X_va, y_va)],
        verbose=False,
    )
    oof["xgb"][idx_va] = xgbm.predict_proba(X_va)
    test_probs["xgb"] += xgbm.predict_proba(X_test) / N_SPLITS

    s_lgb = balanced_accuracy_score(y_va, oof["lgb"][idx_va].argmax(axis=1))
    s_xgb = balanced_accuracy_score(y_va, oof["xgb"][idx_va].argmax(axis=1))
    print(f"fold {fold}: lgb={s_lgb:.5f}  xgb={s_xgb:.5f} (xgb_best_iter={xgbm.best_iteration})")


def bal_acc(pred):
    return np.mean([(pred[y_arr == c] == c).mean() for c in range(3)])


print(f"\nOOF lgb : {bal_acc(oof['lgb'].argmax(axis=1)):.5f}")
print(f"OOF xgb : {bal_acc(oof['xgb'].argmax(axis=1)):.5f}")

best_alpha, best_score = 1.0, 0.0
for alpha in np.linspace(0, 1, 21):
    s = bal_acc((alpha * oof["lgb"] + (1 - alpha) * oof["xgb"]).argmax(axis=1))
    if s > best_score:
        best_alpha, best_score = alpha, s

print(f"\nbest blend: alpha={best_alpha:.2f} (lgb share) -> OOF {best_score:.5f}")

blend_oof_pred = (best_alpha * oof["lgb"] + (1 - best_alpha) * oof["xgb"]).argmax(axis=1)
print("\nper-class recall (OOF, blend):")
for cls_idx, cls_name in INV_LABEL_MAP.items():
    mask = y_arr == cls_idx
    print(f"  {cls_name:10s} {(blend_oof_pred[mask] == cls_idx).mean():.5f}")

blend_test = best_alpha * test_probs["lgb"] + (1 - best_alpha) * test_probs["xgb"]
submission = pd.DataFrame(
    {
        "id": test["id"],
        "health_condition": [INV_LABEL_MAP[i] for i in blend_test.argmax(axis=1)],
    }
)
submission.to_csv("submission_v06.csv", index=False)
print("\npredicted class distribution:")
print(submission["health_condition"].value_counts(normalize=True).round(3))
