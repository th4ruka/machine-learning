"""s_06_ep_07 v11: blend tuned LightGBM with CatBoost.

Requires: uv add catboost

Concepts in this version
------------------------
- v06 (LGB+XGB) failed because both are the same algorithm at heart: histogram
  GBDTs that split raw categorical codes. CatBoost is a different recipe:
  it converts categoricals into ORDERED TARGET STATISTICS (running per-category
  mean of the target, computed causally to avoid leakage) and uses ordered
  boosting. Different representation -> a real chance of decorrelated errors
  on the 6 categorical columns.
- CatBoost quirks handled here: it refuses NaN in categorical features (we
  fill "missing" as an explicit category); class balancing is
  auto_class_weights="Balanced"; a custom metric is a small class, and we
  evaluate it every 10 iterations (metric_period) since a Python callback
  every round is slow.
- Blend weight alpha tuned on OOF as in v06. If CatBoost alone is close to
  LGBM and the blend beats both, diversity worked; if alpha pins to 1.0,
  the categorical columns simply don't carry extra signal and we're done
  with model-side ideas.
"""

import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold

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

LGB_PARAMS = {
    "learning_rate": 0.05,
    "num_leaves": 41,
    "min_child_samples": 35,
    "feature_fraction": 0.8401695624538926,
    "bagging_fraction": 0.8092497041550049,
    "bagging_freq": 1,
    "reg_alpha": 0.38651187383382324,
    "reg_lambda": 0.06292313532864219,
}

train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    feature_cols = [c for c in df.columns if c not in ("id", "health_condition")]
    df["n_missing"] = df[feature_cols].isna().sum(axis=1)
    for col in feature_cols:
        if df[col].isna().any():
            df[f"{col}_missing"] = df[col].isna().astype(int)
    df["cal_per_step"] = df["calorie_expenditure"] / (df["step_count"] + 1)
    df["steps_per_exercise_min"] = df["step_count"] / (df["exercise_duration"] + 1)
    df["cal_per_exercise_min"] = df["calorie_expenditure"] / (df["exercise_duration"] + 1)
    df["water_per_cal"] = df["water_intake"] / (df["calorie_expenditure"] + 1)
    df["hr_x_bmi"] = df["heart_rate"] * df["bmi"]
    df["sleep_x_exercise"] = df["sleep_duration"] * df["exercise_duration"]
    for col in CAT_COLS:
        df[col] = df[col].astype("category")
    return df


train = add_features(train)
test = add_features(test)

X = train.drop(columns=["id", "health_condition"])
y = train["health_condition"].map(LABEL_MAP)
X_test = test.drop(columns=["id"])
y_arr = y.to_numpy()

# CatBoost view of the data: categorical NaN -> explicit "missing" string
X_cb = X.copy()
X_test_cb = X_test.copy()
for col in CAT_COLS:
    X_cb[col] = X_cb[col].cat.add_categories("missing").fillna("missing").astype(str)
    X_test_cb[col] = X_test_cb[col].cat.add_categories("missing").fillna("missing").astype(str)


def balanced_accuracy_eval(y_true, y_prob):
    return "balanced_accuracy", balanced_accuracy_score(y_true, y_prob.argmax(axis=1)), True


class CatBoostBalancedAccuracy:
    def get_final_error(self, error, weight):
        return error

    def is_max_optimal(self):
        return True

    def evaluate(self, approxes, target, weight):
        preds = np.array(approxes).argmax(axis=0)
        t = np.asarray(target).astype(int)
        score = np.mean([(preds[t == c] == c).mean() for c in range(3)])
        return score, 1.0


skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
oof = {"lgb": np.zeros((len(train), 3)), "cat": np.zeros((len(train), 3))}
test_probs = {"lgb": np.zeros((len(test), 3)), "cat": np.zeros((len(test), 3))}

for fold, (idx_tr, idx_va) in enumerate(skf.split(X, y)):
    lgbm = lgb.LGBMClassifier(
        n_estimators=8000,
        class_weight="balanced",
        metric="None",
        random_state=42,
        verbose=-1,
        **LGB_PARAMS,
    )
    lgbm.fit(
        X.iloc[idx_tr],
        y.iloc[idx_tr],
        eval_set=[(X.iloc[idx_va], y.iloc[idx_va])],
        eval_metric=balanced_accuracy_eval,
        callbacks=[lgb.early_stopping(200, verbose=False)],
    )
    oof["lgb"][idx_va] = lgbm.predict_proba(X.iloc[idx_va])
    test_probs["lgb"] += lgbm.predict_proba(X_test) / N_SPLITS

    cbm = CatBoostClassifier(
        iterations=8000,
        learning_rate=0.05,
        auto_class_weights="Balanced",
        eval_metric=CatBoostBalancedAccuracy(),
        metric_period=10,
        early_stopping_rounds=200,
        cat_features=CAT_COLS,
        random_seed=42,
        verbose=0,
    )
    cbm.fit(
        X_cb.iloc[idx_tr],
        y.iloc[idx_tr],
        eval_set=(X_cb.iloc[idx_va], y.iloc[idx_va]),
    )
    oof["cat"][idx_va] = cbm.predict_proba(X_cb.iloc[idx_va])
    test_probs["cat"] += cbm.predict_proba(X_test_cb) / N_SPLITS

    s_lgb = balanced_accuracy_score(y.iloc[idx_va], oof["lgb"][idx_va].argmax(axis=1))
    s_cat = balanced_accuracy_score(y.iloc[idx_va], oof["cat"][idx_va].argmax(axis=1))
    print(f"fold {fold}: lgb={s_lgb:.5f}  cat={s_cat:.5f} (cat_best_iter={cbm.get_best_iteration()})")


def bal_acc(pred):
    return np.mean([(pred[y_arr == c] == c).mean() for c in range(3)])


print(f"\nOOF lgb : {bal_acc(oof['lgb'].argmax(axis=1)):.5f}")
print(f"OOF cat : {bal_acc(oof['cat'].argmax(axis=1)):.5f}")

best_alpha, best_score = 1.0, 0.0
for alpha in np.linspace(0, 1, 21):
    s = bal_acc((alpha * oof["lgb"] + (1 - alpha) * oof["cat"]).argmax(axis=1))
    if s > best_score:
        best_alpha, best_score = alpha, s

print(f"\nbest blend: alpha={best_alpha:.2f} (lgb share) -> OOF {best_score:.5f}")
print("(v09 reference: 0.94986)")

blend_test = best_alpha * test_probs["lgb"] + (1 - best_alpha) * test_probs["cat"]
submission = pd.DataFrame(
    {
        "id": test["id"],
        "health_condition": [INV_LABEL_MAP[i] for i in blend_test.argmax(axis=1)],
    }
)
submission.to_csv("submissions/submission_v11.csv", index=False)
print("\npredicted class distribution:")
print(submission["health_condition"].value_counts(normalize=True).round(3))
