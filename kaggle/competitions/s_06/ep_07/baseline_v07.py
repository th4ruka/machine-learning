"""s_06_ep_07 v07: feature engineering (ratios + missingness indicators).

Concepts in this version
------------------------
- Decision trees split on ONE feature at a time (x < threshold), so they can
  approximate sums of effects easily but struggle with RATIOS: to model
  calories-per-step a tree needs a staircase of many splits. Handing the model
  the ratio directly as a column is free information.
- Missingness itself can be signal. In real data, *why* a value is missing
  often correlates with the outcome (e.g. people who don't track sleep). We
  add an is-missing flag per gappy column plus a per-row missing count.
  LightGBM already routes NaN to one side of each split, so flags are partly
  redundant - the CV score tells us whether they add anything on top.
- Model config is frozen at v03's (our proven setup) so any score change is
  attributable to the features alone. Change one thing at a time.
"""

import lightgbm as lgb
import numpy as np
import pandas as pd
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

train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    feature_cols = [c for c in df.columns if c not in ("id", "health_condition")]

    # missingness signal
    df["n_missing"] = df[feature_cols].isna().sum(axis=1)
    for col in feature_cols:
        if df[col].isna().any():
            df[f"{col}_missing"] = df[col].isna().astype(int)

    # ratios trees can't easily build from raw columns
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
print(f"features: {X.shape[1]}")


def balanced_accuracy_eval(y_true, y_prob):
    return "balanced_accuracy", balanced_accuracy_score(y_true, y_prob.argmax(axis=1)), True


skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
scores = []
test_probs = np.zeros((len(test), 3))
oof_pred = np.zeros(len(train), dtype=int)
importances = np.zeros(X.shape[1])

for fold, (idx_tr, idx_va) in enumerate(skf.split(X, y)):
    model = lgb.LGBMClassifier(
        n_estimators=8000,
        learning_rate=0.05,
        num_leaves=63,
        class_weight="balanced",
        metric="None",
        random_state=42,
        verbose=-1,
    )
    model.fit(
        X.iloc[idx_tr],
        y.iloc[idx_tr],
        eval_set=[(X.iloc[idx_va], y.iloc[idx_va])],
        eval_metric=balanced_accuracy_eval,
        callbacks=[lgb.early_stopping(200, verbose=False)],
    )
    pred = model.predict(X.iloc[idx_va])
    oof_pred[idx_va] = pred
    score = balanced_accuracy_score(y.iloc[idx_va], pred)
    scores.append(score)
    test_probs += model.predict_proba(X_test) / N_SPLITS
    importances += model.feature_importances_ / N_SPLITS
    print(f"fold {fold}: {score:.5f} (best_iter={model.best_iteration_})")

print(f"\nCV balanced accuracy: {np.mean(scores):.5f} +/- {np.std(scores):.5f}")
print("(v03 reference: 0.94948)")

print("\ntop 15 features by split importance:")
order = np.argsort(importances)[::-1][:15]
for i in order:
    print(f"  {X.columns[i]:28s} {importances[i]:8.0f}")

submission = pd.DataFrame(
    {
        "id": test["id"],
        "health_condition": [INV_LABEL_MAP[i] for i in test_probs.argmax(axis=1)],
    }
)
submission.to_csv("submission_v07.csv", index=False)
print("\npredicted class distribution:")
print(submission["health_condition"].value_counts(normalize=True).round(3))
