"""s_06_ep_07 v12: endgame pipeline - 10-fold x 3-seed tuned LightGBM.

Concepts in this version
------------------------
- With k folds, each model trains on (k-1)/k of the data. Moving 5 -> 10
  folds means 90% instead of 80% - and every gain in this competition since
  v03 has come from data quantity and variance reduction, not model
  cleverness (v11: even CatBoost's different categorical view added nothing).
- CAUTION when reading the CV number: 10-fold scores run slightly higher
  than 5-fold scores for the same pipeline, partly because fold models
  genuinely see more data (real - it transfers to test) and partly because
  smaller validation folds make early stopping's peak-picking more
  flattering (doesn't transfer). Compare v12 only against other 10-fold
  runs; the LB is the fair arbiter against v09/v10.
- The test prediction now averages 30 models (10 folds x 3 seeds) - maximal
  variance reduction, which v09's LB jump showed pays off beyond what CV
  can measure.
- Params are v09's Optuna winners, frozen. Roughly 2x v10's runtime.
"""

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold

N_SPLITS = 10
SEEDS = [42, 101, 2026]
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

TUNED_PARAMS = {
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


def balanced_accuracy_eval(y_true, y_prob):
    return "balanced_accuracy", balanced_accuracy_score(y_true, y_prob.argmax(axis=1)), True


skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
scores = []
test_probs = np.zeros((len(test), 3))
oof_probs = np.zeros((len(train), 3))

for fold, (idx_tr, idx_va) in enumerate(skf.split(X, y)):
    fold_va_probs = np.zeros((len(idx_va), 3))
    iters = []
    for seed in SEEDS:
        model = lgb.LGBMClassifier(
            n_estimators=8000,
            class_weight="balanced",
            metric="None",
            random_state=seed,
            verbose=-1,
            **TUNED_PARAMS,
        )
        model.fit(
            X.iloc[idx_tr],
            y.iloc[idx_tr],
            eval_set=[(X.iloc[idx_va], y.iloc[idx_va])],
            eval_metric=balanced_accuracy_eval,
            callbacks=[lgb.early_stopping(200, verbose=False)],
        )
        fold_va_probs += model.predict_proba(X.iloc[idx_va]) / len(SEEDS)
        test_probs += model.predict_proba(X_test) / (N_SPLITS * len(SEEDS))
        iters.append(model.best_iteration_)
    oof_probs[idx_va] = fold_va_probs
    score = balanced_accuracy_score(y.iloc[idx_va], fold_va_probs.argmax(axis=1))
    scores.append(score)
    print(f"fold {fold}: {score:.5f} (best_iters={iters})")

print(f"\nCV balanced accuracy: {np.mean(scores):.5f} +/- {np.std(scores):.5f}")
print("(5-fold v09 reference: 0.94986 - not directly comparable, see docstring)")

oof_pred = oof_probs.argmax(axis=1)
print("\nper-class recall (OOF):")
for cls_idx, cls_name in INV_LABEL_MAP.items():
    mask = y_arr == cls_idx
    print(f"  {cls_name:10s} {(oof_pred[mask] == cls_idx).mean():.5f}")

submission = pd.DataFrame(
    {
        "id": test["id"],
        "health_condition": [INV_LABEL_MAP[i] for i in test_probs.argmax(axis=1)],
    }
)
submission.to_csv("submissions/submission_v12.csv", index=False)
print("\npredicted class distribution:")
print(submission["health_condition"].value_counts(normalize=True).round(3))
