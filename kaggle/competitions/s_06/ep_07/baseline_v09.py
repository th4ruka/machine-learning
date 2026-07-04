"""s_06_ep_07 v09: hyperparameter tuning with Optuna on the v07 pipeline.

Requires: uv add optuna

Concepts in this version
------------------------
- Parameters (tree splits) are learned from data; HYPERparameters (tree size,
  regularization) shape how learning happens and must be chosen from outside.
  So far num_leaves=63 etc. were educated guesses.
- Optuna does Bayesian-ish search (TPE): it fits a model of "which regions of
  hyperparameter space score well" and samples promising regions more often -
  much more efficient than grid search when knobs interact.
- Budget trick: searching on ONE fold (~1-2 min/trial) instead of full 5-fold
  CV lets us afford ~40 trials. Risk: with many trials we can overfit the
  search to that fold's quirks. Guard: the winner is re-scored with full
  5-fold CV, and only that confirmed number counts against v07's 0.94970.
- v07/v08 verdicts baked in: engineered features kept, original dataset
  dropped (it consistently hurt across folds).
"""

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold

N_SPLITS = 5
N_TRIALS = 40
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


def balanced_accuracy_eval(y_true, y_prob):
    return "balanced_accuracy", balanced_accuracy_score(y_true, y_prob.argmax(axis=1)), True


skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
folds = list(skf.split(X, y))


def fit_and_score(params, idx_tr, idx_va):
    model = lgb.LGBMClassifier(
        n_estimators=8000,
        class_weight="balanced",
        metric="None",
        random_state=42,
        verbose=-1,
        **params,
    )
    model.fit(
        X.iloc[idx_tr],
        y.iloc[idx_tr],
        eval_set=[(X.iloc[idx_va], y.iloc[idx_va])],
        eval_metric=balanced_accuracy_eval,
        callbacks=[lgb.early_stopping(200, verbose=False)],
    )
    return model


def objective(trial):
    params = {
        "learning_rate": 0.05,
        "num_leaves": trial.suggest_int("num_leaves", 15, 255, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 300, log=True),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
        "bagging_freq": 1,
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
    }
    idx_tr, idx_va = folds[0]  # search on one fold for speed
    model = fit_and_score(params, idx_tr, idx_va)
    pred = model.predict(X.iloc[idx_va])
    return balanced_accuracy_score(y.iloc[idx_va], pred)


optuna.logging.set_verbosity(optuna.logging.WARNING)
study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(objective, n_trials=N_TRIALS, timeout=5400, show_progress_bar=True)

print(f"\nbest fold-0 score: {study.best_value:.5f} (v07 fold 0 was 0.95044)")
print("best params:", study.best_params)

best_params = {
    "learning_rate": 0.05,
    "bagging_freq": 1,
    **study.best_params,
}

# confirm with full 5-fold CV - only this number is comparable to v07
scores = []
test_probs = np.zeros((len(test), 3))
oof_pred = np.zeros(len(train), dtype=int)
for fold, (idx_tr, idx_va) in enumerate(folds):
    model = fit_and_score(best_params, idx_tr, idx_va)
    pred = model.predict(X.iloc[idx_va])
    oof_pred[idx_va] = pred
    score = balanced_accuracy_score(y.iloc[idx_va], pred)
    scores.append(score)
    test_probs += model.predict_proba(X_test) / N_SPLITS
    print(f"fold {fold}: {score:.5f} (best_iter={model.best_iteration_})")

print(f"\nCV balanced accuracy: {np.mean(scores):.5f} +/- {np.std(scores):.5f}")
print("(v07 reference: 0.94970)")

submission = pd.DataFrame(
    {
        "id": test["id"],
        "health_condition": [INV_LABEL_MAP[i] for i in test_probs.argmax(axis=1)],
    }
)
submission.to_csv("submission_v09.csv", index=False)
print("\npredicted class distribution:")
print(submission["health_condition"].value_counts(normalize=True).round(3))
