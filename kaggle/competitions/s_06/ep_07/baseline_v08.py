"""s_06_ep_07 v08: augment training with the original 50k-row dataset.

Concepts in this version
------------------------
- Playground data is synthetic, generated from a real "original" dataset
  (here: ziya07/college-student-health-behavior-dataset). Adding the real
  rows gives the model extra genuine signal - historically the single
  biggest lever in these competitions.
- THE CRUCIAL RULE: original rows go into the TRAINING side of each fold
  only, never into validation. We are scored on competition-distribution
  data ("close to, but not exactly the same" as the original), so validation
  must stay purely competition rows - otherwise CV would measure performance
  on a mixture we're not graded on, and we couldn't compare to v03's 0.94948.
- Model config stays frozen at v03's: any score change = the extra data.
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
orig = pd.read_csv("data/student_health_dataset_50k.csv")

feature_cols = [c for c in train.columns if c not in ("id", "health_condition")]

# stack competition train + original so categorical codes are shared, then split
X_all = pd.concat([train[feature_cols], orig[feature_cols]], ignore_index=True)
for col in CAT_COLS:
    X_all[col] = X_all[col].astype("category")
    test[col] = test[col].astype(X_all[col].dtype)

y_all = pd.concat(
    [train["health_condition"], orig["health_condition"]], ignore_index=True
).map(LABEL_MAP)

n_comp = len(train)  # rows 0..n_comp-1 are competition data, the rest is original
orig_idx = np.arange(n_comp, len(X_all))
X_test = test[feature_cols]

print(f"competition rows: {n_comp}, original rows: {len(orig_idx)}")


def balanced_accuracy_eval(y_true, y_prob):
    return "balanced_accuracy", balanced_accuracy_score(y_true, y_prob.argmax(axis=1)), True


skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
scores = []
test_probs = np.zeros((len(test), 3))
oof_pred = np.zeros(n_comp, dtype=int)

# folds are defined on competition rows only; original rows join every training set
for fold, (idx_tr, idx_va) in enumerate(skf.split(X_all.iloc[:n_comp], y_all.iloc[:n_comp])):
    idx_tr_aug = np.concatenate([idx_tr, orig_idx])
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
        X_all.iloc[idx_tr_aug],
        y_all.iloc[idx_tr_aug],
        eval_set=[(X_all.iloc[idx_va], y_all.iloc[idx_va])],
        eval_metric=balanced_accuracy_eval,
        callbacks=[lgb.early_stopping(200, verbose=False)],
    )
    pred = model.predict(X_all.iloc[idx_va])
    oof_pred[idx_va] = pred
    score = balanced_accuracy_score(y_all.iloc[idx_va], pred)
    scores.append(score)
    test_probs += model.predict_proba(X_test) / N_SPLITS
    print(f"fold {fold}: {score:.5f} (best_iter={model.best_iteration_})")

print(f"\nCV balanced accuracy: {np.mean(scores):.5f} +/- {np.std(scores):.5f}")
print("(v03 reference: 0.94948)")

y_comp = y_all.iloc[:n_comp].to_numpy()
print("\nper-class recall (OOF):")
for cls_idx, cls_name in INV_LABEL_MAP.items():
    mask = y_comp == cls_idx
    print(f"  {cls_name:10s} {(oof_pred[mask] == cls_idx).mean():.5f}")

submission = pd.DataFrame(
    {
        "id": test["id"],
        "health_condition": [INV_LABEL_MAP[i] for i in test_probs.argmax(axis=1)],
    }
)
submission.to_csv("submission_v08.csv", index=False)
print("\npredicted class distribution:")
print(submission["health_condition"].value_counts(normalize=True).round(3))
