"""s_06_ep_07 v05: lower learning rate (0.05 -> 0.02).

Concepts in this version
------------------------
- Each boosting round adds a tree that corrects the current errors; the
  learning rate scales how big that correction step is. v03 stopped at only
  ~110-320 trees, meaning we reached the peak in very few, very coarse steps -
  and likely stepped over the true optimum.
- Halving the learning rate roughly doubles the number of trees needed, but
  the finer steps usually land closer to the peak. This is the classic
  step-size/steps-count trade: slower but more precise.
- Early-stopping patience is raised 200 -> 400 because with smaller steps,
  the metric plateaus longer before genuinely peaking; a short patience would
  quit during a lull.
- v04's lesson is baked in: no decision-rule tuning (multipliers were exactly
  1.0 - class_weight='balanced' already aligns argmax with balanced accuracy).
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

for df in (train, test):
    for col in CAT_COLS:
        df[col] = df[col].astype("category")

X = train.drop(columns=["id", "health_condition"])
y = train["health_condition"].map(LABEL_MAP)
X_test = test.drop(columns=["id"])


def balanced_accuracy_eval(y_true, y_prob):
    return "balanced_accuracy", balanced_accuracy_score(y_true, y_prob.argmax(axis=1)), True


skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
scores = []
test_probs = np.zeros((len(test), 3))
oof_pred = np.zeros(len(train), dtype=int)

for fold, (idx_tr, idx_va) in enumerate(skf.split(X, y)):
    model = lgb.LGBMClassifier(
        n_estimators=8000,
        learning_rate=0.02,
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
        callbacks=[lgb.early_stopping(400, verbose=False)],
    )
    pred = model.predict(X.iloc[idx_va])
    oof_pred[idx_va] = pred
    score = balanced_accuracy_score(y.iloc[idx_va], pred)
    scores.append(score)
    test_probs += model.predict_proba(X_test) / N_SPLITS
    print(f"fold {fold}: {score:.5f} (best_iter={model.best_iteration_})")

print(f"\nCV balanced accuracy: {np.mean(scores):.5f} +/- {np.std(scores):.5f}")

print("\nper-class recall (OOF):")
for cls_idx, cls_name in INV_LABEL_MAP.items():
    mask = y == cls_idx
    print(f"  {cls_name:10s} {(oof_pred[mask] == cls_idx).mean():.5f}")

submission = pd.DataFrame(
    {
        "id": test["id"],
        "health_condition": [INV_LABEL_MAP[i] for i in test_probs.argmax(axis=1)],
    }
)
submission.to_csv("submissions/submission_v05.csv", index=False)
print("\npredicted class distribution:")
print(submission["health_condition"].value_counts(normalize=True).round(3))
