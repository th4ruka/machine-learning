"""s_06_ep_07 v01: LightGBM baseline with balanced class weights.

Target: health_condition (at-risk / unhealthy / fit), metric: balanced accuracy.
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

skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
scores = []
test_probs = np.zeros((len(test), 3))

for fold, (idx_tr, idx_va) in enumerate(skf.split(X, y)):
    model = lgb.LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=63,
        class_weight="balanced",
        random_state=42,
        verbose=-1,
    )
    model.fit(
        X.iloc[idx_tr],
        y.iloc[idx_tr],
        eval_set=[(X.iloc[idx_va], y.iloc[idx_va])],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    pred = model.predict(X.iloc[idx_va])
    score = balanced_accuracy_score(y.iloc[idx_va], pred)
    scores.append(score)
    test_probs += model.predict_proba(X_test) / N_SPLITS
    print(f"fold {fold}: {score:.5f} (best_iter={model.best_iteration_})")

print(f"\nCV balanced accuracy: {np.mean(scores):.5f} +/- {np.std(scores):.5f}")

submission = pd.DataFrame(
    {
        "id": test["id"],
        "health_condition": [INV_LABEL_MAP[i] for i in test_probs.argmax(axis=1)],
    }
)
submission.to_csv("submission_v01.csv", index=False)
print("\npredicted class distribution:")
print(submission["health_condition"].value_counts(normalize=True).round(3))
