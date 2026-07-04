"""s_06_ep_07 v04: v03 model + per-class probability multipliers tuned on OOF.

Concepts in this version
------------------------
- A classifier outputs *probabilities*; turning them into a class label is a
  separate decision rule. Plain argmax is optimal for accuracy, but not
  necessarily for balanced accuracy.
- We tune one multiplier per class (w applied as argmax(prob * w)) to maximize
  balanced accuracy on the OUT-OF-FOLD predictions - rows the model never saw
  during training - so the tuning is honest, not read off the training fit.
- Scaling all multipliers by a constant doesn't change the argmax, so we can
  fix w[at-risk]=1 and grid-search only the other two: a 2D search instead of 3D.
- The same multipliers are then applied to the averaged test probabilities.
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
y_arr = y.to_numpy()


def balanced_accuracy_eval(y_true, y_prob):
    return "balanced_accuracy", balanced_accuracy_score(y_true, y_prob.argmax(axis=1)), True


skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
scores = []
test_probs = np.zeros((len(test), 3))
oof_probs = np.zeros((len(train), 3))

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
    oof_probs[idx_va] = model.predict_proba(X.iloc[idx_va])
    score = balanced_accuracy_score(y.iloc[idx_va], oof_probs[idx_va].argmax(axis=1))
    scores.append(score)
    test_probs += model.predict_proba(X_test) / N_SPLITS
    print(f"fold {fold}: {score:.5f} (best_iter={model.best_iteration_})")

print(f"\nCV balanced accuracy (plain argmax): {np.mean(scores):.5f} +/- {np.std(scores):.5f}")


def bal_acc(pred):
    return np.mean([(pred[y_arr == c] == c).mean() for c in range(3)])


def search(w1_grid, w2_grid):
    best = (bal_acc(oof_probs.argmax(axis=1)), 1.0, 1.0)
    for w1 in w1_grid:
        for w2 in w2_grid:
            pred = (oof_probs * [1.0, w1, w2]).argmax(axis=1)
            s = bal_acc(pred)
            if s > best[0]:
                best = (s, w1, w2)
    return best


# coarse grid, then refine around the best point
s, w1, w2 = search(np.geomspace(0.5, 2.0, 21), np.geomspace(0.5, 2.0, 21))
s, w1, w2 = search(np.geomspace(w1 * 0.85, w1 * 1.18, 21), np.geomspace(w2 * 0.85, w2 * 1.18, 21))

print(f"\ntuned multipliers: at-risk=1.0, unhealthy={w1:.4f}, fit={w2:.4f}")
print(f"OOF balanced accuracy (tuned): {s:.5f}")

oof_pred = (oof_probs * [1.0, w1, w2]).argmax(axis=1)
print("\nper-class recall (OOF, tuned):")
for cls_idx, cls_name in INV_LABEL_MAP.items():
    mask = y_arr == cls_idx
    print(f"  {cls_name:10s} {(oof_pred[mask] == cls_idx).mean():.5f}")

test_pred = (test_probs * [1.0, w1, w2]).argmax(axis=1)
submission = pd.DataFrame(
    {"id": test["id"], "health_condition": [INV_LABEL_MAP[i] for i in test_pred]}
)
submission.to_csv("submission_v04.csv", index=False)
print("\npredicted class distribution:")
print(submission["health_condition"].value_counts(normalize=True).round(3))
