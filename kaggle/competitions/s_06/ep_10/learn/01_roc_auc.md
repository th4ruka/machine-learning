# 01 — The metric: ROC AUC

Competition: Playground S6E10, *Predicting Airline Satisfaction* (binary target `satisfaction`, ~44% True).
Metric: **ROC AUC**. The submission is one probability (score) per passenger.

## 1. What AUC measures

> **AUC = P(score of a random positive > score of a random negative)**

- **ROC curve** = TPR (y-axis) plotted against FPR (x-axis), with one point per threshold.
  - TPR = TP / (TP + FN): the share of actual positives we flag.
  - FPR = FP / (FP + TN): the share of actual negatives we wrongly flag.
- **AUC = 0.5**: random ranking (the diagonal line).
  **AUC < 0.5**: worse than random. The ranking is inverted, so flipping the scores gives 1 − AUC.
  **AUC = 1.0**: every positive is ranked above every negative.

## 2. Only the order of scores matters

AUC never looks at score *values*, only at their **ranking**.

- Any transform that keeps the order (`p**3`, `log p`, ranks) gives the **same AUC**.
- AUC = 1.0 does **not** mean the probabilities are calibrated. Scores of 0.51 for every positive and
  0.49 for every negative also give AUC = 1.0.
- That's why the submission must be continuous. Hard 0/1 labels collapse everyone into two tied
  groups and throw the ranking away.

## 3. AUC is threshold-independent (my misconception, corrected)

❌ *My first answer:* "The threshold matters, because TPR and FPR depend on it."

✅ *Correct:* a **single ROC point** depends on the threshold. The **curve** is built by sweeping
**every** threshold, and AUC is the area under all of those points. No threshold is ever chosen, so
AUC doesn't depend on one.

### What this means in practice (compared with s6e7, which used balanced accuracy)

| | s6e7 (balanced accuracy) | s6e10 (ROC AUC) |
|---|---|---|
| Decision threshold | Mattered a lot | Irrelevant |
| `class_weight='balanced'` | Essential | Mostly useless (shifts probabilities, barely changes order) |
| Probability calibration | Not scored | Not scored |
| What to improve | Correct argmax per class | **Ranking quality** |

## 4. Worked example

| score | 0.9 | 0.8 | 0.4 | 0.3 |
|---|---|---|---|---|
| satisfied | 1 | 0 | 1 | 0 |

**Pair counting** (2 positives × 2 negatives = 4 pairs):
0.9>0.8 ✓ · 0.9>0.3 ✓ · 0.4>0.8 ✗ · 0.4>0.3 ✓ → **AUC = 3/4 = 0.75**

**ROC curve** (predict positive when score ≥ t):

| threshold t | flagged | TPR | FPR | point (FPR, TPR) |
|---|---|---|---|---|
| > 0.9 | nobody | 0.0 | 0.0 | (0, 0) |
| 0.9 | 0.9 | 0.5 | 0.0 | (0, 0.5) |
| 0.8 | 0.9, 0.8 | 0.5 | 0.5 | (0.5, 0.5) |
| 0.4 | 0.9, 0.8, 0.4 | 1.0 | 0.5 | (0.5, 1.0) |
| 0.3 | everyone | 1.0 | 1.0 | (1, 1) |

Area = 0.5 × 0.5 + 0.5 × 1.0 = **0.75**. Both methods agree.

**Why they agree:** sort passengers by score from high to low and walk down the list. Each positive
moves the curve **up** one step and each negative moves it **right** one step. The column of area
above each negative equals the number of positives ranked above it. Summing those columns counts
exactly the correctly ordered pairs.

## 5. Lessons from `auc_calculation.py`

1. **Floating-point thresholds.** `np.arange(0, 1, 0.1)[3]` is `0.30000000000000004`, so
   `0.3 >= t` is **False**. At "threshold 0.3" the passenger scoring 0.3 was silently excluded. That
   point came out as (0.5, 1.0) when it should have been (1, 1).
   *Fix:* use the **actual scores** as thresholds (`sorted(set(scores), reverse=True)`). The curve
   only changes at those values anyway.
2. **The curve must start at (0, 0).** `arange(0, 1, 0.1)` stops at 0.9 and never goes above the
   highest score, so the (0, 0) point is missing. Here it didn't change the area, because that
   segment is vertical, but in general you need a threshold above the maximum score (e.g. `+inf`).
3. **Naming and order.** `get_auc_point` returns `(tpr, fpr)`. It's a **ROC point**, not an AUC
   point, and plots use the order **(FPR, TPR)**. Mixing the order up flips the curve.

## Still open

- [ ] Part B: check in code that `roc_auc_score(y, s) == roc_auc_score(y, s**3) == roc_auc_score(y, rankdata(s))`,
      and see what thresholding to 0/1 does. Write predictions first.
- [ ] Step 2: EDA questions (class balance, what rating 0 means, best single feature, the two delay
      columns, missing values).
