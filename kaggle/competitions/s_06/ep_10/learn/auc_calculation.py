import numpy as np

scores=[0.9, 0.8, 0.4, 0.3]
labels=[1, 0, 1, 0]

def calculate_tpr(tp, fn):
    return tp / (tp + fn) if (tp + fn) > 0 else 0

def calculate_fpr(fp, tn):
    return fp / (fp + tn) if (fp + tn) > 0 else 0

def get_auc_point(tpr, fpr):
    return (tpr, fpr)

for threshold in np.arange(0, 1, 0.1):
    tp = sum((score >= threshold) and (label == 1) for score, label in zip(scores, labels))
    fn = sum((score < threshold) and (label == 1) for score, label in zip(scores, labels))
    fp = sum((score >= threshold) and (label == 0) for score, label in zip(scores, labels))
    tn = sum((score < threshold) and (label == 0) for score, label in zip(scores, labels))

    tpr = calculate_tpr(tp, fn)
    fpr = calculate_fpr(fp, tn)
    auc_point = get_auc_point(tpr, fpr)

    print(f"Threshold: {threshold}, TPR: {tpr:.2f}, FPR: {fpr:.2f}, AUC Point: {auc_point}")