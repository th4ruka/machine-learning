from sklearn.metrics import roc_auc_score
import numpy as np
from scipy.stats import rankdata

scores=[0.9, 0.8, 0.4, 0.3]
labels=[1, 0, 1, 0]

scores_cubed=[np.power(score, 3) for score in scores]

scores_log=[np.log(score) for score in scores]

auc = roc_auc_score(labels, scores)
print(f"AUC: {auc:.2f}")

auc_cubed = roc_auc_score(labels, scores_cubed)
print(f"AUC (cubed): {auc_cubed:.2f}")


auc_log = roc_auc_score(labels, scores_log)
print(f"AUC (log): {auc_log:.2f}")

score_ranks = rankdata(scores)
print(f"Ranks: {score_ranks}")

auc_ranks = roc_auc_score(labels, score_ranks)
print(f"AUC (ranks): {auc_ranks:.2f}")



scores_astype_int = [int(s > 0.5) for s in scores]
print(f"Binary scores: {scores_astype_int}")

auc_binary = roc_auc_score(labels, scores_astype_int)
print(f"AUC (binary): {auc_binary:.2f}")