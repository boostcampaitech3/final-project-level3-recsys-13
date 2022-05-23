import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score

## 수정 필요!
def get_metric(targets, preds):
    auc = roc_auc_score(targets, preds)
    acc = accuracy_score(targets, np.where(preds >= 0.5, 1, 0))

    return auc, acc
