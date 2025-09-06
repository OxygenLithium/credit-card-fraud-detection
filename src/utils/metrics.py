# Determine metrics and evaluation functions
import numpy as np
from sklearn.metrics import (
    average_precision_score, roc_auc_score, f1_score, 
    precision_score, recall_score, confusion_matrix,
    precision_recall_curve, roc_curve
)

def calculate_metrics(y_true, y_pred, y_proba=None):
    metrics = {
        'f1': f1_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
    }
    
    if y_proba is not None:
        metrics.update({
            'average_precision': average_precision_score(y_true, y_proba),
            'roc_auc': roc_auc_score(y_true, y_proba),
        })
    
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics.update({
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn),
        'true_positives': int(tp),
    })
    
    return metrics


def find_optimal_threshold(y_true, y_proba, metric='f1', search_space=(0.01, 0.99), steps=200):
    thresholds = np.linspace(search_space[0], search_space[1], steps)
    scores = []
    
    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        
        if metric == 'f1':
            score = f1_score(y_true, y_pred)
        elif metric == 'precision':
            score = precision_score(y_true, y_pred, zero_division=0)
        elif metric == 'recall':
            score = recall_score(y_true, y_pred, zero_division=0)
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        scores.append(score)
    
    best_idx = np.argmax(scores)
    best_threshold = thresholds[best_idx]
    best_score = scores[best_idx]
    
    return best_threshold, best_score, list(zip(thresholds, scores))
