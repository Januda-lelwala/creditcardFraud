import json
import os
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import (classification_report, average_precision_score,
                             roc_auc_score, f1_score, precision_score, recall_score)

RESULTS_DIR = 'results'


def _key(name):
    return name.lower().replace(' ', '_').replace('+', 'plus').replace('-', '_')


def evaluate(name, scores, preds, y_true):
    metrics = {
        'AUPRC':     float(average_precision_score(y_true, scores)),
        'ROC-AUC':   float(roc_auc_score(y_true, scores)),
        'F1':        float(f1_score(y_true, preds)),
        'Precision': float(precision_score(y_true, preds, zero_division=0)),
        'Recall':    float(recall_score(y_true, preds)),
    }
    print(f"\n=== {name} ===")
    print(f"AUPRC: {metrics['AUPRC']:.4f} | ROC-AUC: {metrics['ROC-AUC']:.4f} | "
          f"F1: {metrics['F1']:.4f} | Precision: {metrics['Precision']:.4f} | "
          f"Recall: {metrics['Recall']:.4f}")
    print(classification_report(y_true, preds, target_names=['Normal', 'Fraud']))
    return metrics


def save_results(name, metrics, scores, y_test):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    k = _key(name)

    with open(f'{RESULTS_DIR}/{k}.json', 'w') as f:
        json.dump({'name': name, **metrics}, f, indent=2)

    np.save(f'{RESULTS_DIR}/{k}_scores.npy', scores)
    np.save(f'{RESULTS_DIR}/y_test.npy', y_test)

    print(f"  Saved → results/{k}.json  results/{k}_scores.npy", flush=True)
