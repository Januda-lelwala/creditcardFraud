"""
Load saved results from results/ and generate comparison plots.
Run after all train_*.py scripts have completed.
"""
import os
import json
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_curve, roc_curve

RESULTS_DIR = 'results'
PLOTS_DIR   = 'plots'
METRICS     = ['AUPRC', 'ROC-AUC', 'F1', 'Precision', 'Recall']
COLORS      = ['#2196F3', '#4CAF50', '#FF5722', '#9C27B0', '#FF9800', '#00BCD4']

os.makedirs(PLOTS_DIR, exist_ok=True)


def load_results():
    y_test_path = f'{RESULTS_DIR}/y_test.npy'
    if not os.path.exists(y_test_path):
        raise FileNotFoundError("results/y_test.npy not found — run at least one train_*.py first.")
    y_test = np.load(y_test_path)

    rows, scores_map = {}, {}
    for path in sorted(glob.glob(f'{RESULTS_DIR}/*.json')):
        with open(path) as f:
            data = json.load(f)
        name = data['name']
        rows[name] = {m: data[m] for m in METRICS}

        scores_path = path.replace('.json', '_scores.npy')
        if os.path.exists(scores_path):
            scores_map[name] = np.load(scores_path)

    if not rows:
        raise FileNotFoundError("No result JSON files found in results/. Run the train_*.py scripts first.")

    results_df = pd.DataFrame(rows).T[METRICS].sort_values('AUPRC', ascending=False)
    return results_df, scores_map, y_test


def print_table(results_df):
    col_w = 14
    sep = '=' * (28 + col_w * len(METRICS))
    print(f'\n{sep}')
    print(f"{'Model':<28}" + ''.join(f'{m:>{col_w}}' for m in METRICS))
    print(sep)
    for name, row in results_df.iterrows():
        print(f'{name:<28}' + ''.join(f'{row[m]:>{col_w}.4f}' for m in METRICS))
    print(sep)
    print(f'\nBest by AUPRC : {results_df.index[0]}')
    print(f'Best by F1    : {results_df["F1"].idxmax()}')
    print(f'Best by Recall: {results_df["Recall"].idxmax()}')


def save(path):
    plt.savefig(os.path.join(PLOTS_DIR, path), dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: plots/{path}')


def plot_bars(results_df):
    fig, axes = plt.subplots(1, len(METRICS), figsize=(4 * len(METRICS), 5))
    colors = COLORS[:len(results_df)]
    for ax, metric in zip(axes, METRICS):
        bars = ax.bar(results_df.index, results_df[metric], color=colors)
        ax.set_title(metric, fontsize=12, fontweight='bold')
        ax.set_ylim(0, 1.15)
        ax.tick_params(axis='x', rotation=35)
        for bar, val in zip(bars, results_df[metric]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=7)
    plt.suptitle('Fraud Detection — Model Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    save('model_comparison_bars.png')


def plot_heatmap(results_df):
    fig, ax = plt.subplots(figsize=(10, max(3, len(results_df) * 0.7)))
    sns.heatmap(results_df, annot=True, fmt='.4f', cmap='YlOrRd', ax=ax,
                linewidths=0.5, vmin=0, vmax=1)
    ax.set_title('Model Performance Heatmap', fontsize=14, fontweight='bold')
    plt.tight_layout()
    save('model_comparison_heatmap.png')


def plot_pr_curves(results_df, scores_map, y_test):
    fig, ax = plt.subplots(figsize=(10, 7))
    for i, name in enumerate(results_df.index):
        if name not in scores_map:
            continue
        prec, rec, _ = precision_recall_curve(y_test, scores_map[name])
        auprc = results_df.loc[name, 'AUPRC']
        ax.plot(rec, prec, label=f'{name} ({auprc:.3f})', color=COLORS[i % len(COLORS)])
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curves', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    save('pr_curves.png')


def plot_roc_curves(results_df, scores_map, y_test):
    fig, ax = plt.subplots(figsize=(10, 7))
    for i, name in enumerate(results_df.index):
        if name not in scores_map:
            continue
        fpr, tpr, _ = roc_curve(y_test, scores_map[name])
        auc = results_df.loc[name, 'ROC-AUC']
        ax.plot(fpr, tpr, label=f'{name} (AUC={auc:.3f})', color=COLORS[i % len(COLORS)])
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.4)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    save('roc_curves.png')


if __name__ == '__main__':
    results_df, scores_map, y_test = load_results()
    print_table(results_df)
    plot_bars(results_df)
    plot_heatmap(results_df)
    plot_pr_curves(results_df, scores_map, y_test)
    plot_roc_curves(results_df, scores_map, y_test)
    print('\nAll plots saved to plots/')
