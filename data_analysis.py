import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, average_precision_score,
                             roc_auc_score, f1_score, precision_score,
                             recall_score, precision_recall_curve, roc_curve)

from models import (
    IsolationForestModel,
    LOFModel,
    OneClassSVMModel,
    KNNModel,
    XGBoostModel,
    AutoencoderModel,
)

PLOTS_DIR = "plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

COLORS = ['#2196F3', '#4CAF50', '#FF5722', '#9C27B0', '#FF9800', '#00BCD4']

# ─── Load & explore ───────────────────────────────────────────────────────────
df = pd.read_csv("creditcard.csv")
print(df.head())
print(df.info())
print("Shape:", df.shape)
print("Fraudulent transactions:", len(df[df["Class"] == 1]))
print("Normal transactions    :", len(df[df["Class"] == 0]))
print(f"Fraud rate: {492 / 284315 * 100:.4f}%")
print(df.describe())

# ─── Preprocessing ────────────────────────────────────────────────────────────
scaler = StandardScaler()
df['scaled_amount'] = scaler.fit_transform(df[['Amount']])
df['scaled_time']   = scaler.fit_transform(df[['Time']])

features = [c for c in df.columns if c not in ['Time', 'Amount', 'Class']]
X = df[features].values
y = df['Class'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train: {X_train.shape}, fraud={y_train.sum()}")
print(f"Test : {X_test.shape},  fraud={y_test.sum()}")

# ─── Evaluation ───────────────────────────────────────────────────────────────
results = {}

def evaluate(name, model):
    s = model.scores(X_test)
    p = model.predict(X_test)
    results[name] = {
        'AUPRC':     average_precision_score(y_test, s),
        'ROC-AUC':   roc_auc_score(y_test, s),
        'F1':        f1_score(y_test, p),
        'Precision': precision_score(y_test, p, zero_division=0),
        'Recall':    recall_score(y_test, p),
        '_scores':   s,
    }
    r = results[name]
    print(f"\n=== {name} ===")
    print(f"AUPRC: {r['AUPRC']:.4f} | ROC-AUC: {r['ROC-AUC']:.4f} | "
          f"F1: {r['F1']:.4f} | Precision: {r['Precision']:.4f} | Recall: {r['Recall']:.4f}")
    print(classification_report(y_test, p, target_names=['Normal', 'Fraud']))

# ─── Train & evaluate ─────────────────────────────────────────────────────────
models = [
    ('Isolation Forest', IsolationForestModel()),
    ('LOF',              LOFModel()),
    ('One-Class SVM',    OneClassSVMModel()),
    ('KNN + SMOTE',      KNNModel()),
    ('XGBoost + SMOTE',  XGBoostModel()),
    ('Autoencoder',      AutoencoderModel()),
]

for name, model in models:
    print(f"\n--- Training {name} ---")
    model.fit(X_train, y_train)
    evaluate(name, model)

# ─── Comparison table ─────────────────────────────────────────────────────────
METRICS = ['AUPRC', 'ROC-AUC', 'F1', 'Precision', 'Recall']
results_df = (
    pd.DataFrame({k: {m: v for m, v in v.items() if m != '_scores'} for k, v in results.items()})
    .T[METRICS]
    .sort_values('AUPRC', ascending=False)
)

col_w = 14
sep = "=" * (28 + col_w * len(METRICS))
print(f"\n{sep}")
print(f"{'Model':<28}" + "".join(f"{m:>{col_w}}" for m in METRICS))
print(sep)
for model_name, row in results_df.iterrows():
    print(f"{model_name:<28}" + "".join(f"{row[m]:>{col_w}.4f}" for m in METRICS))
print(sep)
print(f"\nBest model by AUPRC : {results_df.index[0]}")
print(f"Best model by F1    : {results_df['F1'].idxmax()}")
print(f"Best model by Recall: {results_df['Recall'].idxmax()}")

# ─── Plots ────────────────────────────────────────────────────────────────────
def save(path):
    plt.savefig(os.path.join(PLOTS_DIR, path), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: plots/{path}")


# Bar chart
fig, axes = plt.subplots(1, len(METRICS), figsize=(4 * len(METRICS), 5))
for ax, metric in zip(axes, METRICS):
    bars = ax.bar(results_df.index, results_df[metric], color=COLORS[:len(results_df)])
    ax.set_title(metric, fontsize=12, fontweight='bold')
    ax.set_ylim(0, 1.15)
    ax.tick_params(axis='x', rotation=35)
    for bar, val in zip(bars, results_df[metric]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f'{val:.3f}', ha='center', va='bottom', fontsize=7)
plt.suptitle('Fraud Detection — Model Comparison', fontsize=14, fontweight='bold')
plt.tight_layout()
save("model_comparison_bars.png")

# Heatmap
fig, ax = plt.subplots(figsize=(10, 5))
sns.heatmap(results_df, annot=True, fmt='.4f', cmap='YlOrRd', ax=ax,
            linewidths=0.5, vmin=0, vmax=1)
ax.set_title('Model Performance Heatmap', fontsize=14, fontweight='bold')
plt.tight_layout()
save("model_comparison_heatmap.png")

# PR curves
fig, ax = plt.subplots(figsize=(10, 7))
for i, (name, _) in enumerate(models):
    prec, rec, _ = precision_recall_curve(y_test, results[name]['_scores'])
    ax.plot(rec, prec, label=f"{name} (AUPRC={results[name]['AUPRC']:.3f})",
            color=COLORS[i % len(COLORS)])
ax.set_xlabel('Recall')
ax.set_ylabel('Precision')
ax.set_title('Precision-Recall Curves', fontsize=14, fontweight='bold')
ax.legend(loc='upper right', fontsize=8)
ax.grid(True, alpha=0.3)
plt.tight_layout()
save("pr_curves.png")

# ROC curves
fig, ax = plt.subplots(figsize=(10, 7))
for i, (name, _) in enumerate(models):
    fpr, tpr, _ = roc_curve(y_test, results[name]['_scores'])
    ax.plot(fpr, tpr, label=f"{name} (AUC={results[name]['ROC-AUC']:.3f})",
            color=COLORS[i % len(COLORS)])
ax.plot([0, 1], [0, 1], 'k--', alpha=0.4)
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curves', fontsize=14, fontweight='bold')
ax.legend(loc='lower right', fontsize=8)
ax.grid(True, alpha=0.3)
plt.tight_layout()
save("roc_curves.png")

print("\nAll done. PNGs saved to plots/")
