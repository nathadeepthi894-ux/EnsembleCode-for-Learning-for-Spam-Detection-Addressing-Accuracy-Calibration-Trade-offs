#!/usr/bin/env python3
"""
Generate all visualization figures for the manuscript
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix
import pickle

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9

# Load results
print("Loading experimental results...")
with open('experimental_results.json', 'r') as f:
    results = json.load(f)

# Load original dataset and recreate test split
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('spambase.csv')
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# Use same split as in experiments (70-30, random_state=42)
_, _, _, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

print(f"Loaded {len(results)} model results")
print(f"Test set size: {len(y_test)}")

# ============================================================================
# FIGURE 1: Confusion Matrices for Base Classifiers (2x3 grid)
# ============================================================================
print("\n" + "="*80)
print("Generating Figure 1: Confusion Matrices (Base Classifiers)")
print("="*80)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Confusion Matrices - Base Classifiers', fontsize=16, fontweight='bold')

base_classifiers = ['Naive Bayes', 'Logistic Regression', 'SVM', 
                    'Random Forest', 'Gradient Boosting', 'XGBoost']

for idx, clf_name in enumerate(base_classifiers):
    row = idx // 3
    col = idx % 3
    ax = axes[row, col]
    
    cm = np.array(results[clf_name]['confusion_matrix'])
    
    # Plot confusion matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                cbar=True, square=True, 
                xticklabels=['Legitimate', 'Spam'],
                yticklabels=['Legitimate', 'Spam'])
    
    # Add metrics
    acc = results[clf_name]['test_accuracy']
    f1 = results[clf_name]['f1_score']
    
    ax.set_title(f'{clf_name}\nAcc: {acc:.4f} | F1: {f1:.4f}', 
                 fontweight='bold', fontsize=11)
    ax.set_xlabel('Predicted Label', fontsize=10)
    ax.set_ylabel('True Label', fontsize=10)

plt.tight_layout()
plt.savefig('figure1_confusion_matrices_base.png', dpi=300, bbox_inches='tight')
print("✓ Saved: figure1_confusion_matrices_base.png")
plt.close()

# ============================================================================
# FIGURE 2: ROC Curves Comparison (All Methods)
# ============================================================================
print("\n" + "="*80)
print("Generating Figure 2: ROC Curves Comparison")
print("="*80)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle('ROC Curves - All Methods', fontsize=16, fontweight='bold')

# Define colors
colors_base = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c']
colors_ensemble = ['#34495e', '#e67e22', '#95a5a6', '#16a085']

# LEFT: Base Classifiers
ax1.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')

for idx, clf_name in enumerate(base_classifiers):
    if results[clf_name].get('roc_auc') is not None:
        y_proba = np.array(results[clf_name]['y_test_proba'])
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = results[clf_name]['roc_auc']
        
        ax1.plot(fpr, tpr, lw=2.5, color=colors_base[idx],
                label=f'{clf_name} (AUC = {roc_auc:.4f})')

ax1.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
ax1.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
ax1.set_title('(a) Base Classifiers', fontsize=13, fontweight='bold')
ax1.legend(loc='lower right', frameon=True, shadow=True)
ax1.grid(True, alpha=0.3)
ax1.set_xlim([-0.02, 1.02])
ax1.set_ylim([-0.02, 1.02])

# RIGHT: Ensemble Methods
ax2.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')

ensemble_methods = ['Soft Voting', 'Stacking (LR)', 'Stacking (GB)']
for idx, ens_name in enumerate(ensemble_methods):
    if results[ens_name].get('roc_auc') is not None:
        y_proba = np.array(results[ens_name]['y_test_proba'])
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = results[ens_name]['roc_auc']
        
        ax2.plot(fpr, tpr, lw=2.5, color=colors_ensemble[idx],
                label=f'{ens_name} (AUC = {roc_auc:.4f})')

ax2.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
ax2.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
ax2.set_title('(b) Ensemble Methods', fontsize=13, fontweight='bold')
ax2.legend(loc='lower right', frameon=True, shadow=True)
ax2.grid(True, alpha=0.3)
ax2.set_xlim([-0.02, 1.02])
ax2.set_ylim([-0.02, 1.02])

plt.tight_layout()
plt.savefig('figure2_roc_curves.png', dpi=300, bbox_inches='tight')
print("✓ Saved: figure2_roc_curves.png")
plt.close()

# ============================================================================
# FIGURE 3: Calibration Plots (Reliability Diagrams)
# ============================================================================
print("\n" + "="*80)
print("Generating Figure 3: Calibration Plots")
print("="*80)

def compute_calibration_curve(y_true, y_prob, n_bins=10):
    """Compute calibration curve"""
    bins = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    bin_indices = np.digitize(y_prob, bins) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)
    
    bin_sums = np.bincount(bin_indices, weights=y_true, minlength=n_bins)
    bin_counts = np.bincount(bin_indices, minlength=n_bins)
    
    # Avoid division by zero
    bin_counts = np.maximum(bin_counts, 1)
    fraction_of_positives = bin_sums / bin_counts
    
    return bin_centers, fraction_of_positives, bin_counts

fig, axes = plt.subplots(2, 3, figsize=(16, 11))
fig.suptitle('Calibration Plots (Reliability Diagrams)', fontsize=16, fontweight='bold')

# Select 6 models for calibration plots
models_to_plot = ['Naive Bayes', 'Logistic Regression', 'SVM', 
                  'Random Forest', 'XGBoost', 'Stacking (LR)']

for idx, model_name in enumerate(models_to_plot):
    row = idx // 3
    col = idx % 3
    ax = axes[row, col]
    
    if results[model_name].get('y_test_proba'):
        y_proba = np.array(results[model_name]['y_test_proba'])
        
        # Compute calibration curve
        prob_pred, prob_true, counts = compute_calibration_curve(y_test, y_proba, n_bins=10)
        
        # Plot perfect calibration
        ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Perfect Calibration')
        
        # Plot actual calibration
        ax.plot(prob_pred, prob_true, 's-', lw=2.5, markersize=8,
                color='#e74c3c', label='Model Calibration')
        
        # Add histogram
        ax2 = ax.twinx()
        ax2.hist(y_proba, bins=20, alpha=0.3, color='gray', label='Prediction Distribution')
        ax2.set_ylabel('Count', fontsize=9)
        ax2.set_ylim([0, max(counts) * 1.5])
        
        # Get calibration metrics
        ece = results['calibration'][model_name]['ece']
        brier = results['calibration'][model_name]['brier_score']
        
        ax.set_xlabel('Mean Predicted Probability', fontsize=10)
        ax.set_ylabel('Fraction of Positives', fontsize=10)
        ax.set_title(f'{model_name}\nECE: {ece:.4f} | Brier: {brier:.4f}', 
                     fontweight='bold', fontsize=11)
        ax.legend(loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])

plt.tight_layout()
plt.savefig('figure3_calibration_plots.png', dpi=300, bbox_inches='tight')
print("✓ Saved: figure3_calibration_plots.png")
plt.close()

# ============================================================================
# FIGURE 4: Performance Comparison Bar Charts
# ============================================================================
print("\n" + "="*80)
print("Generating Figure 4: Performance Comparison")
print("="*80)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Performance Metrics Comparison', fontsize=16, fontweight='bold')

all_methods = base_classifiers + ['Hard Voting', 'Soft Voting', 'Stacking (LR)', 'Stacking (GB)']

# Prepare data
accuracies = [results[m]['test_accuracy'] * 100 for m in all_methods]
precisions = [results[m]['precision'] * 100 for m in all_methods]
recalls = [results[m]['recall'] * 100 for m in all_methods]
f1_scores = [results[m]['f1_score'] * 100 for m in all_methods]

x_pos = np.arange(len(all_methods))
bar_width = 0.6

# Colors: base classifiers in blue, ensembles in orange
colors = ['#3498db'] * 6 + ['#e67e22'] * 4

# (a) Test Accuracy
ax1 = axes[0, 0]
bars1 = ax1.bar(x_pos, accuracies, bar_width, color=colors, alpha=0.8, edgecolor='black')
ax1.set_ylabel('Test Accuracy (%)', fontsize=11, fontweight='bold')
ax1.set_title('(a) Test Accuracy', fontsize=12, fontweight='bold')
ax1.set_xticks(x_pos)
ax1.set_xticklabels(all_methods, rotation=45, ha='right', fontsize=9)
ax1.grid(axis='y', alpha=0.3)
ax1.set_ylim([75, 100])

# Highlight best
best_idx = np.argmax(accuracies)
bars1[best_idx].set_color('#2ecc71')
bars1[best_idx].set_edgecolor('black')
bars1[best_idx].set_linewidth(2)

# Add value labels
for i, v in enumerate(accuracies):
    ax1.text(i, v + 0.5, f'{v:.2f}', ha='center', va='bottom', fontsize=7, fontweight='bold')

# (b) Precision
ax2 = axes[0, 1]
bars2 = ax2.bar(x_pos, precisions, bar_width, color=colors, alpha=0.8, edgecolor='black')
ax2.set_ylabel('Precision (%)', fontsize=11, fontweight='bold')
ax2.set_title('(b) Precision', fontsize=12, fontweight='bold')
ax2.set_xticks(x_pos)
ax2.set_xticklabels(all_methods, rotation=45, ha='right', fontsize=9)
ax2.grid(axis='y', alpha=0.3)
ax2.set_ylim([65, 100])

best_idx = np.argmax(precisions)
bars2[best_idx].set_color('#2ecc71')
bars2[best_idx].set_edgecolor('black')
bars2[best_idx].set_linewidth(2)

# (c) Recall
ax3 = axes[1, 0]
bars3 = ax3.bar(x_pos, recalls, bar_width, color=colors, alpha=0.8, edgecolor='black')
ax3.set_ylabel('Recall (%)', fontsize=11, fontweight='bold')
ax3.set_title('(c) Recall', fontsize=12, fontweight='bold')
ax3.set_xticks(x_pos)
ax3.set_xticklabels(all_methods, rotation=45, ha='right', fontsize=9)
ax3.grid(axis='y', alpha=0.3)
ax3.set_ylim([85, 100])

best_idx = np.argmax(recalls)
bars3[best_idx].set_color('#2ecc71')
bars3[best_idx].set_edgecolor('black')
bars3[best_idx].set_linewidth(2)

# (d) F1-Score
ax4 = axes[1, 1]
bars4 = ax4.bar(x_pos, f1_scores, bar_width, color=colors, alpha=0.8, edgecolor='black')
ax4.set_ylabel('F1-Score (%)', fontsize=11, fontweight='bold')
ax4.set_title('(d) F1-Score', fontsize=12, fontweight='bold')
ax4.set_xticks(x_pos)
ax4.set_xticklabels(all_methods, rotation=45, ha='right', fontsize=9)
ax4.grid(axis='y', alpha=0.3)
ax4.set_ylim([75, 100])

best_idx = np.argmax(f1_scores)
bars4[best_idx].set_color('#2ecc71')
bars4[best_idx].set_edgecolor('black')
bars4[best_idx].set_linewidth(2)

# Add legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#3498db', edgecolor='black', label='Base Classifiers'),
    Patch(facecolor='#e67e22', edgecolor='black', label='Ensemble Methods'),
    Patch(facecolor='#2ecc71', edgecolor='black', label='Best Performer')
]
fig.legend(handles=legend_elements, loc='upper center', ncol=3, 
          bbox_to_anchor=(0.5, 0.98), frameon=True, shadow=True, fontsize=11)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('figure4_performance_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Saved: figure4_performance_comparison.png")
plt.close()

# ============================================================================
# FIGURE 5: Cross-Validation Results with Error Bars
# ============================================================================
print("\n" + "="*80)
print("Generating Figure 5: Cross-Validation Results")
print("="*80)

fig, ax = plt.subplots(figsize=(14, 8))

cv_means = []
cv_stds = []
cv_cis = []

for clf in base_classifiers:
    cv_data = results['cross_validation'][clf]
    cv_means.append(cv_data['mean_accuracy'] * 100)
    cv_stds.append(cv_data['std_accuracy'] * 100)
    
    # ci_95 is already the margin (half-width of CI)
    ci_margin = cv_data['ci_95'] * 100
    cv_cis.append(ci_margin)

x_pos = np.arange(len(base_classifiers))

# Create bar plot with error bars
bars = ax.bar(x_pos, cv_means, bar_width, 
              color='#3498db', alpha=0.8, edgecolor='black', linewidth=1.5,
              yerr=cv_cis, capsize=8, error_kw={'linewidth': 2, 'ecolor': '#e74c3c'})

# Highlight best
best_idx = np.argmax(cv_means)
bars[best_idx].set_color('#2ecc71')
bars[best_idx].set_edgecolor('black')
bars[best_idx].set_linewidth(2)

ax.set_ylabel('Mean Accuracy (%) ± 95% CI', fontsize=13, fontweight='bold')
ax.set_xlabel('Classifier', fontsize=13, fontweight='bold')
ax.set_title('10-Fold Cross-Validation Results', fontsize=15, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels(base_classifiers, rotation=30, ha='right', fontsize=11)
ax.grid(axis='y', alpha=0.3)
ax.set_ylim([75, 100])

# Add value labels
for i, (mean, ci) in enumerate(zip(cv_means, cv_cis)):
    ax.text(i, mean + ci + 0.5, f'{mean:.2f}±{ci:.2f}', 
            ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig('figure5_cross_validation.png', dpi=300, bbox_inches='tight')
print("✓ Saved: figure5_cross_validation.png")
plt.close()

# ============================================================================
# FIGURE 6: Calibration Metrics Comparison
# ============================================================================
print("\n" + "="*80)
print("Generating Figure 6: Calibration Metrics")
print("="*80)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle('Calibration Metrics Comparison', fontsize=16, fontweight='bold')

# Get calibration data
cal_models = sorted(results['calibration'].keys())
eces = [results['calibration'][m]['ece'] for m in cal_models]
briers = [results['calibration'][m]['brier_score'] for m in cal_models]

x_pos = np.arange(len(cal_models))

# (a) Expected Calibration Error (ECE)
bars1 = ax1.barh(x_pos, eces, bar_width, color='#3498db', alpha=0.8, edgecolor='black')
ax1.set_xlabel('Expected Calibration Error (ECE)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Classifier', fontsize=12, fontweight='bold')
ax1.set_title('(a) Expected Calibration Error', fontsize=13, fontweight='bold')
ax1.set_yticks(x_pos)
ax1.set_yticklabels(cal_models, fontsize=10)
ax1.grid(axis='x', alpha=0.3)
ax1.axvline(x=0.05, color='red', linestyle='--', linewidth=2, label='Well-Calibrated Threshold')
ax1.legend(fontsize=10)

# Highlight best (lowest ECE)
best_idx = np.argmin(eces)
bars1[best_idx].set_color('#2ecc71')

# Add value labels
for i, v in enumerate(eces):
    ax1.text(v + 0.002, i, f'{v:.4f}', ha='left', va='center', fontsize=9, fontweight='bold')

# (b) Brier Score
bars2 = ax2.barh(x_pos, briers, bar_width, color='#e67e22', alpha=0.8, edgecolor='black')
ax2.set_xlabel('Brier Score', fontsize=12, fontweight='bold')
ax2.set_ylabel('Classifier', fontsize=12, fontweight='bold')
ax2.set_title('(b) Brier Score', fontsize=13, fontweight='bold')
ax2.set_yticks(x_pos)
ax2.set_yticklabels(cal_models, fontsize=10)
ax2.grid(axis='x', alpha=0.3)

# Highlight best (lowest Brier)
best_idx = np.argmin(briers)
bars2[best_idx].set_color('#2ecc71')

# Add value labels
for i, v in enumerate(briers):
    ax2.text(v + 0.003, i, f'{v:.4f}', ha='left', va='center', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig('figure6_calibration_metrics.png', dpi=300, bbox_inches='tight')
print("✓ Saved: figure6_calibration_metrics.png")
plt.close()

# ============================================================================
# Summary
# ============================================================================
print("\n" + "="*80)
print("ALL FIGURES GENERATED SUCCESSFULLY!")
print("="*80)
print("\nGenerated files:")
print("  1. figure1_confusion_matrices_base.png")
print("  2. figure2_roc_curves.png")
print("  3. figure3_calibration_plots.png")
print("  4. figure4_performance_comparison.png")
print("  5. figure5_cross_validation.png")
print("  6. figure6_calibration_metrics.png")
print("\nAll figures are saved at 300 DPI for publication quality.")
print("="*80)
