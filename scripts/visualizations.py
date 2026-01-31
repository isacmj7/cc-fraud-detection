"""
Visualizations for credit card fraud analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

COLORS = ['#2ecc71', '#e74c3c']  # green=legit, red=fraud


def save_fig(fig, filename, output_dir=None):
    """Save figure."""
    if output_dir is None:
        output_dir = Path(__file__).parent.parent / "visualizations"
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(exist_ok=True)
    fig.savefig(output_dir / filename, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def plot_class_distribution(df, output_dir=None):
    """Fraud vs legitimate distribution."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    counts = df['Class'].value_counts().sort_index()
    labels = ['Legitimate', 'Fraud']
    
    axes[0].bar(labels, counts.values, color=COLORS)
    axes[0].set_title('Transaction Count')
    axes[0].set_ylabel('Count')
    
    axes[1].pie(counts.values, labels=labels, autopct='%1.2f%%', colors=COLORS, explode=(0, 0.1))
    axes[1].set_title('Fraud Rate')
    
    plt.suptitle('Fraud Distribution', fontweight='bold')
    plt.tight_layout()
    save_fig(fig, '01_fraud_distribution.png', output_dir)


def plot_amount_distribution(df, output_dir=None):
    """Amount distribution by class."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    for label, color in zip([0, 1], COLORS):
        name = 'Legitimate' if label == 0 else 'Fraud'
        data = df[df['Class'] == label]['Amount']
        axes[0].hist(data, bins=50, alpha=0.7, label=name, color=color)
    
    axes[0].set_xlabel('Amount ($)')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Amount Distribution')
    axes[0].legend()
    axes[0].set_xlim(0, 2000)
    
    df_plot = df.copy()
    df_plot['Label'] = df_plot['Class'].map({0: 'Legitimate', 1: 'Fraud'})
    sns.boxplot(data=df_plot, x='Label', y='Amount', palette={'Legitimate': COLORS[0], 'Fraud': COLORS[1]}, ax=axes[1])
    axes[1].set_title('Amount Comparison')
    axes[1].set_ylim(0, 500)
    
    plt.suptitle('Amount Analysis', fontweight='bold')
    plt.tight_layout()
    save_fig(fig, '02_amount_analysis.png', output_dir)


def plot_feature_correlation(df, output_dir=None):
    """Feature correlation with fraud."""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    v_cols = [c for c in df.columns if c.startswith('V')] + ['Amount']
    corr = df[v_cols + ['Class']].corr()['Class'].drop('Class').sort_values()
    
    colors = ['#e74c3c' if x < 0 else '#2ecc71' for x in corr.values]
    corr.plot(kind='barh', ax=ax, color=colors)
    
    ax.set_xlabel('Correlation')
    ax.set_title('Feature Correlation with Fraud', fontweight='bold')
    ax.axvline(x=0, color='black', linewidth=0.5)
    
    plt.tight_layout()
    save_fig(fig, '03_feature_correlation.png', output_dir)


def plot_top_features(df, output_dir=None):
    """Top discriminating features."""
    v_cols = [c for c in df.columns if c.startswith('V')]
    corr = {c: abs(df[c].corr(df['Class'])) for c in v_cols}
    top = sorted(corr.items(), key=lambda x: x[1], reverse=True)[:6]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, (feat, _) in enumerate(top):
        for label, color in zip([0, 1], COLORS):
            name = 'Legitimate' if label == 0 else 'Fraud'
            data = df[df['Class'] == label][feat]
            axes[i].hist(data, bins=50, alpha=0.6, label=name, color=color, density=True)
        axes[i].set_title(feat)
        axes[i].legend()
    
    plt.suptitle('Top Features', fontweight='bold')
    plt.tight_layout()
    save_fig(fig, '04_top_features.png', output_dir)


def plot_fraud_by_amount(df, output_dir=None):
    """Fraud by amount category."""
    if 'Amount_Category' not in df.columns:
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    summary = df.groupby('Amount_Category', observed=True)['Class'].agg(['sum', 'count'])
    summary['rate'] = (summary['sum'] / summary['count']) * 100
    
    axes[0].bar(range(len(summary)), summary['sum'], color='#e74c3c')
    axes[0].set_xticks(range(len(summary)))
    axes[0].set_xticklabels(summary.index, rotation=30, ha='right')
    axes[0].set_ylabel('Fraud Count')
    axes[0].set_title('Fraud by Amount')
    
    axes[1].bar(range(len(summary)), summary['rate'], color='#3498db')
    axes[1].set_xticks(range(len(summary)))
    axes[1].set_xticklabels(summary.index, rotation=30, ha='right')
    axes[1].set_ylabel('Fraud Rate (%)')
    axes[1].set_title('Fraud Rate by Amount')
    
    plt.suptitle('Amount Category Analysis', fontweight='bold')
    plt.tight_layout()
    save_fig(fig, '05_fraud_by_amount.png', output_dir)


def plot_correlation_matrix(df, output_dir=None):
    """Correlation matrix."""
    v_cols = [c for c in df.columns if c.startswith('V')]
    corr = {c: abs(df[c].corr(df['Class'])) for c in v_cols}
    top = [x[0] for x in sorted(corr.items(), key=lambda x: x[1], reverse=True)[:8]]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    cols = top + ['Amount', 'Class']
    matrix = df[cols].corr()
    sns.heatmap(matrix, annot=True, fmt='.2f', cmap='RdBu_r', center=0, ax=ax)
    ax.set_title('Correlation Matrix', fontweight='bold')
    
    plt.tight_layout()
    save_fig(fig, '06_correlation_matrix.png', output_dir)


def create_all_visualizations(df, output_dir=None):
    """Create all charts."""
    print("Creating visualizations...")
    plot_class_distribution(df, output_dir)
    plot_amount_distribution(df, output_dir)
    plot_feature_correlation(df, output_dir)
    plot_top_features(df, output_dir)
    plot_fraud_by_amount(df, output_dir)
    plot_correlation_matrix(df, output_dir)
    print("Done!")


if __name__ == "__main__":
    from data_processing import load_data, clean_data
    
    df = load_data()
    df_clean = clean_data(df)
    create_all_visualizations(df_clean)
