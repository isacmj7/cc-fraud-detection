"""
Data processing for credit card fraud analysis.
"""

import pandas as pd
import numpy as np
from pathlib import Path


def load_data(filepath=None):
    """Load the dataset."""
    if filepath is None:
        project_root = Path(__file__).parent.parent
        filepath = project_root / "data" / "creditcard_2023.csv"
    
    df = pd.read_csv(filepath)
    print(f"Loaded {len(df):,} transactions")
    return df


def clean_data(df):
    """Clean and prepare data."""
    df_clean = df.copy()
    
    # Fill missing values if any
    if df_clean.isnull().sum().sum() > 0:
        df_clean = df_clean.fillna(df_clean.median())
    
    # Add normalized amount
    df_clean['Amount_Normalized'] = (df_clean['Amount'] - df_clean['Amount'].mean()) / df_clean['Amount'].std()
    
    # Add amount categories
    df_clean['Amount_Category'] = pd.cut(
        df_clean['Amount'],
        bins=[0, 50, 200, 500, 1000, float('inf')],
        labels=['Very Low', 'Low', 'Medium', 'High', 'Very High']
    )
    
    # Add readable labels
    df_clean['Fraud_Label'] = df_clean['Class'].map({0: 'Legitimate', 1: 'Fraud'})
    
    print(f"Cleaned: {len(df_clean):,} rows")
    return df_clean


def get_fraud_stats(df):
    """Get fraud statistics."""
    total = len(df)
    fraud = df['Class'].sum()
    
    return {
        'total': total,
        'fraud': fraud,
        'legitimate': total - fraud,
        'fraud_rate': (fraud / total) * 100,
        'avg_fraud_amount': df[df['Class'] == 1]['Amount'].mean(),
        'avg_legit_amount': df[df['Class'] == 0]['Amount'].mean()
    }


def export_for_tableau(df, output_dir=None):
    """Export data for Tableau."""
    if output_dir is None:
        output_dir = Path(__file__).parent.parent / "tableau"
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(exist_ok=True)
    
    # Sample data (all fraud + sample of legitimate)
    fraud_df = df[df['Class'] == 1]
    legit_sample = df[df['Class'] == 0].sample(n=min(10000, len(df[df['Class'] == 0])), random_state=42)
    sample_df = pd.concat([fraud_df, legit_sample])
    sample_df.to_csv(output_dir / "fraud_data_for_tableau.csv", index=False)
    
    # Summary by amount
    if 'Amount_Category' in df.columns:
        summary = df.groupby('Amount_Category', observed=True).agg({
            'Class': ['sum', 'count', 'mean']
        }).round(4)
        summary.columns = ['Fraud_Count', 'Total', 'Fraud_Rate']
        summary.to_csv(output_dir / "summary_by_amount.csv")
    
    # Feature comparison
    v_cols = [c for c in df.columns if c.startswith('V')][:10]
    stats = df.groupby('Class')[v_cols + ['Amount']].mean().T
    stats.columns = ['Legitimate_Mean', 'Fraud_Mean']
    stats['Difference'] = stats['Fraud_Mean'] - stats['Legitimate_Mean']
    stats.to_csv(output_dir / "feature_comparison.csv")
    
    print(f"Exported to {output_dir}")


if __name__ == "__main__":
    df = load_data()
    df_clean = clean_data(df)
    
    stats = get_fraud_stats(df_clean)
    print(f"\nFraud rate: {stats['fraud_rate']:.2f}%")
    
    export_for_tableau(df_clean)
