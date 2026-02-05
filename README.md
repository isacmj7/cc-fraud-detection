# Credit Card Fraud Detection

**Ishak Islam** | UMID28072552431 | Unified Mentor Internship

## About

Analysis of credit card transactions to identify fraud patterns. The dataset has 568,630 transactions with a 50/50 balanced distribution (284,315 fraud and 284,315 legitimate).

## How to Run

```bash
pip install -r requirements.txt
jupyter notebook notebooks/01_credit_card_fraud_analysis.ipynb
```

Run all cells to see the analysis.

## Dataset

Download from: https://www.kaggle.com/datasets/nelgiriyewithana/credit-card-fraud-detection-dataset-2023

Place `creditcard_2023.csv` in the `data/` folder.

## Files

```
├── data/           # Put dataset here
├── notebooks/      # Analysis notebook
├── scripts/        # Helper functions
├── visualizations/ # Charts
├── tableau/        # Tableau exports
└── docs/           # Documentation
```

## Results

- Fraud rate: 50% (balanced dataset)
- Key features identified for fraud detection
- Visualizations show fraud patterns
- Balanced data exported for Tableau dashboards (10,000 fraud + 10,000 legitimate)

## Tableau Dashboard

**Live Interactive Dashboard:** [View on Tableau Public](https://public.tableau.com/app/profile/ishak.islam/viz/CCFraudDetection_17696696335300/Dashboard?publish=yes)

## Tech Stack

Python, Pandas, NumPy, Matplotlib, Seaborn, Tableau

## GitHub Repository

**Source Code:** [https://github.com/isacmj7/cc-fraud-detection](https://github.com/isacmj7/cc-fraud-detection)
