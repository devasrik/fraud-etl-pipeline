# Fraud Transaction Feature Engineering ETL Pipeline

An end-to-end data engineering pipeline built on a real 500,000-row
credit card fraud dataset. Demonstrates ETL pipeline design, fraud
detection feature engineering, automated data validation, and SQL
analytics — directly mirroring workflows used on financial services
fraud AI teams.

---

## What this project does

| Stage | File | What happens |
|-------|------|-------------|
| Extract | `extract.py` | Loads 50,000 transactions from CSV + fetches live USD exchange rates from a REST API |
| Transform | `transform.py` | Cleans data, normalizes currency, engineers 7 fraud detection features |
| Validate | `validate.py` | Runs 6 automated data quality checks before any data reaches the database |
| Load | `load.py` | Saves to SQLite, runs post-load SQL verification queries |

---

## The 7 fraud detection features engineered

| Feature | Type | Why it matters for fraud |
|---------|------|--------------------------|
| `amount_bucket` | Categorical | Fraudsters probe stolen cards with small amounts before escalating |
| `is_high_value` | Binary flag | High-value transactions require extra scrutiny |
| `is_nighttime` | Binary flag | Fraud rates are statistically higher between 10pm–6am |
| `is_weekend` | Binary flag | Less human oversight on weekends affects fraud patterns |
| `is_high_risk_combo` | Binary flag | International + high value, or nighttime + far from home |
| `distance_tier` | Categorical | Local / nearby / regional / distant — distance from cardholder's home |
| `risk_score` | Numeric (0–10) | Composite signal combining all above features into one fraud indicator |

---

## Dataset

500,000 anonymized credit card transactions across 8 countries
(USA, UK, Germany, France, India, Australia, Canada, Singapore).
1.5% fraud rate (7,500 fraud cases). 16 columns including merchant
category, card type, transaction type, device type, and distance
from home.

---

## Tech stack

- Python 3.x
- pandas — data transformation and feature engineering
- SQLite — local database for loading and SQL analytics
- requests — REST API calls for live exchange rate enrichment

---

## How to run

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/fraud-etl-pipeline.git
cd fraud-etl-pipeline

# 2. Install dependencies
pip install pandas requests

# 3. Add your dataset
# Place credit_card_fraud_2025.csv inside the data/ folder

# 4. Run the full pipeline
python pipeline.py

# Or run individual stages to test them
python extract.py
python transform.py
python validate.py
python load.py
```

---

## Project structure

```
fraud_etl_pipeline/
├── data/
│   └── credit_card_fraud_2025.csv   ← dataset (not tracked by git)
├── extract.py                        ← Stage 1: pull data
├── transform.py                      ← Stage 2: clean + engineer features
├── validate.py                       ← Stage 3: data quality checks
├── load.py                           ← Stage 4: save to database + verify
├── pipeline.py                       ← Master runner: executes all stages
├── README.md
└── .gitignore
```

---

## Sample SQL queries on the output database

```sql
-- Fraud rate by card type
SELECT Card_Type,
       COUNT(*) AS transactions,
       ROUND(AVG(is_fraud) * 100, 2) AS fraud_rate_pct
FROM transaction_features
GROUP BY Card_Type
ORDER BY fraud_rate_pct DESC;

-- Does the high-risk combo flag actually catch more fraud?
SELECT is_high_risk_combo,
       COUNT(*) AS transactions,
       ROUND(AVG(is_fraud) * 100, 3) AS fraud_rate_pct
FROM transaction_features
GROUP BY is_high_risk_combo;

-- Average risk score: fraud vs non-fraud
SELECT is_fraud,
       ROUND(AVG(risk_score), 4) AS avg_risk_score
FROM transaction_features
GROUP BY is_fraud;
```
