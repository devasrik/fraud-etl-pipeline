"""
extract.py
==========
STAGE 1 of the ETL pipeline — Extract

What this file does:
  1. Loads your credit card fraud CSV dataset into Python
  2. Fetches live currency exchange rates from a free public API

Why two sources?
  Real bank pipelines combine internal data (transactions) with
  external signals (exchange rates, economic data). Doing both
  shows you understand how enterprise data engineering works.
"""

import pandas as pd
import requests


# ─────────────────────────────────────────────
# FUNCTION 1: Load transaction data from CSV
# ─────────────────────────────────────────────

def extract_transactions(filepath="data/credit_card_fraud_2025.csv", sample_size=50000):
    """
    Read the fraud dataset CSV into a pandas DataFrame.

    Parameters
    ----------
    filepath    : path to the CSV file
    sample_size : how many rows to use (we use 50k out of 500k
                  so it runs fast on a laptop — same logic applies
                  at scale with tools like Spark)

    Returns
    -------
    df : a pandas DataFrame with raw transaction data
    """

    print("[EXTRACT] Loading transaction data...")

    # Read the CSV file into a DataFrame
    # A DataFrame is like an Excel spreadsheet in Python — rows and columns
    df = pd.read_csv(filepath)

    # Sample a subset so this runs quickly on your laptop
    # random_state=42 means you get the same sample every time you run it
    df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)

    # Print a quick summary so you can see what was loaded
    total   = len(df)
    frauds  = df["Fraud_Flag"].sum()
    fraud_pct = round(df["Fraud_Flag"].mean() * 100, 2)

    print(f"  Rows loaded  : {total:,}")
    print(f"  Fraud cases  : {frauds:,} ({fraud_pct}%)")
    print(f"  Columns      : {list(df.columns)}")

    return df


# ─────────────────────────────────────────────
# FUNCTION 2: Fetch live exchange rates from API
# ─────────────────────────────────────────────

def extract_exchange_rates():
    """
    Call a free public REST API to get today's USD exchange rates.

    Why this matters:
      Our dataset has transactions from 8 countries (USA, UK, Germany,
      India, Australia, Canada, France, Singapore). Real fraud models
      normalize amounts to a single currency so the AI can compare them
      fairly. We fetch live rates to do that conversion.

    Returns
    -------
    dict : exchange rates and the date they were fetched
    """

    print("[EXTRACT] Fetching live exchange rates from API...")

    url = "https://api.exchangerate-api.com/v4/latest/USD"

    try:
        # Make an HTTP GET request to the API — same as opening a URL in your browser
        response = requests.get(url, timeout=10)
        data = response.json()   # parse the JSON response into a Python dictionary

        # Pull out just the rates we need for our 8 countries
        rates = {
            "USD": 1.0,
            "EUR": data["rates"]["EUR"],   # Germany, France
            "GBP": data["rates"]["GBP"],   # UK
            "INR": data["rates"]["INR"],   # India
            "AUD": data["rates"]["AUD"],   # Australia
            "CAD": data["rates"]["CAD"],   # Canada
            "SGD": data["rates"]["SGD"],   # Singapore
            "fetched_date": data["date"]
        }

        print(f"  Rates fetched for: {data['date']}")
        print(f"  1 USD = {rates['EUR']} EUR | {rates['GBP']} GBP | {rates['INR']} INR")
        return rates

    except Exception as e:
        # If the API is down, fall back to hardcoded rates so the pipeline still runs
        print(f"  API unavailable ({e}). Using fallback rates.")
        return {
            "USD": 1.0, "EUR": 0.92, "GBP": 0.79,
            "INR": 83.5, "AUD": 1.53, "CAD": 1.36,
            "SGD": 1.34, "fetched_date": "fallback"
        }


# ─────────────────────────────────────────────
# Run this file directly to test just this stage
# ─────────────────────────────────────────────

if __name__ == "__main__":
    df    = extract_transactions()
    rates = extract_exchange_rates()

    print("\n--- First 3 rows of raw data ---")
    print(df[["Transaction_ID","Customer_ID","Amount","Merchant_Category",
              "Country","Fraud_Flag"]].head(3).to_string(index=False))
