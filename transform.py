"""
transform.py
============
STAGE 2 of the ETL pipeline — Transform

What this file does:
  1. Cleans the raw data (fix types, remove issues)
  2. Enriches each transaction with USD-normalized amounts
  3. Engineers 7 fraud detection features

What is "feature engineering"?
  The AI fraud model doesn't learn from raw columns like "Amount" or
  "Hour_of_Day" directly. It learns from calculated signals — things
  like "is this transaction unusually large for this customer?" or
  "is this happening at 3am far from home?". Creating those signals
  from raw data is called feature engineering, and it's the core skill
  of a data engineer on a fraud AI team.
"""

import pandas as pd
import numpy as np


# ─────────────────────────────────────────────────────────────
# Country → currency mapping (matches our 8 countries in data)
# ─────────────────────────────────────────────────────────────

COUNTRY_CURRENCY = {
    "USA"       : "USD",
    "UK"        : "GBP",
    "Germany"   : "EUR",
    "France"    : "EUR",
    "India"     : "INR",
    "Australia" : "AUD",
    "Canada"    : "CAD",
    "Singapore" : "SGD",
}


def transform(df, exchange_rates):
    """
    Clean the raw transaction data and engineer fraud detection features.

    Parameters
    ----------
    df             : raw DataFrame from extract.py
    exchange_rates : dict of currency rates from extract.py

    Returns
    -------
    df : enriched DataFrame ready for validation and loading
    """

    print("[TRANSFORM] Starting transformation...")
    original_rows = len(df)

    # ── Step 1: Clean column names ─────────────────────────────────────
    # Rename columns to be clearer and lowercase for consistency
    df = df.rename(columns={
        "Fraud_Flag"         : "is_fraud",
        "Amount"             : "amount_usd",
        "Transaction_Date"   : "txn_datetime",
        "Distance_From_Home" : "distance_from_home_km",
        "Hour_of_Day"        : "hour_of_day",
    })


    # ── Step 2: Fix data types ─────────────────────────────────────────
    # Convert the datetime string to an actual datetime object in Python
    # This lets us do time math — like extracting day of week or month
    df["txn_datetime"] = pd.to_datetime(df["txn_datetime"])

    # Extract useful time components from the datetime
    df["txn_date"]       = df["txn_datetime"].dt.date
    df["txn_month"]      = df["txn_datetime"].dt.month
    df["txn_day_of_week"] = df["txn_datetime"].dt.dayofweek   # 0=Monday, 6=Sunday
    df["txn_year"]       = df["txn_datetime"].dt.year


    # ── Step 3: Enrich with currency normalization ─────────────────────
    # Map each transaction's country to its currency
    df["currency"] = df["Country"].map(COUNTRY_CURRENCY).fillna("USD")

    # Normalize every transaction amount to USD
    # This lets the AI model compare a £50 UK transaction to a ₹5000 Indian one fairly
    def convert_to_usd(row):
        currency = row["currency"]
        rate = exchange_rates.get(currency, 1.0)
        # To convert TO USD: divide by the rate (rate = how many foreign per 1 USD)
        return round(row["amount_usd"] / rate, 2)

    df["amount_normalized_usd"] = df.apply(convert_to_usd, axis=1)
    df["exchange_rates_date"]   = exchange_rates.get("fetched_date", "unknown")


    # ── Step 4: Feature Engineering ───────────────────────────────────
    # This is the most important section — we create 7 fraud signals

    # ── FEATURE 1: Amount bucket ──────────────────────────────────────
    # Group transaction amounts into 4 categories
    # Fraudsters often test cards with small amounts before going big
    df["amount_bucket"] = pd.cut(
        df["amount_usd"],
        bins   = [0, 20, 100, 500, float("inf")],
        labels = ["micro", "low", "medium", "high"]
    ).astype(str)


    # ── FEATURE 2: Is high value? ──────────────────────────────────────
    # Simple binary flag: 1 if over $500, 0 otherwise
    # High-value transactions get extra scrutiny in real fraud systems
    df["is_high_value"] = (df["amount_usd"] > 500).astype(int)


    # ── FEATURE 3: Is nighttime transaction? ──────────────────────────
    # Fraud disproportionately happens late at night / early morning
    # We flag anything between 10pm (22:00) and 6am as nighttime
    df["is_nighttime"] = df["hour_of_day"].apply(
        lambda h: 1 if (h >= 22 or h <= 6) else 0
    )


    # ── FEATURE 4: Is weekend? ────────────────────────────────────────
    # Weekends have different fraud patterns — less human oversight
    df["is_weekend"] = (df["txn_day_of_week"] >= 5).astype(int)


    # ── FEATURE 5: High risk combination flag ─────────────────────────
    # A transaction is "high risk" if it hits multiple red flags at once:
    #   - It's international AND high value
    #   - OR it's at night AND far from home
    # Combining signals is a hallmark of real fraud feature engineering
    df["is_high_risk_combo"] = (
        ((df["Is_International"] == 1) & (df["is_high_value"] == 1)) |
        ((df["is_nighttime"] == 1) & (df["distance_from_home_km"] > 50))
    ).astype(int)


    # ── FEATURE 6: Distance risk tier ────────────────────────────────
    # How far from home the transaction happened — bucketed into 4 tiers
    # A transaction 200km from home is more suspicious than one 2km away
    df["distance_tier"] = pd.cut(
        df["distance_from_home_km"],
        bins          = [0, 5, 25, 100, float("inf")],
        labels        = ["local", "nearby", "regional", "distant"],
        include_lowest = True    # includes the 0.0 edge so no rows fall outside bins
    ).astype(str)


    # ── FEATURE 7: Composite risk score ──────────────────────────────
    # A single number (0–10) summarizing how suspicious a transaction is
    # This combines multiple signals into one score — just like real
    # fraud scoring systems do (e.g., FICO fraud score)
    df["risk_score"] = (
        df["is_nighttime"]       * 1.5  +   # night = higher risk
        df["Is_International"]   * 2.0  +   # international = higher risk
        df["is_high_value"]      * 1.5  +   # high value = higher risk
        df["is_high_risk_combo"] * 2.0  +   # combo flag = big risk boost
        (df["distance_from_home_km"] / df["distance_from_home_km"].max()) * 3.0
                                            # normalized distance (0–3 range)
    ).round(4)


    # ── Step 5: Final cleanup ──────────────────────────────────────────
    # Drop the raw datetime column — we've extracted everything we need from it
    df = df.drop(columns=["txn_datetime"])

    # Make sure is_fraud is integer (0 or 1)
    df["is_fraud"] = df["is_fraud"].astype(int)

    rows_after = len(df)
    print(f"  Rows in  : {original_rows:,}")
    print(f"  Rows out : {rows_after:,}")
    print(f"  Features added: amount_bucket, is_high_value, is_nighttime,")
    print(f"                  is_weekend, is_high_risk_combo, distance_tier,")
    print(f"                  risk_score, amount_normalized_usd")
    print(f"[TRANSFORM] Done.")

    return df


# ─────────────────────────────────────────────
# Run directly to test just this stage
# ─────────────────────────────────────────────

if __name__ == "__main__":
    from extract import extract_transactions, extract_exchange_rates

    df    = extract_transactions()
    rates = extract_exchange_rates()
    out   = transform(df, rates)

    print("\n--- Sample of engineered features ---")
    cols = ["Transaction_ID", "amount_usd", "amount_bucket",
            "is_high_value", "is_nighttime", "is_weekend",
            "distance_tier", "risk_score", "is_fraud"]
    print(out[cols].head(8).to_string(index=False))

    print("\n--- Fraud vs non-fraud avg risk score ---")
    print(out.groupby("is_fraud")["risk_score"].mean().rename({0:"non-fraud", 1:"fraud"}))
