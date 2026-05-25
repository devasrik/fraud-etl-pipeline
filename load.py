"""
load.py
=======
STAGE 4 of the ETL pipeline — Load

What this file does:
  1. Saves the cleaned, feature-engineered data to a SQLite database
  2. Creates a second summary stats table (useful for dashboards/reporting)
  3. Runs SQL verification queries to confirm everything loaded correctly

What is SQLite?
  SQLite is a database that lives as a single file on your computer —
  no server needed. It's real SQL, just local. Perfect for projects and
  learning. In production at Wells Fargo, this would be a cloud database
  like Snowflake, PostgreSQL, or a data warehouse on OpenShift.

Why verify after loading?
  A real data engineer never assumes the load worked. You always run
  queries AFTER loading to confirm: row counts match, the fraud rate
  looks right, no unexpected nulls snuck in. This is called a
  "post-load validation" and it's standard practice.
"""

import sqlite3
import pandas as pd


def load(df, db_path="fraud_features.db"):
    """
    Load the transformed DataFrame into a SQLite database.

    Creates two tables:
      1. transaction_features — the main feature table (one row per transaction)
      2. pipeline_run_summary — high-level stats about this pipeline run

    Parameters
    ----------
    df      : transformed and validated DataFrame
    db_path : filename for the SQLite database file
    """

    print(f"[LOAD] Connecting to database: {db_path}")

    # sqlite3.connect creates the file if it doesn't exist yet
    conn = sqlite3.connect(db_path)

    # ── Table 1: Main feature table ────────────────────────────────────
    # if_exists="replace" means: if this table already exists, delete it
    # and replace it with fresh data. Good for a pipeline that re-runs daily.
    print(f"[LOAD] Writing {len(df):,} rows to 'transaction_features' table...")
    df.to_sql("transaction_features", conn, if_exists="replace", index=False)
    print(f"  Done.")


    # ── Table 2: Pipeline run summary ─────────────────────────────────
    # This summary table is what a manager or data analyst would look at
    # to quickly understand "what did today's pipeline produce?"
    print("[LOAD] Writing pipeline summary table...")

    summary_data = {
        "total_transactions"    : [len(df)],
        "fraud_cases"           : [int(df["is_fraud"].sum())],
        "fraud_rate_pct"        : [round(df["is_fraud"].mean() * 100, 4)],
        "avg_amount_usd"        : [round(df["amount_usd"].mean(), 2)],
        "max_amount_usd"        : [round(df["amount_usd"].max(), 2)],
        "avg_risk_score"        : [round(df["risk_score"].mean(), 4)],
        "avg_risk_score_fraud"  : [round(df[df["is_fraud"]==1]["risk_score"].mean(), 4)],
        "nighttime_txn_pct"     : [round(df["is_nighttime"].mean() * 100, 2)],
        "international_txn_pct" : [round(df["Is_International"].mean() * 100, 2)],
        "high_risk_combo_pct"   : [round(df["is_high_risk_combo"].mean() * 100, 2)],
        "exchange_rates_date"   : [df["exchange_rates_date"].iloc[0]],
        "pipeline_run_rows"     : [len(df)],
    }

    summary_df = pd.DataFrame(summary_data)
    summary_df.to_sql("pipeline_run_summary", conn, if_exists="replace", index=False)
    print("  Done.")


    # ── Post-load SQL verification ─────────────────────────────────────
    # Now we use SQL to query the database and confirm everything looks right.
    # This is important: we're reading BACK from the database, not from Python.
    # If something went wrong during the write, we'd catch it here.

    print("\n[LOAD] Running post-load SQL verification queries...")

    # Query 1: Row count confirmation
    q1 = pd.read_sql("SELECT COUNT(*) AS total_rows FROM transaction_features", conn)
    print(f"  Row count confirmed: {q1['total_rows'][0]:,}")

    # Query 2: Fraud breakdown by card type
    q2 = pd.read_sql("""
        SELECT
            Card_Type,
            COUNT(*)                         AS transactions,
            SUM(is_fraud)                    AS fraud_cases,
            ROUND(AVG(is_fraud) * 100, 2)   AS fraud_rate_pct,
            ROUND(AVG(risk_score), 3)        AS avg_risk_score
        FROM transaction_features
        GROUP BY Card_Type
        ORDER BY fraud_rate_pct DESC
    """, conn)
    print("\n  Fraud rate by card type:")
    print(q2.to_string(index=False))

    # Query 3: Nighttime vs daytime fraud comparison
    q3 = pd.read_sql("""
        SELECT
            CASE WHEN is_nighttime = 1 THEN 'Nighttime' ELSE 'Daytime' END AS time_period,
            COUNT(*)                         AS transactions,
            SUM(is_fraud)                    AS fraud_cases,
            ROUND(AVG(is_fraud) * 100, 3)   AS fraud_rate_pct
        FROM transaction_features
        GROUP BY is_nighttime
        ORDER BY is_nighttime DESC
    """, conn)
    print("\n  Nighttime vs daytime fraud rate:")
    print(q3.to_string(index=False))

    # Query 4: Top merchant categories by fraud rate
    q4 = pd.read_sql("""
        SELECT
            Merchant_Category,
            COUNT(*)                         AS transactions,
            SUM(is_fraud)                    AS fraud_cases,
            ROUND(AVG(is_fraud) * 100, 3)   AS fraud_rate_pct
        FROM transaction_features
        GROUP BY Merchant_Category
        ORDER BY fraud_rate_pct DESC
        LIMIT 5
    """, conn)
    print("\n  Top 5 merchant categories by fraud rate:")
    print(q4.to_string(index=False))

    # Query 5: High-risk combo effectiveness
    q5 = pd.read_sql("""
        SELECT
            CASE WHEN is_high_risk_combo = 1 THEN 'High-risk combo' ELSE 'Normal' END AS flag,
            COUNT(*)                         AS transactions,
            SUM(is_fraud)                    AS fraud_cases,
            ROUND(AVG(is_fraud) * 100, 3)   AS fraud_rate_pct,
            ROUND(AVG(risk_score), 3)        AS avg_risk_score
        FROM transaction_features
        GROUP BY is_high_risk_combo
    """, conn)
    print("\n  High-risk combo flag effectiveness:")
    print(q5.to_string(index=False))

    conn.close()
    print(f"\n[LOAD] Complete. Database saved to: {db_path}")


# ─────────────────────────────────────────────
# Run directly to test just this stage
# ─────────────────────────────────────────────

if __name__ == "__main__":
    from extract import extract_transactions, extract_exchange_rates
    from transform import transform
    from validate import validate

    df    = extract_transactions()
    rates = extract_exchange_rates()
    out   = transform(df, rates)
    validate(out)
    load(out)
