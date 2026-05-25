"""
pipeline.py
===========
THE MASTER PIPELINE — Run this file to execute the entire project

This file ties all 4 stages together and runs them in order:
  Stage 1 → extract.py   (pull data)
  Stage 2 → transform.py (clean + engineer features)
  Stage 3 → validate.py  (quality checks)
  Stage 4 → load.py      (save to database)

How to run:
  python pipeline.py

Think of this file like a conductor at an orchestra.
Each musician (extract, transform, validate, load) plays their own part.
The conductor just makes sure everyone plays in the right order, at the right time,
and stops everything if something goes wrong.
"""

import time
from extract   import extract_transactions, extract_exchange_rates
from transform import transform
from validate  import validate
from load      import load


def run_pipeline():
    """
    Execute all 4 ETL stages in order.
    Prints timing for each stage and total runtime.
    """

    pipeline_start = time.time()

    print("=" * 60)
    print("  FRAUD TRANSACTION FEATURE ENGINEERING ETL PIPELINE")
    print("=" * 60)


    # ────────────────────────────────────────────
    # STAGE 1: EXTRACT
    # Pull raw data from two sources:
    #   - The CSV file (transaction history)
    #   - The exchange rate API (live external signal)
    # ────────────────────────────────────────────
    print("\n[STAGE 1 / 4]  EXTRACT")
    print("-" * 40)
    t0 = time.time()

    raw_transactions = extract_transactions(
        filepath    = "data/credit_card_fraud_2025.csv",
        sample_size = 50000
    )
    exchange_rates = extract_exchange_rates()

    print(f"  Stage 1 completed in {round(time.time()-t0, 2)}s")


    # ────────────────────────────────────────────
    # STAGE 2: TRANSFORM
    # Clean the data and engineer 7 fraud features
    # ────────────────────────────────────────────
    print("\n[STAGE 2 / 4]  TRANSFORM")
    print("-" * 40)
    t0 = time.time()

    featured_df = transform(raw_transactions, exchange_rates)

    print(f"  Stage 2 completed in {round(time.time()-t0, 2)}s")


    # ────────────────────────────────────────────
    # STAGE 3: VALIDATE
    # Run 6 quality checks — stops the pipeline if any fail
    # ────────────────────────────────────────────
    print("\n[STAGE 3 / 4]  VALIDATE")
    print("-" * 40)
    t0 = time.time()

    validate(featured_df)

    print(f"  Stage 3 completed in {round(time.time()-t0, 2)}s")


    # ────────────────────────────────────────────
    # STAGE 4: LOAD
    # Save to database and run SQL verification queries
    # ────────────────────────────────────────────
    print("\n[STAGE 4 / 4]  LOAD")
    print("-" * 40)
    t0 = time.time()

    load(featured_df, db_path="fraud_features.db")

    print(f"  Stage 4 completed in {round(time.time()-t0, 2)}s")


    # ────────────────────────────────────────────
    # DONE
    # ────────────────────────────────────────────
    total_time = round(time.time() - pipeline_start, 2)

    print("\n" + "=" * 60)
    print(f"  PIPELINE COMPLETE in {total_time}s")
    print(f"  Output database : fraud_features.db")
    print(f"  Tables created  : transaction_features, pipeline_run_summary")
    print("=" * 60)


# Standard Python pattern: only run when this file is called directly
# (not when it's imported by another file)
if __name__ == "__main__":
    run_pipeline()
