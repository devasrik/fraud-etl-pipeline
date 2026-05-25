"""
validate.py
===========
STAGE 3 of the ETL pipeline — Validate

What this file does:
  Runs automated data quality checks BEFORE loading anything to the
  database. If any check fails, the pipeline stops and tells you exactly
  what went wrong.

Why validation matters:
  Imagine the fraud model gets trained on data where the "is_fraud" column
  accidentally has nulls — the model would learn incorrect patterns. Or
  imagine risk scores are negative because of a math bug. Bad data = bad
  AI decisions = real customers get hurt (blocked legitimate transactions,
  or worse, real fraud slips through).

  In production at a bank, data engineers are responsible for making sure
  data quality is guaranteed BEFORE it reaches the data scientists and
  their models. That's exactly what this file does.
"""


def validate(df):
    """
    Run 6 data quality checks on the transformed DataFrame.

    If all checks pass  → returns True and pipeline continues
    If any check fails  → raises a ValueError and pipeline stops

    Parameters
    ----------
    df : transformed DataFrame from transform.py
    """

    print("[VALIDATE] Running data quality checks...")

    errors  = []   # collect all failures before stopping
    passed  = 0    # count how many checks pass

    def ok(msg):
        nonlocal passed
        passed += 1
        print(f"  PASS  {msg}")

    def fail(msg):
        errors.append(msg)
        print(f"  FAIL  {msg}")


    # ── CHECK 1: No missing values in critical columns ─────────────────
    # These columns MUST exist and have values for every row.
    # If any are null, the fraud model can't use the data.
    critical_columns = [
        "amount_usd", "is_fraud", "risk_score",
        "hour_of_day", "amount_bucket", "distance_tier",
        "amount_normalized_usd"
    ]
    for col in critical_columns:
        if col not in df.columns:
            fail(f"Column '{col}' is missing from the dataset entirely")
        else:
            null_count = df[col].isnull().sum()
            if null_count > 0:
                fail(f"'{col}' has {null_count:,} missing values")
            else:
                ok(f"'{col}' has no missing values")


    # ── CHECK 2: Fraud label integrity ─────────────────────────────────
    # is_fraud must ONLY contain 0 (not fraud) or 1 (fraud).
    # Any other value means something went wrong in the transform step.
    invalid_labels = df[~df["is_fraud"].isin([0, 1])].shape[0]
    if invalid_labels > 0:
        fail(f"{invalid_labels:,} rows have invalid fraud labels (not 0 or 1)")
    else:
        ok(f"All fraud labels are valid (0 or 1) — "
           f"{df['is_fraud'].sum():,} fraud cases ({df['is_fraud'].mean()*100:.2f}%)")


    # ── CHECK 3: No negative transaction amounts ───────────────────────
    # A negative dollar amount makes no sense for a transaction.
    # This would indicate corrupted or mis-parsed data.
    negative_amounts = (df["amount_usd"] < 0).sum()
    if negative_amounts > 0:
        fail(f"{negative_amounts:,} rows have negative transaction amounts")
    else:
        ok(f"All transaction amounts are non-negative "
           f"(min=${df['amount_usd'].min():.2f}, max=${df['amount_usd'].max():.2f})")


    # ── CHECK 4: Risk score is within expected range ───────────────────
    # Our formula produces scores between 0 and ~10.
    # Anything below 0 or above 15 would signal a formula bug.
    out_of_range = df[(df["risk_score"] < 0) | (df["risk_score"] > 15)].shape[0]
    if out_of_range > 0:
        fail(f"{out_of_range:,} risk scores are outside expected range [0, 15]")
    else:
        ok(f"All risk scores in valid range "
           f"(avg={df['risk_score'].mean():.3f}, max={df['risk_score'].max():.3f})")


    # ── CHECK 5: Hour of day is between 0 and 23 ──────────────────────
    # Hours must be 0–23. Anything outside that range means bad data.
    bad_hours = df[(df["hour_of_day"] < 0) | (df["hour_of_day"] > 23)].shape[0]
    if bad_hours > 0:
        fail(f"{bad_hours:,} rows have invalid hour_of_day values (must be 0–23)")
    else:
        ok(f"All hour_of_day values are valid (0–23)")


    # ── CHECK 6: Row count sanity check ───────────────────────────────
    # We extracted 50,000 rows. After transforms we should still have
    # at least 45,000 (allowing for some deduplication/cleaning loss).
    # Far fewer than that suggests a serious pipeline error.
    min_expected_rows = 45_000
    if len(df) < min_expected_rows:
        fail(f"Only {len(df):,} rows remaining — expected at least {min_expected_rows:,}. "
             f"Something may have gone wrong in the transform step.")
    else:
        ok(f"Row count looks healthy: {len(df):,} rows")


    # ── Final result ───────────────────────────────────────────────────
    print(f"\n[VALIDATE] Results: {passed} passed, {len(errors)} failed")

    if errors:
        print("\n[VALIDATE] PIPELINE HALTED — fix these issues before loading:")
        for e in errors:
            print(f"  → {e}")
        raise ValueError(f"Data validation failed with {len(errors)} error(s).")

    print("[VALIDATE] All checks passed. Safe to load.\n")
    return True


# ─────────────────────────────────────────────
# Run directly to test just this stage
# ─────────────────────────────────────────────

if __name__ == "__main__":
    from extract import extract_transactions, extract_exchange_rates
    from transform import transform

    df    = extract_transactions()
    rates = extract_exchange_rates()
    out   = transform(df, rates)
    validate(out)
