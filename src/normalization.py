# ============================================================
# BUSINESS ENTITY RESOLUTION
# EDA + FAST DATA NORMALIZATION
# ============================================================

import os
import pandas as pd


# ============================================================
# SETTINGS
# ============================================================

INPUT_DIR = "dataset/train"
OUTPUT_DIR = "output/normalized"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 1. LOAD DATA
# ============================================================

print("=" * 60)
print("LOADING DATA")
print("=" * 60)

s1 = pd.read_csv(
    f"{INPUT_DIR}/train_source1.tsv",
    sep="\t",
    dtype=str,
    keep_default_na=False
)

s2 = pd.read_csv(
    f"{INPUT_DIR}/train_source2.tsv",
    sep="\t",
    dtype=str,
    keep_default_na=False
)

s3 = pd.read_csv(
    f"{INPUT_DIR}/train_source3.tsv",
    sep="\t",
    dtype=str,
    keep_default_na=False
)

print(f"S1 rows: {len(s1):,}")
print(f"S2 rows: {len(s2):,}")
print(f"S3 rows: {len(s3):,}")


# ============================================================
# 2. EDA SUMMARY
# ============================================================

print()
print("=" * 60)
print("EDA SUMMARY")
print("=" * 60)


def eda_report(df, source_name):

    print()
    print("-" * 60)
    print(source_name)
    print("-" * 60)

    print("Rows:", f"{len(df):,}")
    print("Columns:", list(df.columns))

    print()
    print("Missing values:")

    for column in df.columns:

        missing = (
            df[column]
            .isna()
            .sum()
        )

        empty = (
            df[column]
            .eq("")
            .sum()
        )

        print(
            f"{column}: "
            f"{missing + empty:,}"
        )

    print()
    print("Duplicate rows:", f"{df.duplicated().sum():,}")

    print(
        "Duplicate entity IDs:",
        f"{df['entity_id'].duplicated().sum():,}"
    )

    print()
    print("Unique values:")

    for column in df.columns:

        print(
            f"{column}: "
            f"{df[column].nunique():,}"
        )


eda_report(s1, "SOURCE 1")
eda_report(s2, "SOURCE 2")
eda_report(s3, "SOURCE 3")


# ============================================================
# 3. FAST TEXT NORMALIZATION
# ============================================================

def normalize_text(series):

    return (
        series
        .fillna("")
        .astype(str)
        .str.lower()
        .str.replace(
            r"[^a-z0-9\s]",
            " ",
            regex=True
        )
        .str.replace(
            r"\s+",
            " ",
            regex=True
        )
        .str.strip()
    )


# ============================================================
# 4. BUSINESS NAME NORMALIZATION
# ============================================================

def normalize_name(series):

    series = normalize_text(series)

    return (
        series
        .str.replace(
            r"\b("
            r"private limited|"
            r"private ltd|"
            r"pvt ltd|"
            r"incorporated|"
            r"corporation|"
            r"company|"
            r"limited|"
            r"llc|"
            r"inc|"
            r"corp|"
            r"ltd|"
            r"co"
            r")\b",
            " ",
            regex=True
        )
        .str.replace(
            r"\s+",
            " ",
            regex=True
        )
        .str.strip()
    )


# ============================================================
# 5. NORMALIZE SOURCE 1
# ============================================================

print()
print("=" * 60)
print("NORMALIZING SOURCE 1")
print("=" * 60)

s1["normalized_name"] = normalize_name(
    s1["business_name"]
)

s1["normalized_address"] = normalize_text(
    s1["business_address"]
)

s1["normalized_country"] = normalize_text(
    s1["country"]
)

print("S1 normalization complete.")


# ============================================================
# 6. NORMALIZE SOURCE 2
# ============================================================

print()
print("=" * 60)
print("NORMALIZING SOURCE 2")
print("=" * 60)

s2["normalized_name"] = normalize_name(
    s2["business_name"]
)

s2["normalized_address"] = normalize_text(
    s2["business_address"]
)

s2["normalized_country"] = normalize_text(
    s2["country"]
)

print("S2 normalization complete.")


# ============================================================
# 7. NORMALIZE SOURCE 3
# ============================================================

print()
print("=" * 60)
print("NORMALIZING SOURCE 3")
print("=" * 60)

s3["normalized_name"] = normalize_name(
    s3["business_name"]
)

s3["normalized_address"] = normalize_text(
    s3["business_address"]
)

s3["normalized_country"] = normalize_text(
    s3["country"]
)

print("S3 normalization complete.")


# ============================================================
# 8. SAVE NORMALIZED DATA
# ============================================================

print()
print("=" * 60)
print("SAVING NORMALIZED DATA")
print("=" * 60)


s1.to_csv(
    f"{OUTPUT_DIR}/source1_normalized.tsv",
    sep="\t",
    index=False
)

print("Saved S1.")


s2.to_csv(
    f"{OUTPUT_DIR}/source2_normalized.tsv",
    sep="\t",
    index=False
)

print("Saved S2.")


s3.to_csv(
    f"{OUTPUT_DIR}/source3_normalized.tsv",
    sep="\t",
    index=False
)

print("Saved S3.")


# ============================================================
# 9. NORMALIZATION SUMMARY
# ============================================================

print()
print("=" * 60)
print("NORMALIZATION COMPLETE")
print("=" * 60)

print()
print("Output directory:")
print(OUTPUT_DIR)

print()
print("Files created:")

print("1. source1_normalized.tsv")
print("2. source2_normalized.tsv")
print("3. source3_normalized.tsv")

print()
print("S1 records:", f"{len(s1):,}")
print("S2 records:", f"{len(s2):,}")
print("S3 records:", f"{len(s3):,}")

print()
print("DONE.")