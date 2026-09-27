import re
import os
import numpy as np
import pandas as pd

from difflib import SequenceMatcher
from collections import Counter


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_text(value):
    """
    Normalize a single text value.
    """
    if pd.isna(value):
        return ""

    value = str(value).lower()

    # Replace punctuation with spaces
    value = re.sub(r"[^a-z0-9\s]", " ", value)

    # Remove extra spaces
    value = re.sub(r"\s+", " ", value).strip()

    return value


def normalize_name(value):
    """
    Normalize business/company name and remove common
    company suffixes.
    """
    value = normalize_text(value)

    value = re.sub(
        r"\b("
        r"private limited|private ltd|pvt ltd|"
        r"incorporated|corporation|company|"
        r"limited|llc|inc|corp|ltd|co"
        r")\b",
        " ",
        value,
    )

    value = re.sub(r"\s+", " ", value).strip()

    return value


# ============================================================
# BASIC STRING SIMILARITY
# ============================================================

def sequence_similarity(a, b):
    """
    SequenceMatcher similarity between two strings.
    Returns a value between 0 and 1.
    """
    if not a or not b:
        return 0.0

    return SequenceMatcher(None, a, b).ratio()


# ============================================================
# TOKEN-BASED FEATURES
# ============================================================

def token_set(value):
    """
    Convert text into a set of unique tokens.
    """
    if not value:
        return set()

    return set(value.split())


def jaccard_similarity(a, b):
    """
    Jaccard similarity:

        intersection / union

    Returns a value between 0 and 1.
    """
    tokens_a = token_set(a)
    tokens_b = token_set(b)

    if not tokens_a or not tokens_b:
        return 0.0

    intersection = len(tokens_a & tokens_b)
    union = len(tokens_a | tokens_b)

    if union == 0:
        return 0.0

    return intersection / union


def token_overlap(a, b):
    """
    Measures how much of the smaller token set is shared.
    """
    tokens_a = token_set(a)
    tokens_b = token_set(b)

    if not tokens_a or not tokens_b:
        return 0.0

    intersection = len(tokens_a & tokens_b)

    return intersection / min(len(tokens_a), len(tokens_b))


def common_token_count(a, b):
    """
    Number of common tokens.
    """
    return len(token_set(a) & token_set(b))


# ============================================================
# CHARACTER-LEVEL FEATURES
# ============================================================

def normalized_edit_similarity(a, b):
    """
    Approximate edit similarity using SequenceMatcher.
    """
    if not a or not b:
        return 0.0

    max_len = max(len(a), len(b))

    if max_len == 0:
        return 1.0

    return 1.0 - (
        sum(x != y for x, y in zip(a, b)) +
        abs(len(a) - len(b))
    ) / max_len


def prefix_similarity(a, b, prefix_len=3):
    """
    Checks whether the first few characters are the same.
    """
    if not a or not b:
        return 0.0

    if len(a) < prefix_len or len(b) < prefix_len:
        return 0.0

    return float(a[:prefix_len] == b[:prefix_len])


def suffix_similarity(a, b, suffix_len=3):
    """
    Checks whether the last few characters are the same.
    """
    if not a or not b:
        return 0.0

    if len(a) < suffix_len or len(b) < suffix_len:
        return 0.0

    return float(a[-suffix_len:] == b[-suffix_len:])


# ============================================================
# LENGTH FEATURES
# ============================================================

def length_difference(a, b):
    """
    Absolute difference in string lengths.
    """
    return abs(len(a) - len(b))


def length_ratio(a, b):
    """
    Ratio of smaller string length to larger string length.
    """
    if not a or not b:
        return 0.0

    max_len = max(len(a), len(b))

    if max_len == 0:
        return 1.0

    return min(len(a), len(b)) / max_len


# ============================================================
# COUNTRY FEATURES
# ============================================================

def country_match(country_a, country_b):
    """
    Exact normalized country match.
    """
    country_a = normalize_text(country_a)
    country_b = normalize_text(country_b)

    if not country_a or not country_b:
        return 0.0

    return float(country_a == country_b)


# ============================================================
# NAME FEATURES
# ============================================================

def generate_name_features(name1, name2):
    """
    Generate multiple similarity features for business names.
    """

    name1 = normalize_name(name1)
    name2 = normalize_name(name2)

    features = {}

    features["name_exact_match"] = float(
        bool(name1) and bool(name2) and name1 == name2
    )

    features["name_sequence_similarity"] = sequence_similarity(
        name1, name2
    )

    features["name_jaccard_similarity"] = jaccard_similarity(
        name1, name2
    )

    features["name_token_overlap"] = token_overlap(
        name1, name2
    )

    features["name_common_token_count"] = common_token_count(
        name1, name2
    )

    features["name_length_difference"] = length_difference(
        name1, name2
    )

    features["name_length_ratio"] = length_ratio(
        name1, name2
    )

    features["name_prefix_match"] = prefix_similarity(
        name1, name2
    )

    features["name_suffix_match"] = suffix_similarity(
        name1, name2
    )

    return features


# ============================================================
# ADDRESS FEATURES
# ============================================================

def generate_address_features(address1, address2):
    """
    Generate similarity features for addresses.
    """

    address1 = normalize_text(address1)
    address2 = normalize_text(address2)

    features = {}

    features["address_exact_match"] = float(
        bool(address1) and bool(address2) and address1 == address2
    )

    features["address_sequence_similarity"] = sequence_similarity(
        address1, address2
    )

    features["address_jaccard_similarity"] = jaccard_similarity(
        address1, address2
    )

    features["address_token_overlap"] = token_overlap(
        address1, address2
    )

    features["address_common_token_count"] = common_token_count(
        address1, address2
    )

    features["address_length_difference"] = length_difference(
        address1, address2
    )

    features["address_length_ratio"] = length_ratio(
        address1, address2
    )

    return features


# ============================================================
# COMPLETE PAIR FEATURE GENERATION
# ============================================================

def generate_pair_features(row1, row2):
    """
    Generate all features for one candidate pair.

    row1 = entity from Source 1
    row2 = candidate entity from Source 2/3
    """

    features = {}

    # --------------------------------------------------------
    # NAME FEATURES
    # --------------------------------------------------------

    name_features = generate_name_features(
        row1.get("normalized_name", ""),
        row2.get("normalized_name", "")
    )

    features.update(name_features)

    # --------------------------------------------------------
    # ADDRESS FEATURES
    # --------------------------------------------------------

    address_features = generate_address_features(
        row1.get("normalized_address", ""),
        row2.get("normalized_address", "")
    )

    features.update(address_features)

    # --------------------------------------------------------
    # COUNTRY FEATURES
    # --------------------------------------------------------

    features["country_match"] = country_match(
        row1.get("normalized_country", ""),
        row2.get("normalized_country", "")
    )

    # --------------------------------------------------------
    # COMBINED FEATURES
    # --------------------------------------------------------

    features["name_address_similarity"] = (
        0.7 * features["name_sequence_similarity"]
        + 0.3 * features["address_sequence_similarity"]
    )

    features["overall_similarity"] = (
        0.6 * features["name_sequence_similarity"]
        + 0.3 * features["address_sequence_similarity"]
        + 0.1 * features["country_match"]
    )

    return features


# ============================================================
# FEATURE DATASET GENERATION
# ============================================================

def generate_feature_dataset(
    source1_df,
    source2_df,
    candidate_pairs,
):
    """
    Convert candidate pairs generated by the blocking stage
    into a machine-learning feature dataset.

    candidate_pairs format:

        {
            source1_position: {
                source2_position_1,
                source2_position_2,
                ...
            }
        }
    """

    records = []

    for s1_pos, candidate_positions in candidate_pairs.items():

        row1 = source1_df.iloc[s1_pos]

        for s2_pos in candidate_positions:

            row2 = source2_df.iloc[s2_pos]

            features = generate_pair_features(
                row1,
                row2
            )

            # IDs
            features["source1_entity_id"] = row1["entity_id"]
            features["source2_entity_id"] = row2["entity_id"]

            records.append(features)

    return pd.DataFrame(records)


# ============================================================
# FEATURE ENGINEERING FOR BOTH SOURCES
# ============================================================

def build_features(
    s1_df,
    s2_df,
    s3_df,
    candidates_s2,
    candidates_s3,
):
    """
    Generate ML features for:

        Source 1 <-> Source 2
        Source 1 <-> Source 3
    """

    print("=" * 60)
    print("FEATURE ENGINEERING")
    print("=" * 60)

    # --------------------------------------------------------
    # SOURCE 2 FEATURES
    # --------------------------------------------------------

    print("Generating Source 1 <-> Source 2 features...")

    features_s2 = generate_feature_dataset(
        s1_df,
        s2_df,
        candidates_s2
    )

    print(
        f"Source 2 feature pairs: {len(features_s2):,}"
    )

    # --------------------------------------------------------
    # SOURCE 3 FEATURES
    # --------------------------------------------------------

    print("Generating Source 1 <-> Source 3 features...")

    features_s3 = generate_feature_dataset(
        s1_df,
        s3_df,
        candidates_s3
    )

    print(
        f"Source 3 feature pairs: {len(features_s3):,}"
    )

    # --------------------------------------------------------
    # COMBINE
    # --------------------------------------------------------

    features = pd.concat(
        [features_s2, features_s3],
        ignore_index=True
    )

    print(
        f"Total feature pairs: {len(features):,}"
    )

    return features


# ============================================================
# SAVE FEATURES
# ============================================================

def save_features(features, output_path):

    os.makedirs(
        os.path.dirname(output_path) or ".",
        exist_ok=True
    )

    features.to_csv(
        output_path,
        sep="\t",
        index=False
    )

    print()
    print("=" * 60)
    print("FEATURE DATASET SAVED")
    print("=" * 60)

    print(
        f"Rows:    {len(features):,}"
    )

    print(
        f"Columns: {len(features.columns):,}"
    )

    print(
        f"Output:  {output_path}"
    )


# ============================================================
# EXAMPLE
# ============================================================

if __name__ == "__main__":

    # Example usage:
    #
    # features = build_features(
    #     s1,
    #     s2,
    #     s3,
    #     candidates_s2,
    #     candidates_s3
    # )
    #
    # save_features(
    #     features,
    #     "output/features.tsv"
    # )

    print("Feature engineering module loaded successfully.")