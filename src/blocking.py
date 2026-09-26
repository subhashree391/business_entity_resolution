import argparse
import gc
import heapq
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from normalization import normalize_text, normalize_name
    print("[blocking.py] Imported normalize_text/normalize_name from normalization.py")
except Exception as exc:  # noqa: BLE001
    print(
        f"[blocking.py] WARNING: could not import from normalization.py ({exc}). "
        "Falling back to local normalization logic."
    )

    def normalize_text(series):
        return (
            series.fillna("")
            .astype(str)
            .str.lower()
            .str.replace(r"[^a-z0-9\s]", " ", regex=True)
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
        )

    def normalize_name(series):
        series = normalize_text(series)
        return (
            series.str.replace(
                r"\b("
                r"private limited|private ltd|pvt ltd|incorporated|"
                r"corporation|company|limited|llc|inc|corp|ltd|co"
                r")\b",
                " ",
                regex=True,
            )
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
        )


def ensure_normalized(df):
    """Normalize required fields and drop raw string columns to save memory."""
    if "normalized_name" not in df.columns:
        df["normalized_name"] = normalize_name(df["business_name"])
    if "normalized_address" not in df.columns:
        df["normalized_address"] = normalize_text(df["business_address"])
    if "normalized_country" not in df.columns:
        df["normalized_country"] = normalize_text(df["country"])

    raw_cols = ["business_name", "business_address", "country"]
    df.drop(columns=[col for col in raw_cols if col in df.columns], inplace=True)
    return df


COUNTRY_ALIASES = {
    "us": "us",
    "usa": "us",
    "u s a": "us",
    "united states": "us",
    "united states of america": "us",
    "america": "us",
    "india": "india",
    "bharat": "india",
    "republic of india": "india",
    "france": "france",
    "french republic": "france",
}


def canonicalize_country(value):
    v = (value or "").strip().lower()
    return COUNTRY_ALIASES.get(v, v)


# ============================================================
# MEMORY-BOUNDED TOKEN INVERTED INDEX
# ============================================================

def build_inverted_index(
    texts,
    min_token_len=3,
    max_doc_freq_ratio=0.01,
    max_doc_freq_abs=1500,
):
    """Builds token -> np.ndarray[int32].
    Using np.ndarray rather than Python sets drastically drops RAM usage.
    """
    n = len(texts)
    max_doc_freq = min(max(5, int(n * max_doc_freq_ratio)), max_doc_freq_abs)

    temp_dict = defaultdict(list)
    for pos, text in enumerate(texts):
        # Unique tokens per document
        tokens = set(text.split())
        for tok in tokens:
            if len(tok) >= min_token_len:
                temp_dict[tok].append(pos)

    index = {}
    for tok, positions in temp_dict.items():
        if len(positions) <= max_doc_freq:
            index[tok] = np.array(positions, dtype=np.int32)

    del temp_dict
    return index


def update_token_candidates(
    s1_texts,
    inverted_index,
    s1_global_pos,
    other_global_pos,
    result_dict,
    max_cands_per_entity=100,
    min_token_len=3,
):
    """Updates candidate sets in-place with a strict cap per entity to prevent runaway memory."""
    for local_i, text in enumerate(s1_texts):
        gi = s1_global_pos[local_i]
        curr_set = result_dict[gi]

        if len(curr_set) >= max_cands_per_entity:
            continue

        for tok in set(text.split()):
            if len(tok) < min_token_len:
                continue
            hits = inverted_index.get(tok)
            if hits is not None:
                # Add hits up to the budget
                remaining = max_cands_per_entity - len(curr_set)
                if remaining <= 0:
                    break
                curr_set.update(other_global_pos[hits[:remaining]])


# ============================================================
# TF-IDF BLOCKING WITHIN BUCKETS
# ============================================================

def tfidf_bucket_candidates(
    s1_texts,
    other_texts,
    ngram_range=(2, 4),
    max_features=25_000,
    top_k=15,
    threshold=0.35,
    row_batch_size=1000,
    col_batch_size=2000,
):
    n1_total = len(s1_texts)
    n2_total = len(other_texts)
    candidates = defaultdict(set)
    if n1_total == 0 or n2_total == 0:
        return candidates

    # Fast guard: check if all strings are empty or too short for ngram_range[0]
    min_len = ngram_range[0]
    if not any(len(t.strip()) >= min_len for t in s1_texts) or not any(
        len(t.strip()) >= min_len for t in other_texts
    ):
        return candidates

    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=ngram_range,
        max_features=max_features,
        min_df=1,
        dtype=np.float32,
    )

    combined_corpus = np.concatenate([s1_texts, other_texts])
    try:
        vectorizer.fit(combined_corpus)
    except ValueError:
        # Raised if vocabulary is completely empty (no valid n-grams formed)
        del vectorizer, combined_corpus
        return candidates

    del combined_corpus

    x1_bucket = vectorizer.transform(s1_texts).astype(np.float32)
    x2_bucket = vectorizer.transform(other_texts).astype(np.float32)
    del vectorizer

    # Guard: if matrices ended up with 0 non-zero elements
    if x1_bucket.nnz == 0 or x2_bucket.nnz == 0:
        del x1_bucket, x2_bucket
        return candidates

    heaps = [[] for _ in range(n1_total)]

    for col_start in range(0, n2_total, col_batch_size):
        col_end = min(col_start + col_batch_size, n2_total)
        x2_chunk = x2_bucket[col_start:col_end]

        for row_start in range(0, n1_total, row_batch_size):
            row_end = min(row_start + row_batch_size, n1_total)
            sim_block = (x1_bucket[row_start:row_end] @ x2_chunk.T).tocsr()

            for local_i in range(sim_block.shape[0]):
                row = sim_block.getrow(local_i)
                if row.nnz == 0:
                    continue

                idx = row.indices
                data = row.data
                mask = data >= threshold
                idx = idx[mask]
                data = data[mask]
                if idx.size == 0:
                    continue

                global_i = row_start + local_i
                heap = heaps[global_i]
                for local_j, score in zip(idx, data):
                    other_pos = col_start + int(local_j)
                    if len(heap) < top_k:
                        heapq.heappush(heap, (float(score), other_pos))
                    elif score > heap[0][0]:
                        heapq.heappushpop(heap, (float(score), other_pos))

            del sim_block
        del x2_chunk

    del x1_bucket, x2_bucket

    for i, heap in enumerate(heaps):
        if heap:
            candidates[i] = {other_pos for _, other_pos in heap}

    return candidates


# ============================================================
# CANDIDATE GENERATION PER SOURCE
# ============================================================

def generate_source_candidates(s1_df, other_df, config):
    result = defaultdict(set)

    s1_df["_country_key"] = s1_df["normalized_country"].map(canonicalize_country)
    other_df["_country_key"] = other_df["normalized_country"].map(canonicalize_country)

    s1_df["_pos"] = np.arange(len(s1_df), dtype=np.int32)
    other_df["_pos"] = np.arange(len(other_df), dtype=np.int32)

    prefix_len = config["tfidf_prefix_len"]
    max_pair_product = config["tfidf_max_pair_product"]
    max_cands = config["max_candidates_per_entity"]

    if prefix_len > 0:
        s1_df["_prefix"] = s1_df["normalized_name"].str.slice(0, prefix_len)
        other_df["_prefix"] = other_df["normalized_name"].str.slice(0, prefix_len)

    other_country_groups = dict(tuple(other_df.groupby("_country_key", sort=False)))

    for country, s1_group in s1_df.groupby("_country_key", sort=False):
        other_group = other_country_groups.get(country)
        if other_group is None or other_group.empty:
            continue

        s1_global_pos = s1_group["_pos"].to_numpy()
        other_global_pos = other_group["_pos"].to_numpy()

        s1_names = s1_group["normalized_name"].to_numpy()
        other_names = other_group["normalized_name"].to_numpy()

        # 1. Name token blocking (bounded insertion)
        name_index = build_inverted_index(
            other_names,
            min_token_len=config["min_token_len"],
            max_doc_freq_ratio=config["max_doc_freq_ratio"],
            max_doc_freq_abs=config["max_doc_freq_abs"],
        )
        update_token_candidates(
            s1_names,
            name_index,
            s1_global_pos,
            other_global_pos,
            result,
            max_cands_per_entity=max_cands,
            min_token_len=config["min_token_len"],
        )
        del name_index
        gc.collect()

        # 2. Address token blocking (bounded insertion)
        s1_addrs = s1_group["normalized_address"].to_numpy()
        other_addrs = other_group["normalized_address"].to_numpy()

        addr_index = build_inverted_index(
            other_addrs,
            min_token_len=config["addr_min_token_len"],
            max_doc_freq_ratio=config["max_doc_freq_ratio"],
            max_doc_freq_abs=config["max_doc_freq_abs"],
        )
        update_token_candidates(
            s1_addrs,
            addr_index,
            s1_global_pos,
            other_global_pos,
            result,
            max_cands_per_entity=max_cands,
            min_token_len=config["addr_min_token_len"],
        )
        del addr_index
        gc.collect()

        # 3. TF-IDF nearest neighbours within small prefix buckets
        if prefix_len > 0:
            other_sub_groups = dict(tuple(other_group.groupby("_prefix", sort=False)))
            for prefix, s1_sub in s1_group.groupby("_prefix", sort=False):
            # Skip empty/whitespace prefixes
                if not prefix or not prefix.strip():
                    continue

                other_sub = other_sub_groups.get(prefix)
                if other_sub is None or other_sub.empty:
                    continue

                s1_sub_global = s1_sub["_pos"].to_numpy()
                other_sub_global = other_sub["_pos"].to_numpy()

                local_result = tfidf_bucket_candidates(
                    s1_sub["normalized_name"].to_numpy(),
                    other_sub["normalized_name"].to_numpy(),
                    ngram_range=(2, 4),
                    max_features=config["tfidf_max_features"],
                    top_k=config["tfidf_top_k"],
                    threshold=config["tfidf_threshold"],
                    row_batch_size=config["tfidf_row_batch_size"],
                    col_batch_size=config["tfidf_col_batch_size"],
                )

                for local_i, local_ids in local_result.items():
                    gi = s1_sub_global[local_i]
                    if len(result[gi]) < max_cands:
                        result[gi].update(other_sub_global[j] for j in local_ids)

            del other_sub_groups

        gc.collect()

    del other_country_groups
    s1_df.drop(columns=[c for c in ["_country_key", "_pos", "_prefix"] if c in s1_df.columns], inplace=True)
    other_df.drop(columns=[c for c in ["_country_key", "_pos", "_prefix"] if c in other_df.columns], inplace=True)
    gc.collect()

    return result


# ============================================================
# MAIN & I/O
# ============================================================

def load_source(path):
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    return ensure_normalized(df)


def write_candidate_pairs(s1_ids, other2_ids, other3_ids, candidates_s2, candidates_s3, output_path):
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    total_candidates = 0
    zero_candidate_entities = 0
    n = len(s1_ids)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for pos in range(n):
            eid = s1_ids[pos]
            merged_positions_s2 = candidates_s2.get(pos)
            merged_positions_s3 = candidates_s3.get(pos)

            merged_ids = set()
            if merged_positions_s2:
                merged_ids.update(other2_ids[j] for j in merged_positions_s2)
            if merged_positions_s3:
                merged_ids.update(other3_ids[j] for j in merged_positions_s3)

            if not merged_ids:
                zero_candidate_entities += 1

            total_candidates += len(merged_ids)
            f.write(f"{eid}\t{','.join(sorted(merged_ids))}\n")

    print()
    print("=" * 60)
    print("BLOCKING SUMMARY")
    print("=" * 60)
    print(f"Source 1 entities:                {n:,}")
    print(f"Entities with zero candidates:     {zero_candidate_entities:,}")
    print(f"Total candidate pairs generated:   {total_candidates:,}")
    print(f"Avg candidates per S1 entity:      {total_candidates / n:.2f}" if n else "N/A")
    print(f"Output written to:                 {output_path}")


def build_config(args):
    return {
        "min_token_len": args.min_token_len,
        "addr_min_token_len": args.addr_min_token_len,
        "max_doc_freq_ratio": args.max_doc_freq_ratio,
        "max_doc_freq_abs": args.max_doc_freq_abs,
        "max_candidates_per_entity": args.max_candidates_per_entity,
        "tfidf_top_k": args.tfidf_top_k,
        "tfidf_threshold": args.tfidf_threshold,
        "tfidf_prefix_len": args.tfidf_prefix_len,
        "tfidf_max_pair_product": args.tfidf_max_pair_product,
        "tfidf_row_batch_size": args.tfidf_row_batch_size,
        "tfidf_col_batch_size": args.tfidf_col_batch_size,
        "tfidf_max_features": args.tfidf_max_features,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Candidate generation / blocking stage")
    parser.add_argument("--source1", default="dataset/test/test_source1.tsv")
    parser.add_argument("--source2", default="dataset/test/test_source2.tsv")
    parser.add_argument("--source3", default="dataset/test/test_source3.tsv")
    parser.add_argument("--output", default="output/candidate_pairs.tsv")
    parser.add_argument("--ground-truth", default=None)

    parser.add_argument("--min-token-len", type=int, default=3)
    parser.add_argument("--addr-min-token-len", type=int, default=4)
    parser.add_argument("--max-doc-freq-ratio", type=float, default=0.01)
    parser.add_argument("--max-doc-freq-abs", type=int, default=1000)
    parser.add_argument(
        "--max-candidates-per-entity",
        type=int,
        default=100,
        help="Strict upper bound of candidates retained per S1 entity across blocking passes.",
    )
    parser.add_argument("--tfidf-top-k", type=int, default=15)
    parser.add_argument("--tfidf-threshold", type=float, default=0.35)
    parser.add_argument("--tfidf-prefix-len", type=int, default=2)
    parser.add_argument("--tfidf-max-pair-product", type=int, default=10_000_000)
    parser.add_argument("--tfidf-row-batch-size", type=int, default=1000)
    parser.add_argument("--tfidf-col-batch-size", type=int, default=2000)
    parser.add_argument("--tfidf-max-features", type=int, default=25_000)

    return parser.parse_args()


def main():
    args = parse_args()
    config = build_config(args)

    print("=" * 60)
    print("LOADING & NORMALIZING")
    print("=" * 60)
    s1 = load_source(args.source1)
    s2 = load_source(args.source2)
    s3 = load_source(args.source3)
    print(f"S1: {len(s1):,}  S2: {len(s2):,}  S3: {len(s3):,}")

    s1_ids = s1["entity_id"].to_numpy()
    s2_ids = s2["entity_id"].to_numpy()
    s3_ids = s3["entity_id"].to_numpy()

    print()
    print("=" * 60)
    print("BLOCKING AGAINST SOURCE 2")
    print("=" * 60)
    candidates_s2 = generate_source_candidates(s1, s2, config)
    gc.collect()

    print()
    print("=" * 60)
    print("BLOCKING AGAINST SOURCE 3")
    print("=" * 60)
    candidates_s3 = generate_source_candidates(s1, s3, config)
    gc.collect()

    write_candidate_pairs(s1_ids, s2_ids, s3_ids, candidates_s2, candidates_s3, args.output)


if __name__ == "__main__":
    main()