import argparse
import csv
import re
from collections import Counter
 
 
def load_tsv(path):
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        return list(reader)
 
 
def basic_stats(rows, name):
    print(f"\n=== {name} ===")
    print(f"rows: {len(rows)}")
    if not rows:
        return
    cols = rows[0].keys()
    for col in cols:
        n_missing = sum(1 for r in rows if not r.get(col, "").strip())
        print(f"  missing '{col}': {n_missing} ({n_missing/len(rows):.1%})")
 
    if "country" in cols:
        c = Counter(r["country"].strip() for r in rows if r.get("country", "").strip())
        print(f"  country distribution: {dict(c)}")
 
    if "entity_id" in cols:
        n_dupe_ids = len(rows) - len(set(r["entity_id"] for r in rows))
        print(f"  duplicate entity_id count: {n_dupe_ids}")
 
    if "business_name" in cols and "business_address" in cols:
        dupe_key = Counter(
            (r["business_name"].strip().lower(), r["business_address"].strip().lower())
            for r in rows
        )
        n_exact_dupes = sum(v - 1 for v in dupe_key.values() if v > 1)
        print(f"  exact (name,address) duplicate rows: {n_exact_dupes}")
 
        name_lengths = [len(r["business_name"].split()) for r in rows if r.get("business_name")]
        if name_lengths:
            print(f"  name token count: min={min(name_lengths)} max={max(name_lengths)} "
                  f"avg={sum(name_lengths)/len(name_lengths):.1f}")
 
        # Legal-suffix-like token frequency (rough heuristic, last token of name)
        last_tokens = Counter(
            r["business_name"].strip().split()[-1].lower().strip(".,")
            for r in rows if r.get("business_name", "").strip()
        )
        print("  top 20 last-tokens in business_name (candidate legal suffixes):")
        for tok, cnt in last_tokens.most_common(20):
            print(f"    {tok!r}: {cnt}")
 
        # non-ASCII character check
        non_ascii_names = sum(
            1 for r in rows if any(ord(ch) > 127 for ch in r.get("business_name", ""))
        )
        non_ascii_addr = sum(
            1 for r in rows if any(ord(ch) > 127 for ch in r.get("business_address", ""))
        )
        print(f"  names with non-ASCII chars: {non_ascii_names}")
        print(f"  addresses with non-ASCII chars: {non_ascii_addr}")
 
        # landmark phrase frequency
        landmark_re = re.compile(r"\b(near|opposite|behind|next to|beside)\b", re.I)
        n_landmark = sum(
            1 for r in rows if landmark_re.search(r.get("business_address", ""))
        )
        print(f"  addresses containing landmark phrases: {n_landmark}")
 
        # postal code presence (5-6 digit number)
        postal_re = re.compile(r"\b\d{5,6}\b")
        n_postal = sum(
            1 for r in rows if postal_re.search(r.get("business_address", ""))
        )
        print(f"  addresses containing a 5-6 digit code: {n_postal}")
 
 
def ground_truth_stats(rows):
    print("\n=== ground truth ===")
    print(f"rows (S1 entities): {len(rows)}")
    match_counts = []
    for r in rows:
        ids = r.get("matched_entity_ids", "").strip()
        n = 0 if not ids else len(ids.split(","))
        match_counts.append(n)
    c = Counter(match_counts)
    print(f"  match count distribution (0=singleton): {dict(sorted(c.items()))}")
    n_singleton = c.get(0, 0)
    print(f"  singleton rate: {n_singleton/len(rows):.1%}")
    if match_counts:
        print(f"  avg matches per entity: {sum(match_counts)/len(match_counts):.2f}, "
              f"max: {max(match_counts)}")
 
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    parser.add_argument("--ground-truth", action="store_true")
    args = parser.parse_args()
 
    rows = load_tsv(args.path)
    if args.ground_truth:
        ground_truth_stats(rows)
    else:
        basic_stats(rows, args.path)