# Business Entity Resolution

## Overview

This project implements a machine learning based **Business Entity Resolution** pipeline for matching business records across multiple independent data sources.

The objective is to identify which records from **Source 2** and **Source 3** refer to the same real-world business as each record in **Source 1**.

Source 1 acts as the deduplicated reference source. Each Source 1 entity may have:

* No matching records
* One matching record
* Multiple matching records

The input data contains noisy and inconsistent business names, addresses, and other fields. The pipeline therefore performs data normalization, candidate generation, and entity matching to identify probable matches.

---

## Problem Statement

Business information can appear differently across different data sources due to:

* Spelling mistakes
* Abbreviations
* Different legal suffixes
* Punctuation differences
* Word-order changes
* Transliteration
* Partial addresses
* Missing address components
* Different formatting conventions
* Landmark-based addresses

The goal is to resolve these records without relying on external business databases or APIs.

The solution uses only the data provided as part of the challenge.

---

## Dataset

The challenge contains three independent sources.

### Training Data

```text
dataset/
├── train/
│   ├── train_source1.tsv
│   ├── train_source2.tsv
│   ├── train_source3.tsv
│   └── train_ground_truth.tsv
```

### Test Data

```text
dataset/
└── test/
    ├── test_source1.tsv
    ├── test_source2.tsv
    └── test_source3.tsv
```

All datasets are stored as **tab-separated values (TSV)** files.

When loading the data, the separator must therefore be specified explicitly:

```python
import pandas as pd

df = pd.read_csv("file.tsv", sep="\t")
```

---

## Input Columns

The source files contain the following fields:

| Column             | Description                        |
| ------------------ | ---------------------------------- |
| `entity_id`        | Unique identifier of the record    |
| `business_name`    | Name of the business               |
| `business_address` | Address of the business            |
| `country`          | Country associated with the record |

The source can be identified from the `entity_id` prefix:

```text
S1- → Source 1
S2- → Source 2
S3- → Source 3
```

The pipeline treats country as an open set of labels and does not assume that only the countries present in the training data will occur in the test data.

---

## Pipeline

The complete pipeline consists of the following major stages:

```text
Raw TSV Data
     │
     ▼
Data Loading
     │
     ▼
Data Cleaning & Normalization
     │
     ▼
Candidate Generation / Blocking
     │
     ▼
Candidate Feature Generation
     │
     ▼
Entity Matching
     │
     ▼
Match Filtering / Decision
     │
     ▼
Output Generation
     │
     ├── matching_results.tsv
     └── candidate_pairs.tsv
```

---

## 1. Data Loading

The three source files are loaded independently using pandas with a tab separator.

The pipeline keeps the source identifiers so that records can be traced back to their original datasets.

---

## 2. Data Cleaning and Normalization

Business names and addresses can contain significant formatting differences even when they refer to the same entity.

The normalization stage reduces these superficial differences before matching.

Typical preprocessing includes:

* Converting text to a consistent case
* Removing unnecessary punctuation
* Normalizing whitespace
* Standardizing common textual variations
* Handling missing values
* Normalizing business names
* Normalizing business addresses

The normalization process is applied consistently across the different sources.

The normalized datasets are generated before candidate matching.

Example normalized files:

```text
source1_normalized.tsv
source2_normalized.tsv
source3_normalized.tsv
```

---

## 3. Candidate Generation / Blocking

Comparing every Source 1 record with every Source 2 and Source 3 record would result in an extremely large number of comparisons.

Therefore, a **blocking / candidate-generation** stage is used.

The purpose of blocking is to reduce the search space while retaining likely matching records.

Only records that satisfy the candidate-generation conditions are passed to the subsequent matching stage.

Conceptually:

```text
Source 1
   │
   ├── Blocking
   │
   ├── Candidate S2 records
   │
   └── Candidate S3 records
```

The final candidate set is stored in:

```text
output/candidate_pairs.tsv
```

The candidate file represents the records that are actually considered by the final matching stage.

---

## 4. Feature Generation

Candidate pairs are compared using features derived from the available business information.

Potential similarity features include:

* Business name similarity
* Address similarity
* Token overlap
* Character-level similarity
* Country consistency
* Other normalized text-based features

The features are calculated using only the provided challenge data.

No external business databases, APIs, geocoding services, or external identity lookups are used.

---

## 5. Entity Matching

The matching stage evaluates candidate pairs and determines whether they represent the same business entity.

The model/decision process produces the final set of matching Source 2 and Source 3 records for every Source 1 entity.

Because the evaluation metric is precision-heavy, the pipeline is designed to avoid unnecessary false matches.

The challenge evaluates predictions using:

```text
F0.5
```

where precision receives greater weight than recall.

---

## 6. Output Generation

The final pipeline generates two files inside the `output/` directory:

```text
output/
├── matching_results.tsv
└── candidate_pairs.tsv
```

### matching_results.tsv

This contains the final predicted matches.

Format:

```text
source1_entity_id    matched_entity_ids
```

Example:

```text
source1_entity_id	matched_entity_ids
S1-00001	S2-00047,S2-00193,S3-00812
S1-00002	S3-00004
S1-00003	
```

Every Source 1 entity must have exactly one row.

If no matching Source 2 or Source 3 entity is found, the `matched_entity_ids` field is left empty.

---

### candidate_pairs.tsv

This contains the candidate records considered by the matching stage.

Format:

```text
source1_entity_id    candidate_entity_ids
```

Example:

```text
source1_entity_id	candidate_entity_ids
S1-00001	S2-00047,S2-00193,S3-00812,S3-00999
S1-00002	S3-00004
S1-00003	
```

Every final match must also appear in the candidate list.

---

## Project Structure

The recommended project structure is:

```text
business_entity_resolution/
│
├── src/
│   ├── data_loading.py
│   ├── normalization.py
│   ├── candidate_generation.py
│   ├── matching.py
│   └── main.py
│
├── README.md
├── requirements.txt
│
└── output/
    ├── matching_results.tsv
    └── candidate_pairs.tsv
```

If the implementation uses different filenames, the structure can be adjusted accordingly.

---

## Requirements

Python 3.x is required.

Install the required Python packages using:

```bash
pip install -r requirements.txt
```

Example `requirements.txt`:

```text
pandas
numpy
scikit-learn
```

Only include packages that are actually used by the implementation.

---

## Running the Pipeline

From the `business_entity_resolution` directory:

```bash
python src/main.py
```

The pipeline should:

1. Load the source datasets.
2. Normalize the input data.
3. Generate candidate pairs.
4. Calculate matching features.
5. Apply the matching model/decision logic.
6. Generate the final predictions.
7. Save the output files.

The expected output is:

```text
output/matching_results.tsv
output/candidate_pairs.tsv
```

---

## Output Validation

Before submission, validate the generated files using the challenge-provided validation script.

From the `student_resource/` directory:

```bash
python3 utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```

A successful validation should print:

```text
PASS
```

The validator checks the required output format and consistency rules.

---

## Important Output Rules

The generated submission must satisfy the following conditions:

* Every Source 1 test entity must appear exactly once.
* `matched_entity_ids` may contain only Source 2 and Source 3 IDs.
* Source 1 self-matches are not allowed.
* Duplicate entity IDs within an ID list are not allowed.
* IDs must exist in the test data.
* Empty match lists must remain empty.
* Every final match must appear in `candidate_pairs.tsv`.
* Both files must be tab-separated.

---

## Evaluation Metric

The challenge uses the **F0.5 score**:

```text
F0.5 = (1.25 × Precision × Recall)
       / (0.25 × Precision + Recall)
```

The score is calculated per Source 1 entity and then macro-averaged.

The metric places greater importance on precision than recall, meaning incorrect matches are particularly important to avoid.

Singleton entities are also evaluated. Correctly predicting that an entity has no matches receives credit.

---

## Data Usage and Fair Play

This project uses only the datasets supplied with the challenge.

No external entity-resolution services or external business databases are used.

The following are not used:

* Commercial entity-resolution APIs
* Government business-registration databases
* Geocoding APIs
* External business identity lookups
* Internet-based business information
* External data augmentation

All preprocessing, candidate generation, feature engineering, and matching are performed using the provided challenge data.

---

## Reproducibility

The complete pipeline is contained within this project.

To reproduce the results:

```text
1. Install the required dependencies.
2. Place the challenge datasets in the expected dataset directories.
3. Run the main pipeline.
4. Check the generated files in output/.
5. Run the submission validator.
```

The objective is that the complete output can be regenerated from the supplied data and the code contained in this directory.

---

## Final Submission

The final submission package follows the required structure:

```text
<team_name>_submission.zip
│
├── output/
│   ├── matching_results.tsv
│   └── candidate_pairs.tsv
│
├── code/
│   └── business_entity_resolution/
│       ├── src/
│       ├── README.md
│       └── requirements.txt
│
└── Documentation_template.md
```

The methodology document should describe:

* Methodology used
* Candidate generation / blocking strategy
* Model architecture
* Feature engineering
* Other relevant implementation details

---

## Summary

This project provides an end-to-end entity resolution pipeline for identifying matching business records across three noisy data sources.

The main stages are:

```text
Data Loading
      ↓
Normalization
      ↓
Candidate Generation
      ↓
Feature Engineering
      ↓
Entity Matching
      ↓
Output Generation
      ↓
Validation
```

The final outputs are:

```text
matching_results.tsv
candidate_pairs.tsv
```

Both files are generated according to the challenge submission requirements.


