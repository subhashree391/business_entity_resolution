# ============================================================
# MODEL TRAINING AND PREDICTION
# Business Entity Resolution
# ============================================================

import os
import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report


# ============================================================
# TRAIN MODEL
# ============================================================

def train_model(features, target_column="label"):
    """
    Train a Random Forest model using the generated features.

    Parameters:
        features       : pandas DataFrame containing ML features
        target_column  : column containing 0/1 match labels

    Returns:
        trained model
    """

    print("=" * 60)
    print("MODEL TRAINING")
    print("=" * 60)

    if target_column not in features.columns:
        raise ValueError(
            f"Target column '{target_column}' was not found "
            f"in the feature dataset."
        )

    # Separate input features and target
    X = features.drop(columns=[target_column])
    y = features[target_column]

    # Remove non-numeric columns
    X = X.select_dtypes(include=["number"])

    print(f"Training rows: {len(X)}")
    print(f"Number of features: {X.shape[1]}")

    # Create Random Forest model
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced"
    )

    # Train
    model.fit(X, y)

    print("Model trained successfully.")

    return model


# ============================================================
# PREDICT MATCHES
# ============================================================

def predict_matches(model, features, threshold=0.5):
    """
    Predict whether entity pairs are matches.

    Returns:
        DataFrame containing predictions and probabilities.
    """

    print("=" * 60)
    print("PREDICTING ENTITY MATCHES")
    print("=" * 60)

    X = features.select_dtypes(include=["number"]).copy()

    # If label is present, don't use it for prediction
    if "label" in X.columns:
        X = X.drop(columns=["label"])

    probabilities = model.predict_proba(X)[:, 1]

    predictions = (probabilities >= threshold).astype(int)

    results = features.copy()

    results["match_probability"] = probabilities
    results["prediction"] = predictions

    print(f"Total pairs: {len(results)}")
    print(f"Predicted matches: {predictions.sum()}")

    return results


# ============================================================
# SAVE MODEL
# ============================================================

def save_model(model, output_path="output/model.pkl"):
    """
    Save trained model to disk.
    """

    os.makedirs(
        os.path.dirname(output_path) or ".",
        exist_ok=True
    )

    joblib.dump(model, output_path)

    print("=" * 60)
    print("MODEL SAVED")
    print("=" * 60)
    print(f"Output: {output_path}")


# ============================================================
# LOAD MODEL
# ============================================================

def load_model(model_path="output/model.pkl"):
    """
    Load a previously trained model.
    """

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found: {model_path}"
        )

    model = joblib.load(model_path)

    print("Model loaded successfully.")

    return model


# ============================================================
# SAVE PREDICTION RESULTS
# ============================================================

def save_predictions(results, output_path="output/matching_results.tsv"):
    """
    Save predicted entity matches.
    """

    os.makedirs(
        os.path.dirname(output_path) or ".",
        exist_ok=True
    )

    results.to_csv(
        output_path,
        sep="\t",
        index=False
    )

    print("=" * 60)
    print("PREDICTIONS SAVED")
    print("=" * 60)
    print(f"Output: {output_path}")


# ============================================================
# EXAMPLE
# ============================================================

if __name__ == "__main__":

    print("Business Entity Resolution Model Module")
    print("Model module loaded successfully.")