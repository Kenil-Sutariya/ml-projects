"""
Train a few custom test models and save them to /tmp for BYO Model testing.

Trains 3 different model types to exercise the prediction engine:
  1. RandomForest Pipeline   — has predict + predict_proba (the typical case)
  2. Logistic Regression     — has predict + predict_proba (different family)
  3. SVC without probability — has predict ONLY (tests the no-proba path)

All models are trained on a held-out slice of the existing demo dataset,
then saved as both .pkl (pickle) and .joblib for round-trip testing.
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose       import ColumnTransformer
from sklearn.ensemble      import RandomForestClassifier
from sklearn.impute        import SimpleImputer
from sklearn.linear_model  import LogisticRegression
from sklearn.pipeline      import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm           import SVC

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

OUT = Path("/tmp/byom_test_models")
OUT.mkdir(exist_ok=True)

NUM = ["age", "income", "account_age_months", "monthly_activity_score",
       "support_tickets", "num_logins", "feature_usage_score"]
CAT = ["region", "subscription_type"]

# Use the original raw dataset (with target, no predictions yet)
TRAIN_CSV = ROOT.parent / "ai-ml-monitoring-dashboard" / "data" / "raw" / "classification_data.csv"
df = pd.read_csv(TRAIN_CSV)
X, y = df[NUM + CAT], df["target"]

# Standard preprocessing block
preproc = ColumnTransformer([
    ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                      ("sc",  StandardScaler())]), NUM),
    ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                      ("ohe", OneHotEncoder(handle_unknown="ignore"))]), CAT),
])


def train_and_save(model_name: str, classifier) -> None:
    pipeline = Pipeline([("preproc", preproc), ("clf", classifier)])
    pipeline.fit(X, y)
    pkl_path = OUT / f"{model_name}.pkl"
    job_path = OUT / f"{model_name}.joblib"
    joblib.dump(pipeline, pkl_path)
    joblib.dump(pipeline, job_path)

    has_proba = hasattr(pipeline, "predict_proba")
    print(f"  ✓ {model_name:<30s} → predict_proba={has_proba!s:<5s} "
          f"({pkl_path.name}, {job_path.name})")


print("Training test models for BYO Model verification …")
train_and_save("rf_pipeline",          RandomForestClassifier(n_estimators=80, max_depth=8, random_state=42))
train_and_save("logistic_regression",  LogisticRegression(max_iter=1000))
train_and_save("svc_no_proba",         SVC(probability=False, random_state=42))

print(f"\nSaved to: {OUT}")
print("Files:")
for p in sorted(OUT.iterdir()):
    print(f"  {p.name}  ({p.stat().st_size:,} bytes)")
