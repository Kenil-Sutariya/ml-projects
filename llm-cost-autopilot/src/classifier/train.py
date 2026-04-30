"""
Train the complexity classifier and save the artifact.

Run:
    python -m src.classifier.train
    # or
    python src/classifier/train.py
"""

import csv
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.classifier.features import extract_features, PromptFeatures

DATA_PATH = Path("data/labeled_prompts/prompts.csv")
FAILURES_PATH = Path("data/labeled_prompts/failures.csv")
MODEL_PATH = Path("src/classifier/model.pkl")
REPORT_PATH = Path("data/classifier_report.json")


def load_dataset() -> tuple[list[str], list[int]]:
    texts, labels = [], []

    for path in [DATA_PATH, FAILURES_PATH]:
        if not path.exists():
            continue
        with open(path) as f:
            for row in csv.DictReader(f):
                texts.append(row["text"])
                labels.append(int(row["tier"]))

    return texts, labels


def build_feature_matrix(texts: list[str]) -> np.ndarray:
    return np.array([extract_features(t).to_array() for t in texts])


def train(verbose: bool = True) -> dict:
    texts, labels = load_dataset()
    X = build_feature_matrix(texts)
    y = np.array(labels)

    if verbose:
        from collections import Counter
        print(f"Dataset: {len(texts)} samples — {dict(sorted(Counter(labels).items()))}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Try three models, pick the best by cross-val accuracy
    candidates = {
        "random_forest": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")),
        ]),
        "gradient_boosting": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", GradientBoostingClassifier(n_estimators=150, random_state=42)),
        ]),
        "logistic_regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)),
        ]),
    }

    best_name, best_pipeline, best_cv = None, None, -1
    cv_scores = {}

    for name, pipeline in candidates.items():
        scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring="accuracy")
        cv_scores[name] = round(float(scores.mean()), 4)
        if verbose:
            print(f"  {name:<25} CV accuracy: {scores.mean():.3f} ± {scores.std():.3f}")
        if scores.mean() > best_cv:
            best_cv = scores.mean()
            best_name = name
            best_pipeline = pipeline

    if verbose:
        print(f"\nBest model: {best_name} (CV={best_cv:.3f})")

    # Final fit on full training set
    best_pipeline.fit(X_train, y_train)
    y_pred = best_pipeline.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=["tier_1", "tier_2", "tier_3"], output_dict=True)
    cm = confusion_matrix(y_test, y_pred).tolist()

    # Feature importance (only available for tree-based models)
    feature_importance = {}
    clf = best_pipeline.named_steps["clf"]
    if hasattr(clf, "feature_importances_"):
        importance = clf.feature_importances_
        names = PromptFeatures.feature_names()
        feature_importance = dict(sorted(
            zip(names, importance.tolist()),
            key=lambda x: x[1], reverse=True
        ))

    if verbose:
        print(f"\nTest accuracy: {acc:.3f}")
        print(f"\nClassification report:")
        print(classification_report(y_test, y_pred, target_names=["tier_1", "tier_2", "tier_3"]))
        print(f"Confusion matrix:\n{np.array(cm)}")
        if feature_importance:
            print(f"\nTop 5 features:")
            for feat, imp in list(feature_importance.items())[:5]:
                print(f"  {feat:<30} {imp:.4f}")

    # Save model
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_pipeline, MODEL_PATH)

    result = {
        "best_model": best_name,
        "cv_scores": cv_scores,
        "test_accuracy": round(acc, 4),
        "classification_report": report,
        "confusion_matrix": cm,
        "feature_importance": feature_importance,
        "n_samples": len(texts),
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(result, indent=2))

    if verbose:
        print(f"\nModel saved → {MODEL_PATH}")
        print(f"Report saved → {REPORT_PATH}")

    return result


if __name__ == "__main__":
    train()
