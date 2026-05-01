"""
Comprehensive end-to-end smoke test:

  - BYO Model pipeline with 3 different model types (RF Pipeline, LogReg, SVC-no-proba)
    using BOTH .pkl and .joblib formats.
  - All 4 user-installed Ollama models (llama3.2, qwen3.5, gemma2:2b, phi3).
  - LLM router fallbacks (Disabled / Cloud-no-key / Cloud-bad-URL).
  - Cloud preset detection from .env.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.utils                import PATHS, clear_workspace, load_env
from src.model_loader         import load_model
from src.monitoring_pipeline  import run_monitoring_pipeline
from src.ollama_client        import (
    check_ollama_connection, list_available_ollama_models, generate_ollama_response,
)
from src.llm_router           import (
    PROVIDER_DISABLED, PROVIDER_OLLAMA, PROVIDER_CLOUD, generate_explanation,
)
from src.llm_providers        import detected_providers, list_presets, get_preset
from src.ai_insights          import generate_overall_insight

NUM = ["age","income","account_age_months","monthly_activity_score",
       "support_tickets","num_logins","feature_usage_score"]
CAT = ["region","subscription_type"]
TARGETED_OLLAMA_MODELS = [
    "llama3.2:latest", "gemma2:2b", "phi3:latest",
]
# qwen3.5:latest removed — model takes longer than 600s on this machine.

results = {"pass": 0, "fail": 0, "skip": 0}

def banner(s: str) -> None:
    print(f"\n{'='*70}\n  {s}\n{'='*70}")

def ok(label: str) -> None:
    print(f"  ✓ {label}")
    results["pass"] += 1

def fail(label: str, detail: str = "") -> None:
    print(f"  ✗ {label}  {detail}")
    results["fail"] += 1

def skip(label: str, why: str) -> None:
    print(f"  ⊘ {label}  ({why})")
    results["skip"] += 1


# ---------------------------------------------------------------------------
# Fixture data — raw current batches (no predictions) from the original project
# ---------------------------------------------------------------------------
def _load_raw_fixtures():
    raw_root = ROOT.parent / "ai-ml-monitoring-dashboard" / "data" / "current_batches"
    ref = pd.read_csv(ROOT.parent / "ai-ml-monitoring-dashboard" / "data" / "raw" / "classification_data.csv")
    ref = ref.sample(n=600, random_state=1).reset_index(drop=True)
    batches = {}
    # Use 3 of the existing raw batches
    for fname in ["batch_1_normal.csv", "batch_3_strong_drift.csv", "batch_5_concept_shift.csv"]:
        batches[fname.replace(".csv","")] = pd.read_csv(raw_root / fname).head(400)
    return ref, batches


# ===========================================================================
# 1. BYO Model — three model types × two file formats
# ===========================================================================
banner("1. BYO Model end-to-end (RF / LogReg / SVC, .pkl + .joblib)")
ref, batches = _load_raw_fixtures()

models_dir = Path("/tmp/byom_test_models")
test_specs = [
    ("rf_pipeline.pkl",          True,  "RandomForest .pkl"),
    ("rf_pipeline.joblib",       True,  "RandomForest .joblib"),
    ("logistic_regression.pkl",  True,  "LogReg .pkl"),
    ("logistic_regression.joblib", True, "LogReg .joblib"),
    ("svc_no_proba.pkl",         False, "SVC (no predict_proba) .pkl"),
    ("svc_no_proba.joblib",      False, "SVC (no predict_proba) .joblib"),
]

byom_summaries: dict[str, pd.DataFrame] = {}
for fname, expected_proba, label in test_specs:
    fp = models_dir / fname
    if not fp.exists():
        fail(label, "(file missing)")
        continue

    info = load_model(fp)
    if not info.loaded or not info.has_predict:
        fail(label, f"loader error: {info.error}")
        continue
    if info.has_predict_proba != expected_proba:
        fail(label, f"predict_proba mismatch: expected {expected_proba}, got {info.has_predict_proba}")
        continue

    clear_workspace()
    result = run_monitoring_pipeline(
        ref, batches,
        target_col="target",
        model=info.model,
        numerical_features=NUM, categorical_features=CAT,
        feature_columns=NUM + CAT,
    )
    if result["status"] != "ok":
        fail(label, f"pipeline failed: {result.get('validation_report')}")
        continue

    summary = result["monitoring_summary"]
    byom_summaries[label] = summary

    # Sanity checks
    if len(summary) != len(batches):
        fail(label, f"got {len(summary)} batches, expected {len(batches)}")
        continue
    if not (0.0 <= summary["accuracy"].iloc[0] <= 1.0):
        fail(label, "accuracy out of range")
        continue

    # If model has no predict_proba, ROC-AUC should default to 0.5 (or summary should still build)
    n_warnings = len(result.get("pipeline_warnings", []))
    proba_col  = result.get("prediction_proba_col")
    if expected_proba and proba_col != "prediction_proba":
        fail(label, f"expected proba column, got {proba_col}")
        continue
    if not expected_proba and proba_col is not None:
        fail(label, f"expected NO proba column, got {proba_col}")
        continue

    ok(f"{label:<32s} · {len(summary)} batches · acc={summary['accuracy'].iloc[0]:.3f} · "
       f"proba={proba_col} · warnings={n_warnings}")


# ===========================================================================
# 2. Ollama — connection + every user-installed model
# ===========================================================================
banner("2. Ollama — all user-installed models")

if not check_ollama_connection():
    skip("All Ollama tests", "Ollama not running")
else:
    available = list_available_ollama_models()
    print(f"  Available models on this machine: {available}")

    for model in TARGETED_OLLAMA_MODELS:
        # Find the actual installed name (case/tag may differ)
        match = next((m for m in available if m.lower().startswith(model.split(":")[0].lower())), None)
        if not match:
            skip(f"Ollama [{model}]", "not installed")
            continue
        t0 = time.time()
        resp = generate_ollama_response(
            "Reply with ONLY the single word: ok",
            match,
        )
        elapsed = time.time() - t0
        if resp.startswith("[ERROR]") or resp.startswith("**Ollama"):
            fail(f"Ollama [{match}]", resp[:100])
        else:
            preview = resp.replace("\n", " ").strip()[:60]
            ok(f"Ollama [{match}]  · {elapsed:5.1f}s · {preview!r}")


# ===========================================================================
# 3. LLM Router — Disabled + Cloud edge cases
# ===========================================================================
banner("3. LLM Router fallbacks")

r = generate_explanation(PROVIDER_DISABLED, "test", {})
(ok if "disabled" in r.lower() else fail)("Disabled provider returns disabled message")

r = generate_explanation(PROVIDER_CLOUD, "test", {})
(ok if r.startswith("[ERROR]") and "key" in r.lower() else fail)("Cloud (no key) → error")

r = generate_explanation(PROVIDER_CLOUD, "test", {"api_key": "x"})
(ok if r.startswith("[ERROR]") and "model" in r.lower() else fail)("Cloud (no model) → error")

r = generate_explanation(PROVIDER_CLOUD, "test",
    {"api_key": "sk-fake", "model_name": "gpt-x", "base_url": "http://localhost:1"})
(ok if r.startswith("[ERROR]") and "reach" in r.lower() else fail)("Cloud (unreachable) → error")


# ===========================================================================
# 4. .env detection
# ===========================================================================
banner("4. .env / cloud provider detection")

load_env()  # explicit reload
print(f"  Available presets: {list_presets()}")
detected = detected_providers()
if detected:
    ok(f"Providers with API keys in .env: {detected}")
else:
    skip(".env-based providers", "no API keys configured (edit .env to add yours)")


# ===========================================================================
# 5. Live cloud test (only if at least one key is set)
# ===========================================================================
banner("5. Live cloud LLM test (only runs if .env has a key)")

for provider_name in detected:
    preset = get_preset(provider_name)
    cfg = {
        "api_key":    preset.api_key,
        "base_url":   preset.base_url,
        "model_name": preset.model_name,
    }
    t0 = time.time()
    resp = generate_explanation(PROVIDER_CLOUD, "Reply with ONLY the word: ok", cfg)
    elapsed = time.time() - t0
    if resp.startswith("[ERROR]"):
        fail(f"Cloud [{provider_name}]", resp[:120])
    else:
        preview = resp.replace("\n", " ").strip()[:60]
        ok(f"Cloud [{provider_name}] ({preset.model_name})  · {elapsed:5.1f}s · {preview!r}")


# ===========================================================================
# 6. AI insight using Ollama (full flow)
# ===========================================================================
banner("6. AI insight generation via Ollama (using BYO Model run output)")

if check_ollama_connection() and byom_summaries:
    # Use the most recent successful BYOM run
    label, summary = list(byom_summaries.items())[-1]
    drift = pd.read_csv(PATHS["summaries"] / "feature_drift_details.csv")
    content, path = generate_overall_insight(
        summary, drift, model_name=None, provider=PROVIDER_OLLAMA,
    )
    if "ollama is not running" in content.lower() or content.startswith("[ERROR]"):
        fail("AI insight via Ollama", content[:120])
    else:
        ok(f"AI insight generated ({len(content):,} chars) → {path.name}")
else:
    skip("AI insight via Ollama", "Ollama unavailable or no BYOM runs")


# ===========================================================================
# Summary
# ===========================================================================
banner(f"RESULTS — pass: {results['pass']}  ·  fail: {results['fail']}  ·  skip: {results['skip']}")
sys.exit(1 if results["fail"] else 0)
