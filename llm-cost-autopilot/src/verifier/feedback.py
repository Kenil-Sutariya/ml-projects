"""
Feedback loop: appends routing failures to failures.csv so the
classifier can be retrained with real-world mistakes.
"""

import csv
import threading
from pathlib import Path

FAILURES_PATH = Path("data/labeled_prompts/failures.csv")
_LOCK = threading.Lock()


def _ensure_header():
    if not FAILURES_PATH.exists() or FAILURES_PATH.stat().st_size == 0:
        FAILURES_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(FAILURES_PATH, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["text", "tier"])
            writer.writeheader()


def record_failure(prompt: str, correct_tier: int) -> None:
    """Append a misrouted prompt with its corrected tier label."""
    with _LOCK:
        _ensure_header()
        with open(FAILURES_PATH, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["text", "tier"])
            writer.writerow({"text": prompt, "tier": correct_tier})


def retrain_from_failures() -> dict:
    """
    Merge failures.csv into the main dataset and retrain the classifier.
    Called weekly (or manually) to close the flywheel loop.
    """
    from src.classifier.train import train

    if not FAILURES_PATH.exists():
        return {"status": "no failures to retrain from"}

    result = train(verbose=True)
    result["status"] = "retrained"
    return result
