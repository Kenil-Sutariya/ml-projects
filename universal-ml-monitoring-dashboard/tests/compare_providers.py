"""
Side-by-side comparison: same monitoring data → different LLM providers.
Generates a real insight for each available provider and prints a compact summary.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.utils         import load_env, PATHS
from src.llm_providers import detected_providers, get_preset, PRESETS
from src.llm_router    import generate_explanation, PROVIDER_OLLAMA, PROVIDER_CLOUD
from src.ollama_client import check_ollama_connection, select_best_available_model
from src.ai_insights   import build_monitoring_prompt

load_env()


def main() -> None:
    summary_csv = PATHS["summaries"] / "monitoring_summary.csv"
    drift_csv   = PATHS["summaries"] / "feature_drift_details.csv"
    if not summary_csv.exists() or not drift_csv.exists():
        print(f"⚠️  Run a monitoring pipeline first (workspace empty).")
        return

    summary = pd.read_csv(summary_csv)
    drift   = pd.read_csv(drift_csv)
    print(f"Using monitoring data: {len(summary)} batches, {len(drift)} drift rows.\n")

    # Build prompt once
    prompt = build_monitoring_prompt(summary, drift, "ALL")

    runs: list[tuple[str, str, str, float]] = []  # (label, model, response, seconds)

    # 1) Local Ollama
    if check_ollama_connection():
        for model in ["llama3.2:latest", "phi3:latest", "gemma2:2b"]:
            print(f"→ Ollama [{model}] …")
            t0 = time.time()
            r = generate_explanation(PROVIDER_OLLAMA, prompt, {"ollama_model": model})
            runs.append((f"Ollama · {model}", model, r, time.time() - t0))

    # 2) Cloud presets (only those with detected keys)
    for name in detected_providers():
        preset = PRESETS[name]
        # For Gemini, prefer the working free-tier model
        model = "gemini-2.5-flash" if "Gemini" in name else preset.model_name
        cfg = {"api_key": preset.api_key, "base_url": preset.base_url, "model_name": model}
        print(f"→ Cloud [{name}] [{model}] …")
        t0 = time.time()
        r = generate_explanation(PROVIDER_CLOUD, prompt, cfg)
        runs.append((f"Cloud · {name}", model, r, time.time() - t0))

    # ----- Comparison table -----
    print("\n" + "=" * 80)
    print(f"{'PROVIDER · MODEL':<48s} {'ELAPSED':>10s}  {'CHARS':>7s}  STATUS")
    print("=" * 80)
    for label, model, resp, dt in runs:
        ok = not resp.startswith("[ERROR]") and not resp.startswith("**Ollama")
        status = "✓ ok" if ok else "✗ FAIL"
        print(f"{label:<48s} {dt:>9.1f}s  {len(resp):>7,d}  {status}")
    print("=" * 80)

    # ----- Save full responses to /tmp for inspection -----
    out_dir = Path("/tmp/llm_comparison")
    out_dir.mkdir(exist_ok=True)
    for label, model, resp, dt in runs:
        slug = label.replace("·","_").replace(" ","_").replace(":","_").replace("/","_")
        (out_dir / f"{slug}.md").write_text(
            f"# {label}\n\n*Model: {model} · Elapsed: {dt:.1f}s*\n\n{resp}\n",
            encoding="utf-8",
        )
    print(f"\nFull responses saved to: {out_dir}")
    print("Files:")
    for fp in sorted(out_dir.iterdir()):
        print(f"  {fp.name}  ({fp.stat().st_size:,} bytes)")

    # Print first 600 chars of each successful response so user can see quality
    print("\n" + "=" * 80)
    print("RESPONSE PREVIEWS (first 500 chars)")
    print("=" * 80)
    for label, model, resp, dt in runs:
        if resp.startswith("[ERROR]") or resp.startswith("**Ollama"):
            continue
        # Strip the leading "# Header\n\n*Provider…*\n\n" if present
        body = resp.strip()
        print(f"\n── {label} ({dt:.1f}s) ──")
        print(body[:500] + ("…" if len(body) > 500 else ""))


if __name__ == "__main__":
    main()
