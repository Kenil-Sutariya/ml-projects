"""
Universal ML Model Monitoring Platform — Streamlit front-end.

    streamlit run app.py
"""

from __future__ import annotations

import sys
from io import StringIO
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.utils                import (
    PATHS, ensure_dirs, load_config, safe_read_csv,
    list_workspace_summary_paths, clear_workspace, derive_batch_name,
)
from src.schema_validator     import (
    infer_column_types, validate_reference_and_current,
)
from src.data_profiler        import basic_profile
from src.monitoring_pipeline  import run_monitoring_pipeline
from src.ollama_client        import (
    check_ollama_connection, list_available_ollama_models,
    select_best_available_model,
)
from src.ai_insights          import (
    generate_overall_insight, generate_batch_insight, generate_all_batch_insights,
)
from src.model_loader         import load_model
from src.llm_router           import (
    PROVIDER_DISABLED, PROVIDER_OLLAMA, PROVIDER_CLOUD,
)
from src.llm_providers        import (
    PRESETS as CLOUD_PRESETS, list_presets, detected_providers, get_preset,
)

# ===========================================================================
# Setup
# ===========================================================================
ensure_dirs()
CFG = load_config()

st.set_page_config(
    page_title=CFG.get("app", {}).get("name", "ML Monitoring"),
    page_icon=CFG.get("app", {}).get("page_icon", "📊"),
    layout=CFG.get("app", {}).get("layout", "wide"),
    initial_sidebar_state="expanded",
)

# ===========================================================================
# Custom CSS
# ===========================================================================
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {background: #0f1117;}
[data-testid="stSidebar"] {background: #1a1d27;}
[data-testid="stSidebar"] * {color: #cbd5e0;}
h1,h2,h3,h4 {color: #e2e8f0;}
p, li, label {color: #a0aec0;}

.metric-card {
    background: #1e2235; border-radius: 12px; padding: 18px 22px;
    border-left: 4px solid #4a90d9; margin-bottom: 8px;
}
.metric-card.healthy  {border-left-color: #48bb78;}
.metric-card.warning  {border-left-color: #ed8936;}
.metric-card.critical {border-left-color: #fc8181;}
.metric-card .label {font-size: 0.78rem; color: #718096; text-transform: uppercase; letter-spacing: .06em;}
.metric-card .value {font-size: 1.7rem; font-weight: 700; color: #e2e8f0; margin-top: 2px;}
.metric-card .sub   {font-size: 0.78rem; color: #a0aec0; margin-top: 4px;}

.feature-card {
    background: #1e2235; border-radius: 14px; padding: 24px 26px;
    border: 1px solid #2d3748; height: 100%;
}
.feature-card h3 {margin: 0 0 10px 0; color: #e2e8f0; font-size: 1.05rem;}
.feature-card p  {color: #a0aec0; line-height: 1.5; font-size: 0.92rem; margin: 0;}

.badge {display: inline-block; padding: 4px 14px; border-radius: 20px;
        font-size: 0.78rem; font-weight: 700; letter-spacing: .04em;}
.badge-healthy  {background:#1a3a2a; color:#48bb78; border: 1px solid #276749;}
.badge-warning  {background:#3a2e1a; color:#ed8936; border: 1px solid #c05621;}
.badge-critical {background:#3a1a1a; color:#fc8181; border: 1px solid #c53030;}
.badge-info     {background:#1a2a3a; color:#63b3ed; border: 1px solid #2b6cb0;}
.badge-passed   {background:#1a3a2a; color:#48bb78; border: 1px solid #276749;}
.badge-failed   {background:#3a1a1a; color:#fc8181; border: 1px solid #c53030;}

.section-title {
    font-size: 1.05rem; font-weight: 700; color: #e2e8f0;
    border-bottom: 1px solid #2d3748; padding-bottom: 6px; margin: 16px 0 14px 0;
}
.info-panel {
    background: #1e2235; border-radius: 10px; padding: 14px 18px;
    border: 1px solid #2d3748; margin-bottom: 14px; color: #a0aec0;
}
.hero {
    background: linear-gradient(135deg, #1e2235 0%, #2a2f4a 100%);
    border-radius: 16px; padding: 32px 36px; border: 1px solid #2d3748;
    margin-bottom: 24px;
}
.hero h1 {margin: 0; font-size: 2.0rem; color: #f7fafc;}
.hero p  {margin: 8px 0 0 0; color: #a0aec0; font-size: 1.0rem;}
</style>
""", unsafe_allow_html=True)


# ===========================================================================
# Session state defaults
# ===========================================================================
def _init_state() -> None:
    defaults = {
        "mode":                  "Demo Mode",
        "reference_df":          None,
        "current_batches":       {},     # {name: DataFrame}
        "target_col":            None,
        "prediction_col":        None,
        "prediction_proba_col":  None,
        "numerical_features":   [],
        "categorical_features": [],
        "validation_status":     None,
        "validation_report":     None,
        "monitoring_run":        None,
        "ai_insights":           {},     # {batch_or_'overall': content}
        "thresholds_overrides":  {},
        # --- Stage 2 ---
        "uploaded_model":        None,        # in-memory model object
        "model_info":            None,        # ModelInfo dict from model_loader
        "model_trusted":         False,       # security checkbox state
        "llm_provider":          PROVIDER_OLLAMA,
        "cloud_llm_config":      {            # never persisted; session only
            "preset":     "OpenAI",
            "api_key":    "",
            "base_url":   "https://api.openai.com/v1",
            "model_name": "gpt-4o-mini",
        },
        "send_raw_to_llm":       False,       # privacy toggle, default OFF
        "enable_cloud_llm":      True,        # whether cloud LLM is selectable
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


# ===========================================================================
# Plotly defaults
# ===========================================================================
CHART_LAYOUT = dict(
    paper_bgcolor="#0f1117", plot_bgcolor="#0f1117",
    font_color="#a0aec0",
    legend=dict(bgcolor="#1e2235", bordercolor="#2d3748", borderwidth=1),
    margin=dict(l=40, r=20, t=40, b=40),
)
COLOR_SEQ = px.colors.qualitative.Plotly
HEALTH_COLORS = {"Healthy": "#48bb78", "Warning": "#ed8936", "Critical": "#fc8181"}


# ===========================================================================
# UI helpers
# ===========================================================================

def metric_card(label: str, value: str, *, variant: str = "", sub: str = "") -> str:
    cls = f"metric-card {variant.lower()}" if variant else "metric-card"
    sub_html = f'<div class="sub">{sub}</div>' if sub else ""
    return (
        f'<div class="{cls}">'
        f'<div class="label">{label}</div>'
        f'<div class="value">{value}</div>'
        f'{sub_html}</div>'
    )


def badge(text: str, kind: str) -> str:
    return f'<span class="badge badge-{kind}">{text}</span>'


def health_badge(label: str) -> str:
    css = label.lower() if label.lower() in {"healthy", "warning", "critical"} else "info"
    return badge(label, css)


def drift_badge(val: str) -> str:
    return badge("Drift Detected", "warning") if str(val).lower() == "yes" else badge("No Drift", "healthy")


# ===========================================================================
# Sample data loader
# ===========================================================================

def load_sample_data() -> tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    ref_path = PATHS["sample_data"] / "reference_data.csv"
    ref_df   = pd.read_csv(ref_path)
    batches: Dict[str, pd.DataFrame] = {}
    for fp in sorted(PATHS["sample_data"].glob("batch_*.csv")):
        batches[fp.stem] = pd.read_csv(fp)
    return ref_df, batches


# ===========================================================================
# PAGE 1: HOME
# ===========================================================================
def page_home() -> None:
    st.markdown(
        '<div class="hero">'
        '<h1>📊 Universal ML Model Monitoring Platform</h1>'
        '<p>Monitor model performance, data drift, feature changes, and prediction errors using your own CSV batches — fully local, no cloud APIs.</p>'
        '</div>', unsafe_allow_html=True,
    )

    cols = st.columns(4)
    cards = [
        ("🧪 Demo Mode",
         "Built-in sample project with reference data and 5 simulated drifted batches."),
        ("📤 BYO Predictions",
         "Upload reference and current CSV files that already contain target + prediction columns."),
        ("🧠 BYO Model",
         "Upload a trusted scikit-learn .pkl/.joblib pipeline; the platform generates predictions for you."),
        ("🤖 AI Insights",
         "Local Ollama by default. Optional cloud LLM with your own API key. Keys are never saved."),
    ]
    for col, (title, desc) in zip(cols, cards):
        with col:
            st.markdown(
                f'<div class="feature-card"><h3>{title}</h3><p>{desc}</p></div>',
                unsafe_allow_html=True,
            )

    st.markdown("---")
    st.markdown('<div class="section-title">Key Concepts</div>', unsafe_allow_html=True)

    concepts = [
        ("Reference data",     "Old baseline data used for comparison. Typically the held-out test set from when the model was trained."),
        ("Current data",       "New production batch data. Your model has predicted on this data and you want to see how it's behaving."),
        ("Data drift",         "When feature distributions in current data change compared to reference data."),
        ("Model degradation",  "When model quality (accuracy, F1, etc.) drops on new data — sometimes caused by drift, sometimes by concept shift."),
    ]
    for c1, c2 in zip(concepts[::2], concepts[1::2]):
        col_a, col_b = st.columns(2)
        for col, (title, desc) in zip([col_a, col_b], [c1, c2]):
            with col:
                st.markdown(
                    f'<div class="info-panel"><b>{title}</b><br>{desc}</div>',
                    unsafe_allow_html=True,
                )

    st.markdown("---")
    st.markdown(
        '<div class="info-panel">👉 Head to <b>Upload &amp; Configure</b> to either load the demo project '
        'or upload your own CSV files. Then run monitoring and explore the dashboard.</div>',
        unsafe_allow_html=True,
    )


# ===========================================================================
# PAGE 2: UPLOAD & CONFIGURE
# ===========================================================================
def page_upload_configure() -> None:
    st.markdown("## 📤 Upload & Configure")

    # ----- Step 1: choose mode -----
    st.markdown('<div class="section-title">Step 1 — Choose Mode</div>', unsafe_allow_html=True)
    mode_options = ["Demo Mode", "Bring Your Own Predictions", "Bring Your Own Model"]
    # Bind directly to session_state.mode via key= so the page updates on a single click.
    if st.session_state.mode not in mode_options:
        st.session_state.mode = mode_options[0]
    mode = st.radio(
        "Select monitoring mode:",
        mode_options,
        horizontal=True,
        key="mode",
    )

    # ----- Step 2: load data -----
    st.markdown('<div class="section-title">Step 2 — Load Data</div>', unsafe_allow_html=True)

    if mode == "Demo Mode":
        st.markdown(
            '<div class="info-panel">The demo project includes a reference dataset and 5 simulated production batches '
            '(normal, slight drift, strong drift, categorical shift, concept shift).</div>',
            unsafe_allow_html=True,
        )
        if st.button("📦 Load Demo Project", type="primary"):
            ref_df, batches = load_sample_data()
            st.session_state.reference_df    = ref_df
            st.session_state.current_batches = batches
            # auto-fill demo column choices
            st.session_state.target_col           = "target"
            st.session_state.prediction_col       = "prediction"
            st.session_state.prediction_proba_col = "prediction_proba"
            st.session_state.numerical_features = [
                "age", "income", "account_age_months",
                "monthly_activity_score", "support_tickets",
                "num_logins", "feature_usage_score",
            ]
            st.session_state.categorical_features = ["region", "subscription_type"]
            st.success(f"Loaded demo project: 1 reference + {len(batches)} batches.")

    elif mode == "Bring Your Own Predictions":
        st.markdown('**Reference CSV (baseline data with predictions)**')
        ref_file = st.file_uploader(
            "Upload reference CSV", type=["csv"], key="ref_upload",
            label_visibility="collapsed",
        )
        if ref_file is not None:
            try:
                st.session_state.reference_df = pd.read_csv(ref_file)
                st.success(f"Reference loaded: {st.session_state.reference_df.shape}")
            except Exception as exc:
                st.error(f"Could not read reference CSV: {exc}")

        st.markdown('**Current Batch CSV files (one or more)**')
        cur_files = st.file_uploader(
            "Upload current batch CSVs", type=["csv"], key="cur_upload",
            accept_multiple_files=True, label_visibility="collapsed",
        )
        if cur_files:
            batches: Dict[str, pd.DataFrame] = {}
            for f in cur_files:
                try:
                    batches[derive_batch_name(f.name)] = pd.read_csv(f)
                except Exception as exc:
                    st.error(f"Could not read {f.name}: {exc}")
            st.session_state.current_batches = batches
            st.success(f"Loaded {len(batches)} current batch(es).")

    else:  # Bring Your Own Model
        st.markdown(
            '<div class="info-panel">⚠️ <b>Pickle / joblib files can execute arbitrary code on load.</b> '
            'Only upload model files you have created yourself or fully trust. '
            'Never upload model files from unknown sources.</div>',
            unsafe_allow_html=True,
        )
        st.session_state.model_trusted = st.checkbox(
            "I understand and trust this model file.",
            value=st.session_state.model_trusted,
        )

        st.markdown('**Trusted model file (.pkl or .joblib)**')
        model_file = st.file_uploader(
            "Upload model file", type=["pkl", "joblib"], key="model_upload",
            label_visibility="collapsed",
        )
        if model_file is not None:
            if not st.session_state.model_trusted:
                st.warning("Please tick the trust checkbox above before loading the model.")
            else:
                # Persist upload to a temp file so model_loader can read it
                tmp_path = PATHS["workspace"] / "uploaded_model" / model_file.name
                tmp_path.parent.mkdir(parents=True, exist_ok=True)
                tmp_path.write_bytes(model_file.getvalue())

                with st.spinner("Loading model…"):
                    info = load_model(tmp_path)

                if info.loaded:
                    st.session_state.uploaded_model = info.model
                    st.session_state.model_info     = info.to_dict()
                    st.success(
                        f"Model loaded — type: `{info.model_type}` · "
                        f"predict: `{info.has_predict}` · predict_proba: `{info.has_predict_proba}`"
                    )
                    if not info.has_predict_proba:
                        st.warning(
                            "Model has no `predict_proba` method. ROC-AUC and probability charts may be unavailable."
                        )
                else:
                    st.session_state.uploaded_model = None
                    st.session_state.model_info     = info.to_dict()
                    st.error(f"Could not load model: {info.error}")

        st.markdown('**Reference CSV (raw features only — predictions will be generated)**')
        ref_file = st.file_uploader(
            "Upload reference CSV", type=["csv"], key="ref_upload_model",
            label_visibility="collapsed",
        )
        if ref_file is not None:
            try:
                st.session_state.reference_df = pd.read_csv(ref_file)
                st.session_state.prediction_col       = None
                st.session_state.prediction_proba_col = None
                st.success(f"Reference loaded: {st.session_state.reference_df.shape}")
            except Exception as exc:
                st.error(f"Could not read reference CSV: {exc}")

        st.markdown('**Current Batch CSV files (raw features only)**')
        cur_files = st.file_uploader(
            "Upload current batch CSVs", type=["csv"], key="cur_upload_model",
            accept_multiple_files=True, label_visibility="collapsed",
        )
        if cur_files:
            batches: Dict[str, pd.DataFrame] = {}
            for f in cur_files:
                try:
                    batches[derive_batch_name(f.name)] = pd.read_csv(f)
                except Exception as exc:
                    st.error(f"Could not read {f.name}: {exc}")
            st.session_state.current_batches = batches
            st.success(f"Loaded {len(batches)} current batch(es).")

    # Quick file summaries
    if st.session_state.reference_df is not None or st.session_state.current_batches:
        st.markdown('<div class="section-title">Loaded Data Summary</div>', unsafe_allow_html=True)
        col_a, col_b = st.columns(2)
        with col_a:
            if st.session_state.reference_df is not None:
                p = basic_profile(st.session_state.reference_df)
                st.markdown(metric_card(
                    "Reference Rows", f"{p['rows']:,}",
                    sub=f"{p['cols']} cols · {p['missing']} missing",
                ), unsafe_allow_html=True)
        with col_b:
            if st.session_state.current_batches:
                total_rows = sum(b.shape[0] for b in st.session_state.current_batches.values())
                st.markdown(metric_card(
                    "Current Batches",
                    str(len(st.session_state.current_batches)),
                    sub=f"{total_rows:,} total rows",
                ), unsafe_allow_html=True)

    # ----- Step 3: configure columns -----
    if st.session_state.reference_df is not None:
        st.markdown('<div class="section-title">Step 3 — Configure Columns</div>', unsafe_allow_html=True)
        cols_avail = list(st.session_state.reference_df.columns)
        inferred   = infer_column_types(st.session_state.reference_df)

        in_model_mode = (mode == "Bring Your Own Model")

        if in_model_mode:
            tgt_idx = cols_avail.index(st.session_state.target_col) if st.session_state.target_col in cols_avail else 0
            st.session_state.target_col = st.selectbox("Target column", cols_avail, index=tgt_idx)
            st.info("Predictions will be generated by your uploaded model — no prediction column needed.")
            st.session_state.prediction_col       = None
            st.session_state.prediction_proba_col = None
        else:
            c1, c2, c3 = st.columns(3)
            with c1:
                tgt_idx = cols_avail.index(st.session_state.target_col) if st.session_state.target_col in cols_avail else 0
                st.session_state.target_col = st.selectbox("Target column", cols_avail, index=tgt_idx)
            with c2:
                pred_idx = cols_avail.index(st.session_state.prediction_col) if st.session_state.prediction_col in cols_avail else 0
                st.session_state.prediction_col = st.selectbox("Prediction column", cols_avail, index=pred_idx)
            with c3:
                proba_options = ["(none)"] + cols_avail
                cur_proba = st.session_state.prediction_proba_col or "(none)"
                cur_idx = proba_options.index(cur_proba) if cur_proba in proba_options else 0
                chosen = st.selectbox("Prediction probability column (optional)", proba_options, index=cur_idx)
                st.session_state.prediction_proba_col = None if chosen == "(none)" else chosen

        # Suggested feature lists (excluding the special columns)
        special = {st.session_state.target_col, st.session_state.prediction_col, st.session_state.prediction_proba_col or ""}
        suggested_num = [c for c in inferred.get("numerical", [])   if c not in special]
        suggested_cat = [c for c in inferred.get("categorical", []) if c not in special]

        # If user has not set features yet, default to suggestions
        if not st.session_state.numerical_features:
            st.session_state.numerical_features = suggested_num
        if not st.session_state.categorical_features:
            st.session_state.categorical_features = suggested_cat

        c4, c5 = st.columns(2)
        with c4:
            st.session_state.numerical_features = st.multiselect(
                "Numerical features",
                options=[c for c in cols_avail if c not in special],
                default=[c for c in st.session_state.numerical_features if c in cols_avail],
            )
        with c5:
            st.session_state.categorical_features = st.multiselect(
                "Categorical features",
                options=[c for c in cols_avail if c not in special],
                default=[c for c in st.session_state.categorical_features if c in cols_avail],
            )

        # ----- Step 4: validate -----
        st.markdown('<div class="section-title">Step 4 — Validate Setup</div>', unsafe_allow_html=True)
        if st.button("✅ Validate Setup"):
            if in_model_mode:
                # In model mode, target must exist in reference; prediction cols are skipped.
                # We also need the model loaded.
                errors = []
                warnings_ = []
                if st.session_state.uploaded_model is None:
                    errors.append("No model loaded. Upload a trusted .pkl/.joblib file first.")
                else:
                    info = st.session_state.model_info or {}
                    if not info.get("has_predict", False):
                        errors.append("Uploaded model does not have a `predict` method.")
                    if not info.get("has_predict_proba", False):
                        warnings_.append("Model has no `predict_proba` — ROC-AUC and probability charts may be unavailable.")
                if st.session_state.reference_df is None:
                    errors.append("Reference CSV missing.")
                if not st.session_state.current_batches:
                    errors.append("No current batch CSVs uploaded.")
                if not st.session_state.target_col:
                    errors.append("Target column not selected.")
                if not (st.session_state.numerical_features + st.session_state.categorical_features):
                    errors.append("No feature columns selected.")

                # Verify feature columns exist in reference + every batch
                feats = st.session_state.numerical_features + st.session_state.categorical_features
                if st.session_state.reference_df is not None:
                    missing = [c for c in feats if c not in st.session_state.reference_df.columns]
                    if missing:
                        errors.append(f"Reference is missing feature columns: {missing}")
                for bn, bdf in (st.session_state.current_batches or {}).items():
                    missing = [c for c in feats if c not in bdf.columns]
                    if missing:
                        errors.append(f"Batch '{bn}' is missing feature columns: {missing}")

                status = "failed" if errors else ("warning" if warnings_ else "passed")
                st.session_state.validation_status = status
                st.session_state.validation_report = {
                    "status":               status,
                    "errors":               errors,
                    "warnings":             warnings_,
                    "detected_numerical":   [],
                    "detected_categorical": [],
                    "suggested_features":   [],
                }
            else:
                report = validate_reference_and_current(
                    st.session_state.reference_df,
                    st.session_state.current_batches,
                    target_col=st.session_state.target_col,
                    prediction_col=st.session_state.prediction_col,
                    prediction_proba_col=st.session_state.prediction_proba_col,
                    feature_columns=st.session_state.numerical_features + st.session_state.categorical_features,
                )
                st.session_state.validation_status = report.status
                st.session_state.validation_report = report.to_dict()

        # Show last validation result
        if st.session_state.validation_status:
            status = st.session_state.validation_status
            kind   = {"passed": "passed", "warning": "warning", "failed": "failed"}.get(status, "info")
            st.markdown(badge(f"Validation: {status.upper()}", kind), unsafe_allow_html=True)
            r = st.session_state.validation_report or {}
            if r.get("errors"):
                st.error("Errors:\n\n- " + "\n- ".join(r["errors"]))
            if r.get("warnings"):
                st.warning("Warnings:\n\n- " + "\n- ".join(r["warnings"]))
            if status == "passed" or status == "warning":
                st.info("✅ You can now go to **Run Monitoring** in the sidebar.")


# ===========================================================================
# PAGE 3: RUN MONITORING
# ===========================================================================
def page_run_monitoring() -> None:
    st.markdown("## ▶️ Run Monitoring")

    if st.session_state.reference_df is None or not st.session_state.current_batches:
        st.warning("⚠️ Please load data on the **Upload & Configure** page first.")
        return
    if not st.session_state.target_col:
        st.warning("⚠️ Please select target column first.")
        return

    in_model_mode = (st.session_state.mode == "Bring Your Own Model")
    if not in_model_mode and not st.session_state.prediction_col:
        st.warning("⚠️ Please select a prediction column first.")
        return
    if in_model_mode and st.session_state.uploaded_model is None:
        st.warning("⚠️ No model loaded. Go back to **Upload & Configure** and load a trusted model.")
        return

    # Configuration summary
    st.markdown('<div class="section-title">Configuration Summary</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.markdown(metric_card("Mode", st.session_state.mode), unsafe_allow_html=True)
    with c2: st.markdown(metric_card("Batches", str(len(st.session_state.current_batches))), unsafe_allow_html=True)
    with c3: st.markdown(metric_card("Numerical Feats", str(len(st.session_state.numerical_features))), unsafe_allow_html=True)
    with c4: st.markdown(metric_card("Categorical Feats", str(len(st.session_state.categorical_features))), unsafe_allow_html=True)

    if in_model_mode:
        info = st.session_state.model_info or {}
        st.markdown(
            f'<div class="info-panel">Target: <b>{st.session_state.target_col}</b> · '
            f'Model: <b>{info.get("model_type", "uploaded model")}</b> · '
            f'predict_proba: <b>{"yes" if info.get("has_predict_proba") else "no"}</b></div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="info-panel">Target: <b>{st.session_state.target_col}</b> · '
            f'Prediction: <b>{st.session_state.prediction_col}</b> · '
            f'Probability: <b>{st.session_state.prediction_proba_col or "—"}</b></div>',
            unsafe_allow_html=True,
        )

    if st.button("🚀 Run Monitoring", type="primary"):
        progress_bar = st.progress(0.0)
        status_box   = st.empty()

        def _on_progress(label: str, frac: float) -> None:
            progress_bar.progress(frac)
            status_box.info(f"⏳ {label}…")

        result = run_monitoring_pipeline(
            st.session_state.reference_df,
            st.session_state.current_batches,
            target_col=st.session_state.target_col,
            prediction_col=st.session_state.prediction_col,
            prediction_proba_col=st.session_state.prediction_proba_col,
            model=st.session_state.uploaded_model if in_model_mode else None,
            numerical_features=st.session_state.numerical_features,
            categorical_features=st.session_state.categorical_features,
            feature_columns=st.session_state.numerical_features + st.session_state.categorical_features,
            progress=_on_progress,
            config_overrides=st.session_state.thresholds_overrides,
        )
        st.session_state.monitoring_run = result

        if result["status"] == "failed":
            status_box.error("❌ Monitoring failed.")
            st.error("Errors:\n\n- " + "\n- ".join(result["validation_report"].get("errors", [])))
            return

        progress_bar.progress(1.0)
        status_box.success("✅ Monitoring complete.")

        for w in result.get("pipeline_warnings", []):
            st.warning(w)

        # Generate AI insights if Local Ollama is available (cloud opt-in elsewhere)
        if st.session_state.llm_provider == PROVIDER_OLLAMA and check_ollama_connection():
            try:
                with st.spinner("Generating AI insight…"):
                    model_name = select_best_available_model()
                    if model_name:
                        generate_overall_insight(
                            result["monitoring_summary"],
                            result["feature_drift_details"],
                            model_name,
                            provider=PROVIDER_OLLAMA,
                        )
                        st.info(f"🤖 Overall AI insight generated using `{model_name}`.")
            except Exception as exc:
                st.warning(f"AI insight skipped: {exc}")

        st.success("🎉 All artefacts saved to `workspace/`.")
        st.markdown("**Outputs:**")
        st.code(
            f"- {result['monitoring_summary_path']}\n"
            f"- {result['feature_drift_path']}\n"
            f"- workspace/reports/data_drift/*.html\n"
            f"- workspace/reports/model_performance/*.html\n"
            f"- workspace/processed/*.csv"
        )


# ===========================================================================
# Helper: load monitoring artefacts
# ===========================================================================
def _load_monitoring_artefacts() -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    paths = list_workspace_summary_paths()
    summary = safe_read_csv(paths["monitoring_summary"])
    drift   = safe_read_csv(paths["feature_drift_details"])
    return summary, drift


def _need_run_warning() -> None:
    st.warning("⚠️ No monitoring run found. Go to **Run Monitoring** first.")


# ===========================================================================
# PAGE 4: DASHBOARD
# ===========================================================================
def page_dashboard() -> None:
    st.markdown("## 📊 Dashboard")
    summary, drift = _load_monitoring_artefacts()
    if summary is None:
        _need_run_warning()
        return

    latest = summary.iloc[-1]

    # Executive overview
    st.markdown('<div class="section-title">Executive Overview</div>', unsafe_allow_html=True)
    c = st.columns(4)
    cards1 = [
        ("Number of Batches",  str(len(summary))),
        ("Latest Batch",       str(latest["batch_name"])),
        ("Latest Accuracy",    f"{latest['accuracy']:.1%}"),
        ("Latest F1-Score",    f"{latest['f1_score']:.1%}"),
    ]
    for col, (lbl, val) in zip(c, cards1):
        with col: st.markdown(metric_card(lbl, val), unsafe_allow_html=True)

    c2 = st.columns(4)
    cards2 = [
        ("Latest ROC-AUC",     f"{latest['roc_auc']:.1%}"),
        ("Latest Error Rate",  f"{latest['error_rate']:.1%}"),
        ("Drift Detected",     str(latest["drift_detected"])),
        ("Drifted Features",   str(latest["number_of_drifted_features"])),
    ]
    for col, (lbl, val) in zip(c2, cards2):
        with col: st.markdown(metric_card(lbl, val), unsafe_allow_html=True)

    # Health badge
    st.markdown(
        f"**Latest Model Health:** {health_badge(str(latest['model_health']))}",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    # Combined performance chart
    st.markdown('<div class="section-title">Combined Performance Metrics</div>', unsafe_allow_html=True)
    metrics = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
    fig = go.Figure()
    for i, m in enumerate(metrics):
        fig.add_trace(go.Scatter(
            x=summary["batch_name"], y=summary[m], mode="lines+markers",
            name=m.replace("_", " ").title(),
            line=dict(color=COLOR_SEQ[i], width=2), marker=dict(size=8),
        ))
    fig.update_layout(**CHART_LAYOUT, yaxis=dict(range=[0, 1.05], gridcolor="#2d3748"),
                      xaxis=dict(gridcolor="#2d3748"))
    st.plotly_chart(fig, use_container_width=True)

    # Individual metric charts (4 panels = 2x2)
    pairs = [
        ("accuracy",  "Accuracy"),
        ("precision", "Precision"),
        ("recall",    "Recall"),
        ("f1_score",  "F1-Score"),
        ("roc_auc",   "ROC-AUC"),
        ("error_rate","Error Rate"),
    ]
    for row in range(0, len(pairs), 2):
        cols = st.columns(2)
        for col, (key, title) in zip(cols, pairs[row:row+2]):
            with col:
                color = COLOR_SEQ[pairs.index((key, title)) % len(COLOR_SEQ)]
                f = px.line(summary, x="batch_name", y=key, markers=True,
                            title=f"{title} over Batches",
                            color_discrete_sequence=[color])
                f.update_layout(**CHART_LAYOUT,
                                yaxis=dict(gridcolor="#2d3748"),
                                xaxis=dict(gridcolor="#2d3748"))
                st.plotly_chart(f, use_container_width=True)

    # Positive prediction rate + drifted feature count
    cols = st.columns(2)
    with cols[0]:
        f = px.line(summary, x="batch_name", y="positive_prediction_rate",
                    markers=True, title="Positive Prediction Rate",
                    color_discrete_sequence=["#63b3ed"])
        f.update_layout(**CHART_LAYOUT,
                        yaxis=dict(range=[0, 1], gridcolor="#2d3748"),
                        xaxis=dict(gridcolor="#2d3748"))
        st.plotly_chart(f, use_container_width=True)
    with cols[1]:
        f = px.bar(summary, x="batch_name", y="number_of_drifted_features",
                   title="Drifted Features per Batch",
                   color="drift_detected",
                   color_discrete_map={"Yes": "#ed8936", "No": "#48bb78"})
        f.update_layout(**CHART_LAYOUT,
                        yaxis=dict(gridcolor="#2d3748"),
                        xaxis=dict(gridcolor="#2d3748"))
        st.plotly_chart(f, use_container_width=True)

    st.markdown('<div class="section-title">All Batch Summary</div>', unsafe_allow_html=True)
    st.dataframe(
        summary[["batch_name", "accuracy", "precision", "recall", "f1_score",
                 "roc_auc", "error_rate", "drift_detected", "model_health"]]
            .style.format({c: "{:.1%}" for c in
                           ["accuracy", "precision", "recall", "f1_score", "roc_auc", "error_rate"]}),
        use_container_width=True,
    )


# ===========================================================================
# PAGE 5: FEATURE DRIFT
# ===========================================================================
def page_feature_drift() -> None:
    st.markdown("## 🌊 Feature Drift")
    summary, drift = _load_monitoring_artefacts()
    if drift is None:
        _need_run_warning()
        return

    st.markdown(
        '<div class="info-panel">A feature is marked as <b>drifted</b> when its current distribution '
        'changes significantly compared with reference data.</div>',
        unsafe_allow_html=True,
    )

    # Top drifted features (avg drift score across all batches)
    st.markdown('<div class="section-title">Top Drifted Features</div>', unsafe_allow_html=True)
    top = (drift.groupby("feature_name")["drift_score"].mean()
                 .reset_index().sort_values("drift_score", ascending=False).head(15))
    f1 = px.bar(top, x="drift_score", y="feature_name", orientation="h",
                color="drift_score", color_continuous_scale="Oranges",
                title="Average Drift Score (across all batches)")
    f1.update_layout(**CHART_LAYOUT,
                     yaxis=dict(gridcolor="#2d3748"),
                     xaxis=dict(gridcolor="#2d3748"))
    st.plotly_chart(f1, use_container_width=True)

    # Feature drift score by batch (heatmap)
    st.markdown('<div class="section-title">Drift Score by Batch (Heatmap)</div>', unsafe_allow_html=True)
    pivot = drift.pivot_table(
        index="feature_name", columns="batch_name", values="drift_score", aggfunc="mean",
    )
    if not pivot.empty:
        f2 = px.imshow(pivot, color_continuous_scale="YlOrRd",
                       title="Feature × Batch Drift Score", aspect="auto")
        f2.update_layout(**CHART_LAYOUT)
        st.plotly_chart(f2, use_container_width=True)

    # Detail table
    st.markdown('<div class="section-title">Feature Drift Details</div>', unsafe_allow_html=True)
    show_cols = ["batch_name", "feature_name", "feature_type", "drift_score", "drift_detected"]
    st.dataframe(
        drift[show_cols].style.format({"drift_score": "{:.4f}"}),
        use_container_width=True,
    )


# ===========================================================================
# PAGE 6: ERROR ANALYSIS
# ===========================================================================
def page_error_analysis() -> None:
    st.markdown("## ❌ Error Analysis")
    summary, drift = _load_monitoring_artefacts()
    if summary is None:
        _need_run_warning()
        return

    # Aggregate charts
    col_a, col_b = st.columns(2)
    with col_a:
        fig = go.Figure()
        fig.add_trace(go.Bar(name="Correct", x=summary["batch_name"],
                             y=summary["correct_predictions"], marker_color="#48bb78"))
        fig.add_trace(go.Bar(name="Wrong", x=summary["batch_name"],
                             y=summary["wrong_predictions"], marker_color="#fc8181"))
        fig.update_layout(**CHART_LAYOUT, barmode="stack",
                          title="Correct vs Wrong Predictions",
                          yaxis=dict(gridcolor="#2d3748"),
                          xaxis=dict(gridcolor="#2d3748"))
        st.plotly_chart(fig, use_container_width=True)
    with col_b:
        f = px.line(summary, x="batch_name", y="error_rate", markers=True,
                    title="Error Rate over Batches",
                    color_discrete_sequence=["#fc8181"])
        cfg_m = CFG.get("monitoring", {})
        f.add_hline(y=cfg_m.get("warning_error_rate", 0.25),
                    line_dash="dash", line_color="#718096",
                    annotation_text=f"Warning {cfg_m.get('warning_error_rate', 0.25):.0%}")
        f.add_hline(y=cfg_m.get("critical_error_rate", 0.35),
                    line_dash="dash", line_color="#fc8181",
                    annotation_text=f"Critical {cfg_m.get('critical_error_rate', 0.35):.0%}")
        f.update_layout(**CHART_LAYOUT,
                        yaxis=dict(range=[0, 1], gridcolor="#2d3748"),
                        xaxis=dict(gridcolor="#2d3748"))
        st.plotly_chart(f, use_container_width=True)

    col_c, col_d = st.columns(2)
    with col_c:
        f = px.line(summary, x="batch_name", y="positive_prediction_rate", markers=True,
                    title="Positive Prediction Rate",
                    color_discrete_sequence=["#63b3ed"])
        f.update_layout(**CHART_LAYOUT,
                        yaxis=dict(range=[0, 1], gridcolor="#2d3748"),
                        xaxis=dict(gridcolor="#2d3748"))
        st.plotly_chart(f, use_container_width=True)
    with col_d:
        if "average_prediction_probability" in summary.columns and summary["average_prediction_probability"].notna().any():
            f = px.line(summary, x="batch_name", y="average_prediction_probability",
                        markers=True, title="Average Prediction Probability",
                        color_discrete_sequence=["#b794f4"])
            f.update_layout(**CHART_LAYOUT,
                            yaxis=dict(range=[0, 1], gridcolor="#2d3748"),
                            xaxis=dict(gridcolor="#2d3748"))
            st.plotly_chart(f, use_container_width=True)
        else:
            st.info("No probability column was provided — probability chart skipped.")

    # Per-batch deep-dive
    st.markdown('<div class="section-title">Batch Deep-Dive</div>', unsafe_allow_html=True)
    selected = st.selectbox("Select batch", summary["batch_name"].tolist())
    row = summary[summary["batch_name"] == selected].iloc[0]

    if bool(row.get("is_binary", False)):
        cm = pd.DataFrame(
            [[int(row.get("true_negative", 0)),  int(row.get("false_positive", 0))],
             [int(row.get("false_negative", 0)), int(row.get("true_positive", 0))]],
            index=["actual_0", "actual_1"], columns=["predicted_0", "predicted_1"],
        )
        st.markdown("**Confusion Matrix**")
        st.dataframe(cm, use_container_width=True)

        cols = st.columns(4)
        cards = [
            ("True Positive",  str(int(row.get("true_positive",  0)))),
            ("False Positive", str(int(row.get("false_positive", 0)))),
            ("True Negative",  str(int(row.get("true_negative",  0)))),
            ("False Negative", str(int(row.get("false_negative", 0)))),
        ]
        for col, (lbl, val) in zip(cols, cards):
            with col: st.markdown(metric_card(lbl, val), unsafe_allow_html=True)
    else:
        st.warning("⚠️ Confusion matrix is only displayed for binary classification batches.")

    # First 50 rows where prediction != target
    processed_path = PATHS["processed"] / f"{selected}.csv"
    if processed_path.exists():
        df = pd.read_csv(processed_path)
        if st.session_state.target_col in df.columns and st.session_state.prediction_col in df.columns:
            wrong = df[df[st.session_state.target_col] != df[st.session_state.prediction_col]].head(50)
            st.markdown(f"**First 50 wrong predictions in `{selected}`**")
            if wrong.empty:
                st.info("No wrong predictions in this batch.")
            else:
                st.dataframe(wrong, use_container_width=True)


# ===========================================================================
# PAGE 7: EVIDENTLY REPORTS
# ===========================================================================
def page_evidently_reports() -> None:
    st.markdown("## 📋 Evidently Reports")
    summary, _ = _load_monitoring_artefacts()
    if summary is None:
        _need_run_warning()
        return

    st.markdown(
        '<div class="info-panel">Evidently AI reports provide detailed statistical analysis '
        'of drift and model performance. Open the local paths in your browser, or view inline below.</div>',
        unsafe_allow_html=True,
    )

    cols = ["batch_name", "data_drift_report_path", "performance_report_path"]
    if "data_summary_report_path" in summary.columns:
        cols.append("data_summary_report_path")
    st.dataframe(summary[cols], use_container_width=True)

    st.markdown("---")
    st.markdown('<div class="section-title">View Inline</div>', unsafe_allow_html=True)
    selected = st.selectbox("Select batch", summary["batch_name"].tolist(), key="ev_batch")
    options  = ["Data Drift", "Model Performance"]
    if "data_summary_report_path" in summary.columns:
        options.append("Data Summary")
    report_type = st.radio("Report type", options, horizontal=True)

    row = summary[summary["batch_name"] == selected].iloc[0]
    key_map = {
        "Data Drift":        "data_drift_report_path",
        "Model Performance": "performance_report_path",
        "Data Summary":      "data_summary_report_path",
    }
    html_path = Path(str(row[key_map[report_type]]))
    if html_path.exists():
        st.components.v1.html(html_path.read_text(encoding="utf-8"), height=900, scrolling=True)
    else:
        st.info(f"Report not available at `{html_path}`.")


# ===========================================================================
# PAGE 8: AI INSIGHTS
# ===========================================================================
def page_ai_insights() -> None:
    st.markdown("## 🤖 AI Insights")
    st.markdown(
        '<div class="info-panel">Generate plain-English explanations from monitoring metrics. '
        'By default only summarised metrics are sent — never raw rows.</div>',
        unsafe_allow_html=True,
    )

    summary, drift = _load_monitoring_artefacts()
    if summary is None or drift is None:
        _need_run_warning()
        return

    # ---- Provider selector ----
    st.markdown('<div class="section-title">Provider</div>', unsafe_allow_html=True)
    provider_labels = {
        PROVIDER_DISABLED: "Disabled",
        PROVIDER_OLLAMA:   "Local Ollama (recommended)",
        PROVIDER_CLOUD:    "Cloud LLM with my own API key",
    }
    available_providers = [PROVIDER_DISABLED, PROVIDER_OLLAMA]
    if st.session_state.enable_cloud_llm:
        available_providers.append(PROVIDER_CLOUD)

    cur = st.session_state.llm_provider if st.session_state.llm_provider in available_providers else PROVIDER_OLLAMA
    chosen_provider = st.radio(
        "LLM provider",
        available_providers,
        format_func=lambda p: provider_labels[p],
        index=available_providers.index(cur),
        horizontal=True,
    )
    st.session_state.llm_provider = chosen_provider

    chosen_ollama_model: Optional[str] = None
    cloud_cfg = st.session_state.cloud_llm_config

    # ---- Provider-specific config ----
    if chosen_provider == PROVIDER_DISABLED:
        st.info("AI explanations are disabled.")
    elif chosen_provider == PROVIDER_OLLAMA:
        ollama_ok = check_ollama_connection()
        if ollama_ok:
            st.markdown(badge("Ollama Connected", "healthy"), unsafe_allow_html=True)
            models = list_available_ollama_models()
            if models:
                default_model = select_best_available_model() or models[0]
                chosen_ollama_model = st.selectbox(
                    "Ollama model", models,
                    index=models.index(default_model) if default_model in models else 0,
                )
            else:
                st.warning("No Ollama models installed. Try `ollama pull llama3.2`.")
        else:
            st.markdown(badge("Ollama Not Running", "critical"), unsafe_allow_html=True)
            st.warning("Ollama is not running. Start it with: `ollama serve`")
    else:  # PROVIDER_CLOUD
        st.markdown(
            '<div class="info-panel">⚠️ Cloud LLM sends monitoring summaries to your chosen provider. '
            'Your API key is kept only in this Streamlit session — it is never written to disk or logged.</div>',
            unsafe_allow_html=True,
        )

        # --- Provider preset selector ---
        preset_names = list_presets()
        env_detected = detected_providers()
        if env_detected:
            st.info(f"🔑 Detected API keys in `.env` for: **{', '.join(env_detected)}**")

        cur_preset_name = cloud_cfg.get("preset", preset_names[0])
        if cur_preset_name not in preset_names:
            cur_preset_name = preset_names[0]

        chosen_preset_name = st.selectbox(
            "Provider preset",
            preset_names,
            index=preset_names.index(cur_preset_name),
            help="Choose a preset to auto-fill the base URL and try to load the API key from .env.",
        )
        preset = get_preset(chosen_preset_name)

        # If the user just switched preset, refresh the fields from preset/env defaults
        if chosen_preset_name != cur_preset_name:
            cloud_cfg["preset"]     = chosen_preset_name
            cloud_cfg["base_url"]   = preset.base_url
            cloud_cfg["model_name"] = preset.model_name
            cloud_cfg["api_key"]    = preset.api_key  # from .env if present
            st.session_state.cloud_llm_config = cloud_cfg
            st.rerun()

        cloud_cfg["preset"] = chosen_preset_name

        # Status badge for this preset
        if preset.has_key:
            st.markdown(badge(f"{chosen_preset_name} key loaded from .env", "healthy"), unsafe_allow_html=True)
        else:
            st.caption(f"Get a key: {preset.docs_url}" if preset.docs_url else "Enter your API key below.")

        c1, c2 = st.columns(2)
        with c1:
            cloud_cfg["base_url"] = st.text_input(
                "Base URL (OpenAI-compatible)",
                value=cloud_cfg.get("base_url") or preset.default_url,
                placeholder=preset.default_url,
            )
            cloud_cfg["model_name"] = st.text_input(
                "Model name",
                value=cloud_cfg.get("model_name") or preset.default_model,
                placeholder=preset.default_model or "model-name",
            )
        with c2:
            api_key_input = st.text_input(
                "API key",
                value=cloud_cfg.get("api_key") or preset.api_key,
                type="password",
                placeholder="sk-…  (or load from .env)",
            )
            cloud_cfg["api_key"] = api_key_input

        if st.session_state.send_raw_to_llm:
            st.warning(
                "**Raw-data sending is currently ON** in Settings. Turn it off to keep raw rows local."
            )
        st.session_state.cloud_llm_config = cloud_cfg

    # ---- Overall insight (saved file) ----
    overall_path = PATHS["ai_insights"] / "overall_monitoring_insight.md"
    st.markdown('<div class="section-title">Overall Monitoring Insight</div>', unsafe_allow_html=True)
    if overall_path.exists():
        st.markdown(overall_path.read_text(encoding="utf-8"))
    else:
        st.info("No overall insight generated yet.")

    # ---- Generate ----
    st.markdown("---")
    st.markdown('<div class="section-title">Generate / Regenerate Insight</div>', unsafe_allow_html=True)

    can_generate = (
        chosen_provider == PROVIDER_OLLAMA and chosen_ollama_model is not None
    ) or (
        chosen_provider == PROVIDER_CLOUD
        and cloud_cfg.get("api_key") and cloud_cfg.get("model_name")
    )

    scope = st.selectbox("Scope", ["Overall"] + summary["batch_name"].tolist())

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("✨ Generate", type="primary", disabled=not can_generate):
            cfg = {}
            if chosen_provider == PROVIDER_OLLAMA:
                cfg = {"ollama_model": chosen_ollama_model}
            elif chosen_provider == PROVIDER_CLOUD:
                cfg = {
                    "api_key":    cloud_cfg["api_key"],
                    "base_url":   cloud_cfg["base_url"],
                    "model_name": cloud_cfg["model_name"],
                }
            with st.spinner(f"Generating via {provider_labels[chosen_provider]}…"):
                if scope == "Overall":
                    content, _ = generate_overall_insight(
                        summary, drift, chosen_ollama_model,
                        provider=chosen_provider, config=cfg,
                    )
                else:
                    content, _ = generate_batch_insight(
                        summary, drift, scope, chosen_ollama_model,
                        provider=chosen_provider, config=cfg,
                    )
            st.success("Generated.")
            st.markdown(content)
    with col_b:
        if st.button("🚀 Generate ALL (overall + every batch)", disabled=not can_generate):
            cfg = {}
            if chosen_provider == PROVIDER_OLLAMA:
                cfg = {"ollama_model": chosen_ollama_model}
            elif chosen_provider == PROVIDER_CLOUD:
                cfg = {
                    "api_key":    cloud_cfg["api_key"],
                    "base_url":   cloud_cfg["base_url"],
                    "model_name": cloud_cfg["model_name"],
                }
            with st.spinner("Generating all insights…"):
                paths = generate_all_batch_insights(
                    summary, drift, chosen_ollama_model,
                    provider=chosen_provider, config=cfg,
                )
            st.success(f"Generated {len(paths)} insight files.")

    if not can_generate and chosen_provider != PROVIDER_DISABLED:
        st.caption("Generate is disabled until a model is selected (Ollama) or API key + model name are entered (Cloud).")

    # ---- Per-batch viewer ----
    st.markdown("---")
    st.markdown('<div class="section-title">Batch Insight Viewer</div>', unsafe_allow_html=True)
    sel = st.selectbox("Select batch insight", summary["batch_name"].tolist(), key="ai_view_batch")
    p   = PATHS["ai_insights"] / f"{sel}_insight.md"
    if p.exists():
        st.markdown(p.read_text(encoding="utf-8"))
    else:
        st.info("No insight file for this batch yet.")


# ===========================================================================
# PAGE 9: SETTINGS
# ===========================================================================
def page_settings() -> None:
    st.markdown("## ⚙️ Settings")

    st.markdown('<div class="section-title">Monitoring Thresholds</div>', unsafe_allow_html=True)
    cfg_m = CFG.get("monitoring", {})
    overrides = st.session_state.thresholds_overrides

    c1, c2 = st.columns(2)
    with c1:
        num_thr = st.slider(
            "Numeric drift threshold",
            min_value=0.1, max_value=2.0,
            value=float(overrides.get("drift_numeric_threshold", cfg_m.get("drift_numeric_threshold", 0.5))),
            step=0.05,
        )
        warn_err = st.slider(
            "Warning error rate",
            min_value=0.0, max_value=1.0,
            value=float(overrides.get("warning_error_rate", cfg_m.get("warning_error_rate", 0.25))),
            step=0.01,
        )
    with c2:
        cat_thr = st.slider(
            "Categorical drift threshold",
            min_value=0.05, max_value=1.0,
            value=float(overrides.get("drift_categorical_threshold", cfg_m.get("drift_categorical_threshold", 0.25))),
            step=0.05,
        )
        crit_err = st.slider(
            "Critical error rate",
            min_value=0.0, max_value=1.0,
            value=float(overrides.get("critical_error_rate", cfg_m.get("critical_error_rate", 0.35))),
            step=0.01,
        )

    if st.button("💾 Save Thresholds"):
        st.session_state.thresholds_overrides = {
            "drift_numeric_threshold":     num_thr,
            "drift_categorical_threshold": cat_thr,
            "warning_error_rate":          warn_err,
            "critical_error_rate":         crit_err,
        }
        st.success("Thresholds saved for this session. Re-run monitoring to apply.")

    # ---- Privacy & LLM settings ----
    st.markdown("---")
    st.markdown('<div class="section-title">Privacy & LLM</div>', unsafe_allow_html=True)

    st.session_state.enable_cloud_llm = st.checkbox(
        "Enable Cloud LLM provider option",
        value=st.session_state.enable_cloud_llm,
        help="When off, only Local Ollama and Disabled are selectable on the AI Insights page.",
    )

    raw_toggle = st.checkbox(
        "Send raw sample rows to LLM (default OFF)",
        value=st.session_state.send_raw_to_llm,
        help="When OFF, only summarised metrics are sent to LLMs. When ON, raw sample rows may be included.",
    )
    if raw_toggle and not st.session_state.send_raw_to_llm:
        st.warning(
            "⚠️ Raw data may contain sensitive information. Only enable this if you are comfortable "
            "sending data to your selected LLM provider."
        )
    st.session_state.send_raw_to_llm = raw_toggle

    st.caption(
        "API keys for cloud LLM providers are kept only in this Streamlit session — "
        "never written to disk and never logged."
    )

    st.markdown("---")
    st.markdown('<div class="section-title">Workspace</div>', unsafe_allow_html=True)

    c3, c4 = st.columns(2)
    with c3:
        if st.button("🧹 Clear Workspace", type="secondary"):
            clear_workspace()
            st.session_state.monitoring_run = None
            st.success("Workspace cleared.")
    with c4:
        if st.button("🔄 Reset Demo Project"):
            ref_df, batches = load_sample_data()
            st.session_state.reference_df    = ref_df
            st.session_state.current_batches = batches
            st.success("Demo project reloaded into session.")

    st.markdown("---")
    st.markdown('<div class="section-title">App Info</div>', unsafe_allow_html=True)
    st.json({
        "config_file":   str(PATHS["config"] / "app_config.yaml"),
        "workspace":     str(PATHS["workspace"]),
        "ollama_status": "connected" if check_ollama_connection() else "not running",
    })


# ===========================================================================
# Sidebar navigation
# ===========================================================================
PAGES = {
    "🏠 Home":               page_home,
    "📤 Upload & Configure": page_upload_configure,
    "▶️ Run Monitoring":     page_run_monitoring,
    "📊 Dashboard":          page_dashboard,
    "🌊 Feature Drift":      page_feature_drift,
    "❌ Error Analysis":     page_error_analysis,
    "📋 Evidently Reports":  page_evidently_reports,
    "🤖 AI Insights":        page_ai_insights,
    "⚙️ Settings":           page_settings,
}


def render_sidebar() -> str:
    with st.sidebar:
        st.markdown("## 🧠 ML Monitor")
        st.markdown("---")

        choice = st.radio("Navigate", list(PAGES.keys()), label_visibility="collapsed")

        st.markdown("---")
        # Status panel
        st.markdown("**Mode:** " + st.session_state.mode)
        if st.session_state.reference_df is not None:
            st.markdown(f"**Reference:** {st.session_state.reference_df.shape[0]:,} rows")
        if st.session_state.current_batches:
            st.markdown(f"**Batches:** {len(st.session_state.current_batches)}")

        summary, _ = _load_monitoring_artefacts()
        if summary is not None and not summary.empty:
            latest = summary.iloc[-1]
            health = str(latest["model_health"])
            color  = HEALTH_COLORS.get(health, "#a0aec0")
            st.markdown(
                f'**Latest health:** <span style="color:{color};font-weight:700">{health}</span>',
                unsafe_allow_html=True,
            )

        st.markdown("---")
        if check_ollama_connection():
            st.markdown(badge("Ollama Online", "healthy"), unsafe_allow_html=True)
        else:
            st.markdown(badge("Ollama Offline", "warning"), unsafe_allow_html=True)

        st.markdown("---")
        st.caption("All inference is local. No cloud APIs used.")
    return choice


# ===========================================================================
# Main
# ===========================================================================
def main() -> None:
    page = render_sidebar()
    PAGES[page]()


if __name__ == "__main__":
    main()
