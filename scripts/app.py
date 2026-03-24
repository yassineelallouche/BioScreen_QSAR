"""
app.py — BioScreen-QSAR
Interactive Streamlit dashboard for QSAR model development and virtual screening.

Modules available in the sidebar:
  1. Data Curation & Deduplication
  2. Descriptor Calculation (ECFP)
  3. Model Training & Validation
  4. Virtual Screening

Author: BioScreen-QSAR Project
"""

import io
import os
import tempfile

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve

# ── Page configuration ────────────────────────────────────────────────────────

st.set_page_config(
    page_title="BioScreen-QSAR",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar navigation ────────────────────────────────────────────────────────

st.sidebar.image(
    "https://img.shields.io/badge/BioScreen--QSAR-v1.0-blue?style=for-the-badge",
    use_container_width=True,
)
st.sidebar.title("🔬 BioScreen-QSAR")
st.sidebar.caption("A low-code platform for antimicrobial QSAR modelling")
st.sidebar.divider()

MODULE = st.sidebar.selectbox(
    "Select Module",
    [
        "🏠 Home",
        "🧹 Data Curation",
        "🔢 Descriptor Calculation",
        "🤖 Model Training",
        "🔍 Virtual Screening",
    ],
)

st.sidebar.divider()
st.sidebar.markdown("**Platform version:** 1.0.0")
st.sidebar.markdown("**Python:** 3.11 | **RDKit:** 2024.3")


# ── Helper functions ──────────────────────────────────────────────────────────

def _download_csv(df: pd.DataFrame, label: str, filename: str) -> None:
    """Render a Streamlit CSV download button."""
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label=f"⬇️ Download {label}",
        data=csv_bytes,
        file_name=filename,
        mime="text/csv",
    )


def _bar_metrics(metrics: dict, title: str) -> None:
    """Render a horizontal bar chart of model metrics."""
    fig, ax = plt.subplots(figsize=(7, 3.5))
    names = list(metrics.keys())
    vals  = [float(v) for v in metrics.values()]
    colors = ["#1B6CA8" if v >= 0.7 else "#E87230" for v in vals]
    bars = ax.barh(names, vals, color=colors)
    ax.set_xlim(0, 1.05)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.axvline(0.7, color="gray", linestyle="--", linewidth=0.8, label="Threshold 0.70")
    ax.legend(fontsize=8)
    for bar, val in zip(bars, vals):
        ax.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=9)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 0 — Home
# ══════════════════════════════════════════════════════════════════════════════

if MODULE == "🏠 Home":
    st.title("🧬 BioScreen-QSAR")
    st.subheader("A Modular Low-Code Platform for Antimicrobial Activity Prediction")
    st.markdown(
        """
        **BioScreen-QSAR** provides an end-to-end QSAR workflow covering:
        - **Data Curation** — SMILES standardisation, salt removal, tautomer normalisation
        - **Duplicate Resolution** — concordant averaging / discordant removal
        - **Descriptor Calculation** — ECFP4/ECFP6 fingerprints via RDKit
        - **Model Training** — RF, SVM, LightGBM with Bayesian optimisation
        - **Virtual Screening** — ranked predictions on external compound libraries
        
        Use the **sidebar** to navigate between modules.
        
        ---
        > ⚠️ **Cloud mode limitation**: the virtual screening module processes up to 2,000 compounds online.
        > For larger libraries, please run BioScreen-QSAR locally (see GitHub README).
        """
    )
    col1, col2, col3 = st.columns(3)
    col1.metric("Supported algorithms", "3 (RF, SVM, LGBM)")
    col2.metric("Fingerprint types", "ECFP4 / ECFP6")
    col3.metric("Task modes", "Classification + Regression")


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 1 — Data Curation
# ══════════════════════════════════════════════════════════════════════════════

elif MODULE == "🧹 Data Curation":
    st.title("🧹 Data Curation & Deduplication")
    st.markdown(
        "Upload a CSV file containing SMILES strings and biological activity values. "
        "Select the appropriate column names and curation steps."
    )

    uploaded = st.file_uploader("Upload raw dataset (CSV)", type=["csv"])
    if uploaded:
        df_raw = pd.read_csv(uploaded)
        st.write(f"**Loaded:** {len(df_raw)} records")
        st.dataframe(df_raw.head(10), use_container_width=True)

        col1, col2 = st.columns(2)
        smiles_col   = col1.selectbox("SMILES column",   df_raw.columns.tolist())
        activity_col = col2.selectbox("Activity column", df_raw.columns.tolist(),
                                      index=min(1, len(df_raw.columns) - 1))

        data_type = st.radio(
            "Activity data type",
            ["binary (classification)", "continuous (regression)"],
            horizontal=True,
        )
        data_type_key = "binary" if "binary" in data_type else "continuous"

        st.markdown("**Curation steps to apply:**")
        do_mixtures    = st.checkbox("Remove mixtures & salts", value=True)
        do_neutralise  = st.checkbox("Neutralise charges",       value=True)
        do_tautomers   = st.checkbox("Canonical tautomers",      value=True)

        if st.button("▶️ Run Curation", type="primary"):
            with st.spinner("Standardising molecules …"):
                from scripts.data_curation import MolecularStandardiser
                from scripts.duplicate_handling import handle_duplicates

                standardiser = MolecularStandardiser(
                    remove_mixtures=do_mixtures,
                    neutralise_charges=do_neutralise,
                    standardise_tautomers=do_tautomers,
                )
                df_std = standardiser.standardise_dataframe(df_raw, smiles_col=smiles_col)
                n_std = len(df_std)

            with st.spinner("Resolving duplicates …"):
                df_curated, stats = handle_duplicates(
                    df_std,
                    smiles_col="curated_SMILES",
                    activity_col=activity_col,
                    data_type=data_type_key,
                )

            st.success("✅ Curation complete!")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Input records",    stats["n_input"])
            c2.metric("After std.",       n_std)
            c3.metric("Duplicates out",   stats["n_removed_duplicate_records"])
            c4.metric("Final dataset",    stats["n_final"])

            st.dataframe(df_curated.head(20), use_container_width=True)
            st.session_state["curated_df"] = df_curated
            _download_csv(df_curated, "Curated Dataset", "curated_dataset.csv")


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 2 — Descriptor Calculation
# ══════════════════════════════════════════════════════════════════════════════

elif MODULE == "🔢 Descriptor Calculation":
    st.title("🔢 ECFP Descriptor Calculation")
    st.markdown(
        "Compute Morgan / Extended Connectivity Fingerprints (ECFP) for your curated dataset."
    )

    uploaded = st.file_uploader("Upload curated CSV (output from Curation module)", type=["csv"])
    if not uploaded and "curated_df" in st.session_state:
        st.info("Using curated dataset from the Curation module.")
        df_curated = st.session_state["curated_df"]
    elif uploaded:
        df_curated = pd.read_csv(uploaded)
    else:
        df_curated = None

    if df_curated is not None:
        st.write(f"**Dataset:** {len(df_curated)} compounds")

        col1, col2, col3 = st.columns(3)
        smiles_col   = col1.selectbox("SMILES column",   df_curated.columns.tolist())
        activity_col = col2.selectbox("Activity column", df_curated.columns.tolist(),
                                      index=min(1, len(df_curated.columns) - 1))
        radius  = col3.selectbox("ECFP radius (ECFP4=2, ECFP6=3)", [2, 3])
        n_bits  = st.selectbox("Bit-vector length", [1024, 2048], index=1)

        if st.button("▶️ Calculate Descriptors", type="primary"):
            with st.spinner(f"Computing ECFP{2*radius} fingerprints ({n_bits} bits) …"):
                from scripts.descriptors_ecfp import ECFPGenerator
                generator = ECFPGenerator(radius=radius, n_bits=n_bits)
                desc_df = generator.build_descriptor_dataframe(
                    df_curated, smiles_col=smiles_col, activity_col=activity_col
                )

            st.success(
                f"✅ Fingerprints computed: {desc_df.shape[0]} compounds × "
                f"{desc_df.shape[1] - 2} bits."
            )
            st.dataframe(desc_df.iloc[:10, :12], use_container_width=True)
            st.session_state["descriptor_df"] = desc_df
            _download_csv(desc_df, "Descriptor Matrix", "descriptors_ecfp.csv")


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 3 — Model Training
# ══════════════════════════════════════════════════════════════════════════════

elif MODULE == "🤖 Model Training":
    st.title("🤖 Model Training & Validation")

    uploaded = st.file_uploader("Upload descriptor CSV", type=["csv"])
    if not uploaded and "descriptor_df" in st.session_state:
        desc_df = st.session_state["descriptor_df"]
        st.info("Using descriptor matrix from the Descriptor Calculation module.")
    elif uploaded:
        desc_df = pd.read_csv(uploaded)
    else:
        desc_df = None

    if desc_df is not None:
        cols = desc_df.columns.tolist()
        col1, col2 = st.columns(2)
        smiles_col   = col1.selectbox("SMILES column",   cols)
        activity_col = col2.selectbox("Activity column", cols,
                                      index=min(1, len(cols) - 1))
        task      = st.radio("Task", ["classification", "regression"], horizontal=True)
        algorithms = st.multiselect(
            "Algorithms to train",
            ["RandomForest", "SVM", "LightGBM"],
            default=["RandomForest", "LightGBM"],
        )
        optimise  = st.checkbox("Bayesian hyperparameter optimisation", value=True)
        n_trials  = st.slider("Optimisation trials (per algorithm)", 10, 100, 30)
        n_splits  = st.slider("CV folds", 3, 10, 5)

        if st.button("▶️ Train Models", type="primary"):
            from scripts.preprocessing import prepare_matrices
            from scripts.train_classification_models import train_classifiers
            from scripts.train_regression_models import train_regressors

            # Save descriptor CSV to temp file
            with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
                desc_df.to_csv(tmp.name, index=False)
                tmp_path = tmp.name

            with st.spinner("Splitting data …"):
                X_tr, X_te, y_tr, y_te, prep = prepare_matrices(
                    tmp_path,
                    smiles_col=smiles_col,
                    activity_col=activity_col,
                    stratify=(task == "classification"),
                )

            st.info(
                f"Train: {len(X_tr)} | Test: {len(X_te)} | "
                f"Features: {X_tr.shape[1]}"
            )

            with st.spinner("Training models … (this may take a few minutes)"):
                model_dir = tempfile.mkdtemp()
                if task == "classification":
                    results = train_classifiers(
                        X_tr, X_te, y_tr.astype(int), y_te.astype(int),
                        algorithms=algorithms,
                        optimise=optimise,
                        n_trials=n_trials,
                        n_splits=n_splits,
                        model_dir=model_dir,
                    )
                else:
                    results = train_regressors(
                        X_tr, X_te, y_tr, y_te,
                        algorithms=algorithms,
                        optimise=optimise,
                        n_trials=n_trials,
                        n_splits=n_splits,
                        model_dir=model_dir,
                    )

            st.success("✅ Training complete!")
            st.dataframe(results, use_container_width=True)

            if task == "classification":
                display_cols = ["Model", "bacc", "sensitivity", "specificity",
                                "ppv", "mcc", "auc", "f1"]
            else:
                display_cols = ["Model", "r2", "mae", "rmse", "explained_var"]

            display_cols_present = [c for c in display_cols if c in results.columns]
            best_row = results.loc[
                results[display_cols_present[1]].idxmax()
            ]
            st.markdown(f"**🏆 Best model:** `{best_row['Model']}`")

            metric_subset = {
                k: v for k, v in best_row.items()
                if k in display_cols_present[1:]
            }
            _bar_metrics(metric_subset, f"Best Model — {best_row['Model']}")

            st.session_state["model_dir"]   = model_dir
            st.session_state["preprocessor"] = prep
            st.session_state["task"]         = task
            _download_csv(results, "Model Results", "model_results.csv")
            os.unlink(tmp_path)


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 4 — Virtual Screening
# ══════════════════════════════════════════════════════════════════════════════

elif MODULE == "🔍 Virtual Screening":
    st.title("🔍 Ligand-Based Virtual Screening")
    st.markdown(
        "Upload a compound library (CSV with SMILES) and a pre-trained model (.pkl) "
        "to rank compounds by predicted antimicrobial activity."
    )
    st.warning(
        "⚠️ Cloud mode: maximum **2,000 compounds** per run. "
        "For larger libraries, use the local installation."
    )

    col1, col2 = st.columns(2)
    library_file = col1.file_uploader("Compound library CSV", type=["csv"])
    model_file   = col2.file_uploader("Pre-trained model (.pkl)", type=["pkl"])

    task_vs   = st.radio("Model task", ["classification", "regression"], horizontal=True)
    smiles_vs = st.text_input("SMILES column name in library", value="SMILES")

    if st.button("▶️ Run Virtual Screening", type="primary"):
        if not library_file or not model_file:
            st.error("Please upload both a compound library and a model file.")
        else:
            with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as lib_tmp:
                lib_tmp.write(library_file.read())
                lib_path = lib_tmp.name

            with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as mdl_tmp:
                mdl_tmp.write(model_file.read())
                mdl_path = mdl_tmp.name

            preprocessor = st.session_state.get("preprocessor", None)

            with st.spinner("Running virtual screening …"):
                from scripts.virtual_screening import run_virtual_screening
                vs_results = run_virtual_screening(
                    library_path=lib_path,
                    model_path=mdl_path,
                    preprocessor=preprocessor,
                    smiles_col=smiles_vs,
                    task=task_vs,
                    cloud_mode=True,
                )

            os.unlink(lib_path)
            os.unlink(mdl_path)

            if vs_results.empty:
                st.error("Virtual screening returned no results.")
            else:
                n_active = (
                    (vs_results["predicted_class"] == 1).sum()
                    if task_vs == "classification"
                    else None
                )
                st.success(
                    f"✅ Screened {len(vs_results)} compounds"
                    + (f" | Predicted active: {n_active}" if n_active is not None else "")
                )

                # Show top-ranked compounds
                st.subheader("Top-ranked candidates")
                top_n = min(50, len(vs_results))
                st.dataframe(vs_results.head(top_n), use_container_width=True)

                # Distribution plot
                if task_vs == "classification" and "probability_active" in vs_results.columns:
                    fig, ax = plt.subplots(figsize=(6, 3))
                    ax.hist(
                        vs_results["probability_active"], bins=30,
                        color="#1B6CA8", edgecolor="white"
                    )
                    ax.axvline(0.5, color="red", linestyle="--", label="Threshold 0.5")
                    ax.set_xlabel("P(active)")
                    ax.set_ylabel("Count")
                    ax.set_title("Distribution of Predicted Probabilities")
                    ax.legend()
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()
                elif task_vs == "regression" and "predicted_pMIC" in vs_results.columns:
                    fig, ax = plt.subplots(figsize=(6, 3))
                    ax.hist(
                        vs_results["predicted_pMIC"], bins=30,
                        color="#E87230", edgecolor="white"
                    )
                    ax.set_xlabel("Predicted pMIC")
                    ax.set_ylabel("Count")
                    ax.set_title("Distribution of Predicted pMIC Values")
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()

                _download_csv(vs_results, "Screening Results", "virtual_screening_results.csv")
