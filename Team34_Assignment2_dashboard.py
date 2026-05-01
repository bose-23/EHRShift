import json
from datetime import datetime
from pathlib import Path

import altair as alt
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from rapidfuzz import process

OUT = Path("outputs")
MODELS = OUT / "models"
FIGS = OUT / "figures"
REPORTS = OUT / "reports"

st.set_page_config(
    layout="wide",
    page_title="Team34 Assignment 2 Dashboard",
    initial_sidebar_state="expanded",
)

TEAM_NAME = "34"
TEAM_MEMBERS = ["Aryan Kumar", "B.Narayana Chandra Bose Reddy", "Jallu Venkata Sai Eswar", "Abhinay Reddy"]
TARGET_CONDITION = "Diabetes prediction under temporal shift"

PALETTE = {
    "dataset1": "#0f766e",
    "dataset2": "#ea580c",
    "accent": "#1d4ed8",
    "good": "#15803d",
    "warn": "#b45309",
}

NUMERIC_CANDIDATES = [
    "age_at_last",
    "enc_n_encounters",
    "enc_enc_cost_mean",
    "Body Height_last",
    "Body Weight_last",
    "BMI_last",
    "Glucose_last",
    "Hemoglobin A1c_last",
    "HbA1c_last",
    "med_count",
    "med_unique",
    "cond_count",
    "cond_unique",
]


@st.cache_data(show_spinner=False)
def load_aggregated():
    path = OUT / "aggregated_patients.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, dtype={"PATIENT": str}, parse_dates=["last_encounter"])
    if "PATIENT" in df.columns:
        df = df.set_index("PATIENT")
    return df


@st.cache_data(show_spinner=False)
def load_labels():
    path = OUT / "diabetes_labels.csv"
    if not path.exists():
        return None
    labels = pd.read_csv(path, index_col=0)
    if labels.shape[1] == 1:
        labels = labels.iloc[:, 0]
    labels.index = labels.index.astype(str)
    return labels.astype(int)


@st.cache_data(show_spinner=False)
def load_ids(name):
    path = OUT / name
    if not path.exists():
        return []
    return pd.read_csv(path, dtype={"PATIENT": str})["PATIENT"].astype(str).tolist()


@st.cache_resource(show_spinner=False)
def load_optional_artifacts():
    artifacts = {
        "preprocessor": None,
        "metrics_csv": None,
        "models_summary": None,
        "results_stage1": None,
        "results_stage2": None,
        "feature_summary": None,
        "dt_importance": None,
        "mlp_importance": None,
    }

    pre = MODELS / "preprocessor.joblib"
    if pre.exists():
        try:
            artifacts["preprocessor"] = joblib.load(pre)
        except Exception:
            pass

    for key, filename in {
        "results_stage1": "results_stage1.joblib",
        "results_stage2": "results_stage2.joblib",
    }.items():
        path = MODELS / filename
        if path.exists():
            try:
                artifacts[key] = joblib.load(path)
            except Exception:
                pass

    metrics_csv = REPORTS / "models_metrics.csv"
    if metrics_csv.exists():
        try:
            artifacts["metrics_csv"] = pd.read_csv(metrics_csv)
        except Exception:
            pass

    models_summary = REPORTS / "models_summary.json"
    if models_summary.exists():
        try:
            with open(models_summary, "r", encoding="utf-8") as handle:
                artifacts["models_summary"] = json.load(handle)
        except Exception:
            pass

    feature_summary = REPORTS / "feature_importance_summary.txt"
    if feature_summary.exists():
        try:
            artifacts["feature_summary"] = feature_summary.read_text(encoding="utf-8")
        except Exception:
            pass

    dt_importance = FIGS / "dt_feature_importances.csv"
    if dt_importance.exists():
        try:
            artifacts["dt_importance"] = pd.read_csv(dt_importance)
        except Exception:
            pass

    mlp_importance = FIGS / "mlp_permutation_importances.csv"
    if mlp_importance.exists():
        try:
            artifacts["mlp_importance"] = pd.read_csv(mlp_importance)
        except Exception:
            pass

    return artifacts


def make_stage1_metrics_table(stage1_results):
    if not isinstance(stage1_results, dict):
        return None

    rows = []
    metric_map = {
        "auc": "roc_auc",
        "roc_auc": "roc_auc",
        "auprc": "auprc",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "accuracy": "accuracy",
    }
    for key, value in stage1_results.items():
        if not isinstance(value, dict):
            continue
        dataset = "Dataset 1 validation" if key.endswith("_val") else "Dataset 2 test"
        model_name = key.replace("_val", "").replace("_test", "").replace("_", " ").upper()
        row = {"model": model_name, "evaluation_split": dataset, "source": "joblib"}
        for src, dst in metric_map.items():
            if src in value:
                row[dst] = value[src]
        rows.append(row)

    return pd.DataFrame(rows) if rows else None


def make_stage2_metrics_table(stage2_results):
    if not isinstance(stage2_results, dict):
        return None

    rows = []
    for key, value in stage2_results.items():
        if not isinstance(value, dict):
            continue
        rows.append(
            {
                "model": key.replace("_", " ").title(),
                "evaluation_split": "Dataset 2 test",
                "roc_auc": value.get("roc_auc"),
                "auprc": value.get("auprc"),
                "accuracy": value.get("accuracy"),
                "precision": value.get("precision"),
                "recall": value.get("recall"),
                "f1": value.get("f1"),
                "source": "continual learning",
            }
        )

    return pd.DataFrame(rows) if rows else None


def find_available_prediction_models():
    models = {}
    for name in ["dt_retrained.joblib", "mlp_finetuned.joblib", "svc_stage1.joblib"]:
        path = MODELS / name
        if not path.exists():
            continue
        try:
            models[name] = joblib.load(path)
        except Exception:
            models[name] = None
    return models


def classify_feature_source(column_name):
    if column_name in {"BIRTHDATE", "GENDER", "RACE", "ZIP", "INCOME", "age_at_last"}:
        return "Patient demographics"
    if column_name.startswith("enc_"):
        return "Encounter aggregates"
    if column_name.startswith("Body ") or "Glucose" in column_name or "A1c" in column_name or column_name == "BMI_last":
        return "Observation aggregates"
    if column_name.startswith("med_"):
        return "Medication aggregates"
    if column_name.startswith("cond_"):
        return "Condition aggregates"
    if column_name == "last_encounter":
        return "Temporal split field"
    return "Other engineered feature"


def build_dataset_frame(df, labels, train_ids, test_ids):
    if df is None or labels is None:
        return None

    data = df.copy()
    data["label"] = labels.reindex(data.index).fillna(0).astype(int)
    data["dataset"] = "Unassigned"
    data.loc[data.index.intersection(train_ids), "dataset"] = "Dataset 1"
    data.loc[data.index.intersection(test_ids), "dataset"] = "Dataset 2"
    if "last_encounter" in data.columns:
        data["last_encounter"] = pd.to_datetime(data["last_encounter"], errors="coerce")
    return data


def compare_feature_stats(data, feature):
    rows = []
    for dataset_name in ["Dataset 1", "Dataset 2"]:
        series = data.loc[data["dataset"] == dataset_name, feature].dropna()
        rows.append(
            {
                "dataset": dataset_name,
                "count": int(series.shape[0]),
                "mean": float(series.mean()) if not series.empty else np.nan,
                "median": float(series.median()) if not series.empty else np.nan,
                "std": float(series.std()) if series.shape[0] > 1 else np.nan,
            }
        )

    stats = pd.DataFrame(rows)
    mean1 = stats.loc[stats["dataset"] == "Dataset 1", "mean"].iloc[0]
    mean2 = stats.loc[stats["dataset"] == "Dataset 2", "mean"].iloc[0]
    if pd.notna(mean1) and mean1 != 0 and pd.notna(mean2):
        drift = (mean2 - mean1) / mean1
    else:
        drift = np.nan
    return stats, drift


def fuzzy_patient_search(query, choices, limit=50):
    if not query:
        return list(choices)[:limit]
    return [match[0] for match in process.extract(query, choices, limit=limit)]


def render_prevalence_chart(prevalence_df):
    base = (
        alt.Chart(prevalence_df)
        .encode(
            x=alt.X("month:T", title="Month"),
            y=alt.Y("label:Q", title="Diabetes prevalence", axis=alt.Axis(format="%")),
            color=alt.Color(
                "dataset:N",
                scale=alt.Scale(
                    domain=["Dataset 1", "Dataset 2"],
                    range=[PALETTE["dataset1"], PALETTE["dataset2"]],
                ),
                legend=alt.Legend(title=None, orient="top"),
            ),
            tooltip=[
                alt.Tooltip("dataset:N", title="Dataset"),
                alt.Tooltip("month:T", title="Month"),
                alt.Tooltip("label:Q", title="Prevalence", format=".2%"),
            ],
        )
        .properties(height=360)
    )
    chart = (
        base.mark_line(interpolate="monotone", strokeWidth=3)
        + base.mark_point(size=90, filled=True, stroke="white", strokeWidth=1.4)
    ).configure_view(stroke=None).configure_axis(
        gridColor="#e7ecef",
        labelColor="#334155",
        titleColor="#0f172a",
    )
    st.altair_chart(chart, width="stretch")


def render_distribution_chart(data, feature, mode="KDE"):
    plot_df = data.loc[data["dataset"].isin(["Dataset 1", "Dataset 2"]), [feature, "dataset"]].dropna().copy()
    if plot_df.empty:
        st.info("No data available for this plot.")
        return

    color_scale = alt.Scale(
        domain=["Dataset 1", "Dataset 2"],
        range=[PALETTE["dataset1"], PALETTE["dataset2"]],
    )

    if mode == "Histogram":
        chart = (
            alt.Chart(plot_df)
            .mark_bar(opacity=0.55, binSpacing=0)
            .encode(
                x=alt.X(f"{feature}:Q", bin=alt.Bin(maxbins=28), title=feature),
                y=alt.Y("count():Q", stack=None, title="Count"),
                color=alt.Color("dataset:N", scale=color_scale, legend=alt.Legend(title=None, orient="top")),
                tooltip=[
                    alt.Tooltip("dataset:N", title="Dataset"),
                    alt.Tooltip("count():Q", title="Count"),
                ],
            )
            .properties(height=360)
        )
    else:
        chart = (
            alt.Chart(plot_df)
            .transform_density(
                feature,
                as_=[feature, "density"],
                groupby=["dataset"],
                counts=False,
                steps=200,
            )
            .mark_area(opacity=0.28, interpolate="monotone", line={"width": 2.5})
            .encode(
                x=alt.X(f"{feature}:Q", title=feature),
                y=alt.Y("density:Q", title="Density"),
                color=alt.Color("dataset:N", scale=color_scale, legend=alt.Legend(title=None, orient="top")),
                tooltip=[
                    alt.Tooltip("dataset:N", title="Dataset"),
                    alt.Tooltip(f"{feature}:Q", title=feature, format=".2f"),
                    alt.Tooltip("density:Q", title="Density", format=".3f"),
                ],
            )
            .properties(height=360)
        )

    chart = chart.configure_view(stroke=None).configure_axis(
        gridColor="#e7ecef",
        labelColor="#334155",
        titleColor="#0f172a",
    )
    st.altair_chart(chart, width="stretch")


def render_bar_chart(dataframe, x_col, y_col, title, color):
    plot_df = dataframe.copy()
    plot_df[x_col] = plot_df[x_col].astype(str)
    chart = (
        alt.Chart(plot_df)
        .mark_bar(cornerRadiusTopLeft=8, cornerRadiusTopRight=8, color=color)
        .encode(
            x=alt.X(f"{x_col}:N", sort=None, title=x_col.replace("_", " ").title(), axis=alt.Axis(labelAngle=-20)),
            y=alt.Y(f"{y_col}:Q", title=y_col.upper()),
            tooltip=[
                alt.Tooltip(f"{x_col}:N", title=x_col.replace("_", " ").title()),
                alt.Tooltip(f"{y_col}:Q", title=y_col.upper(), format=".3f"),
            ],
        )
        .properties(height=360, title=title)
        .configure_view(stroke=None)
        .configure_axis(gridColor="#e7ecef", labelColor="#334155", titleColor="#0f172a")
    )
    st.altair_chart(chart, width="stretch")


def predict_for_patient(patient_id, df, preprocessor, models):
    if patient_id not in df.index or preprocessor is None:
        return {}

    row = df.loc[[patient_id]].copy()
    if "label" in row.columns:
        row = row.drop(columns=["label"])

    try:
        transformed = preprocessor.transform(row)
    except Exception:
        return {}

    predictions = {}
    for name, model in models.items():
        if model is None:
            continue
        try:
            if hasattr(model, "predict_proba"):
                score = float(model.predict_proba(transformed)[:, 1][0])
            elif hasattr(model, "decision_function"):
                raw = float(model.decision_function(transformed)[0])
                score = 1.0 / (1.0 + np.exp(-raw))
            else:
                continue
            predictions[name.replace(".joblib", "")] = score
        except Exception:
            continue
    return predictions


def artifact_status_rows():
    checks = {
        "Aggregated patient table": OUT / "aggregated_patients.csv",
        "Labels": OUT / "diabetes_labels.csv",
        "Dataset 1 IDs": OUT / "train_ids.csv",
        "Dataset 2 IDs": OUT / "test_ids.csv",
        "Preprocessor": MODELS / "preprocessor.joblib",
        "Stage 1 results": MODELS / "results_stage1.joblib",
        "Stage 2 results": MODELS / "results_stage2.joblib",
        "Model summary": REPORTS / "models_summary.json",
        "Feature summary": REPORTS / "feature_importance_summary.txt",
    }
    return pd.DataFrame(
        [
            {"artifact": label, "present": "Yes" if path.exists() else "No", "path": str(path)}
            for label, path in checks.items()
        ]
    )


def render_missing_outputs():
    st.warning(
        "I could not find the generated files in `outputs/` yet, so this dashboard only has the layout for now."
    )
    st.markdown("**Run these scripts to fill the dashboard with results:**")
    st.code(
        "\n".join(
            [
                "conda activate myenv",
                "python src/04_aggregate.py",
                "python src/05_split_and_label.py",
                "python src/07_train.py",
                "python src/08_continual.py",
                "python src/09_explain_eda.py",
                "python src/10_dump_joblib.py",
                "streamlit run Team34_Assignment2_dashboard.py",
            ]
        ),
        language="bash",
    )
    st.dataframe(artifact_status_rows(), width="stretch", hide_index=True)


def main():
    df = load_aggregated()
    labels = load_labels()
    train_ids = load_ids("train_ids.csv")
    test_ids = load_ids("test_ids.csv")
    artifacts = load_optional_artifacts()
    preprocessor = artifacts["preprocessor"]
    prediction_models = find_available_prediction_models()

    data = build_dataset_frame(df, labels, train_ids, test_ids)
    stage1_metrics = make_stage1_metrics_table(artifacts["results_stage1"])
    stage2_metrics = make_stage2_metrics_table(artifacts["results_stage2"])
    csv_metrics = artifacts["metrics_csv"]

    st.title(f"{TEAM_NAME} Assignment 2 Dashboard")
    st.caption(f"{TARGET_CONDITION} | Streamlit dashboard for BITS F464 Assignment 2")

    with st.sidebar:
        st.header("Team Details")
        st.write(f"**Team:** {TEAM_NAME}")
        for member in TEAM_MEMBERS:
            st.write(f"- {member}")
        st.markdown("**What this dashboard covers**")
        st.write("- Preprocessing and feature engineering")
        st.write("- Dataset 1 vs Dataset 2 split")
        st.write("- Model comparison")
        st.write("- Continual learning")
        st.write("- Feature importance and drift")

    if data is None:
        render_missing_outputs()
        return

    total_patients = len(data)
    total_positive = int(data["label"].sum())
    dataset1_count = int((data["dataset"] == "Dataset 1").sum())
    dataset2_count = int((data["dataset"] == "Dataset 2").sum())
    cutoff = None
    if dataset1_count and dataset2_count and "last_encounter" in data.columns:
        cutoff = data.loc[data["dataset"] == "Dataset 2", "last_encounter"].min()

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Patients", f"{total_patients:,}")
    k2.metric("Positive labels", f"{total_positive:,}", f"{(total_positive / total_patients):.1%}")
    k3.metric("Dataset 1 size", f"{dataset1_count:,}")
    k4.metric("Dataset 2 size", f"{dataset2_count:,}")

    tabs = st.tabs(
        [
            "Overview",
            "Pipeline & Features",
            "EDA & Drift",
            "Models",
            "Continual Learning",
            "Feature Importance",
            "Patient Explorer",
            "Submission Notes",
        ]
    )

    with tabs[0]:
        left, right = st.columns([1.4, 1])
        with left:
            st.subheader("What is on this dashboard")
            st.write(
                "Everything is laid out in the same order we built the project: data prep first, then drift checks, then model results, then the continual-learning step."
            )
            summary_rows = [
                {"requirement": "Dataset integration and feature engineering", "status": "Included"},
                {"requirement": "Historical vs current temporal split", "status": "Included"},
                {"requirement": "Decision Tree / SVM / MLP evaluation", "status": "Included"},
                {"requirement": "Cross-dataset generalization analysis", "status": "Included"},
                {"requirement": "Continual learning on Dataset 2", "status": "Included"},
                {"requirement": "Feature importance and drift analysis", "status": "Included"},
            ]
            st.dataframe(pd.DataFrame(summary_rows), width="stretch", hide_index=True)
        with right:
            st.subheader("Temporal split")
            if cutoff is not None and pd.notna(cutoff):
                st.metric("Dataset split cutoff", cutoff.strftime("%Y-%m-%d"))
            else:
                st.metric("Dataset split cutoff", "Unavailable")
            st.dataframe(artifact_status_rows(), width="stretch", hide_index=True)

        if "last_encounter" in data.columns:
            prevalence = (
                data.dropna(subset=["last_encounter"])
                .assign(month=lambda x: x["last_encounter"].dt.to_period("M").dt.to_timestamp())
                .groupby(["month", "dataset"], dropna=False)["label"]
                .mean()
                .reset_index()
            )
            render_prevalence_chart(prevalence)

    with tabs[1]:
        st.subheader("Pipeline steps")
        steps = pd.DataFrame(
            [
                {"step": "1. Aggregate raw CSV tables", "script": "src/04_aggregate.py", "output": "outputs/aggregated_patients.csv"},
                {"step": "2. Create temporal split and labels", "script": "src/05_split_and_label.py", "output": "train_ids.csv, test_ids.csv, diabetes_labels.csv"},
                {"step": "3. Train Dataset 1 models", "script": "src/07_train.py", "output": "preprocessor.joblib, stage 1 models, results_stage1.joblib"},
                {"step": "4. Continual learning on Dataset 2", "script": "src/08_continual.py", "output": "fine-tuned models, results_stage2.joblib, ROC and confusion matrices"},
                {"step": "5. Interpretation and EDA exports", "script": "src/09_explain_eda.py", "output": "feature importance CSVs, plots, text summary"},
                {"step": "6. Artifact summary", "script": "src/10_dump_joblib.py", "output": "outputs/reports/models_summary.json"},
            ]
        )
        st.dataframe(steps, width="stretch", hide_index=True)

        st.subheader("Feature inventory")
        inventory = pd.DataFrame(
            [{"feature": column, "source_group": classify_feature_source(column)} for column in data.columns]
        )
        st.dataframe(inventory, width="stretch", hide_index=True)

        numeric_cols = [col for col in NUMERIC_CANDIDATES if col in data.columns]
        cat_cols = [col for col in data.columns if col not in numeric_cols + ["label", "dataset", "last_encounter"]]
        c1, c2 = st.columns(2)
        c1.metric("Numeric engineered features", len(numeric_cols))
        c2.metric("Categorical or metadata features", len(cat_cols))

        st.markdown("**Feature representation notes**")
        st.write(
            "The feature set combines demographics, encounter summaries, lab-style observations, medication counts, and condition counts. In practice that means the models see both patient background and a compact summary of recent clinical history."
        )
        if artifacts["models_summary"] is not None:
            with st.expander("Saved model settings"):
                st.json(artifacts["models_summary"])

    with tabs[2]:
        st.subheader("Descriptive statistics and drift")
        numeric_cols = [col for col in NUMERIC_CANDIDATES if col in data.columns]
        if not numeric_cols:
            st.info("No expected numeric columns are available in the aggregated table.")
        else:
            controls_a, controls_b = st.columns([1.2, 1])
            with controls_a:
                feature = st.selectbox("Select a feature to compare", numeric_cols)
            with controls_b:
                distribution_mode = st.segmented_control(
                    "Distribution style",
                    options=["KDE", "Histogram"],
                    default="KDE",
                    key="eda_distribution_mode",
                )
            stats, drift = compare_feature_stats(data, feature)

            left, right = st.columns([1.3, 1])
            with left:
                render_distribution_chart(data, feature, mode=distribution_mode)
            with right:
                st.dataframe(
                    stats.style.format({"mean": "{:.3f}", "median": "{:.3f}", "std": "{:.3f}"}),
                    width="stretch",
                )
                if pd.notna(drift):
                    direction = "increased" if drift > 0 else "decreased" if drift < 0 else "stayed flat"
                    st.write(f"Mean {feature} {direction} by {drift:.1%} from Dataset 1 to Dataset 2.")
                else:
                    st.write("There is not enough data here to compute a reliable shift for this feature.")

        class_balance = (
            data.groupby("dataset")["label"]
            .agg(["count", "sum", "mean"])
            .rename(columns={"count": "patients", "sum": "positive_labels", "mean": "prevalence"})
            .reset_index()
        )
        st.subheader("Class balance by temporal dataset")
        st.dataframe(
            class_balance.style.format({"prevalence": "{:.2%}"}),
            width="stretch",
            hide_index=True,
        )

        if (FIGS / "label_prevalence_over_time.png").exists():
            st.image(str(FIGS / "label_prevalence_over_time.png"), caption="Saved prevalence plot from `src/09_explain_eda.py`")

    with tabs[3]:
        st.subheader("Cross-dataset model evaluation")

        shown_any = False
        if csv_metrics is not None:
            shown_any = True
            st.markdown("**Metrics table from `outputs/reports/models_metrics.csv`**")
            st.dataframe(csv_metrics, width="stretch")

        if stage1_metrics is not None:
            shown_any = True
            st.markdown("**Stage 1 evaluation reconstructed from `results_stage1.joblib`**")
            st.dataframe(
                stage1_metrics.style.format(
                    {
                        "roc_auc": "{:.3f}",
                        "auprc": "{:.3f}",
                        "precision": "{:.3f}",
                        "recall": "{:.3f}",
                        "f1": "{:.3f}",
                        "accuracy": "{:.3f}",
                    }
                ),
                width="stretch",
                hide_index=True,
            )

            if {"model", "evaluation_split", "auprc"}.issubset(stage1_metrics.columns):
                chart_df = stage1_metrics.dropna(subset=["auprc"]).copy()
                chart_df["label"] = chart_df["model"] + " | " + chart_df["evaluation_split"]
                render_bar_chart(chart_df, "label", "auprc", "Stage 1 model performance across datasets", PALETTE["accent"])

                pivot = stage1_metrics.pivot_table(index="model", columns="evaluation_split", values="auprc", aggfunc="first")
                if {"Dataset 1 validation", "Dataset 2 test"}.issubset(pivot.columns):
                    gap_df = (
                        pivot.assign(generalization_gap=lambda x: x["Dataset 2 test"] - x["Dataset 1 validation"])
                        .reset_index()[["model", "generalization_gap"]]
                    )
                    st.markdown("**Generalization gap on AUPRC**")
                    st.dataframe(
                        gap_df.style.format({"generalization_gap": "{:+.3f}"}),
                        width="stretch",
                        hide_index=True,
                    )

        if not shown_any:
            st.info("No model metrics were found yet. Run the training and reporting scripts first, then come back here.")

        roc_path = FIGS / "roc_dataset2_test.png"
        if roc_path.exists():
            st.image(str(roc_path), caption="ROC comparison on Dataset 2 test")

        cm_files = sorted(FIGS.glob("cm_*.png"))
        if cm_files:
            st.markdown("**Confusion matrices**")
            cols = st.columns(3)
            for idx, image_path in enumerate(cm_files):
                cols[idx % 3].image(str(image_path), caption=image_path.stem.replace("cm_", "").replace("_", " "))

        st.subheader("Interpretation")
        st.write(
            "The main thing to watch here is how much performance drops from Dataset 1 to Dataset 2. If that gap is large, the older model is not carrying over cleanly to the newer data."
        )

    with tabs[4]:
        st.subheader("Continual learning results")
        if stage2_metrics is not None:
            st.dataframe(
                stage2_metrics.style.format(
                    {
                        "roc_auc": "{:.3f}",
                        "auprc": "{:.3f}",
                        "accuracy": "{:.3f}",
                        "precision": "{:.3f}",
                        "recall": "{:.3f}",
                        "f1": "{:.3f}",
                    }
                ),
                width="stretch",
                hide_index=True,
            )

            render_bar_chart(
                stage2_metrics.dropna(subset=["auprc"]),
                "model",
                "auprc",
                "Continual learning performance on Dataset 2 test",
                PALETTE["accent"],
            )

            if stage1_metrics is not None:
                baseline = stage1_metrics.loc[stage1_metrics["evaluation_split"] == "Dataset 2 test", ["model", "auprc"]].rename(
                    columns={"model": "baseline_model", "auprc": "baseline_auprc"}
                )
                st.markdown("**Comparison note**")
                st.write(
                    "Compare these numbers with the Stage 1 Dataset 2 scores from the Models tab. That tells you whether fine-tuning or retraining actually helped with the time shift."
                )
                st.dataframe(baseline, width="stretch", hide_index=True)
        else:
            st.info("Continual-learning metrics are not available yet. Run `src/08_continual.py` first.")

    with tabs[5]:
        st.subheader("Feature importance and interpretation")
        shown_importance = False
        if artifacts["dt_importance"] is not None or artifacts["mlp_importance"] is not None:
            shown_importance = True
            cols = st.columns(2)
            if artifacts["dt_importance"] is not None:
                cols[0].markdown("**Decision Tree importance table**")
                cols[0].dataframe(artifacts["dt_importance"].head(15), width="stretch", hide_index=True)
            if artifacts["mlp_importance"] is not None:
                cols[1].markdown("**MLP permutation importance table**")
                cols[1].dataframe(artifacts["mlp_importance"].head(15), width="stretch", hide_index=True)

        for image_name, caption in [
            ("fi_dt_retrained.png", "Decision Tree feature importance"),
            ("fi_mlp_permutation.png", "MLP permutation importance"),
        ]:
            path = FIGS / image_name
            if path.exists():
                shown_importance = True
                st.image(str(path), caption=caption)

        if artifacts["feature_summary"]:
            shown_importance = True
            with st.expander("Short write-up"):
                st.text(artifacts["feature_summary"])

        if not shown_importance:
            st.info("Feature-importance files are missing. Run `python src/09_explain_eda.py` to regenerate them.")

    with tabs[6]:
        st.subheader("Patient-level inspection")
        all_ids = list(data.index.astype(str))
        query = st.text_input("Search by patient ID")
        matches = fuzzy_patient_search(query, all_ids, limit=100)
        patient_id = st.selectbox("Select a patient", matches if matches else all_ids[:50])

        if patient_id:
            patient = data.loc[patient_id]
            left, right = st.columns([1, 1.2])
            with left:
                overview = {
                    "dataset": patient.get("dataset"),
                    "label": int(patient.get("label", 0)),
                    "age_at_last": patient.get("age_at_last"),
                    "GENDER": patient.get("GENDER"),
                    "RACE": patient.get("RACE"),
                    "ZIP": patient.get("ZIP"),
                    "last_encounter": str(patient.get("last_encounter")),
                }
                st.json(overview)
            with right:
                feature_cols = [col for col in numeric_cols if col in patient.index]
                patient_features = pd.DataFrame(
                    [{"feature": col, "value": patient[col]} for col in feature_cols if pd.notna(patient[col])]
                )
                st.dataframe(patient_features, width="stretch", hide_index=True)

            predictions = predict_for_patient(patient_id, data, preprocessor, prediction_models)
            if predictions:
                st.markdown("**Model risk estimates**")
                cols = st.columns(len(predictions))
                for idx, (name, value) in enumerate(predictions.items()):
                    cols[idx].metric(name, f"{value:.1%}")
            else:
                st.info("Prediction artifacts are not available yet for patient-level scoring.")

    with tabs[7]:
        st.subheader("Submission checklist")
        checklist = pd.DataFrame(
            [
                {"item": "Dashboard file name", "expected": "Team34_Assignment2_dashboard.py"},
                {"item": "Video file name", "expected": "Team34_Assignment2_video.mp4"},
                {"item": "Zip name", "expected": "Team34_Assignment2.zip"},
                {"item": "Team details visible in dashboard", "expected": "Required"},
                {"item": "Screenshots of dashboard in video", "expected": "Required"},
                {"item": "Summary of findings in video", "expected": "Required"},
            ]
        )
        st.dataframe(checklist, width="stretch", hide_index=True)

        st.markdown("**What this dashboard includes**")
        st.write("- Temporal split and prevalence overview")
        st.write("- Engineered feature inventory and pipeline steps")
        st.write("- Dataset 1 vs Dataset 2 drift analysis")
        st.write("- Stage 1 model comparison and generalization gap")
        st.write("- Continual learning evaluation and interpretation")
        st.write("- Patient-level exploration")

        st.markdown("**Before final submission**")
        st.write("- Regenerate outputs if any figures or metrics are missing.")
        st.write("- Record dashboard interactions for the required video.")

    st.caption(f"Last refreshed: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')} | Team: {TEAM_NAME}")


if __name__ == "__main__":
    main()
