from pathlib import Path
import json
import importlib
import os
import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import shap
import streamlit as st
from sklearn.impute import SimpleImputer

try:
    # Some Windows Python setups need stdlib distutils behavior for TensorFlow imports.
    os.environ.setdefault("SETUPTOOLS_USE_DISTUTILS", "stdlib")
    _tf_spec = importlib.util.find_spec("tensorflow")
    _keras_spec = importlib.util.find_spec("keras")
    if _tf_spec is not None:
        keras = importlib.import_module("tensorflow.keras")
        HAS_TF = True
    elif _keras_spec is not None:
        keras = importlib.import_module("keras")
        HAS_TF = True
    else:
        HAS_TF = False
except Exception:
    HAS_TF = False

st.set_page_config(page_title="Predicting U.S. Cleft Lip and Palate Cases", layout="wide")

ART_DIR = Path(__file__).parent / "streamlit_artifacts"


def get_artifact_signature():
    tracked = [
        "results_df.csv",
        "eda_snapshot.pkl",
        "X_test_full.pkl",
        "y_test_full.pkl",
        "X_test_safe.pkl",
        "y_test_safe.pkl",
        "best_params_log.json",
        "model_registry.json",
        "interactive_schema.json",
        "data_dictionary.pkl",
        "data_dictionary.csv",
    ]
    signature = []
    for name in tracked:
        p = ART_DIR / name
        if p.exists():
            stat = p.stat()
            signature.append((name, int(stat.st_mtime), stat.st_size))
        else:
            signature.append((name, None, None))
    return tuple(signature)


@st.cache_data
def load_data(_artifact_signature):
    results_df = pd.read_csv(ART_DIR / "results_df.csv")
    eda_df = pd.read_pickle(ART_DIR / "eda_snapshot.pkl")
    x_test_full = pd.read_pickle(ART_DIR / "X_test_full.pkl")
    y_test_full = pd.read_pickle(ART_DIR / "y_test_full.pkl")
    x_test_safe = pd.read_pickle(ART_DIR / "X_test_safe.pkl")
    y_test_safe = pd.read_pickle(ART_DIR / "y_test_safe.pkl")
    with open(ART_DIR / "best_params_log.json", "r", encoding="utf-8") as f:
        best_params = json.load(f)
    with open(ART_DIR / "model_registry.json", "r", encoding="utf-8") as f:
        model_registry = json.load(f)
    with open(ART_DIR / "interactive_schema.json", "r", encoding="utf-8") as f:
        interactive_schema = json.load(f)

    data_dictionary = None
    dd_pkl = ART_DIR / "data_dictionary.pkl"
    dd_csv = ART_DIR / "data_dictionary.csv"
    if dd_pkl.exists():
        data_dictionary = pd.read_pickle(dd_pkl)
    elif dd_csv.exists():
        data_dictionary = pd.read_csv(dd_csv)

    return {
        "results_df": results_df,
        "eda_df": eda_df,
        "X_test_full": x_test_full,
        "y_test_full": y_test_full,
        "X_test_safe": x_test_safe,
        "y_test_safe": y_test_safe,
        "best_params": best_params,
        "model_registry": model_registry,
        "interactive_schema": interactive_schema,
        "data_dictionary": data_dictionary,
    }


@st.cache_resource
def load_models(model_registry):
    def repair_sklearn_compat(obj):
        seen = set()

        def walk(node):
            node_id = id(node)
            if node_id in seen:
                return
            seen.add(node_id)

            if isinstance(node, SimpleImputer):
                if not hasattr(node, "_fill_dtype"):
                    if hasattr(node, "statistics_") and getattr(node.statistics_, "dtype", None) is not None:
                        node._fill_dtype = node.statistics_.dtype
                    elif hasattr(node, "_fit_dtype"):
                        node._fill_dtype = node._fit_dtype
                    else:
                        node._fill_dtype = object
                if not hasattr(node, "_fit_dtype"):
                    node._fit_dtype = node._fill_dtype

            if isinstance(node, dict):
                for v in node.values():
                    walk(v)
                return

            if isinstance(node, (list, tuple, set)):
                for item in node:
                    walk(item)
                return

            for attr in ("steps", "named_steps", "transformers", "transformers_", "transformer_list"):
                if hasattr(node, attr):
                    walk(getattr(node, attr))

        walk(obj)
        return obj

    loaded = {}
    for name, meta in model_registry.items():
        if meta["type"] == "sklearn":
            loaded[name] = repair_sklearn_compat(joblib.load(ART_DIR / meta["path"]))
        elif meta["type"] == "keras" and HAS_TF:
            loaded[name] = {
                "model": keras.models.load_model(ART_DIR / meta["path"]),
                "preprocessor": repair_sklearn_compat(joblib.load(ART_DIR / meta["preprocessor_path"])),
            }
    return loaded


def ensure_object_na(df):
    out = df.copy()
    for c in out.columns:
        if not pd.api.types.is_numeric_dtype(out[c]):
            out[c] = out[c].astype("object").where(out[c].notna(), np.nan)
    return out


def predict_prob(model_name, model_meta, model_obj, x_df):
    x_in = ensure_object_na(x_df)
    if model_meta["type"] == "sklearn":
        return model_obj.predict_proba(x_in)[:, 1]
    if model_meta["type"] == "keras":
        x_mat = model_obj["preprocessor"].transform(x_in)
        return model_obj["model"].predict(x_mat, verbose=0).reshape(-1)
    raise ValueError(f"Unsupported model type for {model_name}")


def section_caption(text):
    st.caption(text)


FRIENDLY_NAME_MAP = {
    "MAGER14": "Mother age recode (14)",
    "MAGER": "Mother age (single years)",
    "FAGECOMB": "Father combined age",
    "PREVIS": "Number of prenatal visits",
    "PREVIS_REC": "Prenatal visits recode",
    "CIG_REC": "Any cigarette use recode",
    "MEDUC": "Mother education",
    "SEX": "Infant sex",
    "WIC": "Received WIC",
    "COMBGEST": "Combined gestation (weeks)",
    "OEGest_Comb": "Obstetric estimate gestation (weeks)",
    "WTGAIN": "Weight gain (lbs)",
    "BMI": "Body Mass Index",
    "M_Ht_In": "Mother height (inches)",
    "PWgt_R": "Pre-pregnancy weight (lbs)",
    "DWgt_R": "Delivery weight (lbs)",
    "FRACEHISP": "Father race/Hispanic origin",
    "MRACEHISP": "Mother race/Hispanic origin",
    "FHISP_R": "Father Hispanic origin recode",
    "MHISP_R": "Mother Hispanic origin recode",
    "RESTATUS": "Residence status",
    "BFED": "Breastfed at discharge",
    "TBO_REC": "Total birth order recode",
    "ILP_R": "Interval since last pregnancy",
    "ILOP_R": "Interval since last other pregnancy",
    "ILLB_R11": "Interval since last live birth recode (11)",
    "cleft_case": "Cleft case",
}


def friendly_name(col):
    return FRIENDLY_NAME_MAP.get(col, col)


def pretty_transformed_name(tname, categorical_source_cols):
    if "__" in tname:
        prefix, rest = tname.split("__", 1)
    else:
        prefix, rest = "", tname

    if prefix == "num":
        return friendly_name(rest)

    if prefix == "cat":
        matches = [c for c in categorical_source_cols if rest == c or rest.startswith(c + "_")]
        if matches:
            base_col = max(matches, key=len)
            base_label = friendly_name(base_col)
            if rest == base_col:
                return base_label
            level = rest[len(base_col) + 1 :]
            return f"{base_label} = {level}"
        return rest

    return friendly_name(rest)


def prevalence_table(df, group_col, min_n=2000):
    g = (
        df.dropna(subset=[group_col, "cleft_case"])
        .groupby(group_col, as_index=False)
        .agg(cases=("cleft_case", "sum"), births=("cleft_case", "count"))
    )
    g = g[g["births"] >= min_n].copy()
    g["prev_per_10k"] = g["cases"] / g["births"] * 10000
    return g.sort_values("prev_per_10k", ascending=False)


def compute_shap_bundle(tree_pipe, x_eval, sample_n=350):
    x_shap_raw = x_eval.sample(min(sample_n, len(x_eval)), random_state=42).copy()

    pre = tree_pipe.named_steps["preprocess"]
    model = tree_pipe.named_steps["model"]

    # Match notebook labeling by using categorical columns from the fitted preprocessor.
    categorical_cols = []
    for t_name, _, cols in getattr(pre, "transformers", []):
        if t_name == "cat":
            if isinstance(cols, (list, tuple, np.ndarray, pd.Index)):
                categorical_cols.extend([str(c) for c in cols])
    if not categorical_cols:
        numeric_ratio = x_shap_raw.apply(lambda s: pd.to_numeric(s, errors="coerce").notna().mean())
        categorical_cols = [c for c in x_shap_raw.columns if numeric_ratio.get(c, 1.0) < 0.98]

    x_trans = pre.transform(x_shap_raw)
    feature_names = pre.get_feature_names_out()
    feature_display_names = [pretty_transformed_name(n, categorical_cols) for n in feature_names]

    explainer = shap.TreeExplainer(model)
    sv_raw = explainer.shap_values(x_trans)
    if isinstance(sv_raw, list):
        sv = sv_raw[1] if len(sv_raw) > 1 else sv_raw[0]
    elif isinstance(sv_raw, np.ndarray) and sv_raw.ndim == 3:
        sv = sv_raw[:, :, 1]
    else:
        sv = sv_raw

    base_val = explainer.expected_value
    if isinstance(base_val, (list, np.ndarray)):
        base_val = float(np.ravel(base_val)[-1])
    else:
        base_val = float(base_val)

    mean_abs = np.abs(sv).mean(axis=0)
    top_idx = np.argsort(mean_abs)[-min(20, len(mean_abs)) :][::-1]
    top_display_names = [feature_display_names[i] for i in top_idx]

    x_top = x_trans[:, top_idx]
    if hasattr(x_top, "toarray"):
        x_top = x_top.toarray()

    return {
        "x_shap_raw": x_shap_raw,
        "x_trans": x_trans,
        "feature_names": feature_names,
        "feature_display_names": feature_display_names,
        "sv": sv,
        "base_val": base_val,
        "top_idx": top_idx,
        "top_display_names": top_display_names,
        "x_top": x_top,
    }


def main():
    data = load_data(get_artifact_signature())
    models = load_models(data["model_registry"])

    st.title("Cleft Lip/Palate Risk Modeling and Explainability")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Executive Summary",
        "Descriptive Analytics",
        "Model Performance",
        "Explainability & Interactive Prediction",
        "Causality Analysis",
    ])

    with tab1:
        st.markdown("## Predicting U.S. Cleft Lip and Palate Cases")
        st.markdown("### Using 2024 NCHS National Natality Microdata")
        st.caption("Executive Summary | Predictive Analytics Project")

        with st.expander("Dataset and Predictions", expanded=True):
            st.markdown("#### About the Data Source")
            st.markdown(
                """
                This project is built on the 2024 National Natality Public Use File, published by the National Center for Health Statistics (NCHS) within the U.S. Centers for Disease Control and Prevention. The file is produced annually through the Vital Statistics Cooperative Program, in which vital registration offices from all 50 states, the District of Columbia, New York City, Puerto Rico, the U.S. Virgin Islands, Guam, American Samoa, and the Commonwealth of the Northern Mariana Islands submit individual birth certificate data to NCHS for compilation and release.

                The 2024 file contains 3,638,436 birth records for the United States, representing nearly all live births registered in the country during the calendar year. Each record corresponds to one live birth and is stored in a fixed-width text format with a record length of 1,330 characters. Across all reporting areas, the NCHS estimates that more than 99 percent of all U.S. births are captured in the registration system, making this one of the most complete administrative datasets available for public health research.

                The data items on each record are drawn from two primary collection instruments: the Mother's Worksheet, which captures self-reported information such as race, Hispanic origin, educational attainment, and smoking behavior; and the Facility Worksheet, which captures medical record data including prenatal care, risk factors, delivery method, and infant health outcomes. Both instruments were introduced as part of the 2003 revision of the U.S. Standard Certificate of Live Birth and have been fully implemented across all states and the District of Columbia since January 1, 2016.
                """
            )

            st.markdown("#### Features Available for Modeling")
            st.markdown(
                """
                The dataset provides a rich set of predictors spanning six substantive domains. Maternal demographics include the mother's age in single years, race using 31-category and 6-category recodes, Hispanic origin, nativity (U.S.-born vs. foreign-born), marital status, and educational attainment coded on an 8-level scale from less than 8th grade through doctorate or professional degree. Paternal demographics provide parallel fields for father's age, race, Hispanic origin, and education, though these carry higher missingness rates due to incomplete reporting for unmarried births.

                Pregnancy history variables capture the number of prior live births now living, prior live births now dead, prior other pregnancy terminations, live birth order, total birth order, and multiple interval-since-last-pregnancy recodes. Prenatal behavior variables include the month prenatal care began (coded 1 through 10, plus no care and unknown), the total number of prenatal visits, WIC food program enrollment during pregnancy, and the number of cigarettes smoked per day before pregnancy and in each of the three trimesters. Maternal physical characteristics include height in inches, pre-pregnancy weight, delivery weight, weight gain, and body mass index (BMI) computed as pre-pregnancy weight divided by height squared, multiplied by 703.

                Infant birth outcomes include gestational age in two formulations: the obstetric estimate (the NCHS standard since 2014) and the combined gestation measure based on the date of last normal menses. Additional outcome fields cover birth weight in grams, infant sex, birth facility type, and clinical status at the time of hospital discharge. The dataset also includes a full section on congenital anomalies of the newborn, in which 12 specific conditions are recorded via checkbox on the Facility Worksheet. This section is the direct source of the prediction target.
                """
            )

            st.markdown("#### Prediction Target")
            st.markdown(
                """
                The prediction target is `cleft_case`, a binary variable derived from two fields in the congenital anomalies section of the birth certificate: `CA_CLEFT` (cleft lip with or without cleft palate, file position 550) and `CA_CLPAL` (cleft palate alone, file position 551). A record is coded as a positive case (`cleft_case = 1`) if either field carries a value of `Y`; otherwise it is coded as `0`. Together, these two fields capture the full clinical spectrum of orofacial clefting conditions reported at birth.

                An important data quality consideration is that congenital anomalies have historically been under-reported on birth certificates. The NCHS User Guide to the 2024 file explicitly notes this limitation, explaining that anomalies other than those that are highly visible or severe may be difficult to detect within the short interval between birth and certificate completion. The 2003 certificate revision attempted to address this by restricting the anomaly checklist to conditions diagnosable within 24 hours of birth using widely available techniques, which includes both forms of clefting. The practical implication for this project is that the true prevalence of cleft conditions may be modestly higher than what the data records, and any model trained on these labels will reflect the completeness of birth certificate reporting rather than clinical ground truth.
                """
            )

            dataset_metrics = pd.DataFrame(
                [
                    {"Metric": "Total U.S. birth records (2024)", "Value": "3,638,436"},
                    {"Metric": "Total features parsed for modeling", "Value": "80+"},
                    {"Metric": "Target variable", "Value": "cleft_case (1 cleft, 0 neither)"},
                    {"Metric": "Positive class rate", "Value": "0.069673% (~0.6967 per 1,000)"},
                    {"Metric": "Class imbalance ratio (0:1)", "Value": "1434.28:1"},
                    {"Metric": "Registration completeness", "Value": ">99% U.S. births captured"},
                    {
                        "Metric": "Leakage-safe feature set",
                        "Value": "Pre-/at-birth only; excludes post-delivery outcomes",
                    },
                ]
            )
            st.table(dataset_metrics)

            with st.expander("Full Data Dictionary", expanded=False):
                dd = data.get("data_dictionary")
                if dd is not None and len(dd) > 0:
                    st.caption(f"Loaded exported data dictionary with {len(dd):,} rows.")
                    st.dataframe(dd, use_container_width=True, height=600)
                else:
                    st.error(
                        "Full data dictionary artifact is missing. "
                        "Re-run the notebook export step to write `streamlit_artifacts/data_dictionary.pkl`."
                    )

        with st.expander("Why This Problem Matters", expanded=False):
            st.markdown(
                """
                Cleft lip and palate is among the most common congenital structural birth defects in the United States, affecting more than 6,000 infants each year. Despite its prevalence, it remains a condition that can go undetected until birth in many cases, delaying the mobilization of highly specialized multidisciplinary care teams that these infants require from the earliest weeks of life. Surgical repair, speech therapy, orthodontic treatment, audiological monitoring, and feeding support are all typically needed beginning in infancy and continuing for years.

                Accurate prediction models built on data that is already collected as part of standard birth certificate processing offer an opportunity to close this gap without requiring any additional clinical burden. By identifying which birth profiles carry elevated risk, health systems, craniofacial centers, and public health agencies can act earlier and more precisely in several ways:

                - Resource planning and staffing: Craniofacial surgical centers can anticipate seasonal and regional demand for procedures and specialist consultations based on predicted case volumes by geography and birth month.
                - Health equity: Cleft prevalence rates are not uniform across racial, ethnic, and socioeconomic groups. A model grounded in demographic and behavioral data can surface disparities that might otherwise remain invisible in aggregate statistics, enabling more targeted outreach to underserved populations.
                - Preventive care communication: Several modifiable risk factors associated with cleft conditions, including tobacco use and inadequate prenatal nutrition, are already known to clinicians. Embedding predictive risk signals into prenatal care workflows could prompt timely counseling for higher-risk expectant mothers.
                - Policy and funding allocation: At the population level, aggregating individual predicted probabilities across expected births can generate forward-looking estimates of annual case burden, supporting federal and state-level maternal and child health budget decisions.

                The core value proposition:
                This is not just a statistical exercise. The data used to train these models is already collected on every U.S. birth certificate. What this project demonstrates is that latent predictive signal exists within existing administrative data. If surfaced responsibly, that signal could translate directly into earlier interventions, better resource allocation, and improved outcomes for thousands of infants each year.
                """
            )

        with st.expander("Analytical Approach", expanded=False):
            st.markdown(
                """
                The project followed a structured three-part analytical pipeline covering descriptive analytics, predictive modeling, and model explainability.

                **Descriptive Analytics**
                The exploratory analysis revealed several patterns in cleft risk across the dataset. Prevalence was not uniform across maternal age groups, with certain age ranges showing meaningfully higher case rates per 10,000 births. Cigarette smoking across pregnancy trimesters was associated with elevated cleft prevalence, consistent with published epidemiological literature. Maternal education level showed a gradient pattern, with lower attainment associated with higher observed rates. A cross-tabulation of infant sex and WIC enrollment status revealed interaction effects that would not be visible from looking at either variable in isolation.

                **Predictive Modeling**
                Five classification models were trained and evaluated on a 70/30 stratified train-test split using the full feature set: Logistic Regression as an interpretable baseline, a Decision Tree tuned via 5-fold grid search, a Random Forest ensemble, an XGBoost gradient-boosted tree model, and a two-hidden-layer neural network (MLP) built in Keras. All models used class reweighting or scale adjustment to address the severe class imbalance. Threshold selection was performed on a held-out validation split using two operating points: an F1-optimized threshold that balances precision and recall, and a recall-first threshold designed to maximize sensitivity to true cases at the cost of accepting more false positives. The latter is particularly relevant for screening applications where missing a case is more costly than a false alarm.

                **Explainability**
                SHAP analysis was applied to the best-performing tree model to identify which features most strongly influenced individual predictions. A leakage-safe version of the model was also trained using only features available before or at birth, excluding post-delivery clinical fields such as birth weight, infant living status, and breastfeeding at discharge. This version is more appropriate for real-world deployment because it reflects only information a clinician or health system would have available at or before the time of birth.
                """
            )

        with st.expander("Key Findings", expanded=True):
            key_findings = pd.DataFrame(
                [
                    {"Metric": "Best models by PR-AUC", "Value": "XGBoost and Random Forest"},
                    {
                        "Metric": "Primary evaluation metric",
                        "Value": "PR-AUC and Recall; standard accuracy is misleading for rare-event tasks",
                    },
                    {
                        "Metric": "Recall-first threshold",
                        "Value": "Targets approximately 80% recall; prioritizes not missing true cleft cases",
                    },
                    {
                        "Metric": "Leakage-safe model result",
                        "Value": "Comparable performance using only pre-birth features; suitable for deployment",
                    },
                    {
                        "Metric": "Top SHAP predictors",
                        "Value": "Maternal age group, tobacco use, WIC status, prenatal care timing, race/Hispanic origin",
                    },
                    {
                        "Metric": "Class imbalance approach",
                        "Value": "Balanced class weights, stratified splits, and threshold optimization",
                    },
                ]
            )
            st.table(key_findings)

            st.markdown(
                """
                A consistent and important finding across all models is that raw accuracy is a deeply misleading metric for this problem. Because non-cases account for over 99.8% of births, a model that simply predicts everyone as negative achieves over 99% accuracy while identifying zero actual cleft cases. The appropriate metrics for this task are PR-AUC, recall, and F1. The specific threshold applied at inference time must be driven by the cost asymmetry of the deployment context: for broad screening, missing a case carries higher costs than a false positive; for resource-intensive clinical referral pathways, precision becomes more important.

                The leakage-safe model's performance being comparable to the full model is a particularly meaningful finding. It demonstrates that the predictive signal in this dataset is not dependent on post-birth clinical measurements. The risk profile of a birth can be meaningfully estimated from information that is available before or at the moment of delivery, which is precisely when early intervention and resource preparation are most impactful.

                **Bottom line for non-technical stakeholders:**
                Publicly available U.S. birth certificate data already contains meaningful predictive signal for identifying births at elevated risk of cleft conditions, with no new data collection required. The models built here are not a clinical diagnosis tool. They represent a viable foundation for a risk-stratification layer that could be embedded in existing maternal health workflows to improve preparedness, resource allocation, and health equity outcomes.
                """
            )

    with tab2:
        vivid_palette = ["#0078D4", "#E81123", "#FF8C00", "#00B7C3", "#7B61FF", "#00CC6A", "#C239B3", "#F7630C"]
        sns.set_theme(
            style="whitegrid",
            context="notebook",
            palette=vivid_palette,
            rc={
                "font.family": "Segoe UI",
                "font.weight": "semibold",
                "axes.labelsize": 11,
                "axes.titlesize": 14,
                "axes.labelweight": "semibold",
                "axes.titleweight": "semibold",
                "xtick.labelsize": 10,
                "ytick.labelsize": 10,
                "legend.fontsize": 10,
                "legend.title_fontsize": 11,
            },
        )
        plt.rcParams["font.family"] = "Segoe UI"
        plt.rcParams["font.weight"] = "semibold"
        plt.rcParams["axes.labelweight"] = "semibold"
        plt.rcParams["axes.titleweight"] = "semibold"

        eda = data["eda_df"].copy()

        for col in [
            "MAGER",
            "PREVIS",
            "COMBGEST",
            "BMI",
            "WTGAIN",
            "FAGECOMB",
            "OEGest_Comb",
            "M_Ht_In",
            "PRECARE",
            "PRECARE5",
            "cleft_case",
        ]:
            if col in eda.columns:
                eda[col] = pd.to_numeric(eda[col], errors="coerce")

        for col in ["MAGER14", "CIG_REC", "MEDUC", "SEX", "WIC"]:
            if col in eda.columns:
                eda[col] = eda[col].astype("string").str.strip()

        eda["cleft_case"] = pd.to_numeric(eda["cleft_case"], errors="coerce").fillna(0).astype(int)

        st.markdown("### Target distribution")
        target_counts = eda["cleft_case"].value_counts().sort_index()
        label_map = {0: "No cleft case (0)", 1: "Cleft case (1)"}
        plot_target = pd.DataFrame(
            {
                "class": [label_map.get(i, str(i)) for i in target_counts.index],
                "count": target_counts.values,
            }
        )

        fig, ax = plt.subplots(figsize=(5.2, 2.9), dpi=120)
        sns.barplot(data=plot_target, x="class", y="count", palette=["#4C72B0", "#C44E52"], ax=ax)
        ax.set_title("Target Distribution: Cleft Case")
        ax.set_xlabel("Class")
        ax.set_ylabel("Count")
        st.pyplot(fig, use_container_width=False)
        plt.close(fig)

        pos_rate = float(eda["cleft_case"].mean() * 100)
        imbalance_ratio = float(target_counts.get(0, 0) / max(target_counts.get(1, 1), 1))
        st.markdown(
            f"**Observation:** Positive class prevalence is `{pos_rate:.4f}%` and the dataset is highly imbalanced. "
            f"Class ratio (0:1) is approximately `{imbalance_ratio:.1f}:1`."
        )
        st.markdown(
            "**Conclusion:** Use stratified splits and recall/PR-AUC-focused evaluation; raw accuracy is not reliable for this rare-event target."
        )

        age_prev = prevalence_table(eda, "MAGER14", min_n=200) if "MAGER14" in eda.columns else pd.DataFrame()
        smoke_prev = prevalence_table(eda, "CIG_REC", min_n=200) if "CIG_REC" in eda.columns else pd.DataFrame()
        educ_prev = prevalence_table(eda, "MEDUC", min_n=200) if "MEDUC" in eda.columns else pd.DataFrame()

        if not age_prev.empty:
            fig, ax = plt.subplots(figsize=(5.2, 2.9), dpi=120)
            sns.barplot(data=age_prev.sort_values("MAGER14"), x="MAGER14", y="prev_per_10k", color="#4C72B0", ax=ax)
            ax.set_title(f"Cleft prevalence by {friendly_name('MAGER14')} (per 10,000)")
            ax.set_xlabel(friendly_name("MAGER14"))
            ax.set_ylabel("Cases per 10,000 births")
            st.pyplot(fig, use_container_width=False)
            plt.close(fig)
            st.markdown(
                f"**Observation:** Prevalence varies across maternal-age groups (range `{age_prev['prev_per_10k'].min():.2f}` to `{age_prev['prev_per_10k'].max():.2f}` per 10,000)."
            )
            st.markdown("**Conclusion:** Maternal age profile contributes useful risk stratification signal.")

        if not smoke_prev.empty:
            fig, ax = plt.subplots(figsize=(5.2, 2.9), dpi=120)
            sns.barplot(data=smoke_prev, x="CIG_REC", y="prev_per_10k", color="#C44E52", ax=ax)
            ax.set_title(f"Cleft prevalence by {friendly_name('CIG_REC')} (per 10,000)")
            ax.set_xlabel(friendly_name("CIG_REC"))
            ax.set_ylabel("Cases per 10,000 births")
            st.pyplot(fig, use_container_width=False)
            plt.close(fig)
            st.caption(
                "Interpretation note: `N` means *no cigarette use recorded* in this field, not zero biological risk. "
                "A non-zero bar under `N` is expected because cleft outcomes are multifactorial and can occur without reported smoking. "
                "The chart compares relative prevalence between groups; it does not imply smoking is the only cause."
            )
            st.markdown(
                "**Observation:** Smoking-recode groups show prevalence differences, while the `N` group still has some cases because risk exists even without reported smoking."
            )
            st.markdown(
                "**Conclusion:** Tobacco-related behavior remains an important predictor, but it should be interpreted as one contributor among multiple maternal, fetal, and socioeconomic factors."
            )

        if not educ_prev.empty:
            fig, ax = plt.subplots(figsize=(5.2, 2.9), dpi=120)
            sns.barplot(data=educ_prev.sort_values("MEDUC"), x="MEDUC", y="prev_per_10k", color="#55A868", ax=ax)
            ax.set_title(f"Cleft prevalence by {friendly_name('MEDUC')} (per 10,000)")
            ax.set_xlabel(friendly_name("MEDUC"))
            ax.set_ylabel("Cases per 10,000 births")
            st.pyplot(fig, use_container_width=False)
            plt.close(fig)
            st.markdown("**Observation:** Maternal education groups show a non-flat gradient in prevalence.")
            st.markdown("**Conclusion:** Socioeconomic context contributes meaningful modeling signal.")

        heat_df = (
            eda.dropna(subset=["SEX", "WIC", "cleft_case"])
            .groupby(["SEX", "WIC"], as_index=False)
            .agg(cases=("cleft_case", "sum"), births=("cleft_case", "count"))
        )
        heat_df["prev_per_10k"] = heat_df["cases"] / heat_df["births"] * 10000
        heat_pivot = heat_df.pivot(index="SEX", columns="WIC", values="prev_per_10k") if not heat_df.empty else pd.DataFrame()

        if not heat_pivot.empty:
            fig, ax = plt.subplots(figsize=(7.2, 5.4), dpi=120)
            sns.heatmap(heat_pivot, annot=True, fmt=".2f", cmap="mako", ax=ax)
            ax.set_title(f"Cleft prevalence heatmap ({friendly_name('SEX')} x {friendly_name('WIC')})")
            ax.set_xlabel(friendly_name("WIC"))
            ax.set_ylabel(friendly_name("SEX"))
            ax.tick_params(axis="x", rotation=0)
            fig.subplots_adjust(left=0.20, right=0.97, bottom=0.16, top=0.90)
            st.pyplot(fig, use_container_width=False)
            plt.close(fig)
            st.markdown("**Observation:** Joint SEX x WIC strata are heterogeneous and not captured by single-variable views.")
            st.markdown("**Conclusion:** Interaction-capable models are justified for this problem.")

        st.markdown("### Additional Visualizations")
        st.markdown("### Visualization Checklist")

        pos_df = eda[eda["cleft_case"] == 1].copy()
        neg_pool = eda[eda["cleft_case"] == 0].copy()
        neg_n = min(len(neg_pool), max(len(pos_df) * 10, 1), 50000)
        neg_df = neg_pool.sample(neg_n, random_state=42) if neg_n > 0 else neg_pool.head(0)
        vis_df = pd.concat([pos_df, neg_df], axis=0).copy()
        vis_df["target_label"] = vis_df["cleft_case"].map({0: "No cleft case", 1: "Cleft case"})

        # Histogram by target
        hist_df = vis_df.dropna(subset=["COMBGEST", "target_label"]) if "COMBGEST" in vis_df.columns else pd.DataFrame()
        if not hist_df.empty:
            fig, ax = plt.subplots(figsize=(5.2, 2.9), dpi=120)
            hue_palette = {"No cleft case": "#00B7C3", "Cleft case": "#E81123"}
            sns.histplot(
                data=hist_df,
                x="COMBGEST",
                hue="target_label",
                palette=hue_palette,
                bins=35,
                stat="density",
                common_norm=False,
                alpha=0.55,
                ax=ax,
            )
            ax.set_title(f"{friendly_name('COMBGEST')} Distribution by Cleft Status")
            ax.set_xlabel(friendly_name("COMBGEST"))
            ax.set_ylabel("Density")
            st.pyplot(fig, use_container_width=False)
            plt.close(fig)
            st.markdown("**Observation:** Gestation distributions differ by cleft status, but overlap remains substantial.")
            st.markdown("**Conclusion:** Gestation contributes signal but cannot classify cases alone.")

        # Boxplot by target
        box_df = vis_df.dropna(subset=["BMI", "target_label"]) if "BMI" in vis_df.columns else pd.DataFrame()
        if not box_df.empty:
            fig, ax = plt.subplots(figsize=(5.2, 2.9), dpi=120)
            sns.boxplot(
                data=box_df,
                x="target_label",
                y="BMI",
                hue="target_label",
                dodge=False,
                palette=["#FF8C00", "#7B61FF"],
                legend=False,
                ax=ax,
            )
            ax.set_title(f"{friendly_name('BMI')} by Cleft Status")
            ax.set_xlabel("Cleft status")
            ax.set_ylabel(friendly_name("BMI"))
            st.pyplot(fig, use_container_width=False)
            plt.close(fig)
            st.markdown("**Observation:** BMI distributions overlap strongly across classes.")
            st.markdown("**Conclusion:** BMI is likely a weak standalone predictor but may help in multivariate models.")

        # Violin by category
        meduc_map = {
            "1": "8th grade or less",
            "2": "9th-12th, no diploma",
            "3": "High school graduate/GED",
            "4": "Some college, no degree",
            "5": "Associate degree",
            "6": "Bachelor's degree",
            "7": "Master's degree",
            "8": "Doctorate/professional degree",
            "9": "Unknown",
        }
        violin_df = vis_df.dropna(subset=["MEDUC", "COMBGEST"]) if set(["MEDUC", "COMBGEST"]).issubset(vis_df.columns) else pd.DataFrame()
        if not violin_df.empty:
            top_meduc = violin_df["MEDUC"].value_counts().head(6).index.tolist()
            violin_df = violin_df[violin_df["MEDUC"].isin(top_meduc)].copy()
            violin_df["MEDUC_label"] = violin_df["MEDUC"].map(lambda x: f"{x}: {meduc_map.get(str(x), 'Unknown')}")
            order_labels = [
                f"{c}: {meduc_map.get(str(c), 'Unknown')}"
                for c in sorted(top_meduc, key=lambda x: int(x) if str(x).isdigit() else 99)
            ]

            fig, ax = plt.subplots(figsize=(5.5, 3.2), dpi=120)
            sns.violinplot(
                data=violin_df,
                x="MEDUC_label",
                y="COMBGEST",
                order=order_labels,
                inner="quartile",
                color="#C239B3",
                saturation=1,
                ax=ax,
            )
            ax.set_title(f"{friendly_name('COMBGEST')} across {friendly_name('MEDUC')}")
            ax.set_xlabel(friendly_name("MEDUC"))
            ax.set_ylabel(friendly_name("COMBGEST"))
            ax.tick_params(axis="x", rotation=20)
            st.pyplot(fig, use_container_width=False)
            plt.close(fig)
            st.markdown("**Observation:** Gestation distributions vary across maternal education categories.")
            st.markdown("**Conclusion:** Subgroup heterogeneity supports retaining sociodemographic features.")

        # Pair plot
        pair_cols = [c for c in ["COMBGEST", "MAGER", "BMI", "PREVIS", "cleft_case"] if c in vis_df.columns]
        pair_df = vis_df[pair_cols].dropna().copy() if len(pair_cols) == 5 else pd.DataFrame()
        if not pair_df.empty:
            pair_df = pair_df.sample(min(1000, len(pair_df)), random_state=42)
            pair_df["target_label"] = pair_df["cleft_case"].map({0: "No cleft case", 1: "Cleft case"})
            rename_map = {c: friendly_name(c) for c in pair_cols if c != "cleft_case"}
            pair_df = pair_df.rename(columns=rename_map)
            pair_vars = [rename_map[c] for c in pair_cols if c != "cleft_case"]

            g = sns.pairplot(
                pair_df,
                vars=pair_vars,
                hue="target_label",
                palette={"No cleft case": "#00B7C3", "Cleft case": "#E81123"},
                corner=True,
                plot_kws={"alpha": 0.35, "s": 10},
                diag_kws={"common_norm": False},
                height=1.80,
                aspect=1.0,
            )
            g.fig.suptitle(
                "Pair Plot of Key Numerical Features by Cleft Status",
                y=1.03,
                fontfamily="Segoe UI",
                fontweight="semibold",
                fontsize=13,
            )
            for ax_row in g.axes:
                for ax_i in ax_row:
                    if ax_i is not None:
                        ax_i.tick_params(axis="x", labelrotation=28, labelsize=8)
                        ax_i.tick_params(axis="y", labelsize=8)
                        ax_i.xaxis.label.set_fontsize(9)
                        ax_i.yaxis.label.set_fontsize(9)
                        ax_i.xaxis.label.set_weight("semibold")
                        ax_i.yaxis.label.set_weight("semibold")
            g.fig.subplots_adjust(bottom=0.18, top=0.93)
            st.pyplot(g.fig, use_container_width=False)
            plt.close(g.fig)
            st.markdown("**Observation:** Pairwise relationships show broad class overlap, expected for rare outcomes.")
            st.markdown("**Conclusion:** Predictive power comes from combining weak patterns and interactions.")

        # Bar chart categorical vs target
        bar_df = (
            eda.dropna(subset=["WIC", "cleft_case"])
            .groupby("WIC", as_index=False)
            .agg(cases=("cleft_case", "sum"), births=("cleft_case", "count"))
        )
        if not bar_df.empty:
            bar_df["prev_per_10k"] = bar_df["cases"] / bar_df["births"] * 10000
            fig, ax = plt.subplots(figsize=(5.2, 2.9), dpi=120)
            sns.barplot(data=bar_df, x="WIC", y="prev_per_10k", color="#00A3FF", ax=ax)
            ax.set_title(f"Cleft Prevalence by {friendly_name('WIC')}")
            ax.set_xlabel(friendly_name("WIC"))
            ax.set_ylabel("Cases per 10,000 births")
            st.pyplot(fig, use_container_width=False)
            plt.close(fig)
            st.markdown("**Observation:** Cleft prevalence per 10,000 differs by WIC category.")
            st.markdown("**Conclusion:** Access/eligibility-related categorical variables carry useful stratification signal.")

        st.markdown("### Correlation Heatmap")
        corr_features = [
            c
            for c in [
                "cleft_case",
                "COMBGEST",
                "OEGest_Comb",
                "MAGER",
                "FAGECOMB",
                "PREVIS",
                "WTGAIN",
                "M_Ht_In",
                "BMI",
                "PRECARE",
                "PRECARE5",
            ]
            if c in eda.columns
        ]

        corr_data = eda[corr_features].apply(pd.to_numeric, errors="coerce")
        corr_matrix = corr_data.corr()

        named = corr_matrix.copy()
        named.columns = [friendly_name(c) for c in corr_matrix.columns]
        named.index = [friendly_name(c) for c in corr_matrix.index]

        fig, ax = plt.subplots(figsize=(8.2, 6.4), dpi=120)
        sns.heatmap(
            named,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            center=0,
            linewidths=0.5,
            annot_kws={"size": 8},
            ax=ax,
        )
        ax.set_title("Correlation Heatmap (Cleft Target + Numeric Predictors)")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
        ax.tick_params(axis="x", labelsize=9)
        ax.tick_params(axis="y", labelsize=9)
        st.pyplot(fig, use_container_width=False)
        plt.close(fig)

        pairs = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)).stack().sort_values(
            key=lambda s: s.abs(), ascending=False
        )
        top5 = pairs.head(5).reset_index()
        top5.columns = ["feature_1", "feature_2", "correlation"]
        top5["feature_1_name"] = top5["feature_1"].map(friendly_name)
        top5["feature_2_name"] = top5["feature_2"].map(friendly_name)
        st.dataframe(top5[["feature_1_name", "feature_2_name", "correlation"]], use_container_width=True)

        target_corr = corr_matrix["cleft_case"].drop("cleft_case").sort_values(key=lambda s: s.abs(), ascending=False).head(5)
        target_corr_df = target_corr.reset_index()
        target_corr_df.columns = ["feature", "corr_with_cleft_case"]
        target_corr_df["feature_name"] = target_corr_df["feature"].map(friendly_name)
        st.dataframe(target_corr_df[["feature_name", "corr_with_cleft_case"]], use_container_width=True)

        if not top5.empty:
            top_pair = top5.iloc[0]
            st.markdown(
                f"**Observation:** The strongest pairwise relationship is between `{top_pair['feature_1_name']}` and `{top_pair['feature_2_name']}` (r={top_pair['correlation']:.2f})."
            )
        if not target_corr_df.empty:
            top_target = target_corr_df.iloc[0]
            st.markdown(
                f"**Observation:** The strongest linear association with cleft target is `{top_target['feature_name']}` (r={top_target['corr_with_cleft_case']:.3f}), but overall magnitudes remain modest."
            )
        st.markdown(
            "**Conclusion:** Multivariate and non-linear models are necessary because no single variable shows a dominant linear relationship with the rare-event target."
        )

    with tab3:
        st.subheader("Model Comparison and Evaluation")
        results_df = data["results_df"].copy()
        st.dataframe(results_df, use_container_width=True)

        fig, ax = plt.subplots(figsize=(9, 4))
        labels = results_df["model"] + "\n" + results_df.get("operating_point", "Default-0.5")
        sns.barplot(x=labels, y=results_df["f1"], ax=ax)
        ax.set_title("F1 Comparison")
        ax.set_ylabel("F1")
        ax.tick_params(axis="x", rotation=20)
        st.pyplot(fig)

        st.subheader("ROC Curves")
        fig, ax = plt.subplots(figsize=(8, 5))
        x_full = data["X_test_full"]
        y_full = pd.to_numeric(data["y_test_full"], errors="coerce").astype(int)
        x_safe = data["X_test_safe"]
        y_safe = pd.to_numeric(data["y_test_safe"], errors="coerce").astype(int)

        for model_name, meta in data["model_registry"].items():
            if model_name not in models:
                continue
            if meta.get("feature_space") == "safe":
                x_eval, y_eval = x_safe, y_safe
            else:
                x_eval, y_eval = x_full, y_full
            y_prob = predict_prob(model_name, meta, models[model_name], x_eval)
            try:
                from sklearn.metrics import roc_curve, roc_auc_score
                fpr, tpr, _ = roc_curve(y_eval, y_prob)
                auc = roc_auc_score(y_eval, y_prob)
                ax.plot(fpr, tpr, label=f"{model_name} (AUC={auc:.3f})")
            except Exception:
                pass

        ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
        ax.set_title("ROC Curves by Model")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend(fontsize=8)
        st.pyplot(fig)

        st.subheader("Best Hyperparameters")
        st.json(data["best_params"])

    with tab4:
        st.subheader("SHAP Explainability")

        def render_shap_section(title, model_name, x_eval):
            st.markdown(f"### {title}")
            if model_name not in models:
                st.warning(f"Model not available: {model_name}")
                return

            tree_pipe = models[model_name]
            bundle = compute_shap_bundle(tree_pipe, x_eval, sample_n=400)

            st.caption(f"Model used: {model_name} | SHAP sample rows: {len(bundle['x_shap_raw'])}")

            plt.figure(figsize=(7.0, 4.2))
            shap.summary_plot(
                bundle["sv"][:, bundle["top_idx"]],
                features=bundle["x_top"],
                feature_names=bundle["top_display_names"],
                max_display=min(20, len(bundle["top_idx"]),),
                plot_size=(7.0, 4.2),
                show=False,
            )
            fig = plt.gcf()
            for ax in fig.axes:
                ax.tick_params(axis="x", labelsize=7, rotation=0)
                ax.tick_params(axis="y", labelsize=7)
                ax.xaxis.label.set_size(8)
                ax.yaxis.label.set_size(8)
                ax.grid(True, color="#E8E8E8", linewidth=0.5, alpha=0.5)
            st.pyplot(fig)
            plt.close(fig)

            plt.figure(figsize=(7.0, 4.2))
            shap.summary_plot(
                bundle["sv"][:, bundle["top_idx"]],
                features=bundle["x_top"],
                feature_names=bundle["top_display_names"],
                plot_type="bar",
                max_display=min(20, len(bundle["top_idx"]),),
                plot_size=(7.0, 4.2),
                show=False,
            )
            fig = plt.gcf()
            for ax in fig.axes:
                ax.tick_params(axis="x", labelsize=7, rotation=0)
                ax.tick_params(axis="y", labelsize=7)
                ax.xaxis.label.set_size(8)
                ax.yaxis.label.set_size(8)
                ax.grid(True, color="#E8E8E8", linewidth=0.5, alpha=0.5)
            st.pyplot(fig)
            plt.close(fig)

            sample_pred_prob = tree_pipe.predict_proba(bundle["x_shap_raw"])[:, 1]
            waterfall_idx = int(np.argmax(sample_pred_prob))

            row = bundle["x_trans"][waterfall_idx]
            if hasattr(row, "toarray"):
                row = row.toarray().ravel()
            else:
                row = np.asarray(row).ravel()

            explanation = shap.Explanation(
                values=bundle["sv"][waterfall_idx],
                base_values=bundle["base_val"],
                data=row,
                feature_names=bundle["feature_display_names"],
            )
            fig = plt.figure(figsize=(7.0, 4.2))
            shap.plots.waterfall(explanation, max_display=15, show=False)
            for ax in fig.axes:
                ax.tick_params(axis="x", labelsize=7, rotation=0)
                ax.tick_params(axis="y", labelsize=7)
                ax.xaxis.label.set_size(8)
                ax.yaxis.label.set_size(8)
                ax.grid(True, color="#E8E8E8", linewidth=0.5, alpha=0.5)
            st.pyplot(fig)
            plt.close(fig)

            st.caption(
                f"Selected high-risk case index: {waterfall_idx} | "
                f"predicted risk: {sample_pred_prob[waterfall_idx]:.4f}"
            )

            interp_rows = []
            top_k = min(10, len(bundle["top_idx"]))
            for j in range(top_k):
                col_idx = int(bundle["top_idx"][j])
                shap_col = bundle["sv"][:, col_idx]

                f_col = bundle["x_trans"][:, col_idx]
                if hasattr(f_col, "toarray"):
                    f_col = f_col.toarray().ravel()
                else:
                    f_col = np.asarray(f_col).ravel()

                if np.nanstd(f_col) > 0 and np.nanstd(shap_col) > 0:
                    corr = np.corrcoef(f_col, shap_col)[0, 1]
                else:
                    corr = np.nan

                if np.isnan(corr):
                    direction = "mixed/flat effect"
                elif corr > 0.05:
                    direction = "higher feature values tend to increase predicted risk"
                elif corr < -0.05:
                    direction = "higher feature values tend to decrease predicted risk"
                else:
                    direction = "weak or mixed direction"

                interp_rows.append(
                    {
                        "feature": bundle["feature_display_names"][col_idx],
                        "mean_abs_shap": float(np.mean(np.abs(shap_col))),
                        "mean_shap": float(np.mean(shap_col)),
                        "value_shap_corr": float(corr) if not np.isnan(corr) else np.nan,
                        "directional_interpretation": direction,
                    }
                )

            interp_df = pd.DataFrame(interp_rows).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
            st.dataframe(interp_df, use_container_width=True)
            st.markdown("- Strongest impact features are ranked by mean absolute SHAP value.")
            st.markdown("- Positive SHAP values push predictions toward higher cleft-case risk; negative values push toward lower risk.")
            st.markdown("- Directional interpretation summarizes whether higher feature values generally raise or lower risk.")

        full_tree_name = None
        tree_priority = ["Random Forest", "XGBoost"]
        tree_rows = data["results_df"][data["results_df"].get("model", pd.Series(dtype=str)).isin(tree_priority)].copy()
        if not tree_rows.empty and "pr_auc" in tree_rows.columns:
            ranked = (
                tree_rows.groupby("model", as_index=False)
                .agg(best_pr_auc=("pr_auc", "max"), best_f1=("f1", "max"), best_auc=("roc_auc", "max"))
                .sort_values(["best_pr_auc", "best_f1", "best_auc"], ascending=False)
            )
            for m in ranked["model"].tolist():
                if m in models:
                    full_tree_name = m
                    break

        if full_tree_name is None:
            for fallback in ["Random Forest", "XGBoost"]:
                if fallback in models:
                    full_tree_name = fallback
                    break

        full_tree_candidates = [m for m in ["Random Forest", "XGBoost"] if m in models]
        if full_tree_candidates:
            default_idx = full_tree_candidates.index(full_tree_name) if full_tree_name in full_tree_candidates else 0
            selected_full_shap_model = st.selectbox(
                "Select SHAP model (full feature space)",
                options=full_tree_candidates,
                index=default_idx,
            )
            st.caption(
                "Tip: choose the same model shown in your notebook SHAP section to align top variables exactly."
            )
            render_shap_section(
                "3.1 SHAP Analysis (Selected Tree Model)",
                selected_full_shap_model,
                data["X_test_full"].copy(),
            )
        else:
            st.warning("No full-feature tree model available for SHAP visualizations.")

        if "Leakage-safe Random Forest" in models:
            render_shap_section("3.2 Leakage-safe SHAP (Pre-birth Features Only)", "Leakage-safe Random Forest", data["X_test_safe"].copy())
        else:
            st.info("Leakage-safe Random Forest artifacts are not available in this build.")

        st.subheader("Interactive Prediction")

        model_options = list(data["model_registry"].keys())
        if not model_options:
            st.error("No models are registered. Verify artifact files and dependencies, then reload the app.")
            return

        model_name = st.selectbox("Select model", options=model_options)
        if model_name not in models:
            st.error(
                f"`{model_name}` is registered but could not be loaded in this runtime. "
                "Verify required dependencies are installed (TensorFlow/Keras for MLP) and restart the app."
            )
            return

        model_meta = data["model_registry"][model_name]

        if model_meta.get("feature_space") == "safe":
            base = data["interactive_schema"].get("safe_baseline", {})
            schema = data["interactive_schema"].get("safe_schema", {})
            eval_df = data["X_test_safe"]
        else:
            base = data["interactive_schema"].get("full_baseline", {})
            schema = data["interactive_schema"].get("full_schema", {})
            eval_df = data["X_test_full"]

        input_features = data["interactive_schema"].get("interactive_features", [])
        custom = dict(base)

        cols = st.columns(3)
        idx = 0
        for f in input_features:
            if f not in schema:
                continue
            s = schema[f]
            col = cols[idx % 3]
            feature_label = friendly_name(f)
            with col:
                if s.get("type") == "numeric":
                    fmin = float(s.get("min", s.get("default", 0.0)))
                    fmax = float(s.get("max", s.get("default", 1.0)))
                    fdef = float(s.get("default", (fmin + fmax) / 2))
                    if fmax <= fmin:
                        fmax = fmin + 1.0
                    custom[f] = st.slider(feature_label, min_value=fmin, max_value=fmax, value=fdef, key=f"slider_{f}")
                else:
                    opts = s.get("options", [str(s.get("default", "Unknown"))])
                    defv = str(s.get("default", opts[0]))
                    if defv not in opts:
                        opts = [defv] + opts
                    custom[f] = st.selectbox(feature_label, options=opts, index=opts.index(defv), key=f"select_{f}")
            idx += 1

        input_df = pd.DataFrame([custom])
        # Keep only columns expected by evaluation feature space.
        input_df = input_df.reindex(columns=eval_df.columns)

        y_prob = predict_prob(model_name, model_meta, models[model_name], input_df)[0]

        op_rows = data["results_df"][data["results_df"]["model"] == model_name]
        op_labels = ["Default-0.5"]
        if "operating_point" in op_rows.columns and len(op_rows) > 0:
            op_labels = op_rows["operating_point"].dropna().unique().tolist()
        selected_op = st.selectbox("Operating point", options=op_labels)

        threshold = 0.5
        if len(op_rows) > 0 and "threshold" in op_rows.columns and selected_op in op_rows.get("operating_point", pd.Series(dtype=str)).values:
            threshold = float(op_rows.loc[op_rows["operating_point"] == selected_op, "threshold"].iloc[0])

        pred_class = int(y_prob >= threshold)
        st.metric("Predicted probability", f"{y_prob:.4f}")
        st.metric("Predicted class", f"{pred_class} (threshold={threshold:.4f})")

        # SHAP waterfall for custom input when model is tree-based sklearn pipeline.
        if model_meta["type"] == "sklearn" and hasattr(models[model_name], "named_steps") and "model" in models[model_name].named_steps:
            est = models[model_name].named_steps["model"]
            pre = models[model_name].named_steps["preprocess"]
            if est.__class__.__name__ in {"RandomForestClassifier", "XGBClassifier", "DecisionTreeClassifier"}:
                x_one = pre.transform(input_df)
                exp = shap.TreeExplainer(est)
                sv_raw = exp.shap_values(x_one)
                if isinstance(sv_raw, list):
                    sv = sv_raw[1] if len(sv_raw) > 1 else sv_raw[0]
                elif isinstance(sv_raw, np.ndarray) and sv_raw.ndim == 3:
                    sv = sv_raw[:, :, 1]
                else:
                    sv = sv_raw

                base_val = exp.expected_value
                if isinstance(base_val, (list, np.ndarray)):
                    base_val = float(np.ravel(base_val)[-1])
                else:
                    base_val = float(base_val)

                row = x_one[0]
                if hasattr(row, "toarray"):
                    row = row.toarray().ravel()
                else:
                    row = np.asarray(row).ravel()

                categorical_source_cols = []
                for t_name, _, cols_used in getattr(pre, "transformers", []):
                    if t_name == "cat" and isinstance(cols_used, (list, tuple, np.ndarray, pd.Index)):
                        categorical_source_cols.extend([str(c) for c in cols_used])
                waterfall_feature_names = [
                    pretty_transformed_name(n, categorical_source_cols)
                    for n in pre.get_feature_names_out()
                ]

                explanation = shap.Explanation(
                    values=sv[0],
                    base_values=base_val,
                    data=row,
                    feature_names=waterfall_feature_names,
                )
                fig = plt.figure(figsize=(7.0, 4.2))
                shap.plots.waterfall(explanation, max_display=15, show=False)
                for ax in fig.axes:
                    ax.tick_params(axis="x", labelsize=7, rotation=0)
                    ax.tick_params(axis="y", labelsize=7)
                    ax.xaxis.label.set_size(8)
                    ax.yaxis.label.set_size(8)
                    ax.grid(True, color="#E8E8E8", linewidth=0.5, alpha=0.5)
                st.pyplot(fig)
                plt.close(fig)
            else:
                st.info("Waterfall SHAP is shown for tree-based models. Select Random Forest, XGBoost, Decision Tree, or Leakage-safe Random Forest.")
        else:
            st.info("Waterfall SHAP is unavailable for this model type.")

    with tab5:
        st.subheader("Causality Analysis (Observational)")
        st.warning(
            "This section evaluates association patterns and confounding checks. "
            "It does not prove causal effects because the data is observational and not randomized."
        )

        caus_df = data["eda_df"].copy()
        for col in ["cleft_case", "MAGER", "PREVIS", "COMBGEST", "BMI", "PRECARE", "PRECARE5"]:
            if col in caus_df.columns:
                caus_df[col] = pd.to_numeric(caus_df[col], errors="coerce")
        for col in ["CIG_REC", "MAGER14", "MEDUC", "WIC", "SEX"]:
            if col in caus_df.columns:
                caus_df[col] = caus_df[col].astype("string").str.strip()
        caus_df["cleft_case"] = pd.to_numeric(caus_df.get("cleft_case"), errors="coerce").fillna(0).astype(int)

        exposure_candidates = [c for c in ["CIG_REC", "WIC", "MEDUC", "MAGER14", "SEX", "PRECARE5"] if c in caus_df.columns]
        st.markdown("### 1) Crude Association: Select Exposure Variable")
        if exposure_candidates:
            exposure_var = st.selectbox(
                "Exposure variable",
                options=exposure_candidates,
                format_func=friendly_name,
            )
            crude_exposure = prevalence_table(caus_df, exposure_var, min_n=200)
            if not crude_exposure.empty:
                fig, ax = plt.subplots(figsize=(6.4, 3.8), dpi=120)
                sns.barplot(data=crude_exposure.sort_values("prev_per_10k", ascending=False), x=exposure_var, y="prev_per_10k", color="#0078D4", ax=ax)
                ax.set_title(f"Cleft prevalence by {friendly_name(exposure_var)}")
                ax.set_xlabel(friendly_name(exposure_var))
                ax.set_ylabel("Cases per 10,000 births")
                ax.tick_params(axis="x", rotation=20)
                st.pyplot(fig, use_container_width=False)
                plt.close(fig)

                ref_row = crude_exposure.sort_values("births", ascending=False).iloc[0]
                ref_prev = float(ref_row["prev_per_10k"])
                ref_level = str(ref_row[exposure_var])

                crude_out = crude_exposure.copy()
                crude_out["risk_diff_vs_ref"] = crude_out["prev_per_10k"] - ref_prev
                crude_out["risk_ratio_vs_ref"] = np.where(ref_prev > 0, crude_out["prev_per_10k"] / ref_prev, np.nan)
                st.dataframe(crude_out, use_container_width=True)
                st.caption(
                    f"Reference level is `{ref_level}` (largest birth count). Crude contrasts are association-only and may be confounded."
                )
            else:
                st.info("Not enough support for stable prevalence estimates on this exposure.")
        else:
            st.info("No supported exposure variables are available in the current artifact snapshot.")

        st.markdown("### 2) Stratified Check for Confounding")
        candidate_strata = [c for c in ["MAGER14", "MEDUC", "WIC", "PRECARE5", "SEX"] if c in caus_df.columns]
        if exposure_candidates and candidate_strata:
            valid_strata = [c for c in candidate_strata if c != exposure_var]
            if valid_strata:
                strat_col = st.selectbox("Potential confounder for stratification", options=valid_strata, format_func=friendly_name)
                exposure_levels = (
                    caus_df[exposure_var]
                    .dropna()
                    .astype(str)
                    .value_counts()
                    .head(6)
                    .index
                    .tolist()
                )

                if len(exposure_levels) >= 2:
                    col_a, col_b = st.columns(2)
                    with col_a:
                        level_a = st.selectbox("Compare level A", options=exposure_levels, index=0)
                    with col_b:
                        default_b = 1 if len(exposure_levels) > 1 else 0
                        level_b = st.selectbox("Compare level B", options=exposure_levels, index=default_b)

                    if level_a != level_b:
                        strat_base = caus_df.dropna(subset=[strat_col, exposure_var, "cleft_case"]).copy()
                        strat_base = strat_base[strat_base[exposure_var].astype(str).isin([level_a, level_b])].copy()

                        strat_agg = (
                            strat_base.groupby([strat_col, exposure_var], as_index=False)
                            .agg(cases=("cleft_case", "sum"), births=("cleft_case", "count"))
                        )
                        strat_agg["prev_per_10k"] = strat_agg["cases"] / strat_agg["births"] * 10000

                        p_prev = strat_agg.pivot(index=strat_col, columns=exposure_var, values="prev_per_10k")
                        p_births = strat_agg.pivot(index=strat_col, columns=exposure_var, values="births")

                        if set([level_a, level_b]).issubset(p_prev.columns) and set([level_a, level_b]).issubset(p_births.columns):
                            strat_effect = pd.DataFrame({
                                "stratum": p_prev.index.astype(str),
                                "prev_a_per_10k": p_prev[level_a].values,
                                "prev_b_per_10k": p_prev[level_b].values,
                                "births_a": p_births[level_a].values,
                                "births_b": p_births[level_b].values,
                            })
                            strat_effect = strat_effect.dropna(subset=["prev_a_per_10k", "prev_b_per_10k"]).copy()
                            strat_effect["risk_diff_per_10k"] = strat_effect["prev_a_per_10k"] - strat_effect["prev_b_per_10k"]
                            strat_effect["risk_ratio"] = np.where(
                                strat_effect["prev_b_per_10k"] > 0,
                                strat_effect["prev_a_per_10k"] / strat_effect["prev_b_per_10k"],
                                np.nan,
                            )
                            strat_effect["weight"] = strat_effect["births_a"] + strat_effect["births_b"]
                            strat_effect = strat_effect[strat_effect["weight"] >= 400].copy()

                            if not strat_effect.empty:
                                weighted_rd = float(np.average(strat_effect["risk_diff_per_10k"], weights=strat_effect["weight"]))
                                st.dataframe(strat_effect.sort_values("weight", ascending=False), use_container_width=True)

                                fig, ax = plt.subplots(figsize=(8.0, 4.8), dpi=120)
                                plot_eff = strat_effect.sort_values("risk_diff_per_10k", ascending=False).head(20)
                                sns.barplot(data=plot_eff, x="stratum", y="risk_diff_per_10k", color="#0078D4", ax=ax)
                                ax.axhline(0, color="gray", linestyle="--", linewidth=1)
                                ax.set_title(
                                    f"Stratum-specific Risk Difference: {friendly_name(exposure_var)} ({level_a} - {level_b})"
                                )
                                ax.set_xlabel(friendly_name(strat_col))
                                ax.set_ylabel("Risk difference per 10,000 births")
                                ax.tick_params(axis="x", rotation=30)
                                st.pyplot(fig, use_container_width=False)
                                plt.close(fig)

                                st.markdown(
                                    f"Weighted stratum-adjusted risk difference for `{level_a}` vs `{level_b}`: `{weighted_rd:.2f}` per 10,000 births."
                                )
                                st.caption(
                                    "If stratified effects differ from crude effects, confounding is likely. "
                                    "This improves interpretation but still does not prove causality."
                                )
                            else:
                                st.info("Not enough stratum support after quality filters to estimate stable adjusted contrasts.")
                        else:
                            st.info("Both selected exposure levels are not present across enough strata.")
                    else:
                        st.info("Choose two different exposure levels for comparison.")
                else:
                    st.info("Exposure has fewer than two supported levels for stratified comparison.")
            else:
                st.info("No valid stratification variable available after excluding the selected exposure.")
        else:
            st.info("Required variables for stratified confounding checks are not available.")

        st.markdown("### 3) Dose-Response Signal Check (Smoking-Specific)")
        cig_cols = [c for c in ["CIG_0", "CIG_1", "CIG_2", "CIG_3"] if c in caus_df.columns]
        if cig_cols:
            dose_df = caus_df.copy()
            for c in cig_cols:
                dose_df[c] = pd.to_numeric(dose_df[c], errors="coerce").fillna(0)
            dose_df["cigs_total"] = dose_df[cig_cols].sum(axis=1)
            dose_df["smoking_dose"] = pd.cut(
                dose_df["cigs_total"],
                bins=[-0.1, 0, 5, 10, np.inf],
                labels=["0", "1-5", "6-10", "11+"],
            )

            dose_prev = (
                dose_df.dropna(subset=["smoking_dose", "cleft_case"]) 
                .groupby("smoking_dose", as_index=False)
                .agg(cases=("cleft_case", "sum"), births=("cleft_case", "count"))
            )
            dose_prev["prev_per_10k"] = dose_prev["cases"] / dose_prev["births"] * 10000
            dose_prev = dose_prev[dose_prev["births"] >= 200].copy()

            if not dose_prev.empty:
                fig, ax = plt.subplots(figsize=(6.4, 3.8), dpi=120)
                sns.barplot(data=dose_prev, x="smoking_dose", y="prev_per_10k", color="#E81123", ax=ax)
                ax.set_title("Cleft prevalence by total smoking dose category")
                ax.set_xlabel("Total cigarettes category")
                ax.set_ylabel("Cases per 10,000 births")
                st.pyplot(fig, use_container_width=False)
                plt.close(fig)

                st.dataframe(dose_prev, use_container_width=True)
                st.caption(
                    "A dose-response pattern can strengthen causal plausibility, but still cannot establish causation without stronger design assumptions."
                )
            else:
                st.info("Insufficient support for stable dose categories in this snapshot.")
        else:
            st.info("Trimester smoking-count fields are not present in the current artifact snapshot.")

        st.markdown("### Practical Interpretation")
        st.markdown(
            "- Use this tab to identify likely confounding and whether associations are robust across subgroups.\n"
            "- Treat all findings as risk associations for prediction and planning.\n"
            "- For causal claims, use a dedicated causal framework (DAG-based adjustment, quasi-experimental design, or external validation)."
        )


if __name__ == "__main__":
    main()

