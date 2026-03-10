# Cleft Lip and Palate Risk Modeling and Explainability

Predicting U.S. cleft lip and palate cases from natality data, with end-to-end modeling, explainability, and an interactive Streamlit application.

## Project Summary

This project builds and evaluates classification models to estimate cleft-case risk using U.S. natality microdata. It includes:

- exploratory/descriptive analysis,
- model training and threshold optimization for severe class imbalance,
- SHAP-based global and local explainability,
- an interactive risk prediction interface,
- an observational causality-focused analysis tab.

The app is artifact-driven: models and analysis outputs are produced in the notebook and loaded by Streamlit from `streamlit_artifacts/`.

## Data

- Source used in project narrative: 2024 NCHS National Natality Public Use File.
- Unit of analysis: individual live birth records.
- Target: binary `cleft_case` indicator.
- Feature scope:
  - `full`: broad feature set for predictive performance.
  - `safe`: leakage-safe subset intended for deployment realism (pre-birth or at-birth available fields).

## Repository Structure

```text
.
|-- app.py
|-- Notebook 2.ipynb
|-- requirements.txt
|-- DEPLOYMENT.md
|-- streamlit_artifacts/
|   |-- best_params_log.json
|   |-- data_dictionary.csv
|   |-- data_dictionary.pkl
|   |-- eda_snapshot.pkl
|   |-- interactive_schema.json
|   |-- model_registry.json
|   |-- results_df.csv
|   |-- sex_wic_heatmap_full.csv
|   |-- X_test_full.pkl
|   |-- X_test_safe.pkl
|   |-- y_test_full.pkl
|   |-- y_test_safe.pkl
|   `-- models/
|       |-- decision_tree.joblib
|       |-- leakage_safe_random_forest.joblib
|       |-- logistic_regression.joblib
|       |-- mlp.keras
|       |-- mlp_preprocessor.joblib
|       |-- random_forest.joblib
|       `-- xgboost.joblib
```

## Modeling Stack

The project includes six shipped models:

- Logistic Regression (`full`)
- Decision Tree (`full`)
- Random Forest (`full`)
- XGBoost (`full`)
- MLP (Keras) (`full`)
- Leakage-safe Random Forest (`safe`)

Model metadata is defined in `streamlit_artifacts/model_registry.json`.

## Training and Evaluation Notes

- Class imbalance is handled via class weighting and threshold tuning.
- Multiple operating points are used for model selection and decision support:
  - default threshold (`0.5`) where applicable,
  - F1-optimized threshold,
  - recall-first threshold (targeting higher sensitivity).
- Core metrics tracked include `accuracy`, `precision`, `recall`, `f1`, `roc_auc`, `pr_auc`, and train time (see `streamlit_artifacts/results_df.csv`).

## Streamlit App

Run the app:

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app has five tabs:

1. `Executive Summary`
	- project context, data source, methodology, and headline findings.

2. `Descriptive Analytics`
	- class distribution and prevalence by key factors,
	- subgroup views (for example, sex and WIC),
	- pair plots and correlation summaries.

3. `Model Performance`
	- model comparison tables/plots,
	- operating-point behavior,
	- best hyperparameter summary from notebook exports.

4. `Explainability & Interactive Prediction`
	- SHAP summary (dot/bar) views for tree models,
	- SHAP waterfall explanations,
	- interactive custom profile scoring with model selection.

5. `Causality Analysis`
	- observational association analysis (not causal proof),
	- crude and stratified prevalence/risk comparisons,
	- exposure-focused interpretation support.

## Artifact Workflow

The notebook (`Notebook 2.ipynb`) is the source of truth for data prep, training, tuning, and artifact export.

`app.py` does not retrain models. It loads serialized outputs from `streamlit_artifacts/` and performs inference, visualization, and interpretation. If artifacts change, rerun notebook export steps before launching the app.

## Key Dependencies

From `requirements.txt`:

- `streamlit`
- `pandas`, `numpy`, `scipy`
- `scikit-learn`
- `xgboost`
- `matplotlib`, `seaborn`
- `shap`
- `joblib`

## Reproducibility

- Python version target for deployment: `3.11`.
- Keep `streamlit_artifacts/` in sync with notebook outputs.
- For deterministic comparisons, preserve the same train/test split and random seeds used during notebook execution.

## Important Interpretation Caveats

- This is an observational modeling workflow.
- High feature importance or strong subgroup association does not establish causation.
- The causality tab is designed for structured interpretation, sensitivity checks, and communication of uncertainty, not for definitive causal claims.

## Deployment

Deployment steps are documented in `DEPLOYMENT.md`.

Typical Streamlit Community Cloud setup:

- repo connected to GitHub,
- main file: `app.py`,
- Python: `3.11`.
