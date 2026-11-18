# Predictive Maintenance Using Machine Learning

Applied Machine Learning coursework: binary classification of the row-level
`Machine failure` label using the **UCI AI4I 2020 Predictive Maintenance Dataset**
(UCI ID 601, synthetic, 10,000 rows). DOI: https://doi.org/10.24432/C5HS5C — licence CC BY 4.0.

> The model predicts the dataset's `Machine failure` label only. It does **not** estimate
> remaining useful life or time to failure. The Streamlit app is an educational prototype,
> not a validated industrial or safety system.

## Setup (Windows PowerShell)
```
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```
macOS/Linux: `python3 -m venv .venv && source .venv/bin/activate`

## Reproduce (run from the repository root)
```
python -m src.download_data   # download data, create restricted modelling table
python -m src.train           # train 4 models, select on validation, save model
python -m src.evaluate        # final one-time test evaluation + figures
python -m src.explain         # permutation feature importance
streamlit run app/app.py      # demo app
```
EDA: open `notebooks/01_data_understanding_and_eda.ipynb`.

## Features and leakage exclusions
Predictors: Type, Air temperature [K], Process temperature [K], Rotational speed [rpm],
Torque [Nm], Tool wear [min]. Target: Machine failure.
Excluded: TWF, HDF, PWF, OSF, RNF (failure-mode labels → target leakage); UDI, Product ID (identifiers).

## Method
Stratified 70/15/15 train/validation/test split (seed 42); preprocessing (OneHotEncoder + StandardScaler)
inside a scikit-learn Pipeline fitted on training data only; Dummy baseline, Logistic Regression,
Decision Tree, Random Forest (class weights for imbalance); model chosen by validation Average Precision;
threshold chosen by maximum validation F1; test set used once.

## Outputs
`reports/validation_model_comparison.csv`, `reports/test_metrics.json`,
`reports/classification_report.txt`, `reports/permutation_feature_importance.csv`, `reports/figures/*.png`.

## Limitations
Synthetic data; random row-level split (no timestamps/machine groups); threshold rule is a coursework
choice, not a cost-based maintenance policy; scores are not calibrated probabilities.

## Citation
AI4I 2020 Predictive Maintenance Dataset (2020). UCI Machine Learning Repository. https://doi.org/10.24432/C5HS5C
