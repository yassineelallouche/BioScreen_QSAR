# BioScreen-QSAR — API Reference

## `scripts.data_curation`

### `MolecularStandardiser`
```python
MolecularStandardiser(
    remove_mixtures=True,
    neutralise_charges=True,
    standardise_tautomers=True
)
```
**Methods:**
- `.standardise_smiles(smiles: str) -> str | None`  
  Standardise a single SMILES. Returns canonical SMILES or None.

- `.standardise_dataframe(df, smiles_col='SMILES', activity_col=None) -> pd.DataFrame`  
  Standardise all rows. Adds `curated_SMILES` column. Drops failed rows.

### `run_curation_pipeline(input_path, output_path, smiles_col, activity_col, ...)`
End-to-end: load CSV → standardise → save.

---

## `scripts.duplicate_handling`

### `handle_duplicates(df, smiles_col, activity_col, data_type, output_curated, output_report)`
Detect and resolve duplicates.

**Parameters:**
- `data_type`: `'binary'` or `'continuous'`
- `output_curated`: path to save deduplicated CSV
- `output_report`: path to save plain-text statistics report

**Returns:** `(pd.DataFrame, dict)` — clean dataset + stats dictionary

**Stats dictionary keys:**
```python
{
    "n_input": int,
    "n_duplicated_structures": int,
    "n_removed_duplicate_records": int,
    "n_discordant_structures": int,
    "n_final": int,
    "pct_removed": float
}
```

---

## `scripts.descriptors_ecfp`

### `ECFPGenerator`
```python
ECFPGenerator(radius=2, n_bits=2048, use_chirality=False, use_bond_types=True)
```
**Methods:**
- `.smiles_to_fp(smiles: str) -> np.ndarray | None`  
  Returns binary array of shape `(n_bits,)`.

- `.dataframe_to_fp_matrix(df, smiles_col, activity_col) -> (X, y, valid_smiles)`  
  Returns fingerprint matrix, labels, and valid SMILES list.

- `.build_descriptor_dataframe(df, smiles_col, activity_col) -> pd.DataFrame`  
  Returns DataFrame with columns: curated_SMILES, activity, Bit_0000...Bit_XXXX.

---

## `scripts.preprocessing`

### `FeaturePreprocessor`
```python
FeaturePreprocessor(
    variance_threshold=0.0,
    test_size=0.20,
    n_splits=5,
    random_state=42,
    stratify=True
)
```
**Methods:**
- `.fit_transform(X, y) -> (X_train, X_test, y_train, y_test)`
- `.transform(X) -> np.ndarray`  (apply fitted filter to new data)
- `.get_cv_splitter(task) -> StratifiedKFold | KFold`

### `prepare_matrices(descriptor_path, smiles_col, activity_col, ...) -> (X_tr, X_te, y_tr, y_te, preprocessor)`
Convenience: load descriptors CSV → filter → split.

---

## `scripts.hyperparameter_optimization`

### `BayesianOptimiser`
```python
BayesianOptimiser(
    algorithm='lgbm',      # 'rf' | 'svm' | 'lgbm'
    task='classification', # 'classification' | 'regression'
    n_trials=50,
    cv_folds=5,
    scoring=None,          # defaults: 'roc_auc' / 'r2'
    random_state=42
)
```
**Methods:**
- `.optimise(X_train, y_train) -> dict`  — returns best params
- `.build_best_model()` — returns fitted-ready estimator

**Attributes after optimise():**
- `.best_params_` — best hyperparameter dict
- `.best_score_` — best CV score
- `.study_` — Optuna study object (for visualisation)

---

## `scripts.train_classification_models`

### `train_classifiers(X_train, X_test, y_train, y_test, algorithms, optimise, n_trials, n_splits, model_dir, results_dir) -> pd.DataFrame`

Trains all specified algorithms and returns a metrics summary DataFrame.

**Columns in output:**  
`Model, cv_auc_mean, cv_auc_std, cv_f1_mean, acc, bacc, sensitivity, specificity, ppv, npv, mcc, auc, f1`

---

## `scripts.train_regression_models`

### `train_regressors(X_train, X_test, y_train, y_test, ...) -> pd.DataFrame`

Same interface as `train_classifiers`.

**Columns in output:**  
`Model, cv_r2_mean, cv_r2_std, cv_rmse_mean, r2, mae, mse, rmse, explained_var`

---

## `scripts.validation_metrics`

### Classification
```python
classification_metrics(y_true, y_pred, y_prob=None) -> dict
# Keys: acc, bacc, sensitivity, specificity, ppv, npv, mcc, auc, f1

plot_roc_curve(y_true, y_prob, model_name, save_path=None)
plot_confusion_matrix(y_true, y_pred, model_name, save_path=None)
```

### Regression
```python
regression_metrics(y_true, y_pred) -> dict
# Keys: r2, mae, mse, rmse, explained_var

plot_predicted_vs_experimental(y_true, y_pred, model_name, activity_label, save_path=None)
```

### Utility
```python
metrics_to_dataframe(metrics_dict, model_name) -> pd.DataFrame
```

---

## `scripts.virtual_screening`

### `run_virtual_screening(library_path, model_path, preprocessor, smiles_col, task, ecfp_radius, ecfp_n_bits, output_path, cloud_mode) -> pd.DataFrame`

**Returns ranked DataFrame with columns:**
- `rank` — 1 = most active (int)
- `curated_SMILES` — standardised SMILES (str)
- `predicted_class` — 0 or 1 (classification only)
- `predicted_label` — "Active" / "Inactive" (classification only)
- `probability_active` — 0.0–1.0 (classifiers with predict_proba)
- `predicted_pMIC` — continuous value (regression only)
- All original metadata columns preserved

---

## `scripts.model_serialization`

```python
save_model(model, filepath, metadata=None)
# Saves .pkl + optional _metadata.json

load_model(filepath) -> estimator
# Raises FileNotFoundError if missing

list_saved_models(model_dir='models') -> list[str]
# Returns sorted list of .pkl paths
```

---

## `scripts.utils`

```python
configure_logger(name, level) -> logging.Logger
load_csv(filepath, smiles_col, activity_col) -> pd.DataFrame
save_csv(df, filepath, index=False)
is_valid_smiles(smiles) -> bool
canonical_smiles(smiles) -> str | None
generate_summary_report(n_input, n_after_std, n_dup, n_discord, n_final, output_path)
ensure_dir(path)
timestamp_str() -> str   # e.g. "20250315_143022"
```
