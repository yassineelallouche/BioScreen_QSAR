# BioScreen-QSAR — Detailed Methodology

## 1. Data Curation Pipeline

### 1.1 Molecular Parsing
Each SMILES string is parsed using `Chem.MolFromSmiles()`. Entries yielding `None`
(invalid SMILES syntax) are removed and reported. Sanitisation (`Chem.SanitizeMol()`)
resolves valence errors and assigns aromaticity flags.

### 1.2 Salt and Mixture Removal
Multi-component SMILES (e.g., `CC(=O)O.Na`) arise from co-crystal structures or
salt forms in ChEMBL. The pipeline applies:
1. `SaltRemover.StripMol()` — removes known counter-ions using RDKit's built-in list.
2. `LargestFragmentChooser.choose()` — retains the principal organic fragment in
   ambiguous cases (handles edge cases not covered by the salt list).

### 1.3 Charge Neutralisation
Formal charges are neutralised via `rdMolStandardize.Uncharger()`. This converts
species such as carboxylates (COO⁻) to carboxylic acids (COOH) and ammonium ions
(NR₄⁺) to amines (NR₃), ensuring consistent molecular representations.

### 1.4 Tautomer Canonicalisation
Tautomeric ambiguity (e.g., keto–enol equilibria, imine–enamine) is resolved using
`rdMolStandardize.TautomerEnumerator().Canonicalize()`, which selects the most stable
tautomeric form according to RDKit's scoring function.

---

## 2. Duplicate Handling Protocol

Based on Fourches et al. (J. Chem. Inf. Model. 2010, 2016):

### Binary Data
```
Identical canonical SMILES detected
         │
    ┌────┴────┐
    │         │
Same label  Different labels
    │         │
  Keep 1    Discard ALL
```

### Continuous Data
```
Identical canonical SMILES detected
         │
    ┌────┴────────────────┐
    │                     │
Range ≤ 0.2 log units   Range > 0.2 log units
    │                     │
  Mean → Keep 1         Discard ALL
```

---

## 3. ECFP Fingerprints

The Morgan algorithm computes circular fingerprints by iteratively hashing the
atomic environments at increasing radii:

```
Radius 0: atom type (atomic number, charge, degree, ...)
Radius 1: atom + immediate neighbours
Radius 2: atom + neighbours-of-neighbours (ECFP4)
Radius 3: atom + 2nd-order neighbourhood  (ECFP6)
```

Each hash is mapped to a position in the bit-vector via folding:
`bit_position = hash(environment) mod nBits`

**Recommended defaults:**
- `radius=2, nBits=2048` (ECFP4/2048) — best balance of coverage and speed
- `radius=3, nBits=2048` (ECFP6/2048) — higher resolution, slightly more overfitting risk

---

## 4. Hyperparameter Search Spaces

### Random Forest
| Parameter | Range | Type |
|-----------|-------|------|
| n_estimators | 100–800 (step 50) | int |
| max_depth | 3–30 | int |
| min_samples_split | 2–20 | int |
| min_samples_leaf | 1–10 | int |
| max_features | sqrt, log2, 0.3, 0.5 | categorical |

### Support Vector Machine
| Parameter | Range | Type |
|-----------|-------|------|
| C | 10⁻³–10³ | float (log) |
| kernel | rbf, linear, poly | categorical |
| gamma | 10⁻⁵–10¹ | float (log) |
| degree | 2–5 (poly only) | int |

### LightGBM
| Parameter | Range | Type |
|-----------|-------|------|
| n_estimators | 50–1000 (step 50) | int |
| num_leaves | 20–300 | int |
| learning_rate | 10⁻³–0.3 | float (log) |
| max_depth | 3–12 | int |
| subsample | 0.5–1.0 | float |
| colsample_bytree | 0.5–1.0 | float |
| reg_alpha | 0.0–1.0 | float |
| reg_lambda | 0.0–1.0 | float |

---

## 5. Validation Strategy

```
Full curated dataset (N compounds)
           │
    ┌──────┴──────┐
    │             │
Train (80%)    Test (20%)  ← never seen during training
    │
    │  ┌─────────────────────────────┐
    │  │  Bayesian HPO (Optuna TPE)  │
    │  │  50 trials × 5-fold CV      │
    │  │  Objective: AUC / R²        │
    │  └─────────────────────────────┘
    │           │
    │     Best params
    │           │
    └──── Final fit on full Train set
           │
    Evaluate on Test set → report metrics
```

---

## 6. Evaluation Metrics

### Classification
```
             Predicted
              0      1
Actual  0  [ TN  |  FP ]
        1  [ FN  |  TP ]

Sensitivity (Se)  = TP / (TP + FN)
Specificity (Sp)  = TN / (TN + FP)
PPV               = TP / (TP + FP)
NPV               = TN / (TN + FN)
ACC               = (TP + TN) / N
BACC              = (Se + Sp) / 2
MCC               = (TP·TN − FP·FN) / √((TP+FP)(TP+FN)(TN+FP)(TN+FN))
AUC               = integral of ROC curve
F1                = 2·PPV·Se / (PPV + Se)
```

### Regression
```
R²   = 1 − Σ(yᵢ − ŷᵢ)² / Σ(yᵢ − ȳ)²
MAE  = mean|yᵢ − ŷᵢ|
MSE  = mean(yᵢ − ŷᵢ)²
RMSE = √MSE
EV   = 1 − Var(y − ŷ) / Var(y)
```

---

## 7. Virtual Screening Ranking

Compounds are sorted by:
- **Classification**: `predicted_class DESC`, then `probability_active DESC`
- **Regression**: `predicted_pMIC DESC` (most potent first)

The output table always includes:
- `rank` — integer position in ranked list (1 = most promising)
- `curated_SMILES` — standardised canonical SMILES
- `predicted_class` / `predicted_pMIC`
- `probability_active` (classifiers only)
- All original metadata columns from the input library
