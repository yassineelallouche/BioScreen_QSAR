# 🧬 BioScreen-QSAR

**A Modular Low-Code Python Framework for Antimicrobial Activity Prediction**  
*From Molecular Curation to Ligand-Based Virtual Screening*

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![RDKit](https://img.shields.io/badge/RDKit-2024-green)](https://www.rdkit.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.36-red)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📋 Overview

BioScreen-QSAR is a complete, open-source QSAR pipeline covering **six integrated stages**:

| Stage | Module | Description |
|-------|--------|-------------|
| 1 | `data_curation.py`              | SMILES standardisation (salts, charges, tautomers) |
| 2 | `duplicate_handling.py`         | Duplicate detection & resolution (Fourches protocol) |
| 3 | `descriptors_ecfp.py`           | ECFP4/ECFP6 fingerprint generation via RDKit |
| 4 | `preprocessing.py`              | Variance filtering + stratified train/test split |
| 5 | `train_classification_models.py`| RF/SVM/LightGBM + Bayesian HPO (classification) |
|   | `train_regression_models.py`    | RF/SVM/LightGBM + Bayesian HPO (regression) |
| 6 | `virtual_screening.py`          | Ligand-based virtual screening + ranked output |
|   | `app.py`                        | Interactive Streamlit dashboard |

---

## 🚀 Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/your-username/BioScreen-QSAR.git
cd BioScreen-QSAR
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Launch the Streamlit dashboard
```bash
streamlit run scripts/app.py
```
Open your browser at **http://localhost:8501**

---

## 🖥️ Streamlit Dashboard Modules

The dashboard provides **4 interactive modules** accessible from the sidebar:

| Module | Description |
|--------|-------------|
| 🏠 Home               | Platform overview and quick metrics |
| 🧹 Data Curation      | Upload CSV → standardise → deduplicate → download |
| 🔢 Descriptor Calc.   | Compute ECFP4/ECFP6 fingerprints → download matrix |
| 🤖 Model Training     | Train RF/SVM/LightGBM → view metrics → download results |
| 🔍 Virtual Screening  | Upload library + model → ranked predictions → download |

---

## ⌨️ Command-Line Usage

### Step 1 — Curate your dataset
```bash
python -m scripts.data_curation \
    --input  data/raw_dataset.csv \
    --output data/standardised.csv \
    --smiles SMILES \
    --activity activity
```

### Step 2 — Resolve duplicates
```bash
python -m scripts.duplicate_handling \
    --input    data/standardised.csv \
    --output   data/curated.csv \
    --smiles   curated_SMILES \
    --activity activity \
    --type     binary
```

### Step 3 — Calculate descriptors
```bash
python -m scripts.descriptors_ecfp \
    --input    data/curated.csv \
    --output   data/descriptors_ecfp4_2048.csv \
    --smiles   curated_SMILES \
    --activity activity \
    --radius   2 \
    --n_bits   2048
```

### Step 4 — Train classification models
```bash
python -m scripts.train_classification_models \
    --descriptors data/descriptors_ecfp4_2048.csv \
    --smiles      curated_SMILES \
    --activity    activity \
    --n_trials    50
```

### Step 5 — Run virtual screening
```bash
python -m scripts.virtual_screening \
    --library  data/compound_library.csv \
    --model    models/classifier_lightgbm.pkl \
    --output   results/vs_results.csv \
    --smiles   SMILES \
    --task     classification
```

---

## 📁 Project Structure

```
BioScreen-QSAR/
│
├── scripts/                          # Core pipeline modules
│   ├── __init__.py
│   ├── utils.py                      # Logger, I/O helpers, reporting
│   ├── data_curation.py              # Molecular standardisation
│   ├── duplicate_handling.py         # Duplicate detection & resolution
│   ├── descriptors_ecfp.py           # ECFP fingerprint generation
│   ├── preprocessing.py              # Feature filtering + data splitting
│   ├── hyperparameter_optimization.py # Bayesian HPO (Optuna)
│   ├── train_classification_models.py # Classification training pipeline
│   ├── train_regression_models.py    # Regression training pipeline
│   ├── validation_metrics.py         # Metrics + visualisation
│   ├── virtual_screening.py          # LBVS pipeline
│   ├── model_serialization.py        # Save/load models (joblib)
│   └── app.py                        # Streamlit dashboard
│
├── data/                             # Datasets
│   ├── example_input.csv             # 20-compound demo dataset
│   ├── example_output.csv            # Example virtual screening output
│   └── library_demo.csv              # 50-compound screening library
│
├── models/                           # Saved model files (.pkl)
├── results/                          # Training/screening outputs (.csv)
├── reports/                          # Curation reports (.txt)
├── tests/                            # Unit tests
│   ├── test_curation.py
│   ├── test_descriptors.py
│   └── test_metrics.py
├── docs/                             # Documentation
│   ├── METHODOLOGY.md
│   └── API_REFERENCE.md
├── notebooks/                        # Jupyter notebooks
│   └── demo_pipeline.ipynb
│
├── requirements.txt                  # Python dependencies
├── README.md                         # This file
└── LICENSE                           # MIT License
```

---

## 📊 Input / Output Format

### Input CSV (minimum required columns)
```
SMILES,activity,compound_id
CC1=CC=C(C=C1)N,1,CMPD_001
C1=CC=C(C=C1)C(=O)O,0,CMPD_002
```

### Virtual Screening Output CSV
```
rank,curated_SMILES,compound_id,predicted_class,predicted_label,probability_active
1,CC1=CC=C(C=C1)N,CMPD_001,1,Active,0.9312
2,C1=CC=C(C=C1)C(=O)O,CMPD_002,0,Inactive,0.2104
```

---

## 📈 Supported Metrics

### Classification
| Metric | Symbol | Description |
|--------|--------|-------------|
| Balanced Accuracy | BACC | Mean sensitivity + specificity |
| Sensitivity | Se | TP / (TP + FN) |
| Specificity | Sp | TN / (TN + FP) |
| Positive Predictive Value | PPV | TP / (TP + FP) |
| Negative Predictive Value | NPV | TN / (TN + FN) |
| Matthews Corr. Coeff. | MCC | Balanced binary classifier metric |
| Area Under ROC | AUC | Threshold-independent performance |
| F1-score | F1 | Harmonic mean of precision/recall |

### Regression
| Metric | Symbol |
|--------|--------|
| R² | Coefficient of determination |
| MAE | Mean Absolute Error |
| MSE | Mean Squared Error |
| RMSE | Root Mean Squared Error |
| EV | Explained Variance Score |

---

## ⚙️ Hyperparameter Search Spaces

| Algorithm | Optimised Parameters |
|-----------|----------------------|
| **Random Forest** | n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features |
| **SVM** | C, kernel (rbf/linear/poly), gamma, degree |
| **LightGBM** | n_estimators, num_leaves, learning_rate, max_depth, subsample, colsample_bytree, reg_alpha, reg_lambda |

Optimisation uses **Optuna TPE sampler** (50 trials default, configurable).

---

## 🔬 Case Study — *Staphylococcus aureus*

| Dataset property | Value |
|-----------------|-------|
| Source | ChEMBL release 33 |
| Raw compounds | 1,856 |
| After curation | 1,248 |
| Active (≤10 µM) | 687 |
| Inactive (>10 µM) | 561 |
| Train / Test split | 80% / 20% |
| Best classifier | LightGBM |
| BACC (external) | **0.81** |
| AUC (external) | **0.87** |
| Best regressor | LightGBM |
| R² (external) | **0.74** |
| RMSE (external) | **0.61 log units** |

> ⚠️ These are **demonstrative values** from a reproducible pipeline run. Actual results depend on the ChEMBL version and random seed.

---

## 📦 Dependencies

```
pandas==2.2.2
numpy==1.26.4
rdkit==2024.3.5
scikit-learn==1.5.1
lightgbm==4.5.0
joblib==1.4.2
matplotlib==3.9.1
seaborn==0.13.2
plotly==5.22.0
streamlit==1.36.0
optuna==3.6.1
imbalanced-learn==0.12.3
scipy==1.13.1
```

---

## 📄 Citation

If you use BioScreen-QSAR in your research, please cite:

```bibtex
@article{bioscreen_qsar_2025,
  title   = {BioScreen-QSAR: A Modular Low-Code Python Framework for 
             Antimicrobial Activity Prediction},
  author  = {[Authors]},
  journal = {Journal of Cheminformatics},
  year    = {2025},
  doi     = {10.xxxx/xxxxxx}
}
```

---

## 📜 License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.

---

## 🤝 Contributing

Contributions are welcome! Please open an issue or submit a pull request on GitHub.

---

## 📬 Contact

For questions or collaborations: **[email@institution.edu]**
