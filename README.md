# IKT Reproducibility Study

**Reproducing and Extending Interpretable Knowledge Tracing: A Multi-Dataset Reproducibility Study with TAN Stability Analysis**

Fadi Alazayem (Student ID: 25002207)
MSc Artificial Intelligence, BUiD — AI501 Assignment 2, February 2026

---

## Overview

This repository contains the complete code, data, and results for reproducing the **Interpretable Knowledge Tracing (IKT)** model from:

> Minn, S., Desmarais, M.C., Zhu, F., Xiao, J., & Wang, J. (2022). *Interpretable Knowledge Tracing: Simple and Efficient Student Modeling with Causal Relations.* Proceedings of the AAAI Conference on Artificial Intelligence, 36(11), 12810–12818.

Original authors' code: [github.com/Simon-tan/IKT](https://github.com/Simon-tan/IKT)

## Dual-Track Evaluation

We evaluate IKT through two complementary tracks:

1. **Independent reproduction** — We modernised the codebase (TF1.x → scikit-learn) and ran our own preprocessing. Result: *partially reproducible* (ASSISTments 2009/2012 match; Algebra shows a gap due to undocumented multi-skill handling).

2. **Exact replication** — Using the authors' preprocessed 5-fold CV data. Result: *fully reproducible* (all three datasets match reported AUC within Δ ≤ 0.004).

### Core Results (Exact Replication Track — 5-Fold CV)

| Dataset           | Our Mean | Std   | Paper  | Δ      | Verdict     |
|-------------------|----------|-------|--------|--------|-------------|
| ASSISTments 2009  | 0.798    | 0.007 | 0.797  | +0.001 | Reproduced  |
| Algebra 2005-06   | 0.847    | 0.005 | 0.851  | −0.004 | Reproduced  |
| ASSISTments 2012  | 0.768    | 0.001 | 0.767  | +0.001 | Reproduced  |

## Repository Structure

```
IKT-Reproducibility/
├── src/                          # All Python scripts
│   ├── BKT.py                    # Bayesian Knowledge Tracing module
│   ├── FeatureEngineering_v2.py  # Main IKT pipeline (modernised)
│   ├── run_5fold_cv.py           # 5-fold CV on authors' data
│   ├── run_multi_seed.py         # Multi-seed robustness (5/20 seeds)
│   ├── run_sensitivity_K.py      # K-cluster sensitivity sweep
│   ├── run_sensitivity_interval.py # BKT interval sensitivity sweep
│   ├── run_ablation.py           # Feature ablation (IKT-1/2/3)
│   ├── run_failure_analysis.py   # Concrete failure case extraction
│   └── run_tan_stability.py      # TAN network stability extension
├── data/                         # Single-split datasets
│   ├── ass09/                    # ASSISTments 2009 (CSV + ARFF)
│   ├── algebra/                  # Algebra 2005-06 (CSV + ARFF)
│   └── ass12/                    # ASSISTments 2012 (CSV + ARFF)
├── data_5fold/                   # Authors' preprocessed 5-fold splits
│   ├── ASS09/                    # train/test_fold1-5.arff
│   ├── ASS12/
│   └── KDD/                      # Algebra
├── results/                      # Output directories (populated at runtime)
│   ├── 5fold_cv/
│   ├── multi_seed/
│   ├── sensitivity_K/
│   ├── sensitivity_interval/
│   ├── ablation/
│   ├── failure/
│   └── tan_stability/
├── weka.jar                      # WEKA 3.8 (TAN classifier)
├── requirements.txt
└── README.md
```

## Requirements

- **Python 3.8+**
- **Java 8+** (for WEKA TAN classifier)
- Python packages: `numpy`, `pandas`, `scikit-learn`

```bash
pip install -r requirements.txt
```

## How to Reproduce

### 1. Core Reproduction (5-Fold CV — Recommended)

This uses the authors' preprocessed data and exactly matches the paper's evaluation protocol.

```bash
cd src
python run_5fold_cv.py
```

**Expected output:** AUC per fold for each dataset, plus mean ± std matching the table above.

### 2. Independent Reproduction (Single Split)

Run the full pipeline from raw CSV data with our own preprocessing:

```bash
cd src
python FeatureEngineering_v2.py
```

By default, this runs on ASSISTments 2009. Edit `DATA_NAME` in the script to switch datasets.

### 3. Multi-Seed Robustness

Run 5 (or 20) random seeds to assess k-means sensitivity:

```bash
cd src
python run_multi_seed.py
```

### 4. Hyperparameter Sensitivity

```bash
cd src
python run_sensitivity_K.py         # K = 4, 5, 6, 7, 8, 10
python run_sensitivity_interval.py  # interval = 10, 15, 20, 25, 30
```

### 5. Ablation Study

Tests IKT-1 (mastery only), IKT-2 (mastery + ability), IKT-3 (full model):

```bash
cd src
python run_ablation.py
```

### 6. Failure Analysis

Extracts high-confidence misclassifications with TAN-based explanations:

```bash
cd src
python run_failure_analysis.py
```

### 7. TAN Stability Extension (Novel Contribution)

Runs the full IKT pipeline 20 times per dataset and analyses Bayesian network topology stability:

```bash
cd src
python run_tan_stability.py
```

**Key finding:** Only 2 topologies emerge across 60 runs; 3/5 edges are perfectly stable.

## Key Findings

- **Exact replication:** Fully reproducible (Δ ≤ 0.004 AUC on all 3 datasets)
- **Independent reproduction:** Partially reproducible (Algebra gap due to KC granularity)
- **K sensitivity:** Negligible effect (ΔAUC < 0.001 across K = 4–10)
- **Interval sensitivity:** Slight trend (range = 0.003)
- **Ablation:** Problem difficulty is the dominant feature (+0.082 AUC on ASS-09)
- **Failure cases:** 75% of high-confidence errors are false positives (overconfident mastery predictions)
- **TAN stability:** Core structure (3/5 edges) 100% stable across 60 runs, AUC variation ≤ 0.002

## Codebase Modernisation

Changes from the original IKT repository:
- Replaced TensorFlow 1.x k-means with scikit-learn `KMeans`
- Removed all `tf.Session()` / `tf.app.run()` dependencies
- Added seed control (`random_state` parameter) for reproducibility
- Added logging and progress indicators
- **No changes** to BKT logic, IRT features, or ARFF generation

## Datasets

| Dataset           | Students | Train/Test | Interactions | Source |
|-------------------|----------|------------|--------------|--------|
| ASSISTments 2009  | 2,004    | 1,587/417  | 191,729      | [ASSISTments](https://sites.google.com/site/assistmaborehek/home) |
| Algebra 2005-06   | 564      | 450/114    | 607,025      | [KDD Cup 2010](https://pslcdatashop.web.cmu.edu/KDDCup/) |
| ASSISTments 2012  | 5,000    | 4,000/1,000| 1,654,219    | [ASSISTments](https://sites.google.com/site/assistmaborehek/home) |

5-fold CV splits from the authors' shared data: [Google Drive](https://drive.google.com/drive/folders/1Wuilcb_ash1r5MT3tgMc78PDPUh0n3uT)

## Ethical Considerations

- All datasets are publicly available and anonymised; no PII was accessed
- False positives (75% of errors) risk false reassurance — missed interventions for struggling students
- False negatives risk unnecessary remediation and student frustration
- ASSISTments 2012 sampling cap (5,000 of 18,000+ students) raises representativeness concerns
- Label leakage identified: problem difficulty is computed using train+test students (matches original code but inflates absolute performance)

## Citation

If you use this code, please cite both the original paper and this reproducibility study:

```bibtex
@inproceedings{minn2022interpretable,
  title={Interpretable Knowledge Tracing: Simple and Efficient Student Modeling with Causal Relations},
  author={Minn, Sein and Desmarais, Michel C and Zhu, Feida and Xiao, Jing and Wang, Jigang},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={36},
  number={11},
  pages={12810--12818},
  year={2022}
}
```

## License

This reproducibility study is for academic purposes (BUiD AI501 Assignment 2). The original IKT code is distributed under its original licence. Datasets are used under their respective academic licences.
