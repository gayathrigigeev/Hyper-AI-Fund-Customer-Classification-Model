# Hyper AI Fund — Customer Classification Model

Multi-class customer classification model built using machine learning to predict customer interest in an AI-focused investment fund.

---

## Background

Built as part of MSc Finance and Financial Technology (ICM520 — Machine Learning, AI and Big Data in Finance) at Henley Business School, University of Reading.

The objective was to classify 2,000 anonymised bank customers into four interest categories for a new AI investment fund, using machine learning to support targeted sales outreach.

A key business constraint drove model selection: correctly identifying Category 0 customers (those who do not wish to be contacted) was prioritised to avoid unnecessary and unwanted outreach.

---

## Dataset

- 2,000 anonymised customer records
- 18 anonymised features (FEAT_0 to FEAT_17)
- 4 target classes:
  - 0 = No interest, do not contact again
  - 1 = No interest, low priority to contact
  - 2 = Low interest, medium priority
  - 3 = High interest, high priority
- 1 column with missing values: FEAT_9 (184 missing values, handled with mean imputation)

**Note on Dataset:** The dataset was provided by Henley Business School as part of coursework and is not publicly available. To run the notebook, replace the CSV path with your own classification dataset of similar structure (18 features, 1 target column with 4 classes).

---

## Methodology

Three models were developed and compared:

**Logistic Regression** — baseline model with linear decision boundary, multinomial solver, max_iter=1000

**K-Nearest Neighbours (KNN)** — k=5, Euclidean distance, non-parametric instance-based classifier

**Random Forest** — 100 trees, ensemble learning, final recommended model

Feature selection was applied using SelectKBest with mutual information to identify the minimum number of features required to maintain 70% accuracy.

---

## Results

| Model | Accuracy | Precision (Cat. 0) | Recall (Cat. 0) |
|-------|----------|--------------------|-----------------|
| Logistic Regression | 64% | 0.76 | 0.74 |
| KNN (k=5) | 52% | 0.56 | 0.74 |
| Random Forest | 79% | 0.87 | 0.85 |

**Final model: 6-feature Random Forest — 81.33% accuracy**

Reducing from 18 features to 6 improved accuracy, confirming that many original features added noise rather than signal.

---

## Feature Reduction Analysis

| Features Used | Accuracy |
|---------------|----------|
| 3 | 58.1% |
| 5 | 68.5% |
| 6 | 81.33% (peak) |
| 7 | 80.5% |
| 10 | 77.5% |
| 18 | 79.2% |

The minimum number of features to exceed 70% accuracy is 6.
Top features identified: FEAT_11, FEAT_2, FEAT_8, FEAT_6, FEAT_4

---

## How to Run

1. Clone the repository
2. Place your dataset CSV in the same folder as the notebook
3. Update the file path in the first data loading cell
4. Open the notebook in Jupyter
5. Run all cells from top to bottom

**Requirements:**
```
pandas
numpy
matplotlib
seaborn
scikit-learn
```

Install with:
```
pip install pandas numpy matplotlib seaborn scikit-learn
```

---

## Tools

Python | scikit-learn | Random Forest | Logistic Regression | KNN | SelectKBest | Mutual Information | pandas | seaborn | matplotlib

---

## Author

**Gayathri Gigeev**
MSc Finance and Financial Technology — Henley Business School (Distinction)
[LinkedIn](https://www.linkedin.com/in/gayathri-gigeev/)
[GitHub Portfolio](https://github.com/gayathrigigeev)
