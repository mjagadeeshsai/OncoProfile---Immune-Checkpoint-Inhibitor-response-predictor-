# OncoProfile (Beta)

## Overview
OncoProfile is an AI/ML model that predicts **immune checkpoint inhibitor (ICI) responses** in **NSCLC and colorectal cancer (CRC) patients** using **genetic, demographic, and clinical parameters**. The model achieves up to **96% accuracy** on unseen validation data.  
Developed as part of **MSc Dissertation at University College London (UCL)** and in collaboration with **Cancer Research UK (CRUK)** as a software development project. This work positions OncoProfile as an AI-driven technical solution in oncology, currently in beta and undergoing further enhancements.
> **Note:** This is a **beta version** of OncoProfile, currently undergoing further development and optimization.

## Dataset & Features
The model uses the following features:

**Genetic:**
- TP53, KRAS, PIK3CA, PTEN mutation status
- TP53, KRAS, PIK3CA, PTEN mutation types

**Clinical:**
- Cancer type (NSCLC/CRC)
- Stage
- ICI class

**Demographic:**
- Age
- Sex

**Molecular:**
- MSI Score
- Tumor Mutation Burden (TMB)
- Mutation count
- Tumor purity

## Model Training & Evaluation
The model was trained using a robust machine learning pipeline:

**Preprocessing**
- Missing value imputation: median for numerical, constant for categorical
- Standard scaling for numerical features
- One-hot encoding for categorical features
- Implemented via a **ColumnTransformer** within a unified **Pipeline**

**Models Evaluated (13 total)**
- Logistic Regression (L1, L2, ElasticNet)
- Ridge Classifier
- Decision Tree
- SVM
- KNN
- Naive Bayes
- MLP (Neural Network)
- XGBoost, LightGBM, CatBoost

**Training Strategy**
- **5-fold Stratified Cross-Validation** to maintain responder/non-responder balance
- Metrics tracked: Accuracy, Precision, Recall, F1-score, ROC-AUC

**Model Selection**
- Models ranked by **F1-score**
- Top 3 models retrained on the full dataset
- Final pipelines saved as **joblib artifacts** for reproducibility

**Performance**
- Achieved up to **96% accuracy** on unseen validation folds
- Demonstrates strong generalization for predicting ICI response in NSCLC and CRC

## Ensemble & Explainability (SHAP Analysis)
To interpret the **predictions of the ensemble**, we performed **SHAP (SHapley Additive exPlanations) analysis** on the component models:

- **Models analyzed:** CatBoost, XGBoost, LightGBM  
- **Purpose:** Identify the most influential **genetic, clinical, and demographic features** driving predictions  
- **Global Explanation:** SHAP summary plots reveal overall feature importance across patients  
- **Local Explanation:** SHAP waterfall plots highlight how individual features contribute to the prediction for a single patient  

> **Note:** Each model in the ensemble is analyzed individually. Ensemble predictions are computed as the **average of these models**, and feature contributions for the ensemble can be inferred from the component models.  

This analysis allows **transparent interpretation**, helping clinicians and researchers understand the **key drivers of immune checkpoint inhibitor response**.

## Results
| Model                  | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|------------------------|----------|-----------|--------|----------|---------|
| Best Model Example     | 0.96     | 0.95      | 0.97   | 0.96     | 0.97    |
| Second Best Model      | ...      | ...       | ...    | ...      | ...     |
| Third Best Model       | ...      | ...       | ...    | ...      | ...     |

*Confusion matrices for each model are available in the code.*

## Tech Stack
- Python
- pandas, numpy
- scikit-learn
- XGBoost, LightGBM, CatBoost
- SHAP
- joblib

## Usage
1. Clone the repository
2. Install dependencies:  
```bash
pip install -r requirements.txt




