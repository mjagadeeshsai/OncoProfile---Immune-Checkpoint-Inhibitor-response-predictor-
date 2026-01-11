import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix)
import joblib
import warnings

warnings.filterwarnings('ignore')

# === 1. Load data ===
input_file = r"C:\Users\mjaga\OneDrive\Documents\UNI-UCL\PHAY0055 - Dissertation\Datasets\Merged data\training_set.xlsx"  # Replace with your file path
df = pd.read_excel(input_file, engine='openpyxl')

# Fill mutation columns missing values
mutation_cols = [
    'TP53_mutation_Mutation_Type',
    'KRAS_mutation_Mutation_Type',
    'PIK3CA_mutation_Mutation_Type',
    'PTEN_mutation_Mutation_Type'
]
df[mutation_cols] = df[mutation_cols].fillna('None')

# Target and Features
target_col = 'Response'
feature_cols = [
    'Cancer Type', 'Current Age', 'Sex', 'MSI Score', 'Mutation Count',
    'Stage (Highest Recorded)', 'TMB (nonsynonymous)', 'Tumor Purity',
    'ICI_Class', 'TP53', 'KRAS', 'PIK3CA', 'PTEN',
    'TP53_mutation_Mutation_Type', 'KRAS_mutation_Mutation_Type',
    'PIK3CA_mutation_Mutation_Type', 'PTEN_mutation_Mutation_Type'
]

# Filter dataframe with only needed columns (optional)
df = df[feature_cols + [target_col]]

# Map target to binary 0/1
df[target_col] = df[target_col].str.strip().str.lower()
df = df[df[target_col].isin(['responder', 'non-responder'])]  # filter only valid rows
df[target_col] = df[target_col].map({'responder': 1, 'non-responder': 0})

X = df[feature_cols]
y = df[target_col]

# === 2. Define categorical and numerical columns ===

categorical_cols = [
    'Cancer Type', 'Sex', 'Stage (Highest Recorded)', 'ICI_Class',
    'TP53_mutation_Mutation_Type', 'KRAS_mutation_Mutation_Type',
    'PIK3CA_mutation_Mutation_Type', 'PTEN_mutation_Mutation_Type',
    'TP53', 'KRAS', 'PIK3CA', 'PTEN'
]

numerical_cols = [
    'Current Age', 'MSI Score', 'Mutation Count', 'TMB (nonsynonymous)', 'Tumor Purity'
]

# === 3. Preprocessing pipeline ===

numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# === 4. Define models to try ===

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Ridge Classifier': RidgeClassifier(random_state=42),
    'Lasso Logistic Regression': LogisticRegression(penalty='l1', solver='saga', max_iter=5000, random_state=42),
    'ElasticNet Logistic Regression': LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=0.5,
                                                         max_iter=5000, random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
    'LightGBM': LGBMClassifier(random_state=42),
    'CatBoost': CatBoostClassifier(verbose=0, random_state=42),
    'SVM': SVC(probability=True, random_state=42),
    'MLP': MLPClassifier(max_iter=1000, random_state=42),
    'Naive Bayes': GaussianNB(),
    'KNN': KNeighborsClassifier()
}

# === 5. Cross-validation and evaluation ===

scoring = {
    'accuracy': 'accuracy',
    'precision': 'precision',
    'recall': 'recall',
    'f1': 'f1',
    'roc_auc': 'roc_auc'
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

results = {}

print("Training and evaluating models...\n")

for name, model in models.items():
    print(f"Model: {name}")
    pipe = Pipeline(steps=[('preprocessor', preprocessor),
                           ('classifier', model)])
    accuracies = []
    precisions = []
    recalls = []
    f1s = []
    roc_aucs = []
    cm_total = np.zeros((2, 2), dtype=int)  # Initialize confusion matrix sum

    for train_idx, test_idx in cv.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        y_prob = None
        if hasattr(pipe.named_steps['classifier'], "predict_proba"):
            y_prob = pipe.predict_proba(X_test)[:, 1]
        elif hasattr(pipe.named_steps['classifier'], "decision_function"):
            y_prob = pipe.decision_function(X_test)

        accuracies.append(accuracy_score(y_test, y_pred))
        precisions.append(precision_score(y_test, y_pred, zero_division=0))
        recalls.append(recall_score(y_test, y_pred, zero_division=0))
        f1s.append(f1_score(y_test, y_pred, zero_division=0))
        if y_prob is not None:
            roc_aucs.append(roc_auc_score(y_test, y_prob))
        else:
            # If probability not available, skip ROC AUC for this fold
            roc_aucs.append(np.nan)

        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
        cm_total += cm

    # Average metrics (ignoring NaNs in ROC AUC)
    mean_roc_auc = np.nanmean(roc_aucs)
    results[name] = {
        'accuracy': np.mean(accuracies),
        'precision': np.mean(precisions),
        'recall': np.mean(recalls),
        'f1': np.mean(f1s),
        'roc_auc': mean_roc_auc,
        'confusion_matrix': cm_total
    }

    print(f"Accuracy: {results[name]['accuracy']:.4f}, Precision: {results[name]['precision']:.4f}, "
          f"Recall: {results[name]['recall']:.4f}, F1 Score: {results[name]['f1']:.4f}, ROC AUC: {mean_roc_auc:.4f}")
    print(f"Confusion Matrix (sum over folds):\n{cm_total}\n")

print("\n=== Summary (sorted by F1 score) ===")
sorted_results = sorted(results.items(), key=lambda x: x[1]['f1'], reverse=True)
for name, metrics in sorted_results:
    print(f"{name}: " + ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items() if k != 'confusion_matrix']))
    print(f"Confusion Matrix:\n{metrics['confusion_matrix']}\n")

# === 6. Save top 3 models ===

top_3 = sorted_results[:3]

for rank, (name, _) in enumerate(top_3, start=1):
    print(f"\nSaving top model #{rank}: {name}")
    model = models[name]
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('classifier', model)])
    pipeline.fit(X, y)  # train on full data before saving
    joblib.dump(pipeline, f'top_model_{rank}_{name.replace(" ", "_")}.joblib')

print("\nAll top 3 models saved as joblib files.")
