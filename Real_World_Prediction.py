import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report

# Load models and pipeline
pipeline = joblib.load("C:/Users/mjaga/PycharmProjects/PythonProject3/Ensemble/preprocessing_pipeline.pkl")
CatBoost_model = joblib.load(r"C:\Users\mjaga\PycharmProjects\PythonProject3\catboost_model.pkl")

# Label mapping
label_map = {0: "Non-responder", 1: "Responder"}

# Load your Excel file with input data
excel_path = r"C:\Users\mjaga\OneDrive\Documents\UNI-UCL\PHAY0055 - Dissertation\Datasets\Merged data\test_set_1.xlsx"
data = pd.read_excel(excel_path)

# Columns
categorical_cols = [
    'Cancer Type', 'Sex', 'Stage (Highest Recorded)', 'ICI_Class',
    'TP53_mutation_Mutation_Type', 'KRAS_mutation_Mutation_Type',
    'PIK3CA_mutation_Mutation_Type', 'PTEN_mutation_Mutation_Type',
    'TP53', 'KRAS', 'PIK3CA', 'PTEN'
]
numerical_cols = [
    'Current Age', 'MSI Score', 'Mutation Count', 'TMB (nonsynonymous)', 'Tumor Purity'
]

# Check if true labels exist for evaluation
has_labels = 'Response' in data.columns

# If labels exist, convert to binary for evaluation
if has_labels:
    y_true = data['Response'].map({'Non-responder': 0, 'Responder': 1})

# Extract features
X_new = data[categorical_cols + numerical_cols]

# Preprocess
X_new_transformed = pipeline.transform(X_new)

# Predict
predictions = CatBoost_model.predict(X_new_transformed)
probabilities = CatBoost_model.predict_proba(X_new_transformed)

# Add predictions and confidence
data['Predicted Response'] = [label_map[p] for p in predictions]
data['Prediction Confidence'] = [f"{100 * max(prob):.2f}%" for prob in probabilities]

# Save predictions
data.to_excel(excel_path, index=False)
print(f"Predictions saved to {excel_path}")

# ----- Evaluation -----
if has_labels:
    accuracy = accuracy_score(y_true, predictions)
    precision = precision_score(y_true, predictions)
    recall = recall_score(y_true, predictions)
    f1 = f1_score(y_true, predictions)
    roc_auc = roc_auc_score(y_true, probabilities[:, 1])
    cm = confusion_matrix(y_true, predictions)
    report = classification_report(y_true, predictions, target_names=['Non-responder', 'Responder'])

    print("\n--- Model Evaluation on Test Set ---")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"ROC AUC:   {roc_auc:.4f}")
    print(f"\nConfusion Matrix:\n{cm}")
    print("\nClassification Report:")
    print(report)
