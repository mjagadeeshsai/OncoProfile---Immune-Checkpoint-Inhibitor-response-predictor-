import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

# Define your categorical and numerical columns
categorical_cols = [
    'Cancer Type', 'Sex', 'Stage (Highest Recorded)', 'ICI_Class',
    'TP53_mutation_Mutation_Type', 'KRAS_mutation_Mutation_Type',
    'PIK3CA_mutation_Mutation_Type', 'PTEN_mutation_Mutation_Type',
    'TP53', 'KRAS', 'PIK3CA', 'PTEN'
]

numerical_cols = [
    'Current Age', 'MSI Score', 'Mutation Count', 'TMB (nonsynonymous)', 'Tumor Purity'
]

# Load the preprocessing pipeline and models
preprocessing_pipeline = joblib.load("C:/Users/mjaga/PycharmProjects/PythonProject3/Ensemble/preprocessing_pipeline.pkl")
catboost_model = joblib.load("C:/Users/mjaga/PycharmProjects/PythonProject3/Ensemble/catboost_model.pkl")
xgb_model = joblib.load("C:/Users/mjaga/PycharmProjects/PythonProject3/Ensemble/xgb_model.pkl")
lgbm_model = joblib.load("C:/Users/mjaga/PycharmProjects/PythonProject3/Ensemble/lgbm_model.pkl")

# Load the dataset
data = pd.read_excel(r"C:\Users\mjaga\OneDrive\Documents\UNI-UCL\PHAY0055 - Dissertation\Datasets\Merged data\training_set.xlsx")
X = data[categorical_cols + numerical_cols]
y = data['Response']

# Transform features using the saved pipeline
X_transformed = preprocessing_pipeline.transform(X)

# Function to extract final feature names after encoding and polynomial expansion
def get_feature_names(preprocessor, poly):
    ohe = preprocessor.named_transformers_['cat'].named_steps['onehot']
    cat_features = ohe.get_feature_names_out(categorical_cols)
    num_features = numerical_cols
    original_features = np.hstack([cat_features, num_features])
    poly_features = poly.get_feature_names_out(original_features)
    return poly_features

# Extract feature names
poly = preprocessing_pipeline.named_steps['poly']
feature_names = get_feature_names(preprocessing_pipeline.named_steps['preprocessor'], poly)

# Function to run SHAP analysis with labeled plots
def shap_analysis(model, model_name, X_trans, feat_names):
    print(f"\nRunning SHAP for {model_name}...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_trans)

    # SHAP summary plot (global importance)
    plt.title(f"SHAP Summary Plot – {model_name}", fontsize=16)
    shap.summary_plot(shap_values, X_trans, feature_names=feat_names, plot_size=(14, 10), show=True)

    # SHAP waterfall plot (local explanation for sample 0)
    plt.figure(figsize=(14, 10))
    plt.title(f"SHAP Waterfall Plot – {model_name} (Sample 0)", fontsize=16)
    shap.plots.waterfall(shap.Explanation(
        values=shap_values[0],
        base_values=explainer.expected_value,
        data=X_trans[0],
        feature_names=feat_names
    ))

# Run SHAP analysis for each model with clear headings
print("===== SHAP Analysis: CatBoost Model =====")
shap_analysis(catboost_model, "CatBoost", X_transformed, feature_names)

print("===== SHAP Analysis: XGBoost Model =====")
shap_analysis(xgb_model, "XGBoost", X_transformed, feature_names)

print("===== SHAP Analysis: LightGBM Model =====")
shap_analysis(lgbm_model, "LightGBM", X_transformed, feature_names)