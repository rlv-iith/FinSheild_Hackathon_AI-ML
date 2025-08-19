import matplotlib
# FIX: Set a non-interactive backend BEFORE importing pyplot
matplotlib.use('Agg')

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, ConfusionMatrixDisplay
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import os
import json

def engineer_features(df):
    """Takes a raw dataframe from the v9 generator and engineers the necessary features."""
    print("\nEngineering new features and risk scores from raw data...")
    age_bins = [17, 25, 35, 50, 66]
    age_labels = ['18-25', '26-35', '36-50', '51+']
    df['age_group'] = pd.cut(df['age'], bins=age_bins, labels=age_labels, right=True).astype(str)

    def assign_risk_score(series, thresholds, ascending=True):
        labels = [0, 1, 2] if ascending else [2, 1, 0]
        return pd.cut(series, bins=[-np.inf] + thresholds + [np.inf], labels=labels, right=False).astype(int)

    df['debt_risk'] = assign_risk_score(df['debt_burden'], [0.4, 0.6])
    df['utility_risk'] = assign_risk_score(df['utility_payment_ratio'], [0.6, 0.9], ascending=False)
    df['bnpl_risk'] = assign_risk_score(df['bnpl_repayment_rate'], [0.5, 0.8], ascending=False)
    df['peer_risk'] = assign_risk_score(df['peer_default_exposure'], [0.2, 0.5])
    df['income_tier_risk'] = 3 - df['income_tier'] # For income tiers 1, 2, 3
    df['device_risk'] = assign_risk_score(df['device_tier'], thresholds=[3, 5], ascending=False)
    print("Feature engineering complete.")
    return df

def main():
    """Main function to train and explain a model from raw data."""
    print("--- Phase 2: ML Modeling & Explainability (from Raw Data) ---")
    while True:
        data_file_path = input("\nPlease enter the path to your training dataset CSV file: ")
        if os.path.exists(data_file_path):
            if input(f"Use file: '{os.path.basename(data_file_path)}'? (y/n): ").lower() == 'y': break
        else: print("ERROR: File not found.")

    print(f"Loading main dataset from: {data_file_path}")
    df = pd.read_csv(data_file_path)

    if 'loan_repaid' in df.columns and df['loan_repaid'].isnull().any():
        print("Found empty 'loan_repaid' column. Simulating 'loan_default' outcomes for training...")
        prob = ((3 - df['income_tier']) / 2) * 0.4 + np.clip(df['debt_burden'], 0, 1) * 0.6
        df['loan_default'] = (np.random.rand(len(df)) < np.clip(prob, 0.05, 0.95)).astype(int)
        df = df.drop('loan_repaid', axis=1, errors='ignore')

    df = engineer_features(df)
    
    dataset_name = os.path.splitext(os.path.basename(data_file_path))[0]
    base_output_dir = f"results_{dataset_name}"
    output_dir = base_output_dir; counter = 1
    while os.path.exists(output_dir):
        output_dir = f"{base_output_dir}_{counter}"; counter += 1
    os.makedirs(output_dir, exist_ok=True)
    print(f"Results for this run will be saved in: '{output_dir}/'")
    model_name_for_plot = os.path.basename(output_dir)

    y = df['loan_default'] 
    X = df.drop(['loan_default', 'user_id'], axis=1, errors='ignore')
    
    X_train, X_test, y_train, y_test = None, None, None, None
    is_live_test_set = False # Flag for special plotting

    while True:
        choice = input("\nChoose test data source:\n  1: Split main dataset.\n  2: Provide a separate test file.\nEnter choice (1 or 2): ")
        if choice in ['1', '2']: break

    if choice == '1':
        test_ratio = float(input("Enter test set size (e.g., 0.2): "))
        X = pd.get_dummies(X, columns=['age_group'], drop_first=True)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio, random_state=42, stratify=y)
    elif choice == '2':
        test_file_path = input("Enter path to your test data CSV file: ")
        test_df = pd.read_csv(test_file_path)
        
        if 'is_live_user' in test_df.columns:
            is_live_test_set = True
            print("Detected 'is_live_user' flag. Will highlight this user in SHAP summary plot.")
            
        test_df = engineer_features(test_df)
        
        y_test_labeled_rows = test_df.dropna(subset=['loan_repaid'])
        y_test = 1 - y_test_labeled_rows['loan_repaid']
        
        X_train = X.loc[y.index]
        y_train = y
        X_test = test_df.reindex(y_test.index)
        
        if y_test.empty:
            print("ERROR: Test file has no labeled rows. Cannot evaluate.")
            return

        X_train = pd.get_dummies(X_train, columns=['age_group'], drop_first=True)
        X_test = pd.get_dummies(X_test, columns=['age_group'], drop_first=True)
        
        train_cols = X_train.columns
        for col in train_cols:
            if col not in X_test.columns: X_test[col] = 0
        X_test = X_test[train_cols]
    
    print(f"\nTraining set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")
    print("\nTraining the XGBoost Classifier...")
    scale_pos_weight = np.sum(y_train == 0) / np.sum(y_train == 1)
    model = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss', n_estimators=150, max_depth=5, learning_rate=0.1, scale_pos_weight=scale_pos_weight, random_state=42)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=25)
    print("\nModel training complete.")
    
    model.save_model(os.path.join(output_dir, 'credit_risk_model.json'))
    config = {'training_columns': X_train.columns.tolist()}
    with open(os.path.join(output_dir, 'model_config.json'), 'w') as f: json.dump(config, f, indent=4)
        
    print("\nEvaluating Model Performance...")
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"AUC Score: {roc_auc_score(y_test, y_pred_proba):.4f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    
    disp = ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, cmap='viridis')
    disp.ax_.set_title(f"Confusion Matrix for {model_name_for_plot}")
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), bbox_inches='tight'); plt.close()
    
    print("\nGenerating SHAP explanations...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    
    X_plot = X_test.copy()
    if is_live_test_set and 'is_live_user' in X_plot.columns:
        color_by = pd.Series(['Live User (You)' if x == 1 else 'Generated Persona' for x in X_plot['is_live_user']], index=X_plot.index)
        X_plot = X_plot.drop('is_live_user', axis=1)
        shap.summary_plot(shap_values, X_plot, feature_names=X_plot.columns, color=color_by, show=False)
        plt.title(f"Global Feature Importance for {model_name_for_plot}\n(Live User Highlighted)")
    else:
        shap.summary_plot(shap_values, X_plot, show=False)
        plt.title(f"Global Feature Importance for {model_name_for_plot}")
    explanation_text = ("How to read this plot:\n"
                        "- Each row is a feature, ordered by importance.\n"
                        "- Each dot is a person in the test set.\n"
                        "- Color indicates feature value (Red=High, Blue=Low).\n"
                        "- Position shows impact on prediction (Right=Default, Left=No Default).")
    plt.figtext(0.5, -0.15, explanation_text, ha="center", fontsize=9, bbox={"facecolor":"lightgray", "alpha":0.5, "pad":5})
    plt.savefig(os.path.join(output_dir, 'shap_summary_plot.png'), bbox_inches='tight'); plt.close()
    
    high_risk_indices = np.where(y_pred == 1)[0]
    low_risk_indices = np.where(y_pred == 0)[0]
    
    if len(high_risk_indices) > 0:
        high_risk_idx = high_risk_indices[0]
        shap.force_plot(explainer.expected_value, shap_values[high_risk_idx,:], X_plot.iloc[high_risk_idx,:], matplotlib=True, show=False, figsize=(20, 4), text_rotation=30)
        plt.title(f"SHAP Explanation for a High-Risk Prediction ({model_name_for_plot})", loc='center')
        plt.savefig(os.path.join(output_dir, 'shap_force_plot_high_risk.png'), bbox_inches='tight'); plt.close()
    else:
        print("\nNote: No 'High Risk' predictions were made on the test set, so no high-risk force plot was generated.")

    if len(low_risk_indices) > 0:
        low_risk_idx = low_risk_indices[0]
        shap.force_plot(explainer.expected_value, shap_values[low_risk_idx,:], X_plot.iloc[low_risk_idx,:], matplotlib=True, show=False, figsize=(20, 4), text_rotation=30)
        plt.title(f"SHAP Explanation for a Low-Risk Prediction ({model_name_for_plot})", loc='center')
        plt.savefig(os.path.join(output_dir, 'shap_force_plot_low_risk.png'), bbox_inches='tight'); plt.close()
    else:
        print("\nNote: No 'Low Risk' predictions were made on the test set, so no low-risk force plot was generated.")
        
    print(f"\n--- Phase 2 Complete --- All results saved in '{output_dir}' folder.")

if __name__ == "__main__":
    main()