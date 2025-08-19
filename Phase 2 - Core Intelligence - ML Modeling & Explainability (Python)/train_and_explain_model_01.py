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
import json # To handle the config file

def get_user_weights():
    """Interactively gets weights for risk categories from the user."""
    print("\n--- Configure Custom Risk Weights ---")
    print("Please provide weights for each major risk category. They should ideally sum to 1.0.")
    
    weights = {}
    try:
        weights['debt_risk'] = float(input("Weight for Debt Risk (e.g., 0.3): "))
        weights['income_tier_risk'] = float(input("Weight for Income Tier Risk (e.g., 0.25): "))
        weights['utility_risk'] = float(input("Weight for Utility Payment Risk (e.g., 0.15): "))
        weights['spending_risk'] = float(input("Weight for Discretionary Spending Risk (e.g., 0.1): "))
        weights['behavioral_risk'] = float(input("Weight for Other Behavioral Risks (e.g., 0.2): "))
    except ValueError:
        print("\nInvalid input detected. Reverting to default weights.")
        return {'debt_risk': 0.3, 'income_tier_risk': 0.25, 'utility_risk': 0.15, 'spending_risk': 0.1, 'behavioral_risk': 0.2}

    print("\nCustom weights configured.")
    return weights

def calculate_granular_debt_risk(debt_burden):
    """Calculates a more sensitive debt risk score. Higher value is higher risk."""
    if debt_burden <= 0.4: return 0
    if debt_burden <= 0.6: return 1
    if debt_burden <= 0.9: return 2
    if debt_burden <= 1.5: return 3
    return 4 # Catastrophic risk for anything over 150% DTI

def main():
    """Main function to run the model training and explanation pipeline."""
    print("--- Phase 2: ML Modeling & Explainability (Expert-Guided) ---")
    while True:
        data_file_path = input("\nPlease enter the path to your training dataset CSV file: ")
        if os.path.exists(data_file_path):
            if input(f"Use file: '{os.path.basename(data_file_path)}'? (y/n): ").lower() == 'y':
                break
        else:
            print("ERROR: File not found. Please try again.")

    weights = get_user_weights()
    print(f"Loading main dataset from: {data_file_path}")
    df = pd.read_csv(data_file_path)

    print("\nEngineering features with granular scoring and custom expert weights...")
    df['debt_risk'] = df['debt_burden'].apply(calculate_granular_debt_risk)
    
    spending_features = ['ott_risk', 'food_risk', 'ride_risk']
    behavioral_features = ['device_risk', 'app_diversity_risk', 'clickstream_risk', 'bnpl_risk', 'peer_risk', 'financial_coping_risk']
    
    df['spending_risk_sum'] = df[spending_features].sum(axis=1)
    df['behavioral_risk_sum'] = df[behavioral_features].sum(axis=1)

    df['custom_weighted_risk'] = (
        df['debt_risk'] * weights['debt_risk'] +
        df['income_tier_risk'] * weights['income_tier_risk'] +
        df['utility_risk'] * weights['utility_risk'] +
        df['spending_risk_sum'] * weights['spending_risk'] +
        df['behavioral_risk_sum'] * weights['behavioral_risk']
    )
    print("New 'custom_weighted_risk' feature created successfully.")

    dataset_name = os.path.splitext(os.path.basename(data_file_path))[0]
    base_output_dir = f"results_{dataset_name}"
    output_dir = base_output_dir
    counter = 1
    while os.path.exists(output_dir):
        output_dir = f"{base_output_dir}_{counter}"
        counter += 1
    os.makedirs(output_dir, exist_ok=True)
    print(f"Results for this run will be saved in a new unique folder: '{output_dir}/'")

    y = df['loan_default']
    X = df.drop(['loan_default', 'user_id', 'spending_risk_sum', 'behavioral_risk_sum'], axis=1, errors='ignore')

    while True:
        choice = input("\nChoose your test data source:\n  1: Split main dataset.\n  2: Provide a separate test file.\nEnter choice (1 or 2): ")
        if choice in ['1', '2']: break
        print("Invalid choice.")

    if choice == '1':
        while True:
            try:
                test_ratio_str = input("Enter test set size as a decimal (e.g., 0.2): ")
                test_ratio = float(test_ratio_str)
                break
            except ValueError:
                print("Invalid input. Please enter a number.")
        
        X = pd.get_dummies(X, columns=['age_group'], drop_first=True)
        print(f"Splitting data with a {test_ratio*100:.0f}/{100-test_ratio*100:.0f} ratio...")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio, random_state=42, stratify=y)
    
    elif choice == '2':
        while True:
            test_file_path = input("Enter the full path to your test data CSV file: ")
            if os.path.exists(test_file_path): break
            else: print("File not found. Please check the path.")
        
        test_df = pd.read_csv(test_file_path)
        y_train = df['loan_default'] # The full original df is used for training
        X_train = X
        y_test = test_df['loan_default']
        X_test = test_df.drop(['loan_default', 'user_id'], axis=1, errors='ignore')
        
        print("Aligning columns between training and test sets...")
        X_train = pd.get_dummies(X_train, columns=['age_group'], drop_first=True)
        X_test = pd.get_dummies(X_test, columns=['age_group'], drop_first=True)
        train_cols = X_train.columns
        for c in set(train_cols) - set(X_test.columns): X_test[c] = 0
        for c in set(X_test.columns) - set(train_cols): X_train[c] = 0
        X_test = X_test[train_cols]
    
    print(f"\nTraining set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")

    print("\nTraining the XGBoost Classifier...")
    scale_pos_weight = np.sum(y_train == 0) / np.sum(y_train == 1)
    model = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss', n_estimators=150, max_depth=5, learning_rate=0.1, scale_pos_weight=scale_pos_weight, random_state=42)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=25)
    print("\nModel training complete.")

    model_filename = os.path.join(output_dir, 'credit_risk_model.json')
    model.save_model(model_filename)
    print(f"Model saved to: {model_filename}")

    config_data = { "model_path": model_filename, "custom_weights_used": weights }
    config_filename = os.path.join(output_dir, 'model_run_config.json')
    with open(config_filename, 'w') as f: json.dump(config_data, f, indent=4)
    print(f"Model configuration saved to '{config_filename}'")
        
    print("\nEvaluating Model Performance...")
    y_pred = model.predict(X_test)
    # --- THIS IS THE CORRECTED LINE ---
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"AUC Score: {roc_auc_score(y_test, y_pred_proba):.4f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, cmap='Blues').plot()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close()

    print("\nGenerating SHAP explanations...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    
    shap.summary_plot(shap_values, X_test, show=False)
    plt.title('Global Feature Importance')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'shap_summary_plot.png'))
    plt.close()

    high_risk_indices, low_risk_indices = np.where(y_pred == 1)[0], np.where(y_pred == 0)[0]
    if len(high_risk_indices) > 0 and len(low_risk_indices) > 0:
        shap.force_plot(explainer.expected_value, shap_values[high_risk_indices[0],:], X_test.iloc[high_risk_indices[0],:], matplotlib=True, show=False)
        plt.tight_layout(); plt.savefig(os.path.join(output_dir, 'shap_force_plot_high_risk.png')); plt.close()
        shap.force_plot(explainer.expected_value, shap_values[low_risk_indices[0],:], X_test.iloc[low_risk_indices[0],:], matplotlib=True, show=False)
        plt.tight_layout(); plt.savefig(os.path.join(output_dir, 'shap_force_plot_low_risk.png')); plt.close()
    else:
        print("Could not generate individual force plots due to lack of diverse predictions in the test set.")

    print(f"\n--- Phase 2 Complete --- All results saved in '{output_dir}' folder.")

if __name__ == "__main__":
    main()