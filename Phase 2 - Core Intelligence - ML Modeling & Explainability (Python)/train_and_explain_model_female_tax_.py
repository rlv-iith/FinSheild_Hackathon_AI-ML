# =============================================================================
# FILE: model_trainer.py
# PURPOSE: To train, evaluate, and explain a credit risk model.
# This is the final, verified script for Phase 2.
# =============================================================================
import pandas as pd
import numpy as np
import os
from sqlalchemy import create_engine
import sys
import matplotlib.pyplot as plt
import joblib
import json

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, f1_score, precision_score, recall_score

import xgboost as xgb
import shap

# --- Configuration ---
# The current directory where this script is located.
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# --- Helper Functions ---
def select_database_file(path):
    """Scans a directory for ML-ready .db files and asks the user to pick one."""
    print(f"\nScanning for ML-ready databases in: {path}...")
    try:
        db_files = [f for f in os.listdir(path) if f.startswith('ml_ready_for_') and f.endswith('.db')]
    except FileNotFoundError:
        print(f"ERROR: Directory not found: {path}"); sys.exit()
    if not db_files:
        print("ERROR: No ML-ready database (.db) files found. Please run ml_preparer.py first."); sys.exit()
    print("\nPlease select the ML-ready database to use for training:")
    for i, fname in enumerate(db_files):
        print(f"  {i+1}: {fname}")
    while True:
        try:
            choice = int(input(f"Enter your choice (1-{len(db_files)}): "))
            if 1 <= choice <= len(db_files):
                return os.path.join(path, db_files[choice-1])
            else:
                print("Invalid number.")
        except ValueError:
            print("Invalid input.")

def generate_target_variable(df):
    """Generates a realistic 'loan_default' target variable."""
    print("\nGenerating realistic 'loan_default' target variable...")
    consistency_risk = 1 - df['income_consistency_score'] 
    debt_risk = (df['debt_burden_ratio'] - df['debt_burden_ratio'].min()) / (df['debt_burden_ratio'].max() - df['debt_burden_ratio'].min())
    repayment_risk = 1 - df['loan_repayment_consistency']
    prob = (0.50 * debt_risk + 0.35 * repayment_risk + 0.15 * consistency_risk)
    noise = np.random.normal(0, 0.05, len(df))
    final_prob = np.clip(prob + noise, 0.01, 0.95)
    df['loan_default'] = (np.random.rand(len(df)) < final_prob).astype(int)
    print(f"Generated {df['loan_default'].sum()} defaults (Default Rate: {df['loan_default'].mean()*100:.2f}%)")
    return df

def save_local_explanation_plot(data_row, explainer, base_value, file_path):
    """Generates and saves a local SHAP force plot as an HTML file."""
    shap_values_single = explainer.shap_values(data_row)
    p = shap.force_plot(base_value, shap_values_single, data_row, show=False)
    shap.save_html(file_path, p)

# --- Main Training Pipeline ---
def run_training_pipeline():
    """Executes the full training, evaluation, and artifact saving pipeline."""
    
    # 1. Load Data
    print("--- Data Loading Step ---")
    db_directory_path = input("Please enter the path to the directory containing your ML-ready database files: ")
    if not os.path.isdir(db_directory_path):
        print(f"\nERROR: The path provided is not a valid directory: '{db_directory_path}'"); sys.exit()

    db_path = select_database_file(db_directory_path)
    print(f"\nLoading data from {os.path.basename(db_path)}...")
    engine = create_engine(f'sqlite:///{db_path}')
    try:
        df = pd.read_sql_table('ml_training_data', engine)
    except Exception as e:
        print(f"ERROR: Could not load data. {e}"); sys.exit()

    # 2. Prepare Data
    df = generate_target_variable(df)
    y = df['loan_default']
    X = df.drop(columns=['loan_default', 'user_id', 'behavior_score'])
    
    X_processed = pd.get_dummies(X, columns=['gender', 'income_stability_type'], drop_first=True)
    
    # --- CRITICAL FIX for TypeError ---
    # Ensure all feature names are strings, required by newer scikit-learn versions.
    X_processed.columns = X_processed.columns.astype(str)
    
    # --- Interactive Data Splitting ---
    while True:
        choice = input("\nChoose test data source:\n  1: Split the loaded dataset.\n  2: Provide a URL to a separate test dataset.\nEnter choice (1 or 2): ")
        if choice in ['1', '2']:
            break
        print("Invalid choice. Please enter 1 or 2.")
    
    if choice == '1':
        while True:
            try:
                test_ratio = float(input("Enter the test set size as a decimal (e.g., 0.2 for 20%): "))
                if 0 < test_ratio < 1:
                    break
                else:
                    print("Please enter a value between 0 and 1.")
            except ValueError:
                print("Invalid input. Please enter a number.")
        X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=test_ratio, random_state=42, stratify=y)

    elif choice == '2':
        test_url = input("Enter the URL of the test dataset CSV file: ")
        try:
            print("Loading test data from URL...")
            test_df = pd.read_csv(test_url)
            print("Test data loaded successfully.")
            
            X_train, y_train = X_processed, y
            
            y_test = test_df['loan_default']
            X_test_raw = test_df.drop(columns=['loan_default', 'user_id', 'behavior_score'])
            X_test = pd.get_dummies(X_test_raw, columns=['gender', 'income_stability_type'], drop_first=True)
            X_test.columns = X_test.columns.astype(str)
            
            X_test = X_test.reindex(columns=X_train.columns, fill_value=0)
            
        except Exception as e:
            print(f"ERROR: Could not load or process the test data from URL. {e}")
            sys.exit()

    print(f"\nData split complete: {len(X_train)} training records, {len(X_test)} testing records.")

    # 3. Train Both Models
    print("\n--- Training Models ---")
    scale_pos_weight = np.sum(y_train == 0) / np.sum(y_train == 1)
    
    xgb_model = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss', scale_pos_weight=scale_pos_weight, random_state=42, use_label_encoder=False)
    xgb_model.fit(X_train, y_train)
    print("XGBoost training complete.")
    
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    rf_model.fit(X_train, y_train)
    print("Random Forest training complete.")

    # 4. Evaluate Models
    print("\n--- Model Evaluation ---")
    models = {'XGBoost': xgb_model, 'Random Forest': rf_model}
    results = {}
    for name, model in models.items():
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        results[name] = {
            'AUC-ROC': roc_auc_score(y_test, y_pred_proba),
            'F1-Score': f1_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred),
            'Recall': recall_score(y_test, y_pred)
        }
        print(f"\n--- {name} ---")
        print(pd.DataFrame(results[name], index=[0]))
        print("\nClassification Report:\n", classification_report(y_test, y_pred))

    # 5. Select Best Model
    best_model_name = max(results, key=lambda name: results[name]['AUC-ROC'])
    best_model = models[best_model_name]
    print(f"\n--- Best Performing Model: {best_model_name} (based on AUC-ROC) ---")

    # 6. Save Artifacts for the Best Model
    print("\n--- Saving Model Artifacts ---")
    dataset_name = os.path.basename(db_path).replace('.db', '')
    base_output_dir = os.path.join(CURRENT_DIR, f"model_results_for_{dataset_name}")
    
    output_dir = base_output_dir
    counter = 1
    while os.path.exists(output_dir):
        output_dir = f"{base_output_dir}_{counter}"
        counter += 1
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"Artifacts for this run will be saved in: '{output_dir}/'")

    joblib.dump(best_model, os.path.join(output_dir, 'best_model.joblib'))
    
    config = {'training_columns': X_train.columns.tolist(), 'model_name': best_model_name}
    with open(os.path.join(output_dir, 'model_config.json'), 'w') as f:
        json.dump(config, f, indent=4)
        
    print(f"Saved best model ({best_model_name}) and config.")

    # 7. Implement and Save SHAP Explanations
    print("\n--- Generating and Saving SHAP Explanations ---")
    explainer = shap.TreeExplainer(best_model)
    joblib.dump(explainer, os.path.join(output_dir, 'shap_explainer.joblib'))
    print("Saved SHAP explainer.")

    shap_values = explainer.shap_values(X_test)
    
    shap.summary_plot(shap_values, X_test, show=False)
    plt.title(f"Global Feature Importance ({best_model_name})")
    plt.savefig(os.path.join(output_dir, 'global_feature_importance.png'), bbox_inches='tight'); plt.close()
    print("Saved global feature importance plot.")
    
    high_risk_indices = np.where(best_model.predict(X_test) == 1)[0]
    if len(high_risk_indices) > 0:
        example_idx = high_risk_indices[0]
        local_explanation_path = os.path.join(output_dir, 'local_explanation_example.html')
        save_local_explanation_plot(X_test.iloc[[example_idx]], explainer, explainer.expected_value, local_explanation_path)
        print(f"Saved example local explanation to '{local_explanation_path}'.")
    
    print("\n--- Training Pipeline Complete ---")

# --- Main Execution Control ---
if __name__ == "__main__":
    run_training_pipeline()