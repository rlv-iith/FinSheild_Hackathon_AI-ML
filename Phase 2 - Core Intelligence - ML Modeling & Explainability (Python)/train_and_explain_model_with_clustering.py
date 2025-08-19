# =============================================================================
# FILE: train_with_clustering.py
# PURPOSE: To identify credit risk profiles using clustering and then train,
#          evaluate, and explain a predictive model.
# METHOD:
#   1. Load ML-ready data.
#   2. Use KMeans clustering to create user segments (Unsupervised).
#   3. Analyze clusters to identify a "high-risk" profile and create a
#      'loan_default' target label.
#   4. Train supervised classifiers (XGBoost, RandomForest) on these labels.
#   5. Evaluate the best model and generate SHAP explanations.
# =============================================================================
import pandas as pd
import numpy as np
import os
from sqlalchemy import create_engine
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib
import json

# ML Imports
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, f1_score, precision_score, recall_score, ConfusionMatrixDisplay

import xgboost as xgb
import shap

# --- Configuration ---
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
        print("ERROR: No ML-ready database (.db) files found."); sys.exit()
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

def perform_clustering_and_create_labels(df):
    """
    Performs KMeans clustering to identify user segments and assigns a 'loan_default' label.
    """
    print("\n--- Step 2: Unsupervised Clustering to Generate Labels ---")

    # Select features for clustering. Exclude identifiers, text, and pre-calculated scores.
    features_for_clustering = df.drop(columns=['user_id', 'gender', 'income_stability_type', 'behavior_score'], errors='ignore')

    # Scale data for KMeans
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features_for_clustering)

    # Determine number of clusters
    while True:
        try:
            k = int(input("Enter the number of clusters (groups) to identify (e.g., 3 or 4): "))
            if k > 1: break
            else: print("Please enter a number greater than 1.")
        except ValueError: print("Invalid input.")

    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    df['cluster'] = kmeans.fit_predict(scaled_features)
    print(f"\nSuccessfully segmented users into {k} clusters.")

    # Analyze the clusters to identify the high-risk group
    cluster_analysis = df.groupby('cluster')[['debt_burden_ratio', 'income_consistency_score', 'loan_repayment_consistency']].mean()
    print("\nCluster Analysis (average values):")
    print(cluster_analysis)

    # Identify the highest risk cluster automatically
    # Heuristic: Highest debt burden and lowest consistency scores are riskiest.
    # We create a composite risk score for each cluster.
    cluster_risk_score = (cluster_analysis['debt_burden_ratio'] - (cluster_analysis['income_consistency_score'] * 0.5) - (cluster_analysis['loan_repayment_consistency'] * 0.5))
    high_risk_cluster_id = cluster_risk_score.idxmax()

    print(f"\nAutomatically identified Cluster #{high_risk_cluster_id} as the highest-risk profile.")
    
    # Create the target variable
    df['loan_default'] = (df['cluster'] == high_risk_cluster_id).astype(int)
    print(f"Generated {df['loan_default'].sum()} defaults (Default Rate: {df['loan_default'].mean()*100:.2f}%)")
    return df

# --- Main Training Pipeline ---
def run_training_pipeline():
    """Executes the full training, evaluation, and artifact saving pipeline."""

    # --- Step 1: Load Data and Define Train/Test Sets ---
    print("\n--- Step 1: Data Loading & Splitting ---")
    data_folder = os.path.normpath(input("Enter path to the folder with ML-ready databases (Phase 1 output): "))
    db_path = select_database_file(data_folder)
    print(f"\nLoading data from {os.path.basename(db_path)}...")
    engine = create_engine(f'sqlite:///{db_path}')
    try:
        full_df = pd.read_sql_table('ml_training_data', engine)
    except Exception as e:
        print(f"ERROR: Could not load 'ml_training_data' table. {e}"); sys.exit()

    df_train, df_test = None, None

    # Ask how to split the data BEFORE any processing
    while True:
        choice = input("\nChoose how to create the test set:\n  1: Split the loaded dataset.\n  2: Load a separate test file.\nEnter choice (1 or 2): ")
        if choice in ['1', '2']: break
        print("Invalid choice.")

    if choice == '1':
        test_ratio = float(input("Enter test set size (e.g., 0.25): "))
        df_train, df_test = train_test_split(full_df, test_size=test_ratio, random_state=42)
        print("Data successfully split into training and testing sets.")
    elif choice == '2':
        df_train = full_df
        test_file_path = input("Enter path to the test dataset file (.csv or .db): ")
        try:
            if test_file_path.endswith('.csv'): df_test = pd.read_csv(test_file_path)
            elif test_file_path.endswith('.db'):
                test_engine = create_engine(f'sqlite:///{test_file_path}')
                df_test = pd.read_sql_table('ml_training_data', test_engine)
            else: raise ValueError("Unsupported file type.")
            print("External test data loaded successfully.")
        except Exception as e:
            print(f"ERROR: Could not load test data. {e}"); sys.exit()

    # --- Step 2: Unsupervised Clustering (on TRAINING data only) ---
    # Use .copy() to avoid SettingWithCopyWarning
    df_train_labeled = perform_clustering_and_create_labels(df_train.copy())

    # --- Step 3: Prepare Data for Supervised Models ---
    print("\n--- Step 3: Preparing Data for Supervised Training ---")
    y_train = df_train_labeled['loan_default']
    X_train_raw = df_train_labeled.drop(columns=['loan_default', 'user_id', 'behavior_score', 'cluster'])
    X_train = pd.get_dummies(X_train_raw, columns=['gender', 'income_stability_type'], drop_first=True)

    X_test_raw = df_test.drop(columns=['user_id', 'behavior_score', 'loan_default', 'cluster'], errors='ignore')
    X_test = pd.get_dummies(X_test_raw, columns=['gender', 'income_stability_type'], drop_first=True)

    # Align columns to ensure test set matches the training set structure
    X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)
    
    # Manually convert every column name to a string using a list comprehension.
    X_train.columns = [str(col) for col in X_train.columns]
    X_test.columns = [str(col) for col in X_test.columns]

    print(f"\nData ready: {len(X_train)} training records, {len(X_test)} testing records.")

    # --- Step 4: Train Supervised Models ---
    print("\n--- Step 4: Training Models ---")
    scale_pos_weight = np.sum(y_train == 0) / np.sum(y_train == 1) if np.sum(y_train == 1) > 0 else 1

    xgb_model = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss', scale_pos_weight=scale_pos_weight, random_state=42)
    xgb_model.fit(X_train, y_train)
    print("XGBoost training complete.")

    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    rf_model.fit(X_train, y_train)
    print("Random Forest training complete.")

    # --- Step 5: Evaluate and Select Best Model ---
    print("\n--- Step 5: Model Evaluation ---")
    X_train_eval, X_eval, y_train_eval, y_eval = train_test_split(X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)

    xgb_model.fit(X_train_eval, y_train_eval)
    rf_model.fit(X_train_eval, y_train_eval)
    
    models = {'XGBoost': xgb_model, 'Random Forest': rf_model}
    results = {}
    print("(Evaluating models on a 20% holdout set from the training data)")
    for name, model in models.items():
        eval_pred = model.predict(X_eval)
        eval_pred_proba = model.predict_proba(X_eval)[:, 1]
        results[name] = {'AUC-ROC': roc_auc_score(y_eval, eval_pred_proba), 'F1-Score': f1_score(y_eval, eval_pred)}
        print(f"\n--- {name} Performance ---")
        print(pd.DataFrame(results[name], index=[0]))
        print("\nClassification Report:\n", classification_report(y_eval, eval_pred))

    best_model_name = max(results, key=lambda name: results[name]['AUC-ROC'])
    best_model = models[best_model_name]
    print(f"\n--- Best Model Selected: {best_model_name} (based on AUC-ROC on evaluation set) ---")

    # --- Step 6: Save Artifacts ---
    print("\n--- Step 6: Saving Model Artifacts ---")
    dataset_name = os.path.basename(db_path).replace('.db', '')
    base_output_dir = os.path.join(CURRENT_DIR, f"results_{dataset_name}")
    output_dir = base_output_dir; counter = 1
    while os.path.exists(output_dir):
        output_dir = f"{base_output_dir}_{counter}"; counter += 1
    os.makedirs(output_dir, exist_ok=True)
    print(f"Artifacts will be saved in: '{output_dir}/'")

    explainer = shap.TreeExplainer(best_model)

    joblib.dump(best_model, os.path.join(output_dir, 'credit_risk_model.joblib'))
    joblib.dump(explainer, os.path.join(output_dir, 'shap_explainer.joblib'))
    config = {'training_columns': X_train.columns.tolist(), 'model_name': best_model_name}
    with open(os.path.join(output_dir, 'model_config.json'), 'w') as f:
        json.dump(config, f, indent=4)
    print("Saved best model, SHAP explainer, and configuration file.")
    
    # --- Step 7: SHAP Explanations on the Actual Test Set ---
    print("\n--- Step 7: Generating SHAP Explanations (on the primary test set) ---")
    shap_values = explainer.shap_values(X_test)

    shap.summary_plot(shap_values, X_test, show=False)
    plt.title(f"SHAP Feature Importance on Test Data ({best_model_name})")
    plt.savefig(os.path.join(output_dir, 'shap_summary_plot.png'), bbox_inches='tight'); plt.close()
    print("Saved SHAP summary plot for the test set.")

    y_pred_test = best_model.predict(X_test)
    high_risk_indices = np.where(y_pred_test == 1)[0]
    low_risk_indices = np.where(y_pred_test == 0)[0]

    if len(high_risk_indices) > 0:
        shap.force_plot(explainer.expected_value, shap_values[high_risk_indices[0],:], X_test.iloc[high_risk_indices[0],:], matplotlib=True, show=False, figsize=(20,4))
        plt.savefig(os.path.join(output_dir, 'shap_force_plot_high_risk.png'), bbox_inches='tight'); plt.close()
        print("Saved SHAP force plot for a high-risk prediction.")
        
    if len(low_risk_indices) > 0:
        shap.force_plot(explainer.expected_value, shap_values[low_risk_indices[0],:], X_test.iloc[low_risk_indices[0],:], matplotlib=True, show=False, figsize=(20,4))
        plt.savefig(os.path.join(output_dir, 'shap_force_plot_low_risk.png'), bbox_inches='tight'); plt.close()
        print("Saved SHAP force plot for a low-risk prediction.")

    print("\n\n--- Training & Explainability Pipeline Complete! ---")
    print(f"All outputs are in the '{output_dir}' folder.")

# --- Main Execution Control ---
if __name__ == "__main__":
    run_training_pipeline()