import matplotlib
# FIX: Set a non-interactive backend BEFORE importing pyplot
matplotlib.use('Agg')

import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import os
import glob # To find files matching a pattern
import json
import matplotlib.pyplot as plt
from datetime import datetime

def engineer_features_for_prediction(df):
    """
    Takes a SINGLE ROW raw dataframe and engineers features.
    THIS MUST MIRROR THE LOGIC IN THE TRAINING SCRIPT.
    """
    # Create the 'age_group' column from 'age'
    age_bins = [17, 25, 35, 50, 66]
    age_labels = ['18-25', '26-35', '36-50', '51+']
    df['age_group'] = pd.cut(df['age'], bins=age_bins, labels=age_labels, right=True).astype(str)

    def assign_risk_score(series, thresholds, ascending=True):
        labels = [0, 1, 2] if ascending else [2, 1, 0]
        return pd.cut(series, bins=[-np.inf] + thresholds + [np.inf], labels=labels, right=False).astype(int)

    # Create the same risk scores as the training script
    df['debt_risk'] = assign_risk_score(df['debt_burden'], [0.4, 0.6])
    df['utility_risk'] = assign_risk_score(df['utility_payment_ratio'], [0.6, 0.9], ascending=False)
    df['bnpl_risk'] = assign_risk_score(df['bnpl_repayment_rate'], [0.5, 0.8], ascending=False)
    df['peer_risk'] = assign_risk_score(df['peer_default_exposure'], [0.2, 0.5])
    df['income_tier_risk'] = 3 - df['income_tier'] # For income tiers 1, 2, 3
    df['device_risk'] = assign_risk_score(df['device_tier'], thresholds=[3, 5], ascending=False)
    
    return df

def get_prediction_and_explanation(user_df_engineered, model_path, training_columns, output_dir, applicant_name):
    """Loads a model, aligns data, predicts, and generates a SHAP plot."""
    model = xgb.XGBClassifier()
    model.load_model(model_path)
    
    # Preprocess data: one-hot encode and align columns perfectly
    user_data_processed = pd.get_dummies(user_df_engineered, columns=['age_group'], drop_first=True)
    
    # Ensure all columns the model was trained on are present
    for col in training_columns:
        if col not in user_data_processed.columns:
            user_data_processed[col] = 0
            
    # Ensure the column order is exactly the same
    user_data_processed = user_data_processed[training_columns]

    prediction = model.predict(user_data_processed)[0]
    probability = model.predict_proba(user_data_processed)[0][1]

    print("Generating SHAP explanation...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(user_data_processed)
    
    plot_path = os.path.join(output_dir, "explanation_plot.png")
    shap.force_plot(explainer.expected_value, shap_values[0,:], user_data_processed.iloc[0,:], matplotlib=True, show=False, figsize=(20, 4), text_rotation=30)
    plt.title(f'SHAP Explanation for {applicant_name}', loc='center')
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()
    
    return {"prediction": int(prediction), "probability": probability, "explanation_plot_path": plot_path}

def interactive_loan_advisor():
    """Main function to run the interactive command-line tool."""
    print("--- Interactive Loan Default Advisor ---")
    
    results_folders = sorted(glob.glob("results_*"))
    if not results_folders:
        print("ERROR: No trained models found. Run the training script first.")
        return

    print("\nPlease choose which model to use for this prediction:")
    available_models = []
    for folder in results_folders:
        model_file = os.path.join(folder, 'credit_risk_model.json')
        config_file = os.path.join(folder, 'model_config.json')
        if os.path.exists(model_file) and os.path.exists(config_file):
            available_models.append((folder, model_file, config_file))

    for i, (_, model_path, _) in enumerate(available_models):
        print(f"  {i + 1}: {model_path}")
    
    choice = int(input(f"Enter your choice (1-{len(available_models)}): "))
    selected_folder, model_path, config_path = available_models[choice - 1]

    with open(config_path, 'r') as f:
        config = json.load(f)
        training_columns = config['training_columns']
    
    print(f"\nUsing model: '{model_path}'")

    print("\nPlease provide the following RAW details for the applicant.")
    applicant_name = input("Enter a name or ID for this applicant (e.g., John_Doe): ")
    
    # Create a dictionary of the raw inputs
    user_inputs_raw = {
        'age': [int(input("Applicant's age?: "))],
        'monthly_income_rs': [float(input("Monthly Income in Rupees?: "))],
        'employment_tenure': [float(input("Employment Tenure (in months)?: "))],
        'debt_burden': [float(input("Debt-to-Income Ratio? (e.g., 0.5): "))],
        # Add other key raw inputs here if you want to make them interactive
    }
    
    # Create a full profile using baseline values for non-interactive features
    # These baselines should represent an "average" applicant
    full_profile_dict = {
        'user_id': [0], 'age': user_inputs_raw['age'], 'monthly_income_rs': user_inputs_raw['monthly_income_rs'],
        'income_tier': [pd.cut(user_inputs_raw['monthly_income_rs'], bins=[-1, 20000, 50000, np.inf], labels=[1, 2, 3])[0]],
        'employment_tenure': user_inputs_raw['employment_tenure'],
        'device_tier': [3], 'app_diversity': [25], 'clickstream_volatility': [0.5], 'peer_default_exposure': [0.3],
        'financial_coping_ability': [3], 'asset_diversity': [2], 'earner_density': [2],
        'urbanization_score': [0.7], 'local_unemployment_rate': [0.15],
        'income_tax_paid': [user_inputs_raw['monthly_income_rs'][0] * 12 * 0.1], 'tax_payment_timeliness': [0.85],
        'debt_burden': user_inputs_raw['debt_burden'],
        'utility_payment_ratio': [0.9], 'bnpl_repayment_rate': [0.9],
        'ott_spending_tier': [1], 'food_delivery_tier': [1], 'ride_hailing_tier': [1], 'skill_spend': [0],
        'transaction_to_income_ratio': [0.7] # Assuming 70% of income is transacted
    }
    user_df_raw = pd.DataFrame(full_profile_dict)
    
    # Now, engineer the features
    user_df_engineered = engineer_features_for_prediction(user_df_raw)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prediction_output_dir = os.path.join(selected_folder, 'single_user_predictions', f"{applicant_name}_{timestamp}")
    os.makedirs(prediction_output_dir, exist_ok=True)
    
    print("\nSending processed data to the selected model...")
    result = get_prediction_and_explanation(
        user_df_engineered, model_path, training_columns, prediction_output_dir, applicant_name
    )

    print("\n------------------------------------")
    print("          ADVISOR'S RESULT          ")
    print("------------------------------------")
    THRESHOLD = 0.50 # A standard threshold
    final_answer = "HIGH RISK (Default Predicted)" if result['prediction'] == 1 else "LOW RISK (Repayment Predicted)"
    print(f"Loan Default Risk: {final_answer}")
    print(f"(Confidence of Default: {result['probability']:.1%})")
    print(f"\n* A detailed explanation plot has been saved to:\n  '{result['explanation_plot_path']}'")
    print("------------------------------------")

if __name__ == "__main__":
    interactive_loan_advisor()