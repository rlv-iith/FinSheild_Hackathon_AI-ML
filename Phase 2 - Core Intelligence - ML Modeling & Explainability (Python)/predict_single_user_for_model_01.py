import matplotlib
matplotlib.use('Agg') # Set non-interactive backend first

import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import os
import glob # To find files matching a pattern
import matplotlib.pyplot as plt
from datetime import datetime

def get_prediction_and_explanation(user_data, model_path, output_dir, applicant_name):
    """
    Loads a model, predicts for a user, and generates a SHAP explanation plot.
    Now with custom title formatting.
    """
    if not os.path.exists(model_path):
        return {"error": f"Model file not found at '{model_path}'"}
        
    # 1. Load the Model
    model = xgb.XGBClassifier()
    model.load_model(model_path)
    
    # 2. Preprocess data
    user_data_processed = pd.get_dummies(user_data, columns=['age_group'], drop_first=True)
    model_features = model.get_booster().feature_names
    
    for c in set(model_features) - set(user_data_processed.columns): user_data_processed[c] = 0
    user_data_processed = user_data_processed[model_features]

    # 3. Make Prediction
    prediction = model.predict(user_data_processed)[0]
    probability = model.predict_proba(user_data_processed)[0][1]

    # 4. Generate SHAP Explanation
    print("Generating SHAP explanation for this user...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(user_data_processed)

    plot_filename = "explanation_plot.png"
    force_plot_path = os.path.join(output_dir, plot_filename)
    
    # Create the force plot but DO NOT show it on screen
    shap.force_plot(
        explainer.expected_value, 
        shap_values[0,:], 
        user_data_processed.iloc[0,:], 
        matplotlib=True, 
        show=False, 
        text_rotation=10
    )
    
    # --- NEW: Custom Title Logic ---
    # We add a custom title at the bottom center of the figure area
    plt.figtext(
        0.5, # x-position (0.5 is horizontal center)
        0.01, # y-position (0.01 is very close to the bottom)
        f"SHAP Explanation for {applicant_name}", # The new dynamic title
        wrap=True, 
        horizontalalignment='center', 
        fontsize=12
    )

    # Save the figure, ensuring the custom title is not cropped
    plt.savefig(force_plot_path, bbox_inches='tight')
    plt.close()
    print(f"SHAP explanation plot saved to: '{force_plot_path}'")
    
    return {
        "prediction": int(prediction), 
        "probability": probability,
        "explanation_plot_path": force_plot_path
    }

def interactive_loan_advisor():
    """Main function to run the interactive command-line tool."""
    print("--- Interactive Loan Default Advisor ---")

    # --- Scan for and select a model to use ---
    print("\nScanning for available trained models...")
    results_folders = sorted(glob.glob("results_*"))
    
    available_models = []
    for folder in results_folders:
        model_file = os.path.join(folder, 'credit_risk_model.json')
        if os.path.exists(model_file):
            available_models.append((folder, model_file))

    if not available_models:
        print("ERROR: No trained models found. Please run 'train_and_explain_model.py' first.")
        return

    print("Please choose which model you want to use for this prediction:")
    for i, (_, model_path) in enumerate(available_models):
        print(f"  {i + 1}: {model_path}")
    
    model_choice = -1
    while not (1 <= model_choice <= len(available_models)):
        try: model_choice = int(input(f"Enter your choice (1-{len(available_models)}): "))
        except ValueError: pass
    
    selected_results_folder, selected_model_path = available_models[model_choice - 1]
    print(f"\nUsing model: '{selected_model_path}'")

    # --- Get user input ---
    print("\nPlease provide the following details for the applicant.")
    try:
        applicant_name = input("Enter a name or ID for this applicant (e.g., John_Doe): ")
        debt_burden = float(input("What is the applicant's Debt-to-Income Ratio? (e.g., 0.5 for 50%): "))
        income_consistency = float(input("How consistent is their income? (0.0 to 1.0): "))
        utility_payment_ratio = float(input("What percentage of utility bills do they pay on time? (e.g., 0.9): "))
        age = int(input("What is the applicant's age?: "))
    except ValueError:
        print("\nERROR: Invalid number entered. Please restart.")
        return

    # --- Construct user profile ---
    print("\nConstructing user profile for the model...")
    # This dictionary must contain every single feature the model was trained on.
    user_data_dict = {
        'debt_burden': [debt_burden], 'income_consistency': [income_consistency], 'utility_payment_ratio': [utility_payment_ratio], 'age': [age],
        'income_tier': [1], 'device_tier': [1], 'app_diversity': [20], 'peer_default_exposure': [0.3], 'financial_shock_coping': [3], 'asset_diversity': [2], 'earner_density': [1.8], 'unemployment_rate': [0.15], 'urban_score': [0.7], 'clickstream_volatility': [0.4], 'bnpl_repayment_rate': [0.7],
        'debt_risk': [2 if debt_burden > 0.6 else (1 if debt_burden > 0.4 else 0)], 'utility_risk': [2 if utility_payment_ratio < 0.6 else (1 if utility_payment_ratio < 0.9 else 0)],
        'bnpl_risk': [1], 'peer_risk': [1], 'device_risk': [1], 'app_diversity_risk': [1], 'clickstream_risk': [1], 'asset_diversity_risk': [1], 'financial_coping_risk': [1], 'earner_density_risk': [1], 'unemployment_risk': [1], 'urban_risk': [1], 'income_tier_risk': [1], 'ott_risk': [1], 'food_risk': [1], 'ride_risk': [1],
        'ott_spend_tier': [1], 'food_delivery_tier': [1], 'ride_hailing_tier': [1], 'age_group': [pd.cut([age], bins=[17, 25, 35, 50, 66], labels=['18-25', '26-35', '36-50', '51+'])[0]], 'total_risk': [0], 'default_prob': [0]
    }
    user_df = pd.DataFrame(user_data_dict)

    # --- Create unique, timestamped output directory for this prediction ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    user_folder_name = f"{applicant_name}_{timestamp}"
    prediction_output_dir = os.path.join(selected_results_folder, 'single_user_predictions', user_folder_name)
    os.makedirs(prediction_output_dir, exist_ok=True)
    print(f"Created a dedicated folder for this prediction: '{prediction_output_dir}/'")
    
    # --- GET PREDICTION AND EXPLANATION ---
    print("Sending data to the selected model...")
    result = get_prediction_and_explanation(
        user_data=user_df, 
        model_path=selected_model_path,
        output_dir=prediction_output_dir,
        applicant_name=applicant_name # Pass the name to the function
    )

    # --- DISPLAY FINAL ANSWER ---
    if "error" in result:
        print(f"\nERROR: {result['error']}")
    else:
        print("\n------------------------------------")
        print("          ADVISOR'S RESULT          ")
        print("------------------------------------")
        
        THRESHOLD = 0.40 
        final_answer = "HIGH RISK" if result['probability'] > THRESHOLD else "LOW RISK"
            
        print(f"Loan Default Risk: {final_answer}")
        print(f"(Confidence of Default: {result['probability']:.1%})")
        print(f"\n* A detailed explanation plot has been saved to:\n  '{result['explanation_plot_path']}'")
        print("------------------------------------")

if __name__ == "__main__":
    interactive_loan_advisor()