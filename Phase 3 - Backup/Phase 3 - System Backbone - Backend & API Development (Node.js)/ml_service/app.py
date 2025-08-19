import pandas as pd
import xgboost as xgb
import json
import os
from flask import Flask, request, jsonify
import numpy as np  # <-- 1. ADD THIS IMPORT

# Initialize the Flask application
app = Flask(__name__)

# --- Model & Configuration Loading ---
MODEL_PATH = 'credit_risk_model.json'
CONFIG_PATH = 'model_config.json'

# Load the trained XGBoost model
model = xgb.XGBClassifier()
model.load_model(MODEL_PATH)

# Load the model configuration (which includes the training columns)
with open(CONFIG_PATH, 'r') as f:
    config = json.load(f)
    TRAINING_COLUMNS = config['training_columns']

def engineer_features_for_prediction(df):
    """
    Engineers features for a single user prediction.
    This logic MUST mirror the training script exactly.
    """
    age_bins = [17, 25, 35, 50, 66]
    age_labels = ['18-25', '26-35', '36-50', '51+']
    df['age_group'] = pd.cut(df['age'], bins=age_bins, labels=age_labels, right=True).astype(str)

    def assign_risk_score(series, thresholds, ascending=True):
        labels = [0, 1, 2] if ascending else [2, 1, 0]
        return pd.cut(series, bins=[-float('inf')] + thresholds + [float('inf')], labels=labels, right=False).astype(int)

    df['debt_risk'] = assign_risk_score(df['debt_burden'], [0.4, 0.6])
    df['utility_risk'] = assign_risk_score(df['utility_payment_ratio'], [0.6, 0.9], ascending=False)
    df['bnpl_risk'] = assign_risk_score(df['bnpl_repayment_rate'], [0.5, 0.8], ascending=False)
    df['peer_risk'] = assign_risk_score(df['peer_default_exposure'], [0.2, 0.5])
    df['income_tier_risk'] = 2 - df['income_tier']
    df['tax_timeliness_risk'] = assign_risk_score(df['tax_payment_timeliness'], [0.7, 0.9], ascending=False)
    df['device_risk'] = assign_risk_score(df['device_tier'], thresholds=[1, 2], ascending=False)

    df['total_risk'] = df['debt_risk'] * 1.3 + df['income_tier_risk'] * 1.2 + df['tax_timeliness_risk'] * 1.5
    
    # --- 2. THIS IS THE CORRECTED LINE ---
    df['default_prob'] = 1 / (1 + np.exp(-0.25 * df['total_risk'] + 5))
    
    return df

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint to receive user data and return a loan default prediction."""
    data = request.get_json()
    user_df = pd.DataFrame(data, index=[0])
    user_df_engineered = engineer_features_for_prediction(user_df.copy())
    user_data_processed = pd.get_dummies(user_df_engineered, columns=['age_group'], drop_first=True)
    
    for col in TRAINING_COLUMNS:
        if col not in user_data_processed.columns:
            user_data_processed[col] = 0
            
    user_data_processed = user_data_processed[TRAINING_COLUMNS]
    
    prediction = model.predict(user_data_processed)[0]
    probability = model.predict_proba(user_data_processed)[0][1]
    
    result = {
        'prediction': int(prediction),
        'probability_of_default': float(probability)
    }
    
    return jsonify(result)

@app.route('/health', methods=['GET'])
def health_check():
    """A simple health check endpoint."""
    return jsonify({"status": "healthy"}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)