# =============================================================================
# FILE: app.py
# PURPOSE: A Flask API to serve the credit risk model.
# =============================================================================
import os
import joblib
import json
import pandas as pd
from flask import Flask, request, jsonify

# Initialize Flask app
app = Flask(__name__)

# --- Load Model and Artifacts ---
# This part runs only once when the service starts.
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))

print("Loading model artifacts...")
try:
    # Load the trained model
    model = joblib.load(os.path.join(MODEL_DIR, 'credit_risk_model.joblib'))
    
    # Load the model configuration (contains the list of training columns)
    with open(os.path.join(MODEL_DIR, 'model_config.json'), 'r') as f:
        config = json.load(f)
        # ------------------- THIS IS THE FIX -------------------
        TRAINING_COLUMNS = config['training_columns'] 
        # -------------------------------------------------------
    
    # Load the SHAP explainer
    explainer = joblib.load(os.path.join(MODEL_DIR, 'shap_explainer.joblib'))
    
    print("Artifacts loaded successfully.")
except Exception as e:
    print(f"Error loading artifacts: {e}")
    model, TRAINING_COLUMNS, explainer = None, None, None

# --- API Endpoint for Prediction ---
@app.route('/predict', methods=['POST'])
def predict():
    if not model:
        return jsonify({"error": "Model is not loaded."}), 500

    # Get user data from the POST request
    data = request.get_json()
    if not data:
        return jsonify({"error": "No input data provided."}), 400

    try:
        # Convert incoming JSON to a pandas DataFrame
        input_df = pd.DataFrame([data])

        # --- Preprocessing ---
        # 1. One-hot encode categorical features
        processed_df = pd.get_dummies(input_df, columns=['gender', 'income_stability_type'], drop_first=True)

        # 2. Align columns with the training data to handle missing/extra columns
        processed_df = processed_df.reindex(columns=TRAINING_COLUMNS, fill_value=0)

        # 3. Ensure column names are strings
        processed_df.columns = [str(col) for col in processed_df.columns]

        # --- Prediction ---
        prediction = model.predict(processed_df)[0]
        prediction_proba = model.predict_proba(processed_df)[0]

        # --- Explanation ---
        shap_values = explainer.shap_values(processed_df)[0]
        feature_names = processed_df.columns

        # Create a dictionary of feature influences
        feature_influence = {feature: f"{value:.4f}" for feature, value in zip(feature_names, shap_values)}

        # --- Prepare Response ---
        output = {
            'prediction': 'High Risk' if int(prediction) == 1 else 'Low Risk',
            'confidence_score': {
                'low_risk': f"{prediction_proba[0]:.4f}",
                'high_risk': f"{prediction_proba[1]:.4f}"
            },
            'feature_influence_shap': feature_influence
        }
        
        return jsonify(output)

    except Exception as e:
        return jsonify({"error": f"An error occurred during prediction: {str(e)}"}), 500

# Run the app
if __name__ == '__main__':
    # Using Gunicorn is recommended for production instead of app.run()
    # This block is for direct execution (e.g., `python app.py`)
    app.run(host='0.0.0.0', port=5000)