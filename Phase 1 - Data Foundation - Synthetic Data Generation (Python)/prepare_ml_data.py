# =============================================================================
# FILE: ml_preparer.py
# =============================================================================
import pandas as pd
import numpy as np
import os
from sqlalchemy import create_engine
import sys
import json

MERCHANT_CATEGORIES = { 'income': ['Salary Credit', 'Business Payout'], 'food': ['Zomato', 'Swiggy', 'Grofers'], 'ott': ['Netflix Subscription', 'Hotstar Payment'], 'loan_emi': ['Loan EMI Payment'], 'bnpl_emi': ['BNPL Repayment'] }

# --- NEW: Helper function to detect maternity leave from transaction history ---
def detect_consecutive_zero_income(user_credit_df, num_months=8):
    """
    Detects if a user has a consecutive period of zero-income months.
    This is used to infer a long leave period like maternity.
    """
    if user_credit_df.empty:
        return False
    
    # Ensure data is sorted by month
    user_credit_df = user_credit_df.sort_values('month')
    
    # Create a grouping key that increments every time the zero-income status changes
    is_zero_income = user_credit_df['amount'] == 0
    grouping_key = is_zero_income.ne(is_zero_income.shift()).cumsum()
    
    # Calculate the size of each consecutive group
    consecutive_counts = user_credit_df.groupby(grouping_key).size()
    
    # Get the counts only for the zero-income periods
    zero_income_stretch = consecutive_counts[user_credit_df.groupby(grouping_key)['amount'].first() == 0]
    
    # Check if any of these zero-income periods is long enough
    return (zero_income_stretch >= num_months).any()


def transform_data(profiles_df, upi_df, weights):
    """
    Extracts, transforms, and engineers features from raw data to create an ML-ready dataset.
    """
    print("\nStarting data transformation and feature engineering...")

    total_income = upi_df[upi_df['type'] == 'CREDIT'].groupby('user_id')['amount'].sum().rename('total_income')
    total_debits = upi_df[upi_df['type'] == 'DEBIT'].groupby('user_id')['amount'].sum().rename('total_debits')
    
    # --- MODIFIED: Income Consistency & Maternity Leave Detection ---
    print("Detecting maternity leave and calculating fair income consistency...")
    
    credit_transactions = upi_df[upi_df['type'] == 'CREDIT']
    
    # Step 1: Detect which female users are on maternity leave
    female_user_ids = profiles_df[profiles_df['gender'] == 'Female']['user_id']
    female_credits = credit_transactions[credit_transactions['user_id'].isin(female_user_ids)]
    
    maternity_leave_flags = female_credits.groupby('user_id').apply(detect_consecutive_zero_income)
    maternity_leave_users = maternity_leave_flags[maternity_leave_flags].index.tolist()

    # Step 2: Calculate consistency, excluding leave periods for those users
    # For everyone else, we still only consider months they were actually paid
    paid_months_income = credit_transactions[credit_transactions['amount'] > 0]
    
    # For users on leave, we'll use this data. For others, it's their normal paid months.
    # This fairly handles both cases.
    monthly_income = paid_months_income.groupby(['user_id', 'month'])['amount'].sum().unstack()
    
    # Identify users who should not be penalized for income gaps due to leave
    # We will simply use their paid-month consistency
    
    income_std_dev = monthly_income.std(axis=1)
    income_mean = monthly_income.mean(axis=1)
    
    # Coefficient of Variation: Lower is more stable
    income_consistency_cv = (income_std_dev / income_mean).fillna(0)
    
    # Step 3: Apply the NEW consistency score logic
    # Start with a base score
    income_consistency_score = 1 / (1 + income_consistency_cv)
    # Award a perfect score to those with very low volatility (less than 5% variation)
    income_consistency_score[income_consistency_cv < 0.05] = 1.0
    
    income_consistency_score = income_consistency_score.rename('income_consistency_score')
    
    # --- Other Features (No changes needed) ---
    num_months = upi_df['month'].nunique()
    food_spending = upi_df[upi_df['description'].isin(MERCHANT_CATEGORIES['food'])].groupby('user_id')['amount'].sum().div(num_months).rename('avg_monthly_food')
    ott_spending = upi_df[upi_df['description'].isin(MERCHANT_CATEGORIES['ott'])].groupby('user_id')['amount'].sum().div(num_months).rename('avg_monthly_ott')
    loan_repayments = upi_df[upi_df['description'].isin(MERCHANT_CATEGORIES['loan_emi'])].groupby('user_id')['month'].nunique().div(num_months).rename('loan_repayment_consistency')
    bnpl_repayments = upi_df[upi_df['description'].isin(MERCHANT_CATEGORIES['bnpl_emi'])].groupby('user_id')['month'].nunique().div(num_months).rename('bnpl_repayment_consistency')
    
    # --- Combine All Features ---
    ml_df = profiles_df.set_index('user_id')
    features_to_merge = [total_income, total_debits, income_consistency_score, food_spending, ott_spending, loan_repayments, bnpl_repayments]
    for feature in features_to_merge:
        ml_df = ml_df.merge(feature, left_index=True, right_index=True, how='left')
    ml_df = ml_df.fillna(0)

    # --- Final Score Calculation ---
    ml_df['debt_burden_ratio'] = (ml_df['total_debits'] / ml_df['total_income']).replace([np.inf, -np.inf], 0).fillna(0)
    ml_df['debt_burden_ratio'] = ml_df['debt_burden_ratio'].clip(0, 5)
    print("Calculating weighted behavior score...")
    ml_df['behavior_score'] = (
        ml_df['income_consistency_score'] * weights['income_consistency_weight'] +
        ml_df['avg_monthly_food'] * weights['food_spending_weight'] +
        ml_df['avg_monthly_ott'] * weights['ott_spending_weight'] +
        ml_df['loan_repayment_consistency'] * weights['loan_repayment_consistency_weight'] +
        ml_df['bnpl_repayment_consistency'] * weights['bnpl_repayment_consistency_weight'] +
        ml_df['debt_burden_ratio'] * weights['debt_burden_risk_weight']
    )
    print("Data transformation complete.")
    return ml_df.reset_index()

# The functions select_database_file, load_data_from_db, load_weights, save_ml_database, and the main execution block
# remain the same as the previous version. You can copy them here to complete the file.
def select_database_file(path):
    print(f"Scanning for databases in: {path}...")
    try: db_files = [f for f in os.listdir(path) if f.endswith('.db') and not f.startswith('ml_ready')]
    except FileNotFoundError: print(f"ERROR: Directory not found: {path}"); sys.exit()
    if not db_files: print("ERROR: No raw database (.db) files found."); sys.exit()
    print("\nPlease select the raw database to process:")
    for i, fname in enumerate(db_files): print(f"  {i+1}: {fname}")
    while True:
        try:
            choice = int(input(f"Enter your choice (1-{len(db_files)}): "))
            if 1 <= choice <= len(db_files): return os.path.join(path, db_files[choice-1])
            else: print("Invalid number.")
        except ValueError: print("Invalid input.")

def load_data_from_db(db_path):
    print(f"\nLoading data from {os.path.basename(db_path)}...")
    engine = create_engine(f'sqlite:///{db_path}')
    try:
        profiles_df = pd.read_sql_table('credit_data', engine)
        upi_df = pd.read_sql_table('upi_transactions', engine)
        print("Successfully loaded 'credit_data' and 'upi_transactions' tables.")
        return profiles_df, upi_df
    except Exception as e: print(f"ERROR: Could not load tables. {e}"); sys.exit()

def load_weights(filepath):
    print("\nLoading feature weights from weights.json...")
    try:
        with open(filepath, 'r') as f: weights = json.load(f)
        print("Weights loaded successfully."); return weights
    except FileNotFoundError: print(f"ERROR: weights.json not found."); sys.exit()
    except json.JSONDecodeError: print(f"ERROR: weights.json is not valid."); sys.exit()

def save_ml_database(df, input_path):
    base_name = os.path.basename(input_path).replace('.db', '')
    output_filename = f"ml_ready_for_{base_name}.db"
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), output_filename)
    print(f"\nSaving processed data to new database: {output_path}")
    engine = create_engine(f'sqlite:///{output_path}')
    df.to_sql('ml_training_data', engine, if_exists='replace', index=False)
    print("Save complete!"); return output_path 

if __name__ == "__main__":
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    RAW_DATA_PATH = r"C:\Users\ramun\Desktop\RLV\Fintech_Hackathon\Phase 1 - Data Foundation - Synthetic Data Generation (Python)" #<-- CHANGE THIS PATH IF NEEDED
    WEIGHTS_FILE_PATH = os.path.join(CURRENT_DIR, 'weights.json')
    selected_db = select_database_file(RAW_DATA_PATH)
    profiles_df, upi_df = load_data_from_db(selected_db)
    weights = load_weights(WEIGHTS_FILE_PATH)
    final_ml_df = transform_data(profiles_df, upi_df, weights)
    saved_filepath = save_ml_database(final_ml_df, selected_db)
    print(f"\nProcess finished successfully. The file '{os.path.basename(saved_filepath)}' is ready for machine learning.")