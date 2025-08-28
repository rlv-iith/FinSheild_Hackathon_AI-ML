import pandas as pd
import numpy as np
import os
from sqlalchemy import create_engine
import sys
import json
from tqdm import tqdm

# --- Dictionaries ---
MERCHANT_CATEGORIES = {
    'income': ['Salary Credit', 'Business Payout', 'No Salary'],
    'utility': ['Electricity Bill', 'Mobile Recharge', 'Gas Bill', 'Rent Payment', 'Utility Bill'],
    'food': ['Zomato', 'Swiggy', 'Grofers'],
    'ott': ['Netflix Subscription', 'Hotstar Payment'],
    'ride_hailing': ['Uber', 'Ola Cabs'],
    'loan_emi': ['Loan EMI Payment'],
    'bnpl_emi': ['BNPL Repayment'],
    'skill_spend': ['Coursera Subscription', 'Udemy Course'], 
    'shopping': ['Flipkart Purchase', 'Myntra Order', 'Nykaa Purchase'],
    'medical': ['Medical'],
    'misc': ['Unidentified Purchase']
}

# --- Helper Functions ---
def calculate_trend_slope(series):
    y = series.dropna()
    if len(y) < 2: return 0.0
    x = np.arange(len(y))
    try:
        slope, _ = np.polyfit(x, y, 1)
        return slope if np.isfinite(slope) else 0.0
    except np.linalg.LinAlgError:
        return 0.0

def count_consecutive_events(series, above_or_below='below', threshold=0):
    if above_or_below == 'below':
        events = series < threshold
    else:
        events = series > threshold
    if not events.any(): return 0
    return events.groupby((events != events.shift()).cumsum()).cumsum().max()

# --- Main Transformation Function ---
def transform_data(profiles_df, upi_df, digital_df):
    print("\nStarting definitive data transformation with advanced behavioral features...")

    # Step 1: Aggregation
    print("Step 1: Aggregating monthly financial data...")
    desc_to_cat = {desc: cat for cat, descs in MERCHANT_CATEGORIES.items() for desc in descs}
    upi_df['category'] = upi_df['description'].map(desc_to_cat).fillna('misc')
    monthly_financials = upi_df.groupby(['user_id', 'month', 'category'])['amount'].sum().unstack(fill_value=0)
    for cat in MERCHANT_CATEGORIES.keys():
        if cat not in monthly_financials.columns:
            monthly_financials[cat] = 0
            
    # Step 2: Granular Standalone Features
    print("Step 2: Engineering granular standalone features...")
    all_feature_sets = []
    # Group by the 'user_id' level of the MultiIndex
    grouped_financials = monthly_financials.groupby(level='user_id')
    for cat in monthly_financials.columns:
        features = grouped_financials[cat].agg(
            avg_monthly=lambda x: x[x>0].mean(),
            std_monthly=lambda x: x[x>0].std(),
            trend=calculate_trend_slope,
            total_months_active=lambda x: (x > 0).sum(),
            max_monthly='max',
            last_month_shock=lambda x: (x.iloc[-1] - x.mean()) / (x.mean() + 1e-6),
            consecutive_zero_months = lambda x: count_consecutive_events(x, 'below', 1)
        ).fillna(0).rename(columns=lambda x: f'{cat}_{x}')
        all_feature_sets.append(features)
        
    # Set user_id as index to join all engineered features
    ml_df = profiles_df.set_index('user_id')
    for feature_df in all_feature_sets:
        ml_df = ml_df.merge(feature_df, left_index=True, right_index=True, how='left')
    
    # Step 3: Advanced Relational Features
    print("Step 3: Engineering advanced relational and behavioral features...")
    monthly_income = monthly_financials['income'].replace(0, np.nan)
    ml_df['medical_spend_pct_of_income'] = (monthly_financials['medical'] / monthly_income).groupby(level='user_id').mean()
    ml_df['shopping_spend_pct_of_income'] = (monthly_financials['shopping'] / monthly_income).groupby(level='user_id').mean()
    ml_df['loan_repayment_pct_of_income'] = (monthly_financials['loan_emi'] / monthly_income).groupby(level='user_id').mean()
    discretionary_spend = monthly_financials[['shopping', 'ott', 'food']].sum(axis=1)
    investment_spend = monthly_financials['skill_spend']
    ml_df['investment_to_discretionary_ratio'] = (investment_spend / (discretionary_spend + 1e-6)).groupby(level='user_id').mean()
    essential_spend = monthly_financials[['utility', 'loan_emi', 'bnpl_emi']].sum(axis=1)
    ml_df['essential_to_discretionary_ratio_trend'] = (essential_spend / (discretionary_spend + 1e-6)).groupby(level='user_id').apply(calculate_trend_slope)
    peer_groups = ml_df.groupby('income_tier')
    peer_avg_shopping_pct = peer_groups['shopping_spend_pct_of_income'].mean().rename('peer_avg_shopping_pct')
    peer_avg_medical_pct = peer_groups['medical_spend_pct_of_income'].mean().rename('peer_avg_medical_pct')
    ml_df = ml_df.merge(peer_avg_shopping_pct, on='income_tier', how='left')
    ml_df = ml_df.merge(peer_avg_medical_pct, on='income_tier', how='left')
    ml_df['shopping_spend_vs_peer_ratio'] = ml_df['shopping_spend_pct_of_income'] / (ml_df['peer_avg_shopping_pct'] + 1e-6)
    ml_df['medical_spend_vs_peer_ratio'] = ml_df['medical_spend_pct_of_income'] / (ml_df['peer_avg_medical_pct'] + 1e-6)
    monthly_shocks = monthly_financials.diff().div(monthly_financials.shift() + 1e-6).fillna(0)
    ml_df['consecutive_months_increasing_misc_spend'] = monthly_shocks.groupby(level='user_id')['misc'].apply(count_consecutive_events, 'above', 0.1)
    ml_df['consecutive_months_decreasing_income'] = monthly_shocks.groupby(level='user_id')['income'].apply(count_consecutive_events, 'below', -0.1)
    
    # Step 4: Digital Footprint and Final Touches
    print("Step 4: Adding digital footprint and finalizing dataset...")
    if not digital_df.empty:
        # Check if user_id is already the index, if not, set it.
        if 'user_id' in digital_df.columns:
            digital_df.set_index('user_id', inplace=True)

        digital_features = digital_df.agg(
            digital_avg_screen_time=('avg_daily_screen_time_mins', 'mean'),
            digital_finance_app_pct_of_time=('minutes_finance', lambda x: (x / digital_df.loc[x.index, 'avg_daily_screen_time_mins'].replace(0,1)).mean())
        ).fillna(0)
        ml_df = ml_df.merge(digital_features, left_index=True, right_index=True, how='left')
    
    ml_df['debt_burden_ratio'] = (ml_df['loan_emi_avg_monthly'] + ml_df['bnpl_emi_avg_monthly']) / (ml_df['income_avg_monthly'] + 1e-6)
    ml_df = ml_df.fillna(0).replace([np.inf, -np.inf], 0)
    
    print("Definitive feature engineering complete.")
    
    # --- THIS IS THE PERMANENT FIX ---
    # It takes the 'user_id' from being the DataFrame's index and turns it back into a regular column.
    # This ensures it gets saved correctly to the database.
    return ml_df.reset_index().rename(columns={'index': 'user_id'})

# --- All utility functions (select_database_file, load_data_from_db, etc.) ---
def select_database_file(path):
    print(f"Scanning for databases in: {path}...")
    try:
        db_files = [f for f in os.listdir(path) if f.endswith('.db') and not f.startswith('ml_ready')]
    except FileNotFoundError:
        print(f"ERROR: Directory not found: {path}"); sys.exit()
    if not db_files:
        print("ERROR: No raw database (.db) files found in the specified directory."); sys.exit()
    
    print("\nPlease select the raw database to process:")
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

def load_data_from_db(db_path):
    print(f"\nLoading data from {os.path.basename(db_path)}...")
    engine = create_engine(f'sqlite:///{db_path}')
    try:
        profiles_df = pd.read_sql_table('credit_data', engine)
        upi_df = pd.read_sql_table('upi_transactions', engine)
        digital_df = pd.read_sql_table('digital_footprint', engine)
        print("Successfully loaded 'credit_data', 'upi_transactions', and 'digital_footprint'.")
        return profiles_df, upi_df, digital_df
    except ValueError as e:
        print(f"ERROR: Could not find one of the required tables in {db_path}. Details: {e}"); sys.exit()

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
    RAW_DATA_PATH = r"C:\Users\ramun\Desktop\RLV\Fintech_Hackathon\Phase 1 - Data Foundation - Synthetic Data Generation (Python)"
    
    selected_db = select_database_file(RAW_DATA_PATH)
    profiles_df, upi_df, digital_df = load_data_from_db(selected_db)
    
    final_ml_df = transform_data(profiles_df, upi_df, digital_df)
    
    saved_filepath = save_ml_database(final_ml_df, selected_db)
    print(f"\nProcess finished successfully. The file '{os.path.basename(saved_filepath)}' is now correctly formatted.")