import pandas as pd
import numpy as np
from tqdm import tqdm
import os # Import the os module to interact with the file system

# --- Configuration ---
# You can change this number for different experiments
NUM_RECORDS = int(input("Enter number of records: "))
print(NUM_RECORDS)
TRANSACTION_MONTHS = 3

# --- Dictionaries for UPI Simulation ---
MERCHANT_CATEGORIES = {
    'income': ['Salary Credit', 'Business Payout'],
    'utility': ['Electricity Bill', 'Mobile Recharge', 'Gas Bill', 'Rent Payment'],
    'food': ['Zomato', 'Swiggy', 'Grofers'],
    'ott': ['Netflix Subscription', 'Hotstar Payment'],
    'ride_hailing': ['Uber', 'Ola Cabs'],
    'loan': ['Loan EMI Payment'],
    'misc_debit': ['Amazon Purchase', 'Myntra Order', 'Friend Transfer']
}

# --- PART 1: Generate Base Synthetic Profiles ---
print("Part 1: Generating base synthetic profiles...")
df = pd.DataFrame({
    'user_id': range(NUM_RECORDS),
    'age': np.clip(np.random.gamma(8, 4, NUM_RECORDS), 18, 65).astype(int)
})
df['age_group'] = pd.cut(df['age'], bins=[17, 25, 35, 50, 66], labels=['18-25', '26-35', '36-50', '51+'], right=True).astype(str)
df['income_tier'] = np.random.choice([0, 1, 2], NUM_RECORDS, p=[0.3, 0.5, 0.2])
df['device_tier'] = np.random.choice([0, 1, 2], NUM_RECORDS, p=[0.3, 0.5, 0.2])
df['app_diversity'] = np.random.poisson(15, NUM_RECORDS) + np.where(df['age'] < 30, 10, 0)
df['peer_default_exposure'] = np.random.beta(1.5, 5, NUM_RECORDS)
df['financial_shock_coping'] = np.random.randint(1, 6, NUM_RECORDS)
df['asset_diversity'] = np.random.poisson(0.5 + (df['age']/20), NUM_RECORDS)
df['earner_density'] = np.random.gamma(3, 0.6, NUM_RECORDS)
df['unemployment_rate'] = np.random.beta(2, 8, NUM_RECORDS) * 0.4
df['urban_score'] = np.random.beta(4, 2, NUM_RECORDS)
df['clickstream_volatility'] = np.random.beta(2, 5, NUM_RECORDS)
df['bnpl_repayment_rate'] = np.random.beta(5, 2.5, NUM_RECORDS)

# --- PART 2: Generate Raw UPI Transactions ---
print("\nPart 2: Simulating raw UPI transactions...")
all_transactions = []
income_map = {0: 15000, 1: 35000, 2: 70000}
for _, user in tqdm(df.iterrows(), total=df.shape[0]):
    user_id, income_tier = user['user_id'], user['income_tier']
    monthly_income = income_map[income_tier]
    for month in range(1, TRANSACTION_MONTHS + 1):
        if np.random.rand() > 0.05:
            all_transactions.append({'user_id': user_id, 'date': f'2025-0{month}-01', 'description': 'Salary Credit', 'amount': monthly_income * np.random.normal(1, 0.1), 'type': 'CREDIT'})
        for utility in MERCHANT_CATEGORIES['utility']:
            if np.random.rand() > 0.25:
                all_transactions.append({'user_id': user_id, 'date': f'2025-0{month}-{np.random.randint(5,15)}', 'description': utility, 'amount': np.random.uniform(300, 2000), 'type': 'DEBIT'})
        if income_tier > 0 and np.random.rand() > 0.1:
            all_transactions.append({'user_id': user_id, 'date': f'2025-0{month}-05', 'description': 'Loan EMI Payment', 'amount': monthly_income * np.random.uniform(0.1, 0.35), 'type': 'DEBIT'})
        for _ in range(np.random.poisson(3 + income_tier * 5)):
             all_transactions.append({'user_id': user_id, 'date': f'2025-0{month}-{np.random.randint(1,28)}', 'description': np.random.choice(MERCHANT_CATEGORIES['food']), 'amount': np.random.uniform(150, 600), 'type': 'DEBIT'})

upi_df = pd.DataFrame(all_transactions)
upi_df['amount'] = upi_df['amount'].round(2)
print(f"Generated {len(upi_df)} raw UPI transactions.")

# --- PART 3: Calculate Organic Features from UPI Data ---
print("\nPart 3: Calculating organic features from UPI data...")
# (Logic unchanged)
income_transactions = upi_df[upi_df['type'] == 'CREDIT'].groupby('user_id')
income_consistency = 1 - (income_transactions['amount'].std() / income_transactions['amount'].mean()).fillna(1)
utility_transactions = upi_df[upi_df['description'].isin(MERCHANT_CATEGORIES['utility'])]
unique_utilities_paid = utility_transactions.groupby('user_id')['description'].nunique()
utility_payment_ratio = unique_utilities_paid / len(MERCHANT_CATEGORIES['utility'])
loan_payments = upi_df[upi_df['description'] == 'Loan EMI Payment'].groupby('user_id')['amount'].sum()
total_income = upi_df[upi_df['type'] == 'CREDIT'].groupby('user_id')['amount'].sum()
debt_burden = (loan_payments / total_income).fillna(0)
def calculate_spending_tier(df, category_merchants):
    category_spend = df[df['description'].isin(category_merchants)].groupby('user_id')['amount'].sum() / TRANSACTION_MONTHS
    return pd.cut(category_spend, bins=[-1, 0.99, 400, np.inf], labels=[0, 1, 2])

# --- PART 4: Merge, Finalize, and Calculate Risk Scores ---
print("\nPart 4: Merging data, calculating risk scores, and finalizing...")
# (Logic unchanged)
calculated_features_df = pd.DataFrame({ 'income_consistency': income_consistency, 'utility_payment_ratio': utility_payment_ratio, 'debt_burden': debt_burden, 'ott_spend_tier': calculate_spending_tier(upi_df, MERCHANT_CATEGORIES['ott']), 'food_delivery_tier': calculate_spending_tier(upi_df, MERCHANT_CATEGORIES['food']), 'ride_hailing_tier': calculate_spending_tier(upi_df, MERCHANT_CATEGORIES['ride_hailing']), }).reset_index()
final_df = pd.merge(df, calculated_features_df, on='user_id', how='left')
newly_calculated_cols = [ 'income_consistency', 'utility_payment_ratio', 'debt_burden', 'ott_spend_tier', 'food_delivery_tier', 'ride_hailing_tier' ]
final_df[newly_calculated_cols] = final_df[newly_calculated_cols].fillna(0)
tier_cols = ['ott_spend_tier', 'food_delivery_tier', 'ride_hailing_tier']
final_df[tier_cols] = final_df[tier_cols].astype(int)
def assign_risk_score(series, thresholds, ascending=True):
    labels = [0, 1, 2] if ascending else [2, 1, 0]
    return pd.cut(series, bins=[-np.inf] + thresholds + [np.inf], labels=labels, right=False).astype(int)
final_df['debt_risk'] = assign_risk_score(final_df['debt_burden'], [0.4, 0.6])
final_df['utility_risk'] = assign_risk_score(final_df['utility_payment_ratio'], [0.6, 0.9], ascending=False)
final_df['bnpl_risk'] = assign_risk_score(final_df['bnpl_repayment_rate'], [0.5, 0.8], ascending=False)
final_df['peer_risk'] = assign_risk_score(final_df['peer_default_exposure'], [0.2, 0.5])
final_df['device_risk'] = final_df['device_tier']
final_df['app_diversity_risk'] = assign_risk_score(final_df['app_diversity'], [10, 25], ascending=False)
final_df['clickstream_risk'] = assign_risk_score(final_df['clickstream_volatility'], [0.3, 0.6])
final_df['asset_diversity_risk'] = assign_risk_score(final_df['asset_diversity'], [1, 4], ascending=False)
final_df['financial_coping_risk'] = assign_risk_score(final_df['financial_shock_coping'], [3, 4], ascending=False)
final_df['earner_density_risk'] = assign_risk_score(final_df['earner_density'], [1.5, 2.0], ascending=False)
final_df['unemployment_risk'] = assign_risk_score(final_df['unemployment_rate'], [0.12, 0.2])
final_df['urban_risk'] = assign_risk_score(final_df['urban_score'], [0.6, 0.8], ascending=False)
final_df['income_tier_risk'] = 2 - final_df['income_tier']
def get_contextual_risk(row):
    spend_tier, income_tier = row[0], row[1]
    if spend_tier == 0: return 0
    elif spend_tier == 1: return 2 if income_tier == 0 else (1 if income_tier == 1 else 0)
    else: return 2 if income_tier < 2 else 1
final_df['ott_risk'] = final_df[['ott_spend_tier', 'income_tier']].apply(get_contextual_risk, axis=1)
final_df['food_risk'] = final_df[['food_delivery_tier', 'income_tier']].apply(get_contextual_risk, axis=1)
final_df['ride_risk'] = final_df[['ride_hailing_tier', 'income_tier']].apply(get_contextual_risk, axis=1)
final_df['total_risk'] = (1.3 * (final_df['debt_risk'] + final_df['utility_risk'] + final_df['asset_diversity_risk'] + final_df['earner_density_risk']) + 1.2 * final_df['income_tier_risk'] + 0.8 * (final_df['ott_risk'] + final_df['food_risk'] + final_df['ride_risk']) + 1.0 * (final_df['device_risk'] + final_df['app_diversity_risk'] + final_df['clickstream_risk'] + final_df['bnpl_risk'] + final_df['peer_risk'] + final_df['financial_coping_risk']) + 0.7 * (final_df['unemployment_risk'] + final_df['urban_risk']))
final_df['default_prob'] = 1 / (1 + np.exp(-0.25 * final_df['total_risk'] + 5))
final_df['loan_default'] = (np.random.rand(NUM_RECORDS) < final_df['default_prob']).astype(int)

# --- PART 5: DYNAMIC FILE SAVING ---
print("\nPart 5: Saving files with dynamic naming...")

# Define the base names including the number of records
base_final_name = f'credit_data_final_v3_{NUM_RECORDS}k'
base_upi_name = f'raw_upi_transactions_v3_{NUM_RECORDS}k'

# Construct the initial file path attempts
final_output_path = f'{base_final_name}.csv'
upi_output_path = f'{base_upi_name}.csv'

# Check for existence and add a counter if necessary
counter = 1
while os.path.exists(final_output_path):
    # If the file already exists, create a new name with a counter
    final_output_path = f'{base_final_name}_{counter}.csv'
    upi_output_path = f'{base_upi_name}_{counter}.csv'
    counter += 1

# Save the dataframes with the final, unique filenames
final_df.to_csv(final_output_path, index=False)
upi_df.to_csv(upi_output_path, index=False)

print("\n--- Process Complete ---")
print(f"Saved final merged dataset to: '{final_output_path}'")
print(f"Saved raw UPI transaction data to: '{upi_output_path}'")
print("\nFinal Data Head:")
print(final_df.head())