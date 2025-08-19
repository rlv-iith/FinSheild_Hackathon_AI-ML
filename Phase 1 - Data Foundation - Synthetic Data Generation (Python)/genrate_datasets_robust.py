import pandas as pd
import numpy as np
from tqdm import tqdm
import os

# --- Configuration ---
NUM_RECORDS = int(input("Enter number of records: "))
TRANSACTION_MONTHS = int(input("Enter number of months of UPI records: "))

# --- Dictionaries & Mappings ---
MERCHANT_CATEGORIES = {
    'income': ['Salary Credit', 'Business Payout'],
    'utility': ['Electricity Bill', 'Mobile Recharge', 'Gas Bill', 'Rent Payment'],
    'food': ['Zomato', 'Swiggy', 'Grofers'],
    'ott': ['Netflix Subscription', 'Hotstar Payment'],
    'ride_hailing': ['Uber', 'Ola Cabs'],
    'loan_emi': ['Loan EMI Payment'],
    'bnpl_emi': ['BNPL Repayment'],
    'skill_spend': ['Coursera Subscription', 'Udemy Course'],
    'savings': ['Fixed Deposit', 'Savings Account Transfer'],
    'misc_debit': ['Amazon Purchase', 'Myntra Order', 'Friend Transfer']
}
INCOME_BINS = [-1, 20000, 50000, np.inf] # Bins for Low, Mid, High income
INCOME_LABELS = [0, 1, 2] # 0=Weak, 1=Mid, 2=Strong


# --- PART 1: Generate Base Synthetic Profile Features ---
print("Part 1: Generating comprehensive base profiles...")

df = pd.DataFrame({'user_id': range(NUM_RECORDS)})

# --- NEW: Generate income in Rupees first, then derive tier ---
df['monthly_income_rs'] = np.random.lognormal(mean=np.log(35000), sigma=0.5, size=NUM_RECORDS).round(-2)
df['income_tier'] = pd.cut(df['monthly_income_rs'], bins=INCOME_BINS, labels=INCOME_LABELS).astype(int)

df['age'] = np.clip(np.random.gamma(8, 4, NUM_RECORDS), 18, 65).astype(int)
df['employment_tenure'] = np.random.gamma(1.5 + (df['age']/15), 12, NUM_RECORDS)
df['employment_tenure'] = np.minimum(df['employment_tenure'], (df['age'] - 20) * 12).round(1) # Adjusted min working age to 18
df['device_tier'] = np.random.choice([0, 1, 2], NUM_RECORDS, p=[0.3, 0.5, 0.2])
df['app_diversity'] = np.random.poisson(15, NUM_RECORDS) + np.where(df['age'] < 30, 10, 0)
df['clickstream_volatility'] = np.random.beta(2, 5, NUM_RECORDS)
df['peer_default_exposure'] = np.random.beta(1.5, 5, NUM_RECORDS)
df['financial_coping_ability'] = np.random.randint(1, 6, NUM_RECORDS)
df['asset_diversity'] = np.random.poisson(0.5 + (df['age']/20), NUM_RECORDS)
# --- FIX: earner_density is now an integer ---
df['earner_density'] = (np.random.gamma(3, 0.6, NUM_RECORDS)).astype(int) + 1
df['urbanization_score'] = np.random.beta(4, 2, NUM_RECORDS)
df['local_unemployment_rate'] = np.random.beta(2, 8, NUM_RECORDS) * 0.4

# --- NEW: Add Income Tax features ---
df['income_tax_paid'] = df['monthly_income_rs'] * 12 * np.random.uniform(0.05, 0.20)
df['tax_payment_timeliness'] = np.random.beta(8, 2, NUM_RECORDS)


# --- PART 2: Generate Raw UPI Transactions ---
print("\nPart 2: Simulating raw UPI transactions...")
all_transactions = []
for _, user in tqdm(df.iterrows(), total=df.shape[0]):
    user_id, income, tenure = user['user_id'], user['monthly_income_rs'], user['employment_tenure']
    income_noise = 0.15 - (tenure / 300)
    for month in range(1, TRANSACTION_MONTHS + 1):
        income_amount = income * np.random.normal(1, max(0.01, income_noise))
        all_transactions.append({'user_id': user_id, 'description': 'Salary Credit', 'amount': income_amount, 'type': 'CREDIT'})
        if user['age'] < 40 and user['income_tier'] > 0 and np.random.rand() > 0.7: all_transactions.append({'user_id': user_id, 'description': np.random.choice(MERCHANT_CATEGORIES['skill_spend']), 'amount': np.random.uniform(500, 2000), 'type': 'DEBIT'})
        if np.random.rand() > 0.3: all_transactions.append({'user_id': user_id, 'description': 'BNPL Repayment', 'amount': income * np.random.uniform(0.02, 0.08), 'type': 'DEBIT'})
        if np.random.rand() > 0.2: all_transactions.append({'user_id': user_id, 'description': 'Loan EMI Payment', 'amount': income * np.random.uniform(0.1, 0.4), 'type': 'DEBIT'})
        for _ in range(np.random.poisson(3 + user['income_tier'] * 5)): all_transactions.append({'user_id': user_id, 'description': np.random.choice(MERCHANT_CATEGORIES['food']), 'amount': np.random.uniform(150, 600), 'type': 'DEBIT'})
        if np.random.rand() > 0.3: all_transactions.append({'user_id': user_id, 'description': np.random.choice(MERCHANT_CATEGORIES['ott']), 'amount': np.random.uniform(199, 799), 'type': 'DEBIT'})
        if np.random.rand() > 0.4: all_transactions.append({'user_id': user_id, 'description': np.random.choice(MERCHANT_CATEGORIES['ride_hailing']), 'amount': np.random.uniform(100, 500), 'type': 'DEBIT'})
        if np.random.rand() > 0.1: all_transactions.append({'user_id': user_id, 'description': 'Utility Bill', 'amount': np.random.uniform(300, 3000), 'type': 'DEBIT'})

upi_df = pd.DataFrame(all_transactions)


# --- PART 3: Calculate Transactional Features ---
print("\nPart 3: Calculating all transactional features...")
def calculate_spending_tier(category_merchants, bins, labels):
    category_spend = upi_df[upi_df['description'].isin(category_merchants)].groupby('user_id')['amount'].sum() / TRANSACTION_MONTHS
    return pd.cut(category_spend.reindex(df['user_id']).fillna(0), bins=bins, labels=labels, right=False).astype(int)

total_income = upi_df[upi_df['type'] == 'CREDIT'].groupby('user_id')['amount'].sum().reindex(df['user_id']).fillna(1)
total_debits = upi_df[upi_df['type'] == 'DEBIT'].groupby('user_id')['amount'].sum().reindex(df['user_id']).fillna(0)
loan_payments = upi_df[upi_df['description'].isin(MERCHANT_CATEGORIES['loan_emi'])].groupby('user_id')['amount'].sum().reindex(df['user_id']).fillna(0)

transactional_features = {
    'ott_spending_tier': calculate_spending_tier(MERCHANT_CATEGORIES['ott'], bins=[-1, 400, 1000, np.inf], labels=[0, 1, 2]),
    'food_delivery_tier': calculate_spending_tier(MERCHANT_CATEGORIES['food'], bins=[-1, 1000, 5000, np.inf], labels=[0, 1, 2]),
    'ride_hailing_tier': calculate_spending_tier(MERCHANT_CATEGORIES['ride_hailing'], bins=[-1, 500, 2000, np.inf], labels=[0, 1, 2]),
    'skill_spend': upi_df[upi_df['description'].isin(MERCHANT_CATEGORIES['skill_spend'])].groupby('user_id')['amount'].sum().reindex(df['user_id']).fillna(0),
    'bnpl_repayment_rate': np.where(df['user_id'].isin(upi_df[upi_df['description'] == 'BNPL Repayment']['user_id'].unique()), np.random.beta(5, 2, len(df)), 0),
    'debt_burden': loan_payments / total_income,
    'utility_payment_ratio': np.where(df['user_id'].isin(upi_df[upi_df['description'] == 'Utility Bill']['user_id'].unique()), np.random.beta(7, 2, len(df)), 0),
    'transaction_to_income_ratio': total_debits / total_income,
}

# --- PART 4: Finalize and Merge ---
print("\nPart 4: Merging all data sources...")
for col_name, col_data in transactional_features.items():
    df[col_name] = col_data

final_df = df.fillna(0)


# --- PART 5: Calculate Risk Scores & Final Target Variable ---
print("\nPart 5: Calculating risk scores and generating final default labels...")
def assign_risk_score(series, thresholds, ascending=True):
    labels = [0, 1, 2] if ascending else [2, 1, 0]
    return pd.cut(series, bins=[-np.inf] + thresholds + [np.inf], labels=labels, right=False).astype(int)

# Use your comprehensive risk logic here
final_df['debt_risk'] = assign_risk_score(final_df['debt_burden'], [0.4, 0.6])
final_df['utility_risk'] = assign_risk_score(final_df['utility_payment_ratio'], [0.6, 0.9], ascending=False)
final_df['bnpl_risk'] = assign_risk_score(final_df['bnpl_repayment_rate'], [0.5, 0.8], ascending=False)
final_df['peer_risk'] = assign_risk_score(final_df['peer_default_exposure'], [0.2, 0.5])
final_df['income_tier_risk'] = 2 - final_df['income_tier']
final_df['tax_timeliness_risk'] = assign_risk_score(final_df['tax_payment_timeliness'], [0.7, 0.9], ascending=False)
# Add other risk scores as per your project document...

final_df['total_risk'] = (
    final_df['debt_risk'] * 1.3 + 
    final_df['income_tier_risk'] * 1.2 + 
    final_df['tax_timeliness_risk'] * 1.5
    # Add other weighted risk components here
)

final_df['default_prob'] = 1 / (1 + np.exp(-0.25 * final_df['total_risk'] + 5))
final_df['loan_default'] = (np.random.rand(NUM_RECORDS) < final_df['default_prob']).astype(int)


# --- PART 6: DYNAMIC FILE SAVING ---
print("\nPart 6: Saving files with dynamic naming...")
base_final_name = f'credit_data_final_v7_full_{int(NUM_RECORDS/1000)}k'
upi_base_name = f'raw_upi_transactions_v7_{int(NUM_RECORDS/1000)}k'

final_output_path = f'{base_final_name}.csv'
upi_output_path = f'{upi_base_name}.csv'

counter = 1
while os.path.exists(final_output_path):
    final_output_path = f'{base_final_name}_{counter}.csv'
    upi_output_path = f'{upi_base_name}_{counter}.csv'
    counter += 1

final_df.to_csv(final_output_path, index=False)
upi_df.to_csv(upi_output_path, index=False)

print("\n--- Process Complete ---")
print(f"Generated a new comprehensive dataset: '{final_output_path}'")
print(f"Generated the corresponding raw transaction log: '{upi_output_path}'")
print(f"Dataset contains {len(final_df.columns)} columns.")
print(f"Overall default rate: {final_df['loan_default'].mean() * 100:.2f}%")
print(f"\nVerification of fixes:")
print(f"  - Earner Density is type: {final_df['earner_density'].dtype}")
print(f"  - Debt Risk distribution: {final_df['debt_risk'].value_counts().to_dict()}")