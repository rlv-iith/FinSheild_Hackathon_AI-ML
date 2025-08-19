import pandas as pd
import numpy as np
from tqdm import tqdm
import os
# NEW: Import the library to connect to the database
from sqlalchemy import create_engine

# --- Configuration ---
NUM_RECORDS = int(input("Enter number of records: "))
TRANSACTION_MONTHS = 3

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

# --- Generate income in Rupees first, then derive tier ---
df['monthly_income_rs'] = np.random.lognormal(mean=np.log(35000), sigma=0.5, size=NUM_RECORDS).round(-2)
df['income_tier'] = pd.cut(df['monthly_income_rs'], bins=INCOME_BINS, labels=INCOME_LABELS).astype(int)

df['age'] = np.clip(np.random.gamma(8, 4, NUM_RECORDS), 18, 65).astype(int)
df['employment_tenure'] = np.random.gamma(1.5 + (df['age']/15), 12, NUM_RECORDS)
df['employment_tenure'] = np.minimum(df['employment_tenure'], (df['age'] - 20) * 12).round(1)
df['device_tier'] = np.random.choice([0, 1, 2], NUM_RECORDS, p=[0.3, 0.5, 0.2])
df['app_diversity'] = np.random.poisson(15, NUM_RECORDS) + np.where(df['age'] < 30, 10, 0)
df['clickstream_volatility'] = np.random.beta(2, 5, NUM_RECORDS)
df['peer_default_exposure'] = np.random.beta(1.5, 5, NUM_RECORDS)
df['financial_coping_ability'] = np.random.randint(1, 6, NUM_RECORDS)
df['asset_diversity'] = np.random.poisson(0.5 + (df['age']/20), NUM_RECORDS)
df['earner_density'] = (np.random.gamma(3, 0.6, NUM_RECORDS)).astype(int) + 1
df['urbanization_score'] = np.random.beta(4, 2, NUM_RECORDS)
df['local_unemployment_rate'] = np.random.beta(2, 8, NUM_RECORDS) * 0.4

# --- Add Income Tax features ---
df['income_tax_paid'] = df['monthly_income_rs'] * 12 * np.random.uniform(0.05, 0.20)
df['tax_payment_timeliness'] = np.random.beta(8, 2, NUM_RECORDS)


# --- PART 2: Generate Raw UPI Transactions ---
print("\nPart 2: Simulating raw UPI transactions...")
all_transactions = []
for _, user in tqdm(df.iterrows(), total=df.shape[0]):
    user_id, income, tenure = user['user_id'], user['monthly_income_rs'], user['employment_tenure']
    income_noise = 0.15 - (tenure / 300)
    for month in range(1, TRANSACTION_MONTHS + 1):
    # Add 'month': month to every transaction dictionary
        income_amount = income * np.random.normal(1, max(0.01, income_noise))
        all_transactions.append({'user_id': user_id, 'month': month, 'description': 'Salary Credit', 'amount': income_amount, 'type': 'CREDIT'})
        if user['age'] < 40 and user['income_tier'] > 0 and np.random.rand() > 0.7: all_transactions.append({'user_id': user_id, 'month': month, 'description': np.random.choice(MERCHANT_CATEGORIES['skill_spend']), 'amount': np.random.uniform(500, 2000), 'type': 'DEBIT'})
        if np.random.rand() > 0.3: all_transactions.append({'user_id': user_id, 'month': month, 'description': 'BNPL Repayment', 'amount': income * np.random.uniform(0.02, 0.08), 'type': 'DEBIT'})
        if np.random.rand() > 0.2: all_transactions.append({'user_id': user_id, 'month': month, 'description': 'Loan EMI Payment', 'amount': income * np.random.uniform(0.1, 0.4), 'type': 'DEBIT'})
        for _ in range(np.random.poisson(3 + user['income_tier'] * 5)): all_transactions.append({'user_id': user_id, 'month': month, 'description': np.random.choice(MERCHANT_CATEGORIES['food']), 'amount': np.random.uniform(150, 600), 'type': 'DEBIT'})
        if np.random.rand() > 0.3: all_transactions.append({'user_id': user_id, 'month': month, 'description': np.random.choice(MERCHANT_CATEGORIES['ott']), 'amount': np.random.uniform(199, 799), 'type': 'DEBIT'})
        if np.random.rand() > 0.4: all_transactions.append({'user_id': user_id, 'month': month, 'description': np.random.choice(MERCHANT_CATEGORIES['ride_hailing']), 'amount': np.random.uniform(100, 500), 'type': 'DEBIT'})
        if np.random.rand() > 0.1: all_transactions.append({'user_id': user_id, 'month': month, 'description': 'Utility Bill', 'amount': np.random.uniform(300, 3000), 'type': 'DEBIT'})

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


# --- PART 5: DYNAMIC FILE SAVING (TO A SINGLE SQLITE DATABASE) ---
print("\nPart 5: Saving data to a single SQL database file...")
base_name = f'raw_credit_database_{int(NUM_RECORDS/1000)}k'
output_path = f'{base_name}.db' # The file extension is now .db

counter = 1
while os.path.exists(output_path):
    output_path = f'{base_name}_{counter}.db'
    counter += 1

# Create a connection engine to the single database file
# The file will be created if it doesn't exist.
engine = create_engine(f'sqlite:///{output_path}')

# Save the final_df as a table named 'credit_data'.
# if_exists='replace' will overwrite the table if you run the script again.
print(f"Writing user profiles to 'credit_data' table in {output_path}...")
final_df.to_sql('credit_data', engine, if_exists='replace', index=False)

# Save the upi_df as a table named 'upi_transactions' in the SAME database file.
print(f"Writing transactions to 'upi_transactions' table in {output_path}...")
upi_df.to_sql('upi_transactions', engine, if_exists='replace', index=False)


print("\n--- Process Complete ---")
print(f"Generated a single database file: '{output_path}'")
print("It contains two tables: 'credit_data' and 'upi_transactions'.")