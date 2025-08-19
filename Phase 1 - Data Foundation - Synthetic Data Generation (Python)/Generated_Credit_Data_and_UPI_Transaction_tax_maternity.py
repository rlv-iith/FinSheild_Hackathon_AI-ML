# =============================================================================
# FILE: data_generator.py (FINAL VERSION)
# =============================================================================
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
from sqlalchemy import create_engine

# --- Configuration ---
NUM_RECORDS = int(input("Enter number of records: "))
TRANSACTION_MONTHS = int(input("Enter number of transaction months to simulate (e.g., 12 for maternity/tax): "))

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
    'tax': ['Income Tax Payment'],
    'misc_debit': ['Amazon Purchase', 'Myntra Order', 'Friend Transfer']
}
INCOME_BINS = [-1, 20000, 50000, np.inf]
INCOME_LABELS = [0, 1, 2]


# --- Helper function to calculate Indian income tax based on the New Regime ---
def calculate_indian_tax(annual_income):
    """Calculates income tax based on the FY 2023-24 New Regime slabs."""
    if annual_income <= 300000:
        return 0
    tax = 0
    if annual_income > 1500000:
        tax += (annual_income - 1500000) * 0.30
        annual_income = 1500000
    if annual_income > 1200000:
        tax += (annual_income - 1200000) * 0.20
        annual_income = 1200000
    if annual_income > 900000:
        tax += (annual_income - 900000) * 0.15
        annual_income = 900000
    if annual_income > 600000:
        tax += (annual_income - 600000) * 0.10
        annual_income = 600000
    if annual_income > 300000:
        tax += (annual_income - 300000) * 0.05
    return round(tax, 2)


# --- PART 1: Generate Base Synthetic Profile Features ---
print("Part 1: Generating comprehensive base profiles...")
df = pd.DataFrame({'user_id': range(NUM_RECORDS)})
df['monthly_income_rs'] = np.random.lognormal(mean=np.log(35000), sigma=0.5, size=NUM_RECORDS).round(-2)
df['income_tier'] = pd.cut(df['monthly_income_rs'], bins=INCOME_BINS, labels=INCOME_LABELS).astype(int)
df['age'] = np.clip(np.random.gamma(8, 4, NUM_RECORDS), 18, 65).astype(int)
df['gender'] = np.random.choice(['Male', 'Female'], NUM_RECORDS, p=[0.5, 0.5])
df['income_stability_type'] = np.random.choice(['Stable', 'Variable'], NUM_RECORDS, p=[0.6, 0.4])
df['device_tier'] = np.random.choice([0, 1, 2], NUM_RECORDS, p=[0.3, 0.5, 0.2])
df['app_diversity'] = np.random.poisson(15, NUM_RECORDS) + np.where(df['age'] < 30, 10, 0)
df['clickstream_volatility'] = np.random.beta(2, 5, NUM_RECORDS)
df['peer_default_exposure'] = np.random.beta(1.5, 5, NUM_RECORDS)
df['financial_coping_ability'] = np.random.randint(1, 6, NUM_RECORDS)
df['asset_diversity'] = np.random.poisson(0.5 + (df['age']/20), NUM_RECORDS)
df['earner_density'] = (np.random.gamma(3, 0.6, NUM_RECORDS)).astype(int) + 1
df['urbanization_score'] = np.random.beta(4, 2, NUM_RECORDS)
df['local_unemployment_rate'] = np.random.beta(2, 8, NUM_RECORDS) * 0.4


# --- PART 2: Generate Raw UPI Transactions ---
print("\nPart 2: Simulating raw UPI transactions...")
all_transactions = []
for _, user in tqdm(df.iterrows(), total=df.shape[0]):
    user_id, income, stability_type = user['user_id'], user['monthly_income_rs'], user['income_stability_type']
    income_noise = 0.05
    user_transactions = []
    
    is_on_maternity_leave = False
    leave_duration = 6
    leave_start_month = -1
    if user['gender'] == 'Female' and 25 <= user['age'] <= 40 and TRANSACTION_MONTHS >= leave_duration and np.random.rand() < 0.15:
        is_on_maternity_leave = True
        leave_start_month = np.random.randint(1, TRANSACTION_MONTHS - leave_duration + 2)

    for month in range(1, TRANSACTION_MONTHS + 1):
        is_on_leave_this_month = 1 if is_on_maternity_leave and leave_start_month <= month < leave_start_month + leave_duration else 0
        
        income_gap_this_month = np.random.rand() <= 0.05
        
        if is_on_leave_this_month or income_gap_this_month:
            user_transactions.append({'user_id': user_id, 'month': month, 'description': 'No Salary', 'amount': 0, 'type': 'CREDIT', 'on_maternity_leave': is_on_leave_this_month})
        else:
            income_amount = income if stability_type == 'Stable' else income * np.random.normal(1, max(0.01, income_noise))
            user_transactions.append({'user_id': user_id, 'month': month, 'description': 'Salary Credit', 'amount': income_amount, 'type': 'CREDIT', 'on_maternity_leave': is_on_leave_this_month})

        if user['age'] < 40 and user['income_tier'] > 0 and np.random.rand() > 0.7: user_transactions.append({'user_id': user_id, 'month': month, 'description': np.random.choice(MERCHANT_CATEGORIES['skill_spend']), 'amount': np.random.uniform(500, 2000), 'type': 'DEBIT', 'on_maternity_leave': is_on_leave_this_month})
        if np.random.rand() > 0.3: user_transactions.append({'user_id': user_id, 'month': month, 'description': 'BNPL Repayment', 'amount': income * np.random.uniform(0.02, 0.08), 'type': 'DEBIT', 'on_maternity_leave': is_on_leave_this_month})
        if np.random.rand() > 0.2: user_transactions.append({'user_id': user_id, 'month': month, 'description': np.random.choice(MERCHANT_CATEGORIES['loan_emi']), 'amount': income * np.random.uniform(0.1, 0.4), 'type': 'DEBIT', 'on_maternity_leave': is_on_leave_this_month})
        for _ in range(np.random.poisson(3 + user['income_tier'] * 5)): user_transactions.append({'user_id': user_id, 'month': month, 'description': np.random.choice(MERCHANT_CATEGORIES['food']), 'amount': np.random.uniform(150, 600), 'type': 'DEBIT', 'on_maternity_leave': is_on_leave_this_month})
        if np.random.rand() > 0.3: user_transactions.append({'user_id': user_id, 'month': month, 'description': np.random.choice(MERCHANT_CATEGORIES['ott']), 'amount': np.random.uniform(199, 799), 'type': 'DEBIT', 'on_maternity_leave': is_on_leave_this_month})
        if np.random.rand() > 0.4: user_transactions.append({'user_id': user_id, 'month': month, 'description': np.random.choice(MERCHANT_CATEGORIES['ride_hailing']), 'amount': np.random.uniform(100, 500), 'type': 'DEBIT', 'on_maternity_leave': is_on_leave_this_month})
        if np.random.rand() > 0.1: user_transactions.append({'user_id': user_id, 'month': month, 'description': 'Utility Bill', 'amount': np.random.uniform(300, 3000), 'type': 'DEBIT', 'on_maternity_leave': is_on_leave_this_month})

    if TRANSACTION_MONTHS >= 12:
        realized_income = sum(t['amount'] for t in user_transactions if t['type'] == 'CREDIT')
        tax_due = calculate_indian_tax(realized_income)
        if tax_due > 0 and np.random.rand() < 0.85:
            payment_month = np.random.choice([TRANSACTION_MONTHS - 2, TRANSACTION_MONTHS - 1, TRANSACTION_MONTHS])
            user_transactions.append({'user_id': user_id, 'month': payment_month, 'description': 'Income Tax Payment', 'amount': tax_due, 'type': 'DEBIT', 'on_maternity_leave': 0})
            
    all_transactions.extend(user_transactions)

upi_df = pd.DataFrame(all_transactions)
final_df = df

# --- PART 3: Saving ---
print("\nPart 3: Saving data to a single SQL database file...")
base_name = f'raw_credit_database_{int(NUM_RECORDS/1000)}k_{TRANSACTION_MONTHS}m'
output_path = f'{base_name}.db'
counter = 1
while os.path.exists(output_path):
    output_path = f'{base_name}_{counter}.db'
    counter += 1
engine = create_engine(f'sqlite:///{output_path}')
final_df.to_sql('credit_data', engine, if_exists='replace', index=False)
upi_df.to_sql('upi_transactions', engine, if_exists='replace', index=False)
print("\n--- Process Complete ---")
print(f"Generated a single database file: '{output_path}'")
print("It contains two tables: 'credit_data' and 'upi_transactions'.")