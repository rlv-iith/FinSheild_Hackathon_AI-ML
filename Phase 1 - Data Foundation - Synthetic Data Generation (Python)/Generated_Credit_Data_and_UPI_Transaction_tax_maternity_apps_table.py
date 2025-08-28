import pandas as pd
import numpy as np
from tqdm import tqdm
import os
from sqlalchemy import create_engine

# --- Configuration ---
NUM_RECORDS = int(input("Enter number of records: "))
TRANSACTION_MONTHS = int(input("Enter number of transaction months to simulate (e.g., 12 for maternity/tax): "))

while True:
    try:
        low_consistency_pct = float(input("Enter desired percentage of low-consistency earners (e.g., 15 for 15%): "))
        if 0 <= low_consistency_pct <= 100:
            break
        else:
            print("ERROR: Please enter a number between 0 and 100.")
    except ValueError:
        print("ERROR: Invalid input. Please enter a number.")

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
    'misc_debit': [
        'Flipkart Purchase',
        'Myntra Order',
        'Nykaa Purchase',
        'Medical',
        'Unidentified Purchase'
    ]
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

# Calculate dynamic probabilities for earner types based on user input
p_highly_variable = low_consistency_pct / 100.0
# The remainder is split between Stable and Variable, maintaining the original ~60:25 ratio
p_remainder = 1.0 - p_highly_variable
# Original ratio was 60:25 (total 85 parts)
p_stable = (60/85) * p_remainder
p_variable = (25/85) * p_remainder

# Ensure probabilities sum to 1 to avoid floating point errors
if (p_stable + p_variable + p_highly_variable) != 1.0:
    p_stable = 1.0 - p_variable - p_highly_variable


df['income_stability_type'] = np.random.choice(
    ['Stable', 'Variable', 'Highly Variable'],
    NUM_RECORDS,
    p=[p_stable, p_variable, p_highly_variable]
)

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
    # MODIFIED: Changed upper age for maternity leave from 40 to 45
    if user['gender'] == 'Female' and 25 <= user['age'] <= 45 and TRANSACTION_MONTHS >= leave_duration and np.random.rand() < 0.15:
        is_on_maternity_leave = True
        leave_start_month = np.random.randint(1, TRANSACTION_MONTHS - leave_duration + 2)

    for month in range(1, TRANSACTION_MONTHS + 1):
        is_on_mat_leave_this_month = 1 if is_on_maternity_leave and leave_start_month <= month < leave_start_month + leave_duration else 0
        
        income_gap_this_month = np.random.rand() <= 0.05
        
        if is_on_mat_leave_this_month or income_gap_this_month:
            user_transactions.append({'user_id': user_id, 'month': month, 'description': 'No Salary', 'amount': 0, 'type': 'CREDIT', 'on_maternity_leave': is_on_mat_leave_this_month})
        else:
            income_amount = income if stability_type == 'Stable' else income * np.random.normal(1, max(0.01, income_noise))
            user_transactions.append({'user_id': user_id, 'month': month, 'description': 'Salary Credit', 'amount': income_amount, 'type': 'CREDIT', 'on_maternity_leave': is_on_mat_leave_this_month})

        # Debit Transactions
        if user['age'] < 40 and user['income_tier'] > 0 and np.random.rand() > 0.7: user_transactions.append({'user_id': user_id, 'month': month, 'description': np.random.choice(MERCHANT_CATEGORIES['skill_spend']), 'amount': np.random.uniform(500, 2000), 'type': 'DEBIT', 'on_maternity_leave': is_on_mat_leave_this_month})
        if np.random.rand() > 0.3: user_transactions.append({'user_id': user_id, 'month': month, 'description': 'BNPL Repayment', 'amount': income * np.random.uniform(0.02, 0.08), 'type': 'DEBIT', 'on_maternity_leave': is_on_mat_leave_this_month})
        if np.random.rand() > 0.2: user_transactions.append({'user_id': user_id, 'month': month, 'description': 'Loan EMI Payment', 'amount': income * np.random.uniform(0.1, 0.4), 'type': 'DEBIT', 'on_maternity_leave': is_on_mat_leave_this_month})
        
        for _ in range(np.random.poisson(3 + user['income_tier'] * 5)):
             user_transactions.append({'user_id': user_id, 'month': month, 'description': np.random.choice(MERCHANT_CATEGORIES['food']), 'amount': np.random.uniform(150, 600), 'type': 'DEBIT', 'on_maternity_leave': is_on_mat_leave_this_month})

        if np.random.rand() > 0.3: user_transactions.append({'user_id': user_id, 'month': month, 'description': np.random.choice(MERCHANT_CATEGORIES['ott']), 'amount': np.random.uniform(199, 799), 'type': 'DEBIT', 'on_maternity_leave': is_on_mat_leave_this_month})
        
        if np.random.rand() > 0.4:
            user_transactions.append({'user_id': user_id, 'month': month, 'description': np.random.choice(MERCHANT_CATEGORIES['ride_hailing']), 'amount': np.random.uniform(100, 500), 'type': 'DEBIT', 'on_maternity_leave': is_on_mat_leave_this_month})

        if np.random.rand() > 0.1: user_transactions.append({'user_id': user_id, 'month': month, 'description': 'Utility Bill', 'amount': np.random.uniform(300, 3000), 'type': 'DEBIT', 'on_maternity_leave': is_on_mat_leave_this_month})

        if np.random.rand() < 0.1:
            user_transactions.append({'user_id': user_id, 'month': month, 'description': 'Unidentified Purchase', 'amount': np.random.uniform(100, 1500), 'type': 'DEBIT', 'on_maternity_leave': is_on_mat_leave_this_month})

    # Tax payment logic
    if TRANSACTION_MONTHS >= 12:
        realized_income = sum(t['amount'] for t in user_transactions if t['type'] == 'CREDIT')
        tax_due = calculate_indian_tax(realized_income)
        if tax_due > 0 and np.random.rand() < 0.85:
            payment_month = np.random.choice([TRANSACTION_MONTHS - 2, TRANSACTION_MONTHS - 1, TRANSACTION_MONTHS])
            user_transactions.append({'user_id': user_id, 'month': payment_month, 'description': 'Income Tax Payment', 'amount': tax_due, 'type': 'DEBIT', 'on_maternity_leave': 0})
            
    all_transactions.extend(user_transactions)

upi_df = pd.DataFrame(all_transactions)
final_df = df

# --- NEW CODE START ---
# --- PART 2.5: Generate Digital Footprint Data ---
print("\nPart 2.5: Generating digital footprint (app usage) data...")
APP_CATEGORIES = ['Social', 'Entertainment', 'Productivity', 'Finance', 'Shopping', 'Utility', 'Other']
digital_footprint_data = []

for _, user in tqdm(df.iterrows(), total=df.shape[0]):
    # 1. Calculate total screen time (influenced by age and device tier)
    base_screen_time = np.random.normal(loc=240, scale=80) # Avg 4 hours
    age_factor = -1.5 * (user['age'] - 35) # Younger users have more screen time
    device_factor = 20 * user['device_tier'] # Better device, more usage
    total_minutes = max(30, base_screen_time + age_factor + device_factor)
    
    # 2. Determine app usage profile based on user features
    weights = np.ones(len(APP_CATEGORIES))
    
    # Younger users: more social and entertainment
    if user['age'] < 25:
        weights[APP_CATEGORIES.index('Social')] += 3
        weights[APP_CATEGORIES.index('Entertainment')] += 2.5
    # Middle-aged users: more productivity and finance
    elif 30 <= user['age'] < 50:
        weights[APP_CATEGORIES.index('Productivity')] += 2
        weights[APP_CATEGORIES.index('Finance')] += 1.5
        weights[APP_CATEGORIES.index('Utility')] += 1
    # Older users: less social, more utility
    else:
        weights[APP_CATEGORIES.index('Social')] -= 0.5
        weights[APP_CATEGORIES.index('Utility')] += 1.5

    # Higher income tiers: more finance, productivity, and shopping
    if user['income_tier'] == 1:
        weights[APP_CATEGORIES.index('Finance')] += 1
        weights[APP_CATEGORIES.index('Shopping')] += 1
    elif user['income_tier'] == 2:
        weights[APP_CATEGORIES.index('Finance')] += 2
        weights[APP_CATEGORIES.index('Productivity')] += 1.5
        weights[APP_CATEGORIES.index('Shopping')] += 1.5

    # High app diversity users use more varied apps
    if user['app_diversity'] > 30:
        weights[APP_CATEGORIES.index('Other')] += 1.5

    # 3. Distribute total time across categories
    weights = np.maximum(0.1, weights) # Ensure no category has zero or negative weight
    time_distribution = np.random.dirichlet(weights)
    
    user_app_usage = {
        'user_id': user['user_id'],
        'avg_daily_screen_time_mins': round(total_minutes, 2),
        'app_diversity_count': user['app_diversity']
    }
    
    # Assign minutes to each category
    for i, category in enumerate(APP_CATEGORIES):
        user_app_usage[f'minutes_{category.lower()}'] = round(total_minutes * time_distribution[i], 2)
        
    digital_footprint_data.append(user_app_usage)

digital_footprint_df = pd.DataFrame(digital_footprint_data)
# --- NEW CODE END ---


# --- PART 3: Saving ---
print("\nPart 3: Saving data to a single SQL database file...")
base_name = f'raw_credit_database_{int(NUM_RECORDS/1000)}k_{TRANSACTION_MONTHS}m'
output_path = f'{base_name}.db'
counter = 1
while os.path.exists(output_path):
    output_path = f'{base_name}_{counter}.db'
    counter += 1
engine = create_engine(f'sqlite:///{output_path}')

# Save all three tables
final_df.to_sql('credit_data', engine, if_exists='replace', index=False)
upi_df.to_sql('upi_transactions', engine, if_exists='replace', index=False)
digital_footprint_df.to_sql('digital_footprint', engine, if_exists='replace', index=False) # NEW: Saving the third table

print("\n--- Process Complete ---")
print(f"Generated a single database file: '{output_path}'")
# MODIFIED: Updated completion message
print("It contains three tables: 'credit_data', 'upi_transactions', and 'digital_footprint'.")