import pandas as pd
import numpy as np

# Define the number of records to generate
NUM_RECORDS = 50000

# --- Phase 0: Demographic Feature Simulation ---
print("Phase 0: Generating Age demographics...")

# Generate a slightly younger-skewed age distribution from 18 to 65
ages = np.random.gamma(8, 4, NUM_RECORDS)
df = pd.DataFrame({'age': np.clip(ages, 18, 65).astype(int)})

# Create age groups for conditional logic
age_bins = [17, 25, 35, 50, 66]
age_labels = ['18-25', '26-35', '36-50', '51+']
df['age_group'] = pd.cut(df['age'], bins=age_bins, labels=age_labels, right=True)

# --- Phase 1: Income & Employment Feature Simulation ---
print("Phase 1: Generating Income & Employment features (influenced by age)...")

# Simulate base features, now with age as a factor
df['income_consistency'] = np.random.beta(5 + df['age']/20, 2, NUM_RECORDS) # Consistency improves slightly with age
df['formal_income_evidence'] = np.random.randint(0, 4, NUM_RECORDS)

# Skill spend is higher for younger users
skill_spend_base = np.random.gamma(1.5, 1000, NUM_RECORDS)
df['skill_spend'] = np.where(df['age'] < 40, skill_spend_base, skill_spend_base / 3)

df['employment_tenure'] = np.random.gamma(1.5 + (df['age']/15), 12, NUM_RECORDS)
# CRITICAL: Ensure tenure is not longer than possible years of work (in months)
df['employment_tenure'] = np.minimum(df['employment_tenure'], (df['age'] - 18) * 12).round(1)

# Calculate the composite income_tier
conditions = [
    (df['income_consistency'] >= 0.7),
    (df['formal_income_evidence'] == 3),
    (df['skill_spend'] >= 1000),
    (df['employment_tenure'] > 24)
]
income_score = np.sum(conditions, axis=0)
df['income_tier'] = pd.cut(income_score, bins=[-1, 0, 2, 4], labels=[0, 1, 2]).astype(int)

# --- Phase 2: Contextual Discretionary Spending Risk ---
print("Phase 2: Generating Contextual Spending features...")
df['ott_spend_tier'] = np.random.choice([0, 1, 2], NUM_RECORDS, p=[0.2, 0.5, 0.3])
df['food_delivery_tier'] = np.random.choice([0, 1, 2], NUM_RECORDS, p=[0.3, 0.4, 0.3])
df['ride_hailing_tier'] = np.random.choice([0, 1, 2], NUM_RECORDS, p=[0.4, 0.4, 0.2])

def get_contextual_risk(row):
    spend_tier = row[0]
    income_tier = row[1]
    if spend_tier == 0: return 0
    elif spend_tier == 1: return 2 if income_tier == 0 else (1 if income_tier == 1 else 0)
    else: return 2 if income_tier < 2 else 1

df['ott_risk'] = df[['ott_spend_tier', 'income_tier']].apply(get_contextual_risk, axis=1)
df['food_risk'] = df[['food_delivery_tier', 'income_tier']].apply(get_contextual_risk, axis=1)
df['ride_risk'] = df[['ride_hailing_tier', 'income_tier']].apply(get_contextual_risk, axis=1)

# --- Phase 3: Independent Risk Features (now with some age influence) ---
print("Phase 3: Generating Independent Risk features...")
df['debt_burden'] = np.random.beta(2, 3, NUM_RECORDS) * 1.2
df['utility_payment_ratio'] = np.clip(np.random.beta(7 + df['age']/25, 2, NUM_RECORDS), 0, 1) # Becomes more consistent with age
df['bnpl_repayment_rate'] = np.random.beta(5, 2.5, NUM_RECORDS)
df['peer_default_exposure'] = np.random.beta(1.5, 5, NUM_RECORDS)
df['device_tier'] = np.random.choice([0, 1, 2], NUM_RECORDS, p=[0.3, 0.5, 0.2])

# Younger users have higher app diversity
base_app_diversity = np.random.poisson(15, NUM_RECORDS)
df['app_diversity'] = np.where(df['age'] < 30, base_app_diversity + 10, base_app_diversity)

df['clickstream_volatility'] = np.random.beta(2, 5, NUM_RECORDS)
df['asset_diversity'] = np.random.poisson(0.5 + (df['age']/20), NUM_RECORDS) # Assets increase with age
df['financial_shock_coping'] = np.random.randint(1, 6, NUM_RECORDS)
df['earner_density'] = np.random.gamma(3, 0.6, NUM_RECORDS)
df['unemployment_rate'] = np.random.beta(2, 8, NUM_RECORDS) * 0.4
df['urban_score'] = np.random.beta(4, 2, NUM_RECORDS)

# Convert features to risk scores (0, 1, 2)
def assign_risk_score(series, thresholds, ascending=True):
    bins = [-np.inf] + thresholds + [np.inf]
    labels = [0, 1, 2] if ascending else [2, 1, 0]
    return pd.cut(series, bins=bins, labels=labels, right=False).astype(int)

df['debt_risk'] = assign_risk_score(df['debt_burden'], [0.4, 0.6])
df['utility_risk'] = assign_risk_score(df['utility_payment_ratio'], [0.6, 0.9], ascending=False)
df['bnpl_risk'] = assign_risk_score(df['bnpl_repayment_rate'], [0.5, 0.8], ascending=False)
df['peer_risk'] = assign_risk_score(df['peer_default_exposure'], [0.2, 0.5])
df['device_risk'] = df['device_tier']
df['app_diversity_risk'] = assign_risk_score(df['app_diversity'], [10, 25], ascending=False)
df['clickstream_risk'] = assign_risk_score(df['clickstream_volatility'], [0.3, 0.6])
df['asset_diversity_risk'] = assign_risk_score(df['asset_diversity'], [1, 4], ascending=False)
df['financial_coping_risk'] = assign_risk_score(df['financial_shock_coping'], [3, 4], ascending=False)
df['earner_density_risk'] = assign_risk_score(df['earner_density'], [1.5, 2.0], ascending=False)
df['unemployment_risk'] = assign_risk_score(df['unemployment_rate'], [0.12, 0.2])
df['urban_risk'] = assign_risk_score(df['urban_score'], [0.6, 0.8], ascending=False)
df['income_tier_risk'] = 2 - df['income_tier']

# --- Phase 4: Final Risk Scoring System ---
print("Phase 4: Calculating final weighted risk score (age is NOT a direct factor)...")
df['total_risk'] = (
    1.3 * (df['debt_risk'] + df['utility_risk'] + df['asset_diversity_risk'] + df['earner_density_risk']) +
    1.2 * (df['income_tier_risk']) +
    0.8 * (df['ott_risk'] + df['food_risk'] + df['ride_risk']) +
    1.0 * (df['device_risk'] + df['app_diversity_risk'] + df['clickstream_risk'] + df['bnpl_risk'] + df['peer_risk'] + df['financial_coping_risk']) +
    0.7 * (df['unemployment_risk'] + df['urban_risk'])
)

# --- Phase 5: Default Generation Logic ---
print("Phase 5: Generating default labels...")
df['default_prob'] = 1 / (1 + np.exp(-0.25 * df['total_risk'] + 5))
df['loan_default'] = (np.random.rand(NUM_RECORDS) < df['default_prob']).astype(int)

# --- Finalization ---
print("\nData Synthesis Complete.")
output_filename = 'synthetic_alternate_credit_data_v2.csv'

# Reorder columns for clarity
final_cols = ['age', 'age_group'] + [col for col in df.columns if col not in ['age', 'age_group']]
df = df[final_cols]
df.to_csv(output_filename, index=False)

print(f"Successfully generated {NUM_RECORDS} records.")
print(f"Saved dataset to '{output_filename}'")
print("\n--- Dataset Summary ---")
print("Dataframe head with new age columns:")
print(df.head())
print(f"\nOverall default rate: {df['loan_default'].mean() * 100:.2f}%")
print(f"\nAverage age: {df['age'].mean():.1f} years")
print("\nAge Group Distribution:")
print(df['age_group'].value_counts(normalize=True).round(2))