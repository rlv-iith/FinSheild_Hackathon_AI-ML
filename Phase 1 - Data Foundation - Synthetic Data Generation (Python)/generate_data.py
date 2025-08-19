import pandas as pd
import numpy as np

# Define the number of records to generate
NUM_RECORDS = 50000

# --- Phase 1: Income & Employment Feature Simulation ---
print("Phase 1: Generating Income & Employment features...")

# Simulate base features using realistic distributions
data = {
    'income_consistency': np.random.beta(5, 2, NUM_RECORDS),  # Skewed towards 1.0
    'formal_income_evidence': np.random.randint(0, 4, NUM_RECORDS), # 0-3 scale
    'skill_spend': np.random.gamma(1, 1000, NUM_RECORDS), # Skewed, most spend little
    'employment_tenure': np.random.gamma(2, 12, NUM_RECORDS), # Skewed, most have shorter tenures
}
df = pd.DataFrame(data)

# Calculate the composite income_tier
# (income_tier = Strong (2), Mid (1), Weak (0))
conditions = [
    (df['income_consistency'] >= 0.7),
    (df['formal_income_evidence'] == 3),
    (df['skill_spend'] >= 1000),
    (df['employment_tenure'] > 24)
]
# Sum of conditions met (True=1, False=0)
income_score = np.sum(conditions, axis=0)
df['income_tier'] = pd.cut(income_score, bins=[-1, 0, 2, 4], labels=[0, 1, 2]).astype(int) # Weak: 0, Mid: 1-2, Strong: 3-4

# --- Phase 2: Contextual Discretionary Spending Risk ---
print("Phase 2: Generating Contextual Spending features...")

# Simulate spending tiers
df['ott_spend_tier'] = np.random.choice([0, 1, 2], NUM_RECORDS, p=[0.2, 0.5, 0.3])
df['food_delivery_tier'] = np.random.choice([0, 1, 2], NUM_RECORDS, p=[0.3, 0.4, 0.3])
df['ride_hailing_tier'] = np.random.choice([0, 1, 2], NUM_RECORDS, p=[0.4, 0.4, 0.2])

# Define the function to calculate contextual risk
def get_contextual_risk(spend_tier, income_tier):
    if spend_tier == 0:
        return 0
    elif spend_tier == 1:
        return 2 if income_tier == 0 else (1 if income_tier == 1 else 0)
    else: # spend_tier == 2
        return 2 if income_tier < 2 else 1

# Apply the function to each spending category
df['ott_risk'] = df.apply(lambda row: get_contextual_risk(row['ott_spend_tier'], row['income_tier']), axis=1)
df['food_risk'] = df.apply(lambda row: get_contextual_risk(row['food_delivery_tier'], row['income_tier']), axis=1)
df['ride_risk'] = df.apply(lambda row: get_contextual_risk(row['ride_hailing_tier'], row['income_tier']), axis=1)

# --- Phase 3: Independent Risk Features ---
print("Phase 3: Generating Independent Risk features...")

# Simulate features
df['debt_burden'] = np.random.beta(2, 3, NUM_RECORDS) * 1.2 # Peaking around 0.4-0.5
df['utility_payment_ratio'] = np.random.beta(7, 2, NUM_RECORDS) # Skewed towards high payment
df['bnpl_repayment_rate'] = np.random.beta(5, 2.5, NUM_RECORDS)
df['peer_default_exposure'] = np.random.beta(1.5, 5, NUM_RECORDS) # Skewed towards low exposure
df['device_tier'] = np.random.choice([0, 1, 2], NUM_RECORDS, p=[0.3, 0.5, 0.2]) # 0=High, 1=Mid, 2=Low
df['app_diversity'] = np.random.poisson(20, NUM_RECORDS)
df['clickstream_volatility'] = np.random.beta(2, 5, NUM_RECORDS) # Skewed towards low volatility
df['asset_diversity'] = np.random.poisson(1.5, NUM_RECORDS)
df['financial_shock_coping'] = np.random.randint(1, 6, NUM_RECORDS) # 1-5 scale
df['earner_density'] = np.random.gamma(3, 0.6, NUM_RECORDS) # Centered around 1.8
df['unemployment_rate'] = np.random.beta(2, 8, NUM_RECORDS) * 0.4 # Skewed towards low unemployment
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
df['device_risk'] = df['device_tier'] # Already in 0, 1, 2 format (High, Mid, Low end device)
df['app_diversity_risk'] = assign_risk_score(df['app_diversity'], [10, 25], ascending=False)
df['clickstream_risk'] = assign_risk_score(df['clickstream_volatility'], [0.3, 0.6])
df['asset_diversity_risk'] = assign_risk_score(df['asset_diversity'], [1, 4], ascending=False) # >3 is low risk
df['financial_coping_risk'] = assign_risk_score(df['financial_shock_coping'], [3, 4], ascending=False) # >=4 is low risk
df['earner_density_risk'] = assign_risk_score(df['earner_density'], [1.5, 2.0], ascending=False) # >2.0 is low risk
df['unemployment_risk'] = assign_risk_score(df['unemployment_rate'], [0.12, 0.2])
df['urban_risk'] = assign_risk_score(df['urban_score'], [0.6, 0.8], ascending=False)
df['income_tier_risk'] = 2 - df['income_tier'] # Invert income tier for risk

# --- Phase 4: Final Risk Scoring System ---
print("Phase 4: Calculating final weighted risk score...")

df['total_risk'] = (
    1.3 * (df['debt_risk'] + df['utility_risk'] + df['asset_diversity_risk'] + df['earner_density_risk']) +
    1.2 * (df['income_tier_risk']) +
    0.8 * (df['ott_risk'] + df['food_risk'] + df['ride_risk']) +
    1.0 * (df['device_risk'] + df['app_diversity_risk'] + df['clickstream_risk'] + df['bnpl_risk'] + df['peer_risk'] + df['financial_coping_risk']) +
    0.7 * (df['unemployment_risk'] + df['urban_risk'])
)

# --- Phase 5: Default Generation Logic ---
print("Phase 5: Generating default labels...")

# Apply logistic function to map score to probability
# Tuned the intercept (-5) to get desired default rate
df['default_prob'] = 1 / (1 + np.exp(-0.25 * df['total_risk'] + 5))

# Generate default flag based on probability
df['loan_default'] = (np.random.rand(NUM_RECORDS) < df['default_prob']).astype(int)

# --- Finalization ---
print("\nData Synthesis Complete.")
# Save the final dataset to a CSV file
output_filename = 'synthetic_alternate_credit_data.csv'
df.to_csv(output_filename, index=False)

print(f"Successfully generated {NUM_RECORDS} records.")
print(f"Saved dataset to '{output_filename}'")

# Display a summary
print("\n--- Dataset Summary ---")
print("Dataframe head:")
print(df.head())
print(f"\nOverall default rate: {df['loan_default'].mean() * 100:.2f}%")
print(f"\nAverage Total Risk Score: {df['total_risk'].mean():.2f}")