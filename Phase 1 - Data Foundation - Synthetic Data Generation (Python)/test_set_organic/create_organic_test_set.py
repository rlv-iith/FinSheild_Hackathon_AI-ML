import pandas as pd
import numpy as np

def create_persona(user_id, archetype):
    """Generates a user profile with slight variations based on a specific archetype."""
    
    # Base dictionary with all required columns, initialized to defaults
    profile = {
        'user_id': user_id, 'name': f"{archetype}_{user_id}", 'age': 35, 'monthly_income_rs': 50000, 
        'employment_tenure': 60, 'earner_density': 1, 'device_tier': 3, 'app_diversity': 30, 
        'financial_coping_ability': 3, 'asset_diversity': 2, 'tax_payment_timeliness': 0.8, 
        'urbanization_score': 0.7, 'peer_default_exposure': 0.3, 'debt_ratio': 0.4, 
        'loan_repaid': 1, 'is_live_user': 0 # Default flag is 0
    }
    
    if archetype == "Retiree":
        profile.update({
            'age': np.random.randint(65, 80), 'monthly_income_rs': np.random.randint(20, 35)*1000,
            'employment_tenure': 400, 'device_tier': np.random.randint(1, 3), 'app_diversity': np.random.randint(5, 15),
            'financial_coping_ability': 4, 'asset_diversity': np.random.randint(3, 6), 
            'tax_payment_timeliness': 1.0, 'peer_default_exposure': np.random.uniform(0.0, 0.1), 
            'debt_ratio': np.random.uniform(0.0, 0.1), 'loan_repaid': 1
        })
    elif archetype == "Established Professional":
        profile.update({
            'age': np.random.randint(40, 55), 'monthly_income_rs': np.random.randint(90, 200)*1000,
            'employment_tenure': np.random.randint(120, 240), 'earner_density': 2, 'device_tier': 5,
            'app_diversity': np.random.randint(30, 50), 'financial_coping_ability': 5,
            'asset_diversity': np.random.randint(4, 7), 'tax_payment_timeliness': np.random.uniform(0.9, 1.0),
            'peer_default_exposure': np.random.uniform(0.05, 0.2), 'debt_ratio': np.random.uniform(0.2, 0.4), 'loan_repaid': 1
        })
    elif archetype == "Young Salaried (New Employee)":
        profile.update({
            'age': np.random.randint(22, 28), 'monthly_income_rs': np.random.randint(30, 50)*1000,
            'employment_tenure': np.random.randint(3, 24), 'device_tier': 4,
            'app_diversity': np.random.randint(40, 70), 'financial_coping_ability': 3,
            'asset_diversity': np.random.randint(0, 2), 'tax_payment_timeliness': np.random.uniform(0.85, 1.0),
            'peer_default_exposure': np.random.uniform(0.2, 0.4), 'debt_ratio': np.random.uniform(0.1, 0.3), 'loan_repaid': 1
        })
    elif archetype == "Gig Worker / Freelancer":
        profile.update({
            'age': np.random.randint(25, 40), 'monthly_income_rs': np.random.randint(20, 60)*1000,
            'employment_tenure': np.random.randint(6, 48), 'device_tier': 3,
            'app_diversity': np.random.randint(50, 80), 'financial_coping_ability': 2,
            'asset_diversity': np.random.randint(0, 2), 'tax_payment_timeliness': np.random.uniform(0.4, 0.8),
            'peer_default_exposure': np.random.uniform(0.4, 0.7), 'debt_ratio': np.random.uniform(0.3, 0.6), 'loan_repaid': 0
        })
    elif archetype == "Over-leveraged Spender":
        profile.update({
            'age': np.random.randint(28, 45), 'monthly_income_rs': np.random.randint(70, 110)*1000,
            'employment_tenure': 60, 'device_tier': 5, 'app_diversity': np.random.randint(40, 70),
            'financial_coping_ability': 2, 'asset_diversity': 2, 'tax_payment_timeliness': np.random.uniform(0.6, 0.9),
            'peer_default_exposure': np.random.uniform(0.3, 0.5), 'debt_ratio': np.random.uniform(0.7, 1.2), 'loan_repaid': 0
        })
    elif archetype == "Maternity Leave Returner":
         profile.update({
            'age': 32, 'monthly_income_rs': 65000, 'employment_tenure': 72, 'earner_density': 2,
            'device_tier': 4, 'app_diversity': 40, 'financial_coping_ability': 3, 'asset_diversity': 3,
            'tax_payment_timeliness': 0.9, 'peer_default_exposure': 0.2,
            'debt_ratio': 0.35, 'loan_repaid': 1
        })
    return profile
    
def create_large_organic_test_set():
    """Creates a 100+ person organic test set with dynamic personas and a live user flag."""
    print("--- Creating Large (100-person) Organic Test Set with Live User Flag ---")
    
    # 1. --- INTERVIEW SECTION ---
    my_profile = {}
    print("\nPlease provide details for your profile (this will be the 'live user'):")
    my_profile['user_id'] = 0
    my_profile['name'] = input("1. Your Name/ID: ")
    my_profile['age'] = int(input("2. Your Age: "))
    my_profile['monthly_income_rs'] = float(input("3. Your Approx. Monthly Income (in Rupees): "))
    my_profile['employment_tenure'] = float(input("4. Employment Tenure (in months): "))
    my_profile['earner_density'] = int(input("5. Number of Earners in Household: "))
    my_profile['device_tier'] = int(input("6. Device Tier (1-5, 5 is best): "))
    my_profile['app_diversity'] = int(input("7. Number of Active Apps: "))
    my_profile['financial_coping_ability'] = int(input("8. Financial Coping Ability (1-5, 5 is best): "))
    my_profile['asset_diversity'] = int(input("9. Asset Diversity (count of asset types): "))
    my_profile['tax_payment_timeliness'] = float(input("10. Tax Payment Timeliness (0.0 to 1.0): "))
    my_profile['urbanization_score'] = float(input("11. Urbanization Score (0.0 to 1.0): "))
    my_profile['peer_default_exposure'] = float(input("12. Peer Default Exposure (0.0 to 1.0): "))
    my_profile['debt_ratio'] = float(input("13. Your approx. Debt-to-Income ratio (e.g., 0.3 for 30%): "))
    my_profile['loan_repaid'] = np.nan # Your outcome is unknown
    my_profile['is_live_user'] = 1 # Flag this profile
    
    # 2. --- GENERATE 99 DYNAMIC PERSONAS ---
    personas = []
    archetypes = ["Retiree", "Established Professional", "Young Salaried (New Employee)", "Gig Worker / Freelancer", "Over-leveraged Spender", "Maternity Leave Returner"]
    archetype_probs = [0.1, 0.3, 0.2, 0.2, 0.15, 0.05]
    print("\nGenerating 99 additional diverse personas...")
    for i in range(1, 100):
        archetype = np.random.choice(archetypes, p=archetype_probs)
        personas.append(create_persona(i, archetype))

    all_profiles = [my_profile] + personas
    df = pd.DataFrame(all_profiles)
    
    # 3. --- ADD BASELINE & TRANSACTIONAL COLUMNS ---
    df['income_tier'] = pd.cut(df['monthly_income_rs'], bins=[-1, 30000, 70000, np.inf], labels=[1, 2, 3]).astype(int)
    df['clickstream_volatility'] = np.random.beta(2,5, size=len(df))
    df['income_tax_paid'] = df['monthly_income_rs'] * 12 * 0.15
    df['local_unemployment_rate'] = np.random.uniform(0.05, 0.15, size=len(df))
    df['debt_burden'] = df['debt_ratio']
    df['utility_payment_ratio'] = np.random.uniform(0.7, 1.0, size=len(df))
    total_income_proxy = df['monthly_income_rs'] * 3
    total_debits_proxy = total_income_proxy * (df['debt_burden'] + np.random.uniform(0.1, 0.4, size=len(df)))
    df['transaction_to_income_ratio'] = (total_debits_proxy / total_income_proxy).fillna(0)
    df['ott_spending_tier'] = np.random.choice([0,1,2], size=len(df), p=[0.2,0.5,0.3])
    df['food_delivery_tier'] = np.random.choice([0,1,2], size=len(df), p=[0.2,0.5,0.3])
    df['ride_hailing_tier'] = np.random.choice([0,1,2], size=len(df), p=[0.4,0.4,0.2])
    df['skill_spend'] = np.random.choice([0, 500, 1500, 2000], size=len(df), p=[0.7,0.1,0.1,0.1])
    df['bnpl_repayment_rate'] = np.random.beta(5,2, size=len(df))
    
    # 4. --- ENSURE ALL COLUMNS MATCH for consistency ---
    final_column_order = [
        'user_id', 'monthly_income_rs', 'income_tier', 'age', 'employment_tenure', 
        'device_tier', 'app_diversity', 'clickstream_volatility', 'peer_default_exposure', 
        'financial_coping_ability', 'asset_diversity', 'earner_density', 
        'urbanization_score', 'local_unemployment_rate', 'income_tax_paid', 
        'tax_payment_timeliness', 'ott_spending_tier', 'food_delivery_tier', 
        'ride_hailing_tier', 'skill_spend', 'bnpl_repayment_rate', 
        'debt_burden', 'utility_payment_ratio', 'transaction_to_income_ratio', 
        'loan_repaid', 'is_live_user'
    ]
    df_final = df.reindex(columns=final_column_order).fillna(0)
    
    # 5. --- SAVE TO CSV ---
    df_final.drop(['name'], axis=1, errors='ignore').to_csv('large_organic_test_set_with_flag.csv', index=False)
    
    print("\n--- Process Complete ---")
    print(f"Successfully created 'large_organic_test_set_with_flag.csv' with {len(df_final)} users.")
    print("This file contains an 'is_live_user' flag to highlight your profile in the SHAP analysis.")

if __name__ == "__main__":
    create_large_organic_test_set()