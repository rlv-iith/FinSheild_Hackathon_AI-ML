# FILE: categorize_with_gemini.py

import pandas as pd
import google.generativeai as genai
import os
from tqdm import tqdm

# --- Configuration ---
# IMPORTANT: Set your API key as an environment variable for security
# os.environ['GEMINI_API_KEY'] = 'YOUR_API_KEY'
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))

MERCHANT_CATEGORIES = {
    'income': ['Salary Credit', 'Business Payout'],
    'utility': ['Electricity Bill', 'Mobile Recharge', 'Gas Bill', 'Rent Payment'],
    'food': ['Zomato', 'Swiggy', 'Grofers', 'blinkit'],
    'ott': ['Netflix Subscription', 'Hotstar Payment', 'Spotify'],
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
        'Medical', 'Redbus', 'Westside', 'Trent', 'Google Play',
        'Unidentified Purchase'
    ]
}

def categorize_transactions_with_gemini(csv_path):
    """
    Reads cleaned transactions and uses Gemini API to categorize them.
    """
    df = pd.read_csv(csv_path)
    
    model = genai.GenerativeModel('gemini-pro')
    
    categories = []
    category_list_str = str(list(MERCHANT_CATEGORIES.keys()))

    print("Categorizing transactions using Gemini API...")
    for description in tqdm(df['Description']):
        prompt = f"""
            You are a financial transaction classifier. Based on the following transaction description, classify it into ONLY ONE of these categories:
            {category_list_str}

            Do not provide any explanation or extra text. If you cannot determine a category, return 'misc_debit'. Return only the category name.

            Description: "{description}"
            Category:
        """
        try:
            response = model.generate_content(prompt)
            # Clean up the response to get just the category text
            category = response.text.strip().lower().replace("'", "").replace('"', '')
            if category not in MERCHANT_CATEGORIES:
                category = 'misc_debit' # Default fallback
            categories.append(category)
        except Exception as e:
            print(f"An error occurred: {e}. Defaulting to 'misc_debit'.")
            categories.append('misc_debit')
            
    df['Category'] = categories
    
    output_path = 'categorized_transactions.csv'
    df.to_csv(output_path, index=False)
    print(f"\nCategorization complete. Enriched data saved to '{output_path}'")
    
    return df

# --- Execution ---
if __name__ == "__main__":
    cleaned_csv = 'cleaned_transactions.csv' # Assumes Stage 1 has been run
    if os.path.exists(cleaned_csv):
        categorize_transactions_with_gemini(cleaned_csv)
    else:
        print(f"Error: {cleaned_csv} not found. Please run the data cleaning script first.")