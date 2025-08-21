# FILE: process_bank_statement.py

import pandas as pd
import fitz  # PyMuPDF
import re
import sqlite3

def extract_and_clean_statement(pdf_path):
    """
    Parses a bank statement PDF, saves sensitive data to a database,
    and returns a cleaned DataFrame of transactions.
    """
    # 1. PDF Parsing
    doc = fitz.open(pdf_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text()

    # 2. PII Extraction
    pii = {
        'name': re.search(r"Name\s*\n(.+)", full_text),
        'account_no': re.search(r"account number\s*(\d+)", full_text, re.IGNORECASE),
        'ifsc_code': re.search(r"IFSC Code\s*\n(\w+)", full_text)
    }
    # Convert matches to text or None
    pii_data = {k: v.group(1).strip() if v else None for k, v in pii.items()}

    # 3. Secure PII Storage
    conn = sqlite3.connect('user_data.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS sensitive_info (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            account_no TEXT,
            ifsc_code TEXT
        )
    ''')
    cursor.execute(
        "INSERT INTO sensitive_info (name, account_no, ifsc_code) VALUES (?, ?, ?)",
        (pii_data['name'], pii_data['account_no'], pii_data['ifsc_code'])
    )
    conn.commit()
    conn.close()
    print("Successfully saved sensitive data to user_data.db")

    # 4. Transaction Data Cleaning (Simplified logic for demonstration)
    # In a real scenario, you'd parse tables more robustly.
    lines = full_text.split('\n')
    transactions = []
    # This regex identifies a transaction line based on the date format
    transaction_line_regex = re.compile(r"^\d{2}-\d{2}-\d{4}")

    for line in lines:
        if transaction_line_regex.match(line):
            parts = line.split()
            date = parts[0]
            description = " ".join(parts[1:-3]) # Assumes Withdrawals, Deposits, Balance are last 3
            
            # Clean the description
            clean_desc = re.sub(r"UPI:\d+:", "", description) # Remove UTR
            clean_desc = re.sub(r"@\w+", "", clean_desc)      # Remove VPA handle
            
            transactions.append({'Date': date, 'Description': clean_desc.strip()})

    df = pd.DataFrame(transactions)
    
    # 5. Output Generation
    output_csv_path = 'cleaned_transactions.csv'
    df.to_csv(output_csv_path, index=False)
    print(f"Cleaned transactions saved to {output_csv_path}")
    
    return output_csv_path

# --- Execution ---
if __name__ == "__main__":
    # Replace with the actual path to your bank statement PDF
    bank_statement_pdf = "path/to/your/bank_statement.pdf"
    process_bank_statement(bank_statement_pdf)