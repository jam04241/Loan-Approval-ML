import sqlite3

conn = sqlite3.connect("loan.db")
cursor = conn.cursor()

# LOAN APPLICATION TABLE
cursor.execute("""
CREATE TABLE IF NOT EXISTS LoanApplications (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    age INTEGER,
    years_employed INTEGER,
    annual_income REAL,
    savings_assets REAL,
    current_debt REAL,
    defaults_on_file INTEGER,
    delinquencies INTEGER,
    product_type TEXT,
    loan_intent TEXT,
    loan_amount REAL,
    payment_ratio REAL,
    status TEXT,
    date_submitted TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
""")

conn.commit()
conn.close()

print("Database created successfully.")