import streamlit as st
import sqlite3
import pandas as pd
import os
import joblib

# -------------------------------
# Database path & init
# -------------------------------
DB_PATH = os.path.join(os.getcwd(), "loan.db")

def get_connection():
    return sqlite3.connect(DB_PATH)
def migrate_loanapplications_schema(conn):
    """Add missing columns to LoanApplications for older DB files."""
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(LoanApplications)")
    existing_cols = {row[1] for row in cursor.fetchall()}

    required_columns = {
        "years_employed": "REAL",
        "annual_income": "REAL",
        "credit_score": "INTEGER",
        "credit_history_years": "REAL",
        "savings_assets": "REAL",
        "defaults_on_file": "INTEGER",
        "delinquencies_last_2yrs": "INTEGER",
        "product_type": "TEXT",
        "loan_intent": "TEXT",
        "loan_amount": "REAL",
        "status": "TEXT",
        "date_submitted": "TIMESTAMP DEFAULT CURRENT_TIMESTAMP",
    }

    for col, col_type in required_columns.items():
        if col not in existing_cols:
            cursor.execute(f"ALTER TABLE LoanApplications ADD COLUMN {col} {col_type}")

    conn.commit()

def init_db():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS LoanApplications (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        years_employed REAL,
        annual_income REAL,
        credit_score INTEGER,
        credit_history_years REAL,
        savings_assets REAL,
        defaults_on_file INTEGER,
        delinquencies_last_2yrs INTEGER,
        product_type TEXT,
        loan_intent TEXT,
        loan_amount REAL,
        status TEXT,
        date_submitted TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS ModelRegistry (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        filename TEXT NOT NULL,
        model_blob BLOB NOT NULL,
        is_active INTEGER DEFAULT 1,
        uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    migrate_loanapplications_schema(conn)
    conn.close()

init_db()

# -------------------------------
# Model loading function
# -------------------------------
@st.cache_resource
def load_uploaded_model():
    """Loads the model from model.pkl – can be a full pipeline or a dict."""
    try:
        if os.path.exists('model.pkl'):
            obj = joblib.load('model.pkl')
            if isinstance(obj, dict) and "model" in obj and "le" in obj and "columns" in obj:
                return obj
            elif hasattr(obj, 'predict'):
                return obj
        return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def import_model_from_pkl(uploaded_file):
    """Import a trained model from an uploaded .pkl file."""
    if uploaded_file is None:
        return None
    try:
        with open("model.pkl", "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.cache_resource.clear()
        st.sidebar.success("✅ Model uploaded successfully as model.pkl!")
        return load_uploaded_model()
    except Exception as e:
        st.sidebar.error(f"❌ Error importing model: {e}")
        return None

def predict_loan_status(input_dict, model_obj):
    """Predict loan status using the loaded model (if any)."""
    if model_obj is None:
        return "Manual Review (No Model)"
    try:
        if hasattr(model_obj, 'predict'):
            input_df = pd.DataFrame([input_dict])
            pred = model_obj.predict(input_df)[0]
            return "Approved" if pred == 1 else "Rejected"
        elif isinstance(model_obj, dict):
            model = model_obj["model"]
            le = model_obj["le"]
            columns = model_obj["columns"]
            scaler = model_obj.get("scaler", None)

            input_df = pd.DataFrame([input_dict])
            input_df['loan_intent'] = le.transform(input_df['loan_intent'])
            # LabelEncoder maps alphabetically: Credit Card=0, Line of Credit=1, Personal Loan=2
            product_map = {"Credit Card": 0, "Line of Credit": 1, "Personal Loan": 2}
            input_df['product_type'] = input_df['product_type'].map(product_map)
            input_df = input_df[columns]

            # Scale features if a scaler was saved with the model
            if scaler is not None:
                input_scaled = scaler.transform(input_df)
            else:
                input_scaled = input_df

            pred = model.predict(input_scaled)[0]
            return "Approved" if pred == 1 else "Rejected"
        else:
            return "Unknown model format"
    except Exception as e:
        return f"Error: {str(e)}"

# -------------------------------
# Helper function to insert a loan record
# -------------------------------
def insert_loan_record(record):
    """Insert a single loan record into the database."""
    required_fields = [
        'years_employed', 'annual_income', 'credit_score',
        'credit_history_years', 'savings_assets', 'defaults_on_file',
        'delinquencies_last_2yrs', 'product_type', 'loan_intent',
        'loan_amount', 'status'
    ]
    for field in required_fields:
        if field not in record:
            return False
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO LoanApplications 
            (years_employed, annual_income, credit_score, credit_history_years,
             savings_assets, defaults_on_file, delinquencies_last_2yrs,
             product_type, loan_intent, loan_amount, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            record['years_employed'],
            record['annual_income'],
            record['credit_score'],
            record['credit_history_years'],
            record['savings_assets'],
            record['defaults_on_file'],
            record['delinquencies_last_2yrs'],
            record['product_type'],
            record['loan_intent'],
            record['loan_amount'],
            record['status']
        ))
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        st.error(f"Insert error: {e}")
        return False

# -------------------------------
# Page config & Sidebar
# -------------------------------
st.set_page_config(page_title="Finance Loan Approval System", layout="wide")

if "page" not in st.session_state:
    st.session_state.page = "Applicant Form"

st.sidebar.title("Navigation")
if st.sidebar.button("📝 Applicant Form", use_container_width=True):
    st.session_state.page = "Applicant Form"
if st.sidebar.button("📊 Loan History", use_container_width=True):
    st.session_state.page = "Loan History"
if st.sidebar.button("📤 Upload CSV", use_container_width=True):
    st.session_state.page = "Upload CSV"

# --- MODEL UPLOAD SECTION ---
st.sidebar.markdown("---")
st.sidebar.subheader("Model Management")
uploaded_pkl = st.sidebar.file_uploader("Upload Trained Model (PKL)", type="pkl", key="model_uploader")

if uploaded_pkl:
    # Only import when this is a new upload (not a rerun with the same file)
    upload_id = f"{uploaded_pkl.name}_{uploaded_pkl.size}"
    if st.session_state.get("last_uploaded_model_id") != upload_id:
        imported_model = import_model_from_pkl(uploaded_pkl)
        if imported_model is not None:
            st.session_state["last_uploaded_model_id"] = upload_id
            st.rerun()

current_model = load_uploaded_model()

# -------------------------------
# Render Pages
# -------------------------------
if st.session_state.page == "Applicant Form":
    st.title("Loan Application Form")
    with st.form("loan_form"):
        col1, col2 = st.columns(2)
        with col1:
            years_employed = st.number_input("Years Employed", 0.0, 60.0, step=0.1, format="%.1f")
            annual_income = st.number_input("Annual Income ($)", 0.0, step=1000.0, format="%.2f")
            credit_score = st.number_input("Credit Score", 300, 850, step=1)
            credit_history_years = st.number_input("Credit History (years)", 0.0, 50.0, step=0.1, format="%.1f")
            savings_assets = st.number_input("Savings & Assets ($)", 0.0, step=1000.0, format="%.2f")
        with col2:
            defaults_on_file = st.selectbox("Defaults on File", [0, 1], format_func=lambda x: "Yes" if x else "No")
            delinquencies_last_2yrs = st.number_input("Delinquencies (last 2 years)", 0, 20, step=1)
            product_type = st.selectbox("Product Type", ["Credit Card", "Personal Loan", "Line of Credit"])
            loan_intent = st.selectbox("Loan Intent", ["Debt Consolidation", "Home Improvement", "Business", "Education", "Medical", "Personal"])
            loan_amount = st.number_input("Loan Amount ($)", 0.0, step=1000.0, format="%.2f")

        submitted = st.form_submit_button("Submit Application")
        if submitted:
            input_data = {
                'years_employed': years_employed,
                'annual_income': annual_income,
                'credit_score': credit_score,
                'credit_history_years': credit_history_years,
                'savings_assets': savings_assets,
                'defaults_on_file': defaults_on_file,
                'delinquencies_last_2yrs': delinquencies_last_2yrs,
                'product_type': product_type,
                'loan_intent': loan_intent,
                'loan_amount': loan_amount
            }
            status = predict_loan_status(input_data, current_model)
            record = input_data.copy()
            record['status'] = status
            if insert_loan_record(record):
                st.success(f"Application Submitted — Status: {status}")
            else:
                st.error("Failed to save application.")

elif st.session_state.page == "Loan History":
    st.title("Loan Application History")

    col_filter1, col_filter2 = st.columns(2)
    with col_filter1:
        sort_order = st.selectbox(
            "Sort by Date Submitted",
            ["Latest First", "Oldest First"],
            index=0
        )
    with col_filter2:
        status_filter = st.selectbox(
            "Filter by Loan Status",
            ["All", "Approved", "Rejected"],
            index=0
        )

    order_sql = "DESC" if sort_order == "Latest First" else "ASC"
    if status_filter == "All":
        where_clause = ""
        params = ()
    else:
        where_clause = "WHERE status = ?"
        params = (status_filter,)

    conn = get_connection()
    df = pd.read_sql_query(
        f"""SELECT years_employed, annual_income, credit_score, credit_history_years,
                  savings_assets, defaults_on_file, delinquencies_last_2yrs,
                  product_type, loan_intent, loan_amount, status, date_submitted
           FROM LoanApplications {where_clause} ORDER BY date_submitted {order_sql}""", conn, params=params)
    conn.close()

    if not df.empty:
        # Display record counts
        total = len(df)
        approved = (df['status'] == 'Approved').sum()
        rejected = (df['status'] == 'Rejected').sum()
        other = total - approved - rejected

        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        col_m1.metric("Total Records", total)
        col_m2.metric("Approved", approved)
        col_m3.metric("Rejected", rejected)
        if other > 0:
            col_m4.metric("Other", other)

        st.dataframe(df, use_container_width=True)
    else:
        st.info("No records found.")

elif st.session_state.page == "Upload CSV":
    st.title("Bulk Upload Loan Applications from CSV")
    st.markdown("""
    Upload a CSV file containing loan applications.  
    The CSV must include the following columns (names exactly as shown):
    - years_employed
    - annual_income
    - credit_score
    - credit_history_years
    - savings_assets
    - defaults_on_file (0 or 1)
    - delinquencies_last_2yrs
    - product_type (one of: Credit Card, Personal Loan, Line of Credit)
    - loan_intent (one of: Debt Consolidation, Home Improvement, Business, Education, Medical, Personal)
    - loan_amount
    - loan_status (1 for Approved, 0 for Rejected)

    Additional columns will be ignored.
    """)

    uploaded_csv = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_csv is not None:
        try:
            df = pd.read_csv(uploaded_csv)
            # Only show columns relevant to the model (matches notebook feature selection)
            display_cols = [
                'years_employed', 'annual_income', 'credit_score',
                'credit_history_years', 'savings_assets', 'defaults_on_file',
                'delinquencies_last_2yrs', 'product_type', 'loan_intent',
                'loan_amount', 'loan_status'
            ]
            preview_cols = [c for c in display_cols if c in df.columns]
            st.write("Preview of uploaded data (first 5 rows):")
            st.dataframe(df[preview_cols].head())

            required_cols = [
                'years_employed', 'annual_income', 'credit_score',
                'credit_history_years', 'savings_assets', 'defaults_on_file',
                'delinquencies_last_2yrs', 'product_type', 'loan_intent',
                'loan_amount', 'loan_status'
            ]
            missing = [col for col in required_cols if col not in df.columns]
            if missing:
                st.error(f"Missing columns in CSV: {', '.join(missing)}")
            else:
                status_map = {1: "Approved", 0: "Rejected"}
                success_count = 0
                error_count = 0
                error_rows = []

                for idx, row in df.iterrows():
                    try:
                        record = {
                            'years_employed': float(row['years_employed']),
                            'annual_income': float(row['annual_income']),
                            'credit_score': int(row['credit_score']),
                            'credit_history_years': float(row['credit_history_years']),
                            'savings_assets': float(row['savings_assets']),
                            'defaults_on_file': int(row['defaults_on_file']),
                            'delinquencies_last_2yrs': int(row['delinquencies_last_2yrs']),
                            'product_type': str(row['product_type']).strip(),
                            'loan_intent': str(row['loan_intent']).strip(),
                            'loan_amount': float(row['loan_amount']),
                            'status': status_map.get(int(row['loan_status']), "Unknown")
                        }
                        # Validate categoricals
                        valid_product = ["Credit Card", "Personal Loan", "Line of Credit"]
                        valid_intent = ["Debt Consolidation", "Home Improvement", "Business", "Education", "Medical", "Personal"]
                        if record['product_type'] not in valid_product:
                            raise ValueError(f"Invalid product_type: {record['product_type']}")
                        if record['loan_intent'] not in valid_intent:
                            raise ValueError(f"Invalid loan_intent: {record['loan_intent']}")
                        if record['status'] == "Unknown":
                            raise ValueError(f"Invalid loan_status: {row['loan_status']} (must be 0 or 1)")

                        if insert_loan_record(record):
                            success_count += 1
                        else:
                            error_count += 1
                            error_rows.append(idx + 2)
                    except Exception as e:
                        error_count += 1
                        error_rows.append(idx + 2)
                        st.warning(f"Row {idx+2} error: {e}")

                st.success(f"Upload complete. Inserted {success_count} records.")
                if error_count > 0:
                    st.warning(f"Failed to insert {error_count} records. Check rows: {error_rows[:10]}" + ("..." if len(error_rows)>10 else ""))
        except Exception as e:
            st.error(f"Error reading CSV file: {e}")