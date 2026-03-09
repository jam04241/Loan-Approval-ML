import streamlit as st
import sqlite3
import pandas as pd
import numpy as np
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
    """Loads the model from model.pkl – supports dict bundles, sklearn pipelines, and standalone models."""
    try:
        if not os.path.exists('model.pkl'):
            return None
        obj = joblib.load('model.pkl')
        # Dict bundle (e.g. {'model': ..., 'columns': ...})
        if isinstance(obj, dict) and "model" in obj:
            return obj
        # Sklearn Pipeline or any estimator with .predict
        if hasattr(obj, 'predict'):
            return obj
        st.warning("⚠️ Uploaded PKL is not a recognised model format (expected a dict with 'model' key, or an sklearn-compatible estimator).")
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

def apply_hard_rejection_rules(input_dict):
    """Return a list of (status, reason) tuples for every hard rule that triggers."""
    violations = []
    if input_dict.get('defaults_on_file', 0) == 1:
        violations.append("Applicant has defaults on file.")
    if input_dict.get('credit_score', 1000) < 580:
        violations.append(f"Credit score ({input_dict.get('credit_score')}) is below the minimum threshold of 580.")
    if input_dict.get('delinquencies_last_2yrs', 0) >= 3:
        violations.append(f"Too many delinquencies in the last 2 years ({input_dict.get('delinquencies_last_2yrs')}).")
    return violations

def _get_model_probability(input_dict, model_obj):
    """Run the model and return the approval probability, or None on failure."""
    if model_obj is None:
        return None
    try:
        # Apply log1p to monetary/skewed features (matches notebook training preprocessing)
        LOG1P_COLS = {'annual_income', 'savings_assets', 'loan_amount'}
        transformed = {
            k: (np.log1p(v) if k in LOG1P_COLS else v)
            for k, v in input_dict.items()
        }

        if isinstance(model_obj, dict) and "model" in model_obj:
            model = model_obj["model"]
            columns = model_obj.get("columns")
            scaler = model_obj.get("scaler")

            input_df = pd.DataFrame([transformed])
            if columns is not None:
                input_df = input_df.reindex(columns=columns, fill_value=0)
            if scaler is not None:
                input_df = pd.DataFrame(scaler.transform(input_df), columns=input_df.columns)

            if hasattr(model, 'predict_proba'):
                return float(model.predict_proba(input_df)[0][1])
            return float(model.predict(input_df)[0])

        elif hasattr(model_obj, 'predict'):
            input_df = pd.DataFrame([transformed])
            if hasattr(model_obj, 'predict_proba'):
                return float(model_obj.predict_proba(input_df)[0][1])
            return float(model_obj.predict(input_df)[0])

    except Exception as e:
        st.warning(f"⚠️ Model evaluation warning: {e}")
    return None

def predict_loan_status(input_dict, model_obj):
    """Predict loan status. Returns (status, reasons_list, probability)."""
    violations = apply_hard_rejection_rules(input_dict)

    # Always attempt to get model probability, even for hard rejections
    probability = _get_model_probability(input_dict, model_obj)

    if violations:
        return ("Rejected", violations, probability)

    if model_obj is None:
        return ("Approved", [], None)

    if probability is None:
        return ("Rejected", ["Could not evaluate application — model returned no result."], None)

    if probability >= 0.65:
        return ("Approved", [], probability)
    else:
        return ("Rejected", ["Application did not meet the lending criteria based on risk assessment."], probability)

# -------------------------------
# Helper function to insert a loan record
# -------------------------------
def insert_loan_record(record):
    """Insert a single loan record into the database."""
    required_fields = [
        'years_employed', 'annual_income', 'credit_score',
        'savings_assets', 'defaults_on_file',
        'delinquencies_last_2yrs', 'loan_amount', 'status'
    ]
    for field in required_fields:
        if field not in record:
            return False
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO LoanApplications 
            (years_employed, annual_income, credit_score,
             savings_assets, defaults_on_file, delinquencies_last_2yrs,
             loan_amount, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            record['years_employed'],
            record['annual_income'],
            record['credit_score'],
            record['savings_assets'],
            record['defaults_on_file'],
            record['delinquencies_last_2yrs'],
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
            credit_score = st.number_input("Credit Score", 300, 1000, step=1)
            savings_assets = st.number_input("Savings & Assets ($)", 0.0, step=1000.0, format="%.2f")
        with col2:
            defaults_on_file = st.selectbox("Defaults on File", [0, 1], format_func=lambda x: "Yes" if x else "No")
            delinquencies_last_2yrs = st.number_input("Delinquencies (last 2 years)", 0, 20, step=1)
            loan_amount = st.number_input("Loan Amount ($)", 0.0, step=1000.0, format="%.2f")

        submitted = st.form_submit_button("Submit Application")
        if submitted:
            input_data = {
                'years_employed': years_employed,
                'annual_income': annual_income,
                'credit_score': credit_score,
                'savings_assets': savings_assets,
                'defaults_on_file': defaults_on_file,
                'delinquencies_last_2yrs': delinquencies_last_2yrs,
                'loan_amount': loan_amount
            }
            status, reasons, probability = predict_loan_status(input_data, current_model)
            record = input_data.copy()
            record['status'] = status
            if insert_loan_record(record):
                if status == "Approved":
                    st.success(f"✅ Application Submitted — **Approved** (Approval: {probability:.2%} | Rejection: {1 - probability:.2%})" if probability is not None else "✅ Application Submitted — **Approved**")
                else:
                    st.error(f"❌ Application Submitted — **Rejected** (Approval: {probability:.2%} | Rejection: {1 - probability:.2%})" if probability is not None else "❌ Application Submitted — **Rejected**")
                    for r in reasons:
                        st.warning(f"**Reason:** {r}")
                if probability is not None:
                    col_prob1, col_prob2 = st.columns(2)
                    col_prob1.metric("✅ Approval Probability", f"{probability:.2%}")
                    col_prob2.metric("❌ Rejection Probability", f"{1 - probability:.2%}")
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
        where_clause = "WHERE status IN ('Approved', 'Rejected')"
        params = ()
    else:
        where_clause = "WHERE status = ?"
        params = (status_filter,)

    conn = get_connection()
    df = pd.read_sql_query(
        f"""SELECT years_employed, annual_income, credit_score,
                  savings_assets, defaults_on_file, delinquencies_last_2yrs,
                  loan_amount, status, date_submitted
           FROM LoanApplications {where_clause} ORDER BY date_submitted {order_sql}""", conn, params=params)
    conn.close()

    if not df.empty:
        # Display record counts
        total = len(df)
        approved = (df['status'] == 'Approved').sum()
        rejected = (df['status'] == 'Rejected').sum()
        other = total - approved - rejected

        col_m1, col_m2, col_m3 = st.columns(3)
        col_m1.metric("Total Records", total)
        col_m2.metric("Approved", approved)
        col_m3.metric("Rejected", rejected)

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
    - savings_assets
    - defaults_on_file (0 or 1)
    - delinquencies_last_2yrs
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
                'savings_assets', 'defaults_on_file',
                'delinquencies_last_2yrs', 'loan_amount', 'loan_status'
            ]
            preview_cols = [c for c in display_cols if c in df.columns]
            st.write("Preview of uploaded data (first 5 rows):")
            st.dataframe(df[preview_cols].head())

            required_cols = [
                'years_employed', 'annual_income', 'credit_score',
                'savings_assets', 'defaults_on_file',
                'delinquencies_last_2yrs', 'loan_amount', 'loan_status'
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
                            'savings_assets': float(row['savings_assets']),
                            'defaults_on_file': int(row['defaults_on_file']),
                            'delinquencies_last_2yrs': int(row['delinquencies_last_2yrs']),
                            'loan_amount': float(row['loan_amount']),
                            'status': status_map.get(int(row['loan_status']), "Unknown")
                        }
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