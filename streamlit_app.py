# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# --- Paths (relative to this file) ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "churn_model.pkl")
DEFAULT_CSV_PATH = os.path.join(BASE_DIR, "WA_Fn-UseC_-Telco-Customer-Churn.csv")

# --- Load model and data ---

@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    return model

@st.cache_data
def load_default_data():
    df = pd.read_csv(DEFAULT_CSV_PATH)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna()
    return df

def predict_for_df(model, df_features: pd.DataFrame):
    """
    Takes raw feature dataframe (same structure as training X),
    returns predictions (0/1) and probabilities.
    """
    probs = model.predict_proba(df_features)[:, 1]
    preds = (probs >= 0.5).astype(int)
    return preds, probs

# --- Streamlit App ---

def main():
    st.set_page_config(
        page_title="Telco Churn Prediction Dashboard",
        layout="wide"
    )

    st.title("📊 Telco Customer Churn Prediction Dashboard")

    # Try loading model
    try:
        model = load_model()
    except Exception as e:
        st.error(f"❌ Model not found. Please run train_model.py first.\n\n{e}")
        return

    # Load default dataset
    df = load_default_data()
    # Columns used during training (same as in train_model.py: df.drop(['customerID','Churn'], axis=1))
    feature_cols = df.drop(columns=["customerID", "Churn"]).columns.tolist()

    # Sidebar menu
    page = st.sidebar.radio(
        "Navigation",
        ["Overview / Analysis", "Single Customer Prediction", "Bulk Prediction (CSV)"]
    )

    # -------------- PAGE 1: OVERVIEW ----------------
    if page == "Overview / Analysis":
        st.header("📈 Dataset Overview & Churn Analysis")

        st.subheader("Sample Data")
        st.dataframe(df.head())

        st.subheader("Churn Distribution")
        churn_counts = df["Churn"].value_counts()
        st.bar_chart(churn_counts)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Contract Type vs Churn")
            contract_churn = (
                df.groupby(["Contract", "Churn"])["customerID"]
                .count()
                .unstack(fill_value=0)
            )
            st.bar_chart(contract_churn)

        with col2:
            st.subheader("Internet Service vs Churn")
            internet_churn = (
                df.groupby(["InternetService", "Churn"])["customerID"]
                .count()
                .unstack(fill_value=0)
            )
            st.bar_chart(internet_churn)

        st.subheader("Numeric Feature Summary")
        st.write(df[["tenure", "MonthlyCharges", "TotalCharges"]].describe())

    # -------------- PAGE 2: SINGLE CUSTOMER ---------
    elif page == "Single Customer Prediction":
        st.header("👤 Single Customer Churn Prediction")

        st.write("Choose an existing customer from the dataset, or enter details manually.")

        mode = st.radio(
            "Input mode:",
            ["Select existing customerID", "Enter manually"]
        )

        # --- Option 1: use existing customer row ---
        if mode == "Select existing customerID":
            customer_ids = df["customerID"].tolist()
            selected_id = st.selectbox("Select customerID", customer_ids)
            customer_row = df[df["customerID"] == selected_id].iloc[0]

            st.write("Selected customer data:")
            st.write(customer_row)

            X_single = customer_row.drop(labels=["customerID", "Churn"])
            X_single_df = pd.DataFrame([X_single])
            # Ensure same column order
            X_single_df = X_single_df[feature_cols]

        # --- Option 2: manual input ---
        else:
            st.subheader("Enter Customer Details")

            col1, col2, col3 = st.columns(3)

            with col1:
                gender = st.selectbox("Gender", ["Female", "Male"])
                senior = st.selectbox("SeniorCitizen", [0, 1])
                partner = st.selectbox("Partner", ["Yes", "No"])
                dependents = st.selectbox("Dependents", ["Yes", "No"])
                tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=1)

            with col2:
                phone_service = st.selectbox("PhoneService", ["Yes", "No"])
                multiple_lines = st.selectbox("MultipleLines", ["No", "Yes", "No phone service"])
                internet_service = st.selectbox("InternetService", ["DSL", "Fiber optic", "No"])
                online_security = st.selectbox("OnlineSecurity", ["Yes", "No", "No internet service"])
                online_backup = st.selectbox("OnlineBackup", ["Yes", "No", "No internet service"])
                device_protection = st.selectbox("DeviceProtection", ["Yes", "No", "No internet service"])
                tech_support = st.selectbox("TechSupport", ["Yes", "No", "No internet service"])

            with col3:
                streaming_tv = st.selectbox("StreamingTV", ["Yes", "No", "No internet service"])
                streaming_movies = st.selectbox("StreamingMovies", ["Yes", "No", "No internet service"])
                contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
                paperless = st.selectbox("PaperlessBilling", ["Yes", "No"])
                payment_method = st.selectbox(
                    "PaymentMethod",
                    [
                        "Electronic check",
                        "Mailed check",
                        "Bank transfer (automatic)",
                        "Credit card (automatic)"
                    ]
                )
                monthly_charges = st.number_input("MonthlyCharges", min_value=0.0, value=70.0)
                total_charges = st.number_input("TotalCharges", min_value=0.0, value=70.0)

            # Build a single-row DataFrame with ALL features
            manual_data = {
                "gender": gender,
                "SeniorCitizen": senior,
                "Partner": partner,
                "Dependents": dependents,
                "tenure": tenure,
                "PhoneService": phone_service,
                "MultipleLines": multiple_lines,
                "InternetService": internet_service,
                "OnlineSecurity": online_security,
                "OnlineBackup": online_backup,
                "DeviceProtection": device_protection,
                "TechSupport": tech_support,
                "StreamingTV": streaming_tv,
                "StreamingMovies": streaming_movies,
                "Contract": contract,
                "PaperlessBilling": paperless,
                "PaymentMethod": payment_method,
                "MonthlyCharges": monthly_charges,
                "TotalCharges": total_charges
            }

            X_single_df = pd.DataFrame([manual_data])

            # Match training columns exactly (order + presence)
            # If any training column missing in manual input, add with default
            for col in feature_cols:
                if col not in X_single_df.columns:
                    X_single_df[col] = 0
            X_single_df = X_single_df[feature_cols]

        if st.button("Predict Churn"):
            preds, probs = predict_for_df(model, X_single_df)
            churn_label = "Yes" if int(preds[0]) == 1 else "No"
            prob_pct = probs[0] * 100

            st.success(f"Predicted Churn: **{churn_label}**")
            st.info(f"Churn Probability: **{prob_pct:.2f}%**")

    # -------------- PAGE 3: BULK PREDICTION ---------
    elif page == "Bulk Prediction (CSV)":
        st.header("📂 Bulk Churn Prediction from CSV")

        st.write(
            "Upload a CSV file with the same columns as the Telco dataset "
            "(it may or may not contain the Churn column)."
        )

        uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

        if uploaded_file is not None:
            # Original uploaded data
            user_df = pd.read_csv(uploaded_file)

            st.subheader("Uploaded Data Preview")
            st.dataframe(user_df.head())

            # Work on a copy that will be aligned with features after dropna, etc.
            work_df = user_df.copy()

            # Build features dataframe
            features_df = work_df.copy()
            for col in ["customerID", "Churn"]:
                if col in features_df.columns:
                    features_df = features_df.drop(columns=[col])

            # Handle TotalCharges cleanly if present
            if "TotalCharges" in features_df.columns:
                features_df["TotalCharges"] = pd.to_numeric(
                    features_df["TotalCharges"], errors="coerce"
                )

            # Drop rows where TotalCharges is NaN (and align work_df accordingly)
            if "TotalCharges" in features_df.columns:
                valid_idx = features_df["TotalCharges"].dropna().index
                features_df = features_df.loc[valid_idx]
                work_df = work_df.loc[valid_idx]
            else:
                valid_idx = features_df.index  # all rows

            # Ensure all training columns exist in features_df
            for col in feature_cols:
                if col not in features_df.columns:
                    # if missing, create default value (0 or "No internet service" etc.)
                    # here we simply fill with 0
                    features_df[col] = 0

            # Reorder columns
            features_df = features_df[feature_cols]

            if st.button("Run Bulk Prediction"):
                preds, probs = predict_for_df(model, features_df)

                result_df = work_df.copy()
                result_df["PredictedChurn"] = np.where(preds == 1, "Yes", "No")
                result_df["ChurnProbability"] = probs

                st.subheader("Prediction Results (First 10 Rows)")
                st.dataframe(result_df.head(10))

                csv_out = result_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="Download Results as CSV",
                    data=csv_out,
                    file_name="churn_predictions.csv",
                    mime="text/csv"
                )

if __name__ == "__main__":
    main()
