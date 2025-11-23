import os
import pickle
import numpy as np
import streamlit as st

BASE_DIR = os.path.dirname(__file__)


@st.cache_resource
def load_artifacts():
    processed_path = os.path.join(BASE_DIR, "data", "processed", "processed_churn_data.pkl")
    model_path = os.path.join(BASE_DIR, "outputs", "xgb_churn_model.pkl")

    with open(processed_path, "rb") as f:
        data = pickle.load(f)
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    return data, model


def build_single_feature_vector(data, numeric_inputs, categorical_inputs):
    scaler = data["scaler"]
    ohe = data["ohe"]
    num_cols = data["feature_names_num"]
    cat_cols = data["categorical_cols"]

    num_vals = np.array([[numeric_inputs[col] for col in num_cols]], dtype=float)
    num_scaled = scaler.transform(num_vals)

    cat_vals = [[categorical_inputs[col] for col in cat_cols]]
    cat_enc = ohe.transform(cat_vals)

    X = np.hstack([num_scaled, cat_enc])
    return X


def main():
    st.set_page_config(page_title="SaaS Churn Prediction - XGBoost", layout="centered")
    st.title("📉 SaaS Churn Prediction – XGBoost")
    st.write(
        "Interactive churn prediction demo on a synthetic SaaS dataset. "
        "Adjust the sliders and dropdowns to simulate a customer profile and estimate churn probability."
    )

    data, model = load_artifacts()

    st.subheader("Customer Profile")

    col1, col2 = st.columns(2)

    with col1:
        tenure_days = st.slider("Tenure (days since signup)", min_value=1, max_value=365, value=120, step=5)
        sessions_last_30d = st.slider("Sessions in last 30 days", min_value=0, max_value=200, value=30, step=1)
        projects_created = st.slider("Total projects created", min_value=0, max_value=50, value=5, step=1)
        team_size = st.selectbox("Team size", options=[1, 2, 3, 4, 5, 10, 20], index=3)

    with col2:
        support_tickets_last_90d = st.slider("Support tickets (90 days)", min_value=0, max_value=15, value=1, step=1)
        nps_score = st.slider("NPS score", min_value=-100, max_value=100, value=20, step=5)
        monthly_spend = st.slider("Monthly spend ($)", min_value=0.0, max_value=300.0, value=49.0, step=1.0)

    st.subheader("Account Context")
    col3, col4 = st.columns(2)
    with col3:
        country = st.selectbox("Country", options=["DE", "IT", "CH", "TR", "US", "FR", "NL", "SE"], index=0)
        plan_type = st.selectbox("Plan type", options=["free", "pro", "enterprise"], index=1)
    with col4:
        acquisition_channel = st.selectbox("Acquisition channel", options=["ads", "organic", "referral", "partner"], index=1)

    if st.button("Predict churn probability"):
        numeric_inputs = {
            "tenure_days": tenure_days,
            "sessions_last_30d": sessions_last_30d,
            "projects_created": projects_created,
            "team_size": team_size,
            "support_tickets_last_90d": support_tickets_last_90d,
            "nps_score": nps_score,
            "monthly_spend": monthly_spend,
        }
        categorical_inputs = {
            "country": country,
            "plan_type": plan_type,
            "acquisition_channel": acquisition_channel,
        }

        X = build_single_feature_vector(data, numeric_inputs, categorical_inputs)
        proba = model.predict_proba(X)[0, 1]
        pred = model.predict(X)[0]

        st.markdown("---")
        st.subheader("Prediction")
        st.metric(
            "Churn probability",
            f"{proba * 100:.1f}%",
            help="Probability that this account will churn in the near term.",
        )
        if pred == 1:
            st.error("Model prediction: **Likely to churn**")
        else:
            st.success("Model prediction: **Likely to stay**")

        st.markdown("### Suggested actions")
        if proba > 0.7:
            st.write("- High churn risk: consider prioritized outreach, discount, or success call.")
        elif proba > 0.4:
            st.write("- Medium risk: monitor product usage, send targeted educational content.")
        else:
            st.write("- Low risk: keep engagement steady with value-oriented updates.")


if __name__ == "__main__":
    main()
