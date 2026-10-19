"""Credit Risk Prediction: Streamlit app.

Run locally:  streamlit run app.py
"""
import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = Path("model/credit_risk_model.joblib")
INFO_PATH = Path("model/model_info.json")

# The notebook's ml_data.csv stores categories as integers. These mappings assume
# alphabetical label encoding (sklearn's LabelEncoder default). Verify them against
# your cleaning notebook before deploying; see the README checklist.
HOME_OWNERSHIP = {"Mortgage": 0, "Other": 1, "Own": 2, "Rent": 3}
LOAN_INTENT = {
    "Debt consolidation": 0,
    "Education": 1,
    "Home improvement": 2,
    "Medical": 3,
    "Personal": 4,
    "Venture": 5,
}
LOAN_GRADE = {g: i for i, g in enumerate("ABCDEFG")}
DEFAULT_ON_FILE = {"No": 0, "Yes": 1}

# Column order must match the training data exactly
FEATURES = [
    "person_age",
    "person_income",
    "person_home_ownership",
    "person_emp_length",
    "loan_intent",
    "loan_grade",
    "loan_amnt",
    "loan_int_rate",
    "cb_person_default_on_file",
    "cb_person_cred_hist_length",
]


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


def load_threshold() -> float:
    if INFO_PATH.exists():
        return json.loads(INFO_PATH.read_text()).get("threshold", 0.6)
    return 0.6


st.set_page_config(page_title="Credit Risk Predictor", page_icon="💳")
st.title("💳 Credit Risk Predictor")
st.write(
    "Enter an applicant's details to estimate the probability that the loan "
    "will default. Built with a tuned Random Forest trained on historical loan data."
)

if not MODEL_PATH.exists():
    st.error("Model file not found. Run `python train_model.py` first.")
    st.stop()

model = load_model()
threshold = load_threshold()

col1, col2 = st.columns(2)
with col1:
    st.subheader("Applicant")
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    income = st.number_input("Annual income ($)", min_value=0, value=50000, step=1000)
    home = st.selectbox("Home ownership", list(HOME_OWNERSHIP))
    emp_length = st.number_input(
        "Employment length (years)", min_value=0, max_value=60, value=5
    )
    cred_hist = st.number_input(
        "Credit history length (years)", min_value=0, max_value=60, value=5
    )
    prior_default = st.selectbox("Previous default on file?", list(DEFAULT_ON_FILE))

with col2:
    st.subheader("Loan")
    amount = st.number_input("Loan amount ($)", min_value=500, value=10000, step=500)
    intent = st.selectbox("Loan purpose", list(LOAN_INTENT))
    grade = st.selectbox("Loan grade (A best, G worst)", list(LOAN_GRADE))
    # The training data stores whole-number interest rates
    rate = st.number_input(
        "Interest rate (%)", min_value=1, max_value=30, value=11, step=1
    )

if st.button("Predict default risk", type="primary"):
    row = pd.DataFrame(
        [
            [
                age,
                income,
                HOME_OWNERSHIP[home],
                emp_length,
                LOAN_INTENT[intent],
                LOAN_GRADE[grade],
                amount,
                rate,
                DEFAULT_ON_FILE[prior_default],
                cred_hist,
            ]
        ],
        columns=FEATURES,
    )
    prob = float(model.predict_proba(row)[0, 1])

    st.metric("Estimated probability of default", f"{prob:.1%}")
    st.progress(min(prob, 1.0))
    if prob >= threshold:
        st.error(f"High risk: probability is at or above the {threshold:.0%} cut-off.")
    else:
        st.success(f"Lower risk: probability is below the {threshold:.0%} cut-off.")

st.caption(
    "Portfolio demo only. Predictions come from a model trained on a public "
    "dataset and must not be used for real lending decisions."
)
