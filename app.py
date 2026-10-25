"""Streamlit interface for the tourism package purchase propensity model."""

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = Path(__file__).resolve().parent / "model" / "best_model.joblib"
FEATURE_OPTIONS = {
    "TypeofContact": ["Company Invited", "Self Enquiry"],
    "Occupation": ["Salaried", "Small Business", "Large Business", "Free Lancer"],
    "Gender": ["Male", "Female", "Fe Male"],
    "ProductPitched": ["Basic", "Deluxe", "King", "Standard", "Super Deluxe"],
    "MaritalStatus": ["Married", "Single", "Divorced", "Unmarried"],
    "Designation": ["Executive", "Manager", "Senior Manager", "AVP", "VP"],
}
BOOLEAN_FEATURES = {"Passport", "OwnCar"}


@st.cache_resource
def load_model():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Model artifact not found at {MODEL_PATH}. Run the training workflow first."
        )
    return joblib.load(MODEL_PATH)


st.set_page_config(page_title="Visit with Us | Purchase propensity", page_icon="✈️")
st.title("Tourism Package Purchase Propensity")
st.write(
    "Estimate whether a customer may purchase a tourism package based on the "
    "available customer and sales-interaction data."
)
st.info(
    "The source label is `ProdTaken` for a general package purchase, not a "
    "Wellness-specific purchase. This estimate should support—not replace—"
    "responsible marketing judgment."
)

try:
    model = load_model()
except FileNotFoundError as error:
    st.error(str(error))
    st.stop()

feature_names = list(model.feature_names_in_)
numeric_defaults = {
    "Age": 35.0,
    "CityTier": 2,
    "DurationOfPitch": 10.0,
    "NumberOfPersonVisiting": 2,
    "NumberOfFollowups": 3,
    "PreferredPropertyStar": 3.0,
    "NumberOfTrips": 3.0,
    "PitchSatisfactionScore": 3,
    "NumberOfChildrenVisiting": 0.0,
    "MonthlyIncome": 25000.0,
}

with st.form("customer_inputs"):
    st.subheader("Customer profile")
    inputs = {}
    columns = st.columns(2)
    for index, feature in enumerate(feature_names):
        with columns[index % 2]:
            if feature in FEATURE_OPTIONS:
                inputs[feature] = st.selectbox(feature, FEATURE_OPTIONS[feature])
            elif feature in BOOLEAN_FEATURES:
                inputs[feature] = st.selectbox(feature, [0, 1])
            elif feature in numeric_defaults:
                default = numeric_defaults[feature]
                min_value = 0.0 if isinstance(default, float) else 0
                max_value = 100000.0 if feature == "MonthlyIncome" else (
                    100.0 if isinstance(default, float) else 100
                )
                if feature == "CityTier":
                    max_value = 3
                elif feature == "PreferredPropertyStar":
                    max_value = 5.0
                elif feature == "PitchSatisfactionScore":
                    max_value = 5
                inputs[feature] = st.number_input(
                    feature,
                    min_value=min_value,
                    max_value=max_value,
                    value=default,
                )
            else:
                st.error(f"No input control is configured for model feature: {feature}")
    submitted = st.form_submit_button("Estimate purchase propensity")

if submitted:
    customer = pd.DataFrame([{name: inputs[name] for name in feature_names}])
    prediction = int(model.predict(customer)[0])
    probability = float(model.predict_proba(customer)[0, 1])
    st.metric("Estimated probability of package purchase", f"{probability:.1%}")
    if prediction == 1:
        st.success("The model flags this customer as more likely to purchase.")
    else:
        st.info("The model does not flag this customer as likely to purchase.")
    st.caption(
        "The class decision uses the model's default 0.5 probability threshold. "
        "Prediction is an estimate and is not a guarantee of customer behavior."
    )
