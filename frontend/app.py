import os
import pickle

import pandas as pd
import streamlit as st

# ---------- Paths (resolved relative to this file, not the cwd) ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))          # .../frontend
BACKEND_DIR = os.path.join(BASE_DIR, "..", "backend")           # go up, into backend
MODEL_PATH = os.path.normpath(os.path.join(BACKEND_DIR, "LinearRegressionModel.pkl"))
DATA_PATH = os.path.normpath(os.path.join(BACKEND_DIR, "cleaned car.csv"))

# ---------- Page settings ----------
st.set_page_config(
    page_title="Car Price Prediction",
    page_icon="🚗",
    layout="centered",
)


# ---------- Cached loaders ----------
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file not found: {MODEL_PATH}")
        st.stop()
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


@st.cache_data
def load_data():
    if not os.path.exists(DATA_PATH):
        st.error(f"Data file not found: {DATA_PATH}")
        st.stop()
    df = pd.read_csv(DATA_PATH)
    # Drop any stray index column saved from car.to_csv() without index=False
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
    return df


model = load_model()
car = load_data()

# ---------- Title ----------
st.title("🚗 Car Price Prediction")
st.write("Enter the car details to predict its price.")

# ---------- Dropdown values ----------
companies = sorted(car["company"].dropna().unique())
car_models = sorted(car["name"].dropna().unique())
years = sorted(car["year"].dropna().unique(), reverse=True)
fuel_types = sorted(car["fuel_type"].dropna().unique())

# ---------- Input fields ----------
company = st.selectbox("Select Company", companies)
car_model = st.selectbox("Select Car Model", car_models)
year = st.selectbox("Select Year", years)
fuel_type = st.selectbox("Select Fuel Type", fuel_types)
kms_driven = st.number_input(
    "Kilometers Driven",
    min_value=0,
    value=10000,
    step=1000,
)

# ---------- Prediction ----------
if st.button("Predict Price"):
    data = pd.DataFrame(
        [[car_model, company, int(year), int(kms_driven), fuel_type]],
        columns=["name", "company", "year", "kms_driven", "fuel_type"],
    )

    try:
        prediction = model.predict(data)
        price = float(prediction[0])
        price = max(price, 0)  # guard against nonsensical negative output
        st.success(f"Estimated Car Price: ₹ {price:,.2f}")
    except Exception as e:
        st.error(
            "Couldn't generate a prediction for this combination of inputs. "
            f"Details: {e}"
        )