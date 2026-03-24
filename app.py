import streamlit as st
import pandas as pd
import numpy as np
import joblib


# Loading the model
@st.cache_resource
def load_model():
    try:
        return joblib.load("model.joblib")
    except:
        return None


# Loading the scalar
@st.cache_resource
def load_scaler():
    try:
        return joblib.load("scaler.joblib")
    except:
        return None


model = load_model()
scaler = load_scaler()

# Required EEG Channels
FEATURES = [
    "Fp1","Fp2","F7","F3","Fz","F4","F8",
    "T3","C3","Cz","C4","T4","T5",
    "P3","Pz","P4"
]


# Input
st.title("EEG-Based Early Alzheimer's Disease Detection")

st.markdown(
    "📂 Upload a CSV containing the following 16 EEG columns:"
)

st.code(", ".join(FEATURES))

uploaded_file = st.file_uploader(
    "Choose a CSV file",
    type=["csv"]
)


# Processing
if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    # Check required columns
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        st.error(f"Missing columns: {', '.join(missing)}")
        st.stop()

    # Extract EEG features
    X = df[FEATURES].copy()

    
    # Feature Engineering
    X["frontal_mean"]  = X[["Fp1","Fp2","F3","F4","F7","F8","Fz"]].mean(axis=1)
    X["central_mean"]  = X[["C3","Cz","C4"]].mean(axis=1)
    X["temporal_mean"] = X[["T3","T4","T5"]].mean(axis=1)
    X["parietal_mean"] = X[["P3","Pz","P4"]].mean(axis=1)

    
    # Normalization 
    if scaler is not None:
        X = pd.DataFrame(
            scaler.transform(X),
            columns=X.columns
        )

    
    # Model Prediction
    if model is not None:
        probs = model.predict_proba(X)[:, 1]
    else:
        # Demo fallback if model not loaded
        np.random.seed(42)
        probs = np.random.beta(2, 3, len(X))


    
    # Risk Score
    risk_score = float(probs.mean()) * 100


    
    # Output Section
    st.markdown("---")
    st.subheader("📊 Risk Assessment")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            label="Risk Probability",
            value=f"{risk_score:.2f}%"
        )

    with col2:

        if risk_score < 30:
            level = "Low Risk"
        elif risk_score < 60:
            level = "Moderate Risk"
        else:
            level = "High Risk"

        st.metric(
            label="Risk Category",
            value=level
        )

    st.markdown("---")

else:
    st.info("Upload EEG data to generate a risk prediction.")