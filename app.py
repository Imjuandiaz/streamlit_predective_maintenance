# -*- coding: utf-8 -*-
"""Enhanced Predictive Maintenance Dashboard"""

# app_enhanced.py

import os
import json
import joblib
from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, accuracy_score

st.set_page_config(page_title="Predictive Maintenance Dashboard", layout="wide")
st.title("🧠 Predictive Maintenance Dashboard")
st.markdown("**By Juan Díaz | Dual Model: Classification (Failure) + Regression (MTBF)**")

ARTIFACTS_DIR = Path("artifacts")

# ========= LOAD ARTIFACTS =========
@st.cache_resource
def load_artifacts():
    out = {}
    # Models
    try:
        out['clf'] = joblib.load(ARTIFACTS_DIR / "model_failure_classifier.pkl")
    except:
        out['clf'] = None
    try:
        out['reg'] = joblib.load(ARTIFACTS_DIR / "model_mtbf_randomforest.pkl")
    except:
        out['reg'] = None

    # Features
    try:
        out['features_cls'] = json.load(open(ARTIFACTS_DIR / "features_classifier.json"))
    except:
        out['features_cls'] = None
    try:
        out['features_reg'] = json.load(open(ARTIFACTS_DIR / "features_mtbf.json"))
    except:
        out['features_reg'] = None

    # Importances
    try:
        out['imp_cls'] = pd.read_csv(ARTIFACTS_DIR / "feature_importances_classifier.csv", index_col=0)
    except:
        out['imp_cls'] = None
    try:
        out['imp_reg'] = pd.read_csv(ARTIFACTS_DIR / "feature_importances_mtbf.csv", index_col=0)
    except:
        out['imp_reg'] = None

    # Clean Data
    try:
        out['clean_data'] = pd.read_csv(ARTIFACTS_DIR / "clean_data.csv")
    except:
        out['clean_data'] = None

    return out

ARTS = load_artifacts()

# ========= HELPERS =========
def plot_feature_importances(df_imp, title):
    if df_imp is None:
        st.write("No feature importances found.")
        return
    ser = df_imp.iloc[:,0] if isinstance(df_imp, pd.DataFrame) else pd.Series(df_imp)
    ser = ser.sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(6, max(3, 0.3*len(ser))))
    ser.plot(kind='barh', ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Importance")
    ax.grid(axis='x', linestyle='--', alpha=0.4)
    plt.tight_layout()
    st.pyplot(fig)

DEFAULT_RANGES = {
    "Air temperature [K]": (295,305),
    "Process temperature [K]": (305,315),
    "Rotational speed [rpm]": (1150,2900),
    "Torque [Nm]": (3.8,77),
    "Tool wear [min]": (0,260),
    "Temp_diff": (7.6,13),
    "Wear_per_torque": (0,40)
}

def infer_ranges_from_df(clean_df, features):
    ranges = {}
    for f in features:
        if f in DEFAULT_RANGES:
            mn,mx = DEFAULT_RANGES[f]
            step = max((mx-mn)/100,0.01)
            ranges[f]=(mn,mx,step,(mn+mx)/2)
        else:
            ranges[f]=(0,100,0.1,0)
    return ranges

def build_input_form(features, clean_df=None):
    st.sidebar.header("🔧 Input parameters")
    inputs = {}
    ranges = infer_ranges_from_df(clean_df, features)
    for feat in features:
        mn,mx,step,default = ranges[feat]
        val = st.sidebar.number_input(f"{feat} ({mn}–{mx})", min_value=mn, max_value=mx, value=default, step=step, format="%.3f")
        inputs[feat]=val
    return pd.DataFrame([inputs])


# ========= SIDEBAR =========
st.sidebar.header("⚙️ Select Model Type")
mode = st.sidebar.radio("Choose Model:", ["Classification – Machine Failure", "Regression – MTBF (Time to Failure)"])

features_cls = ARTS.get('features_cls') or []
features_reg = ARTS.get('features_reg') or []

if mode.startswith("Classification"):
    X_input = build_input_form(features_cls, ARTS.get('clean_data'))
else:
    X_input = build_input_form(features_reg, ARTS.get('clean_data'))


# ========= MAIN PREDICTION PANEL =========
st.markdown("## 🔍 Prediction")

# ---------------------------------------------------------
#                  CLASSIFICATION MODEL
# ---------------------------------------------------------
if mode.startswith("Classification"):
    st.subheader("🟠 Classification – Will a failure occur within 30 days?")
    if st.button("🔍 Predict Failure"):
        model = ARTS.get('clf')
        X_in = X_input[features_cls]

        pred = model.predict(X_in)[0]
        proba = model.predict_proba(X_in)[0][1]

        st.metric("Failure Probability", f"{proba*100:.2f}%")
        st.write("Predicted Class:", "⚠️ Failure" if pred==1 else "✅ No Failure")

    st.write("---")
    st.write("### 🔎 Feature Importances (Classification)")
    plot_feature_importances(ARTS.get('imp_cls'), "Classifier feature importances")

    # *** ADDED: CLASSIFICATION ACCURACY ***
    df = ARTS.get('clean_data')
    clf = ARTS.get('clf')
    if df is not None and clf is not None and 'split' in df.columns:
        test_df = df[df['split']=='test']
        if len(test_df)>0 and 'Machine failure' in test_df.columns:
            X_test = test_df[features_cls]
            y_test = test_df['Machine failure']
            y_pred = clf.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            st.success(f"🔵 Classification Accuracy (test): **{acc*100:.2f}%**")
        else:
            st.info("Accuracy unavailable (clean_data.csv missing test split).")
    else:
        st.info("Accuracy unavailable (no clean_data.csv).")


# ---------------------------------------------------------
#                  REGRESSION MODEL
# ---------------------------------------------------------
else:
    st.subheader("🔵 Regression – Estimated time until failure (MTBF)")
    if st.button("⏱️ Predict MTBF"):
        model = ARTS.get('reg')
        X_in = X_input[features_reg]
        prediction = model.predict(X_in)[0]
        st.metric("Estimated Time to Failure", f"{prediction:.2f} minutes")

    st.write("---")
    st.write("### 🔎 Feature Importances (Regression)")
    plot_feature_importances(ARTS.get('imp_reg'), "Regression feature importances")

    # *** ADDED: REGRESSION RMSE ***
    df = ARTS.get('clean_data')
    reg = ARTS.get('reg')
    if df is not None and reg is not None and 'split' in df.columns:
        test_df = df[df['split']=='test']
        if len(test_df)>0 and 'Tool wear [min]' in test_df.columns:
            X_test = test_df[features_reg]
            y_test = test_df['Tool wear [min]']
            y_pred = reg.predict(X_test)
            rmse = mean_squared_error(y_test, y_pred, squared=False)
            st.success(f"🔵 Regression RMSE (test): **{rmse:.2f} minutes**")
        else:
            st.info("RMSE unavailable (clean_data.csv missing test split).")
    else:
        st.info("RMSE unavailable (no clean_data.csv).")


# ========= FOOTER =========
st.markdown("---")
st.caption("© 2025 Juan Díaz — Predictive Maintenance Dashboard | Built with Streamlit & Scikit-learn")
