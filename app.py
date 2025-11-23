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

# --- REPLACE infer_ranges_from_df and build_input_form WITH THIS SAFER VERSION ---

def _is_integer_like(x):
    """Return True if x is integer-like (e.g. 100.0 or 100 or numpy 100)."""
    try:
        # convert to float first (handles numpy types)
        fx = float(x)
        return fx.is_integer()
    except Exception:
        return False

def _to_python_number(x, prefer_int=False):
    """Convert numpy numbers to python int/float. If prefer_int True and x is integer-like return int."""
    # protect from None / nan
    try:
        if x is None:
            return None
        # convert numpy types to Python scalars
        if isinstance(x, (np.generic,)):
            x = x.item()
        # Now decide int/float
        if prefer_int and _is_integer_like(x):
            return int(round(float(x)))
        # default float
        return float(x)
    except Exception:
        return None

def infer_ranges_from_df(clean_df, features):
    """
    Safer range inference:
    returns dict feature -> (mn, mx, step, default) where mn/mx/step/default are python numbers (int or float)
    """
    ranges = {}
    for f in features:
        # 1) If we have explicit defaults (DEFAULT_RANGES) use them and cast properly
        if f in DEFAULT_RANGES:
            mn_raw, mx_raw = DEFAULT_RANGES[f]
            # decide if integer-like
            prefer_int = _is_integer_like(mn_raw) and _is_integer_like(mx_raw)
            mn = _to_python_number(mn_raw, prefer_int=prefer_int)
            mx = _to_python_number(mx_raw, prefer_int=prefer_int)
            # step: choose int step if integer-like range, else float step
            if prefer_int:
                step = max(int(round((mx - mn) / 100.0)) or 1, 1)
                default = int(round((mn + mx) / 2.0))
            else:
                step = max((mx - mn) / 100.0, 0.01)
                default = (mn + mx) / 2.0
            ranges[f] = (mn, mx, step, default)
            continue

        # 2) Try infer from clean_df
        if clean_df is not None and f in clean_df.columns:
            try:
                col = pd.to_numeric(clean_df[f].dropna(), errors='coerce')
                col = col[~col.isna()]
                if len(col) > 0:
                    mn_raw = float(col.min())
                    mx_raw = float(col.max())
                    prefer_int = _is_integer_like(mn_raw) and _is_integer_like(mx_raw)
                    mn = _to_python_number(mn_raw, prefer_int=prefer_int)
                    mx = _to_python_number(mx_raw, prefer_int=prefer_int)
                    if prefer_int:
                        step = max(int(round((mx - mn) / 100.0)) or 1, 1)
                        default = int(round(float(col.median())))
                    else:
                        step = max((mx - mn) / 100.0, 0.01)
                        default = float(col.median())
                    ranges[f] = (mn, mx, step, default)
                    continue
            except Exception:
                pass

        # 3) Generic fallback (float)
        ranges[f] = (0.0, 100.0, 0.1, 0.0)

    return ranges


def build_input_form(features, clean_df=None):
    """
    Builds the sidebar input form, but enforces consistent numeric types for Streamlit number_input.
    """
    st.sidebar.header("🔧 Input parameters")
    inputs = {}
    ranges = infer_ranges_from_df(clean_df, features)

    for feat in features:
        mn, mx, step, default = ranges.get(feat, (0.0, 100.0, 0.1, 0.0))

        # Ensure python native types
        # Decide if this field should be integer (all values integer-like)
        prefer_int = _is_integer_like(mn) and _is_integer_like(mx) and _is_integer_like(default) and _is_integer_like(step)
        if prefer_int:
            mn_py = _to_python_number(mn, prefer_int=True)
            mx_py = _to_python_number(mx, prefer_int=True)
            step_py = int(_to_python_number(step, prefer_int=True)) if step is not None else 1
            default_py = _to_python_number(default, prefer_int=True)
            # final safe fallback
            if mn_py is None: mn_py = 0
            if mx_py is None: mx_py = mn_py + 100
            if default_py is None: default_py = mn_py
            try:
                val = st.sidebar.number_input(f"{feat} ({mn_py}–{mx_py})", min_value=int(mn_py), max_value=int(mx_py), value=int(default_py), step=int(step_py), format="%d")
            except Exception as e:
                # fallback to float-safe input
                val = st.sidebar.number_input(f"{feat} ({mn_py}–{mx_py})", min_value=float(mn_py), max_value=float(mx_py), value=float(default_py), step=float(step_py), format="%.3f")
        else:
            mn_py = _to_python_number(mn, prefer_int=False) or 0.0
            mx_py = _to_python_number(mx, prefer_int=False) or 100.0
            step_py = _to_python_number(step, prefer_int=False) or 0.1
            default_py = _to_python_number(default, prefer_int=False) if default is not None else (mn_py + mx_py) / 2.0
            # final safe fallback
            if mn_py >= mx_py:
                mx_py = mn_py + abs(mn_py) + 1.0
            try:
                val = st.sidebar.number_input(f"{feat} ({mn_py:.3f}–{mx_py:.3f})", min_value=mn_py, max_value=mx_py, value=default_py, step=step_py, format="%.3f")
            except Exception as e:
                # last-resort coerce to simple float range
                val = st.sidebar.number_input(f"{feat}", value=float(default_py), step=0.1, format="%.3f")

        inputs[feat] = val

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
