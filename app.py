# app_enhanced.py
# Predictive Maintenance Dashboard - Enhanced version
# By Juan Díaz (enhanced)
# Replaces / extends original app.py: adds metrics, input validation, CSV batch, survival note, improved plots

import os
import json
import joblib
from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import io
import math
from datetime import datetime

from sklearn.metrics import (
    f1_score, precision_recall_curve, average_precision_score,
    confusion_matrix, classification_report,
    mean_absolute_error, mean_squared_error, r2_score, accuracy_score
)

# ====== CONFIG ======
st.set_page_config(page_title="Predictive Maintenance Dashboard", layout="wide")
st.title("🧠 Predictive Maintenance Dashboard")
st.markdown("**By Juan Díaz | Dual Model: Classification (Failure) + Regression (MTBF)**")
ARTIFACTS_DIR = Path("artifacts")

# ====== UTIL: load artifacts safely ======
@st.cache_resource
def load_artifacts():
    out = {}
    # models
    try:
        out['clf'] = joblib.load(ARTIFACTS_DIR / "model_failure_classifier.pkl")
    except Exception as e:
        out['clf'] = None
        st.error(f"Warning: cannot load classifier model: {e}")
    try:
        out['reg'] = joblib.load(ARTIFACTS_DIR / "model_mtbf_randomforest.pkl")
    except Exception as e:
        out['reg'] = None
        st.error(f"Warning: cannot load regressor model: {e}")
    # features
    try:
        out['features_cls'] = json.load(open(ARTIFACTS_DIR / "features_classifier.json"))
    except Exception:
        out['features_cls'] = None
    try:
        out['features_reg'] = json.load(open(ARTIFACTS_DIR / "features_mtbf.json"))
    except Exception:
        out['features_reg'] = None
    # importances
    try:
        out['imp_cls'] = pd.read_csv(ARTIFACTS_DIR / "feature_importances_classifier.csv", index_col=0)
    except Exception:
        out['imp_cls'] = None
    try:
        out['imp_reg'] = pd.read_csv(ARTIFACTS_DIR / "feature_importances_mtbf.csv", index_col=0)
    except Exception:
        out['imp_reg'] = None
    # clean data (optional)
    try:
        out['clean_data'] = pd.read_csv(ARTIFACTS_DIR / "clean_data.csv")
    except Exception:
        out['clean_data'] = None
    return out

ARTS = load_artifacts()

# ====== HELPERS ======
def plot_feature_importances(df_imp, title="Feature importances"):
    if df_imp is None:
        st.write("No feature importances found.")
        return
    # ensure series
    if isinstance(df_imp, pd.DataFrame):
        try:
            ser = df_imp.iloc[:,0]
        except Exception:
            ser = pd.Series(df_imp.values.flatten(), index=df_imp.index)
    else:
        ser = pd.Series(df_imp)
    ser = ser.sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(6, max(3, 0.3*len(ser))))
    ser.plot(kind='barh', ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Importance")
    ax.grid(axis='x', linestyle='--', alpha=0.4)
    plt.tight_layout()
    st.pyplot(fig)

# ====== METRICS HELPERS ======
def safe_to_numpy(y):
    try:
        return np.asarray(y).astype(float)
    except Exception:
        return np.asarray(y)

def show_single_prediction_metrics(class_true, class_pred, reg_true, reg_pred):
    """
    Display metrics for a single-row case if the true labels are provided (non-None).
    class_true / class_pred: scalar or None
    reg_true / reg_pred: scalar or None
    """
    # Classification metrics (only meaningful if we have a true label)
    if class_true is not None:
        try:
            acc = accuracy_score([class_true], [class_pred])
            f1 = f1_score([class_true], [class_pred], zero_division=0)
            st.write(f"**Classification (single) — Accuracy:** {acc*100:.2f}%  •  **F1:** {f1:.3f}")
        except Exception as e:
            st.info("Could not compute single classification metrics: " + str(e))

    # Regression metrics (only meaningful if we have true reg value)
    if reg_true is not None:
        try:
            rmse = mean_squared_error([reg_true], [reg_pred], squared=False)
            st.write(f"**Regression (single) — RMSE:** {rmse:.3f} minutes")
        except Exception as e:
            st.info("Could not compute single regression metrics: " + str(e))

# CodeSnippet UPDATE_RANGES
# Replace the previous infer_ranges_from_df with this version that uses explicit defaults when provided.
DEFAULT_RANGES = {
    "Air temperature [K]":      (295.0, 305.0),
    "Process temperature [K]":  (305.0, 315.0),
    "Rotational speed [rpm]":  (1150.0, 2900.0),
    "Torque [Nm]":             (3.8, 77.0),
    "Tool wear [min]":         (0.0, 260.0),
    "Temp_diff":               (7.6, 13.0),
    "Wear_per_torque":         (0.0, 40.0)
}

def infer_ranges_from_df(clean_df, features):
    """
    Returns a dict: feature -> (min, max, step, default)
    Uses DEFAULT_RANGES when available; otherwise tries to infer from clean_df;
    otherwise uses reasonable fallbacks.
    """
    ranges = {}
    for f in features:
        # 1) If we have an explicit default range, use it
        if f in DEFAULT_RANGES:
            mn, mx = DEFAULT_RANGES[f]
            step = max((mx - mn) / 100.0, 0.01)
            default = float((mn + mx) / 2.0)
            ranges[f] = (mn, mx, step, default)
            continue

        # 2) Try infer from clean_df if provided
        if clean_df is not None and f in clean_df.columns:
            try:
                col = clean_df[f].dropna().astype(float)
                if len(col) > 0:
                    mn, mx = float(col.min()), float(col.max())
                    step = max((mx - mn) / 100.0, 0.01)
                    default = float(col.median())
                    ranges[f] = (mn, mx, step, default)
                    continue
            except Exception:
                pass

        # 3) Fallback generic values
        ranges[f] = (0.0, 100.0, 0.1, 0.0)

    return ranges
# End CodeSnippet UPDATE_RANGES


def build_input_form(features, clean_df=None):
    st.sidebar.header("🔧 Input parameters")
    inputs = {}
    ranges = infer_ranges_from_df(clean_df, features)
    for feat in features:
        mn, mx, step, default = ranges.get(feat, (0.0, 100.0, 0.1, 0.0))
        try:
            val = st.sidebar.number_input(f"{feat} ({mn:.2f}–{mx:.2f})", value=float(default), min_value=mn, max_value=mx, step=float(step), format="%.3f")
        except Exception:
            val = st.sidebar.number_input(f"{feat}", value=0.0, step=0.1, format="%.3f")
        inputs[feat] = val
    return pd.DataFrame([inputs])

def show_prediction_note_reg(pred_minutes):
    hours = pred_minutes/60.0
    days = pred_minutes/1440.0
    st.markdown("### 📘 Interpretación de la predicción (Regresión)")
    st.write(f"- Estimated time until failure: **{pred_minutes:.1f} minutes** ({hours:.1f} hours, {days:.2f} days).")
    mean = max(pred_minutes, 1.0)
    t_points = np.array([60, 60*24, 60*24*7])
    probs = np.exp(-t_points/mean)
    st.write("- Approx. survival probability (simple exp. model):")
    st.write(f"  - After 1 hour: {probs[0]*100:.1f}%")
    st.write(f"  - After 1 day: {probs[1]*100:.1f}%")
    st.write(f"  - After 7 days: {probs[2]*100:.1f}%")
    st.caption("Nota: esto es una aproximación simple. Para curvas de supervivencia precisas hay que modelar tiempos históricos.")

def show_prediction_note_cls(prob_pos, pred_class):
    st.markdown("### 📘 Interpretación de la predicción (Clasificación)")
    st.write(f"- Predicted class: **{'Failure' if pred_class==1 else 'No Failure'}**")
    st.write(f"- Probability of failure: **{prob_pos*100:.2f}%**")
    if prob_pos >= 0.7:
        st.warning("⚠️ High probability of failure — recommend inspection.")
    elif prob_pos >= 0.3:
        st.info("🟡 Medium risk — monitor and consider preventive measures.")
    else:
        st.success("✅ Low risk — normal conditions.")

def batch_predict_and_download(model, features, uploaded_file, is_classification=True):
    if uploaded_file is None:
        st.info("Upload a CSV with the feature columns to run batch predictions.")
        return
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error("Error reading CSV: " + str(e))
        return
    missing = [c for c in features if c not in df.columns]
    if missing:
        st.error(f"The CSV is missing required columns: {missing}")
        return

    # Limit large files for safety
    MAX_ROWS = 20000
    if len(df) > MAX_ROWS:
        st.warning(f"CSV has {len(df)} rows. Processing first {MAX_ROWS} rows only.")
        df = df.iloc[:MAX_ROWS].copy()

    X = df[features].apply(pd.to_numeric, errors='coerce').fillna(0.0)
    preds = model.predict(X)

    if is_classification:
        df["pred"] = preds.astype(int)
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X)
            df["pred_prob_pos"] = probs[:,1]
        # If truth column exists, compute metrics
        if "Machine failure" in df.columns:
            try:
                y_true = df["Machine failure"].astype(int)
                y_pred = df["pred"].astype(int)
                acc = accuracy_score(y_true, y_pred)
                f1 = f1_score(y_true, y_pred, zero_division=0)
                st.success(f"Batch classification — Accuracy: {acc*100:.2f}%, F1: {f1:.3f}")
            except Exception as e:
                st.info("Could not compute batch classification metrics: " + str(e))
    else:
        df["pred_minutes"] = preds.astype(float)
        df["pred_hours"] = df["pred_minutes"] / 60.0
        # If truth column exists, compute RMSE
        if "Tool wear [min]" in df.columns:
            try:
                y_true_reg = pd.to_numeric(df["Tool wear [min]"], errors='coerce')
                y_pred_reg = df["pred_minutes"].astype(float)
                mask = ~y_true_reg.isna()
                if mask.sum() > 0:
                    rmse = mean_squared_error(y_true_reg[mask], y_pred_reg[mask], squared=False)
                    st.success(f"Batch regression — RMSE: {rmse:.3f} minutes (computed on {mask.sum()} rows)")
                else:
                    st.info("Tool wear column found but contains no valid numeric values to compute RMSE.")
            except Exception as e:
                st.info("Could not compute batch regression RMSE: " + str(e))

    st.dataframe(df.head(20))
    csv_out = df.to_csv(index=False).encode('utf-8')
    st.download_button("Download predictions CSV", data=csv_out, file_name="predictions.csv", mime="text/csv")

def show_trends_if_available(clean_df):
    if clean_df is None:
        st.info("No historical data found for trends.")
        return
    time_cols = [c for c in clean_df.columns if 'time' in c.lower() or 'date' in c.lower() or 'timestamp' in c.lower()]
    if not time_cols:
        st.info("No timestamp column found in clean_data.csv to show trends.")
        return
    tcol = time_cols[0]
    df = clean_df.copy()
    df[tcol] = pd.to_datetime(df[tcol], errors='coerce')
    df = df.dropna(subset=[tcol])
    df = df.sort_values(tcol)
    if 'Tool wear [min]' in df.columns:
        fig, ax = plt.subplots()
        ax.plot(df[tcol], df['Tool wear [min]'], marker='o', linestyle='-')
        ax.set_title("Tool wear over time")
        ax.set_ylabel("Tool wear [min]")
        ax.set_xlabel(tcol)
        st.pyplot(fig)
    else:
        st.info("No 'Tool wear [min]' found to chart.")

# ====== LAYOUT / SIDEBAR ======
st.sidebar.header("⚙️ Select Model Type")
mode = st.sidebar.radio("Choose Model:", ["Classification – Machine Failure", "Regression – MTBF (Time to Failure)"])

# Determine features for forms
features_cls = ARTS.get('features_cls') or []
features_reg = ARTS.get('features_reg') or []

# Build input form based on selection
if mode.startswith("Classification"):
    X_input = build_input_form(features_cls, clean_df=ARTS.get('clean_data'))
else:
    X_input = build_input_form(features_reg, clean_df=ARTS.get('clean_data'))

# ====== MAIN PANEL: Model performance + Features ======
st.sidebar.markdown("---")
if st.sidebar.button("Show model performance"):
    st.subheader("📈 Model performance")
    # Try show metrics from clean_data if present
    df = ARTS.get('clean_data')
    clf = ARTS.get('clf')
    reg = ARTS.get('reg')
    if df is not None:
        # Use test split if exists, otherwise fallback to using the whole dataset
        if 'split' in df.columns:
            test_df = df[df['split'] == 'test']
        else:
            test_df = df.copy()

        with st.expander("Show dataset head"):
            st.dataframe(test_df.head(10))

        # ===== CLASSIFICATION METRICS =====
        if clf and features_cls and 'Machine failure' in test_df.columns:
            try:
                X_test_cls = test_df[features_cls].apply(pd.to_numeric, errors='coerce').fillna(0.0)
                y_test_cls = test_df['Machine failure'].astype(int)
                y_pred = clf.predict(X_test_cls)
                y_proba = clf.predict_proba(X_test_cls)[:,1] if hasattr(clf, "predict_proba") else None

                acc = accuracy_score(y_test_cls, y_pred)
                f1 = f1_score(y_test_cls, y_pred, zero_division=0)

                st.write("**Classification**")
                st.write(f"- Accuracy (test): **{acc*100:.2f}%**")
                st.write(f"- F1 (test): **{f1:.3f}**")
                if y_proba is not None:
                    pr_auc = average_precision_score(y_test_cls, y_proba)
                    st.write(f"- PR-AUC (test): **{pr_auc:.3f}**")

                cm = confusion_matrix(y_test_cls, y_pred)
                fig, ax = plt.subplots()
                im = ax.imshow(cm, cmap="Blues")
                ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
                for (i, j), v in np.ndenumerate(cm):
                    ax.text(j, i, f"{v}", ha='center', va='center', color='white' if v>cm.max()/2 else 'black')
                st.pyplot(fig)

            except Exception as e:
                st.error("Error computing classification metrics: " + str(e))
        else:
            st.info("Classification metrics unavailable (need 'Machine failure' column and classifier loaded).")

        # ===== REGRESSION METRICS =====
        if reg and features_reg and 'Tool wear [min]' in test_df.columns:
            try:
                X_test_reg = test_df[features_reg].apply(pd.to_numeric, errors='coerce').fillna(0.0)
                y_test_reg = pd.to_numeric(test_df['Tool wear [min]'], errors='coerce')
                mask = ~y_test_reg.isna()
                if mask.sum() > 0:
                    y_pred_reg = reg.predict(X_test_reg[mask])
                    mae = mean_absolute_error(y_test_reg[mask], y_pred_reg)
                    rmse = mean_squared_error(y_test_reg[mask], y_pred_reg, squared=False)
                    r2 = r2_score(y_test_reg[mask], y_pred_reg)
                    st.write("**Regression (MTBF)**")
                    st.write(f"- MAE (test): **{mae:.2f}** minutes")
                    st.write(f"- RMSE (test): **{rmse:.2f}** minutes")
                    st.write(f"- R² (test): **{r2:.3f}**")
                    # Plot predicted vs actual
                    fig, ax = plt.subplots()
                    ax.scatter(y_test_reg[mask], y_pred_reg)
                    ax.plot([y_test_reg[mask].min(), y_test_reg[mask].max()],
                            [y_test_reg[mask].min(), y_test_reg[mask].max()], '--')
                    ax.set_xlabel("Actual")
                    ax.set_ylabel("Predicted")
                    ax.set_title("Prediction vs Actual")
                    st.pyplot(fig)
                else:
                    st.info("Tool wear column found but contains no valid numeric values to compute regression metrics.")
            except Exception as e:
                st.error("Error computing regression metrics: " + str(e))
        else:
            st.info("Regression metrics unavailable (need 'Tool wear [min]' column and regressor loaded).")

    else:
        st.info("No clean_data.csv found in artifacts — upload a test CSV to compute metrics here.")


# ====== PREDICTION PANEL ======
st.markdown("## 🔍 Prediction")

if mode.startswith("Classification"):
    st.subheader("🟠 Classification – Will failure occur within 30 days?")
    st.write("Enter current machine parameters and press Predict Failure.")
    if st.button("🔍 Predict Failure"):
        model = ARTS.get('clf')
        if model is None:
            st.error("Classifier model not loaded.")
        else:
            try:
                # ensure same column order
                X_in = X_input[features_cls]
                # Optional: let user input the true label for the current row (for evaluation)
                st.sidebar.markdown("### (Optional) True labels for this input")
                true_class_input = st.sidebar.selectbox("True Machine failure (if known)", options=[None, 0, 1], index=0, format_func=lambda x: "None" if x is None else str(x))
                # predict
                pred_proba = model.predict_proba(X_in)[0][1] if hasattr(model, "predict_proba") else None
                pred_class = model.predict(X_in)[0]
                if pred_proba is not None:
                    st.metric("Failure Probability (%)", f"{pred_proba*100:.2f}")
                else:
                    st.write("Predicted class:", pred_class)
                st.write("Predicted Class:", "⚠️ Failure" if pred_class==1 else "✅ No Failure")
                show_prediction_note_cls(pred_proba if pred_proba is not None else 0.0, pred_class)
                # Show single prediction metrics if user provided the true class
                if true_class_input is not None:
                    true_val = int(true_class_input)
                    show_single_prediction_metrics(class_true=true_val, class_pred=int(pred_class), reg_true=None, reg_pred=None)
            except Exception as e:
                st.error(f"Prediction error: {e}")
    st.write("---")
    st.write("### 🔎 Feature Importances (Classification)")
    plot_feature_importances(ARTS.get('imp_cls'), title="Classifier feature importances")
    # CSV batch
    st.write("---")
    st.subheader("📤 Batch predictions (classification)")
    uploaded_file = st.file_uploader("Upload CSV with feature columns", type=["csv"], key="cls_batch")
    if st.button("Run batch classification"):
        batch_predict_and_download(ARTS.get('clf'), features_cls, uploaded_file, is_classification=True)
else:
    st.subheader("🔵 Regression – Estimated time until failure (MTBF)")
    st.write("Enter current machine parameters and press Predict MTBF.")
    if st.button("⏱️ Predict MTBF"):
        model = ARTS.get('reg')
        if model is None:
            st.error("Regression model not loaded.")
        else:
            try:
                X_in = X_input[features_reg]
                # Optional: let user input the true tool-wear time (minutes) for this row
                st.sidebar.markdown("### (Optional) True labels for this input")
                true_reg_input = st.sidebar.number_input("True Tool wear [min] (if known)", value=float("nan"))
                has_true_reg = not np.isnan(true_reg_input)
                prediction = model.predict(X_in)[0]
                st.metric("Estimated Time to Failure (minutes)", f"{prediction:.2f}")
                show_prediction_note_reg(prediction)
                # Show single prediction regression metric if user provided true value
                if has_true_reg:
                    show_single_prediction_metrics(class_true=None, class_pred=None, reg_true=float(true_reg_input), reg_pred=float(prediction))
            except Exception as e:
                st.error(f"Prediction error: {e}")
    st.write("---")
    st.write("### 🔎 Feature Importances (Regression)")
    plot_feature_importances(ARTS.get('imp_reg'), title="Regression feature importances")
    st.write("---")
    st.subheader("📤 Batch predictions (regression)")
    uploaded_file_reg = st.file_uploader("Upload CSV with feature columns", type=["csv"], key="reg_batch")
    if st.button("Run batch regression"):
        batch_predict_and_download(ARTS.get('reg'), features_reg, uploaded_file_reg, is_classification=False)

# ====== TRENDS & HISTORY ======
st.markdown("---")
st.subheader("📊 Historical trends (if available)")
show_trends_if_available(ARTS.get('clean_data'))

# ====== FOOTER ======
st.markdown("---")
st.caption("© 2025 Juan Díaz — Predictive Maintenance Dashboard | Built with Streamlit & Scikit-learn")
