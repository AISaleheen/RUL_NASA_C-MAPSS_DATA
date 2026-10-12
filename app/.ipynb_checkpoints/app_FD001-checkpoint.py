import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
from streamlit.web import cli as stcli

# ----------------------------------------------------
# 1. App Configuration & Page Layout
# ----------------------------------------------------
st.set_page_config(page_title="Turbofan Fleet Health Monitor", layout="wide")

st.title("Turbofan Fleet Predictive Maintenance Dashboard")
st.markdown("NASA C-MAPSS FD001 — Remaining Useful Life (RUL) Inference Engine")

# ----------------------------------------------------
# 2. Data Ingestion & Caching
# ----------------------------------------------------
@st.cache_data
def load_data():
    cols = ["unit_nr", "time_cycles", "setting_1", "setting_2", "setting_3"] + [f"sensor_{i}" for i in range(1, 22)]
    test_df = pd.read_csv("../data/test_FD001.txt", sep=r"\s+", header=None, names=cols)
    rul_df = pd.read_csv("../data/RUL_FD001.txt", sep=r"\s+", header=None, names=["true_RUL"])
    
    # Load previously exported test predictions DataFrame
    # (Generated in your benchmark step)
    preds_df = pd.read_csv("../data/test_predictions_FD001.csv")
    return test_df, rul_df, preds_df

try:
    test_raw, rul_true, predictions_df = load_data()
except Exception:
    st.error("Data files not found. Ensure test_FD001.txt, RUL_FD001.txt, and test_predictions_FD001.csv exist in ../data/")
    st.stop()

# ----------------------------------------------------
# 3. Sidebar Controls
# ----------------------------------------------------
st.sidebar.header("Fleet Control Panel")
unit_list = sorted(test_raw["unit_nr"].unique())
selected_unit = st.sidebar.selectbox("Select Test Engine Unit:", unit_list, index=30)  # Default Unit 31

# ----------------------------------------------------
# 4. Key Performance Indicators (KPIs)
# ----------------------------------------------------
unit_telemetry = test_raw[test_raw["unit_nr"] == selected_unit]
cycles_recorded = unit_telemetry["time_cycles"].max()

true_rul = rul_true.iloc[selected_unit - 1]["true_RUL"]
clipped_true = min(true_rul, 125)

# 1. Model Selector in Sidebar
available_models = [c for c in predictions_df.columns if c.startswith("pred_")]
default_idx = available_models.index("pred_lstm_ensemble") if "pred_lstm_ensemble" in available_models else 0
selected_model_col = st.sidebar.selectbox("Select Inference Model:", available_models, index=default_idx)
# 2. Extract predicted value dynamically from selected model
predicted_rul = float(predictions_df.loc[predictions_df["unit_nr"] == selected_unit, selected_model_col].values[0])
pred_error = predicted_rul - clipped_true


# Extract predicted value for selected unit
# pred_col = [c for c in predictions_df.columns if "lstm" in c or "xgboost" in c or "pred" in c][0]
# predicted_rul = float(predictions_df.loc[predictions_df["unit_nr"] == selected_unit, pred_col].values[0])
# pred_error = predicted_rul - clipped_true

# Determine Operational Status Badge
if predicted_rul <= 25:
    status_label = "CRITICAL: Ground for Teardown"
    status_color = "red"
elif predicted_rul <= 50:
    status_label = "WARNING: Schedule Depot Inspection"
    status_color = "orange"
else:
    status_label = "HEALTHY: Cleared for Flight"
    status_color = "green"

col1, col2, col3, col4 = st.columns(4)
# col1.metric("Recorded Cycles", f"{cycles_recorded} cycles")
# col2.metric("Predicted RUL", f"{predicted_rul:.1f} cycles", delta=f"{pred_error:+.1f} error")
# col3.metric("True Ground RUL", f"{clipped_true} cycles")
col1.metric("Recorded Cycles", f"{cycles_recorded}")
col2.metric("Predicted RUL (Cycles)", f"{predicted_rul:.1f}", delta=f"{pred_error:+.1f} error")
col3.metric("True Ground RUL (Cycles)", f"{clipped_true}")

col4.markdown(f"**Fleet Status**  \n:{status_color}[**{status_label}**]")

st.divider()

# ----------------------------------------------------
# 5. Sensor Telemetry Visualization
# ----------------------------------------------------
st.subheader(f"Engine {selected_unit}: Key Degradation Telemetry")

fig, axes = plt.subplots(1, 3, figsize=(15, 3.5))

# Sensor 9: HPC Outlet Pressure / Core Speed
axes[0].plot(unit_telemetry["time_cycles"], unit_telemetry["sensor_9"], color="tab:blue")
axes[0].set_title("Sensor 9 (Core Speed Proxy)")
axes[0].set_xlabel("Time Cycles")
axes[0].grid(True, linestyle=":", alpha=0.6)

# Sensor 11: LPT Outlet Temperature
axes[1].plot(unit_telemetry["time_cycles"], unit_telemetry["sensor_11"], color="tab:orange")
axes[1].set_title("Sensor 11 (LPT Outlet Temp)")
axes[1].set_xlabel("Time Cycles")
axes[1].grid(True, linestyle=":", alpha=0.6)

# Sensor 14: LPT Outlet Static Pressure
axes[2].plot(unit_telemetry["time_cycles"], unit_telemetry["sensor_14"], color="tab:green")
axes[2].set_title("Sensor 14 (LPT Outlet Pressure)")
axes[2].set_xlabel("Time Cycles")
axes[2].grid(True, linestyle=":", alpha=0.6)

plt.tight_layout()
st.pyplot(fig)


# if __name__ == "__main__":
#     if st._is_running_with_streamlit:
#         # Code executed when launched via streamlit
#         pass
#     else:
#         # Allows running directly with: python app.py
#         sys.argv = ["streamlit", "run", sys.argv[0]]
#         sys.exit(stcli.main())